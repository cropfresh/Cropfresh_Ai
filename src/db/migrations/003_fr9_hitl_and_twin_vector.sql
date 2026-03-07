-- ═══════════════════════════════════════════════════════════════
-- Migration 003 — FR9: Digital Twin confidence vector + HITL review queue
-- ═══════════════════════════════════════════════════════════════
-- FR9: The system packages and persists vision inference outputs
--      (raw image + bounding boxes + DINOv2 confidence vector) as
--      a reproducible, immutable Digital Twin.
--      AgentStateManager triggers HITL review when:
--        confidence < 0.70  OR  grade = A+  OR  defect_count > 3
-- ═══════════════════════════════════════════════════════════════

-- ! APPLY AFTER 002_business_tables.sql

-- ─────────────────────────────────────────────────────────────
-- 1. Digital Twin — DINOv2 confidence vector column
--    Stores the raw softmax probability vector [p_A+, p_A, p_B, p_C]
--    output by DINOv2 ViT-S/14 for reproducible audit.
-- ─────────────────────────────────────────────────────────────
ALTER TABLE digital_twins
    ADD COLUMN IF NOT EXISTS dinov2_confidence_vector FLOAT[] DEFAULT '{}';

COMMENT ON COLUMN digital_twins.dinov2_confidence_vector IS
    'DINOv2 ViT-S/14 softmax output [p_A+, p_A, p_B, p_C] — immutable after creation';

-- ─────────────────────────────────────────────────────────────
-- 2. Digital Twin — Immutability enforcement
--    Blocks any UPDATE on core quality columns once written.
--    Only status / agent_id administrative fields can be changed.
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION digital_twins_immutable_guard()
RETURNS TRIGGER AS $$
BEGIN
    -- * Core evidence columns are write-once — block any UPDATE
    IF (
        NEW.farmer_photos       IS DISTINCT FROM OLD.farmer_photos OR
        NEW.agent_photos        IS DISTINCT FROM OLD.agent_photos  OR
        NEW.ai_annotations      IS DISTINCT FROM OLD.ai_annotations OR
        NEW.grade               IS DISTINCT FROM OLD.grade         OR
        NEW.confidence          IS DISTINCT FROM OLD.confidence    OR
        NEW.defect_types        IS DISTINCT FROM OLD.defect_types  OR
        NEW.dinov2_confidence_vector IS DISTINCT FROM OLD.dinov2_confidence_vector OR
        NEW.listing_id          IS DISTINCT FROM OLD.listing_id
    ) THEN
        RAISE EXCEPTION
            'digital_twins are immutable evidence records — field "%" cannot be updated after creation (twin_id=%)',
            TG_ARGV[0], OLD.id
            USING ERRCODE = 'integrity_constraint_violation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop and recreate so re-running migration is idempotent
DROP TRIGGER IF EXISTS trg_digital_twins_immutable ON digital_twins;

CREATE TRIGGER trg_digital_twins_immutable
    BEFORE UPDATE ON digital_twins
    FOR EACH ROW EXECUTE FUNCTION digital_twins_immutable_guard();

-- ─────────────────────────────────────────────────────────────
-- 3. HITL Review Queue
--    Tracks every HITL review request raised by the AI pipeline.
--    One row per trigger event; linked to listing and optional agent.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hitl_review_queue (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    listing_id          UUID REFERENCES listings(id) ON DELETE CASCADE NOT NULL,

    -- * Trigger reason — at least one of these conditions was true
    trigger_reason      TEXT NOT NULL
        CHECK (trigger_reason IN ('confidence_low', 'grade_a_plus', 'defects_high', 'combined')),

    -- * Snapshot of the values that triggered the review
    confidence          FLOAT CHECK (confidence >= 0 AND confidence <= 1),
    grade               TEXT,
    defect_count        INT DEFAULT 0 CHECK (defect_count >= 0),

    -- * Review workflow
    status              TEXT DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_review', 'approved', 'rejected', 'auto_resolved')),
    assigned_agent_id   UUID REFERENCES field_agents(id) ON DELETE SET NULL,
    notes               TEXT DEFAULT '',
    resolution          TEXT,                       -- Human resolution text

    -- * Timestamps
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now(),
    resolved_at         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_hitl_listing  ON hitl_review_queue (listing_id);
CREATE INDEX IF NOT EXISTS idx_hitl_status   ON hitl_review_queue (status);
CREATE INDEX IF NOT EXISTS idx_hitl_created  ON hitl_review_queue (created_at DESC);

COMMENT ON TABLE hitl_review_queue IS
    'HITL (Human-In-The-Loop) review requests raised when AI quality confidence is insufficient (FR9)';

-- ─────────────────────────────────────────────────────────────
-- 4. Auto-update updated_at on hitl_review_queue rows
-- ─────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_hitl_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_hitl_updated_at ON hitl_review_queue;

CREATE TRIGGER trg_hitl_updated_at
    BEFORE UPDATE ON hitl_review_queue
    FOR EACH ROW EXECUTE FUNCTION update_hitl_updated_at();
