-- Sprint 06: ADCL district-first persistence contract

ALTER TABLE adcl_reports
    ADD COLUMN IF NOT EXISTS district TEXT DEFAULT 'Bangalore',
    ADD COLUMN IF NOT EXISTS generated_at TIMESTAMPTZ DEFAULT now(),
    ADD COLUMN IF NOT EXISTS summary_en TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS summary_hi TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS summary_kn TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS freshness JSONB DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS source_health JSONB DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb;

UPDATE adcl_reports
SET district = COALESCE(NULLIF(district, ''), 'Bangalore')
WHERE district IS NULL OR district = '';

ALTER TABLE adcl_reports
    ALTER COLUMN district SET NOT NULL;

ALTER TABLE adcl_reports
    DROP CONSTRAINT IF EXISTS adcl_reports_week_start_key;

CREATE UNIQUE INDEX IF NOT EXISTS idx_adcl_reports_week_district
    ON adcl_reports (week_start, district);

CREATE INDEX IF NOT EXISTS idx_adcl_reports_district_week
    ON adcl_reports (district, week_start DESC);
