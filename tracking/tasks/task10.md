# Task 10: Implement Digital Twin Engine

> **Priority:** 🟠 P1 | **Phase:** 3 | **Effort:** 4–5 days  
> **Files:** `src/agents/digital_twin.py` [NEW]  
> **Score Target:** 9/10 — Critical for <2% dispute target

---

## 📌 Problem Statement

Digital Twin is unimplemented but **critical for the <2% dispute target**. It creates a virtual snapshot of produce at departure (farm) and compares it against arrival (buyer) to resolve disputes.

---

## 🔬 Research Findings

### Digital Twin Architecture for Produce
```
DEPARTURE TWIN (at farm):
├── Farmer photos (3-5 angles)
├── Agent verification photos
├── AI grade overlay (bounding boxes)
├── GPS + timestamp
├── Predicted shelf life
└── QR code linking

ARRIVAL COMPARISON:
├── Buyer arrival photos
├── GPS + timestamp
├── AI diff analysis:
│   ├── Color degradation %
│   ├── New defects detected
│   ├── Size/weight variance
│   └── Overall quality delta
└── Liability assignment (farmer/hauler/buyer)
```

### AI Diff Engine
- **Image similarity**: SSIM (Structural Similarity Index) + perceptual hashing
- **Defect delta**: Compare defect counts departure vs arrival
- **Color histogram**: Detect ripening/browning progression
- **Grade delta**: If grade drops > 1 level → hauler likely responsible

### Liability Matrix
| Condition | Likely Responsible |
|-----------|-------------------|
| Grade drop + long transit (>6h) | Hauler (cold chain failure) |
| Grade drop + short transit (<2h) | Farmer (pre-existing issue) |
| Quantity mismatch > 5% | Farmer or hauler |
| Buyer photos show damage not in twin | Buyer fabrication |
| No arrival photos submitted | Buyer (claim rejected) |

---

## 🏗️ Implementation Spec

```python
class DigitalTwinEngine:
    """
    Digital Twin for produce quality tracking.
    
    Creates immutable departure snapshot, compares against arrival,
    generates diff report for dispute resolution.
    """
    
    async def create_departure_twin(
        self,
        listing_id: str,
        farmer_photos: list[str],     # S3 URLs
        agent_photos: list[str],      # Field agent verification
        quality_result: QualityResult, # From CV-QG agent
        gps: tuple[float, float],
    ) -> DigitalTwin:
        """Create departure twin with immutable snapshot."""
    
    async def compare_arrival(
        self,
        twin_id: str,
        arrival_photos: list[str],
        arrival_gps: tuple[float, float],
    ) -> DiffReport:
        """
        Compare departure vs arrival.
        Returns diff report with quality delta and liability recommendation.
        """
    
    async def generate_diff_report(
        self,
        departure_twin: DigitalTwin,
        arrival_data: dict,
    ) -> DiffReport:
        """
        AI-powered diff analysis:
        1. Image similarity (SSIM) between departure/arrival
        2. Defect count delta
        3. Color histogram comparison
        4. Grade assessment of arrival photos
        5. Transit time analysis
        6. Liability recommendation
        """

@dataclass
class DiffReport:
    quality_delta: float      # -1.0 to 0.0 (negative = degraded)
    grade_departure: str      # 'A'
    grade_arrival: str        # 'B'
    new_defects: list[str]    # ['bruise', 'colour_off']
    similarity_score: float   # 0.0–1.0 (SSIM)
    transit_hours: float
    liability: str            # 'farmer', 'hauler', 'buyer', 'shared'
    claim_percent: float      # 0–100%
    confidence: float
    explanation: str          # Human-readable
```

---

## ✅ Acceptance Criteria

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Departure twin captures photos + grade + GPS | 25% |
| 2 | Arrival comparison generates diff report | 25% |
| 3 | Liability recommendation with reasoning | 20% |
| 4 | Image similarity metric (SSIM or perceptual hash) | 15% |
| 5 | Linked to dispute resolution in order service | 15% |
