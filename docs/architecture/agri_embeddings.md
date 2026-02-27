# Agricultural Domain Embeddings — Architecture

> **Status**: Proposed | **Sprint**: 05 (Layer 1) / Phase 4 (Layer 2)  
> **ADR**: [ADR-009](../decisions/ADR-009-agri-embeddings.md)

---

## Overview

CropFresh uses a **two-layer embedding strategy** to progressively improve retrieval precision for Indian agricultural domain text, including multilingual Kannada/Hindi/English queries from smallholder farmers.

- **Layer 1** (Sprint 05): `AgriEmbeddingWrapper` — domain context injection, zero new infra
- **Layer 2** (Phase 4/2027): Fine-tuned `cropfresh-agri-embed-v1` model

---

## Why Domain Embeddings Matter

### Gap Example

**Query**: "Rabi tamatar mein konsa khad dalna chahiye?" (Hindi: "What fertilizer for Rabi tomato?")

**BGE-M3 (generic)** embedding proximity:
1. "tomato fertilizer requirements" ← Correct ✅
2. "Rabi season cereal crop nutrient schedule" ← Partially relevant
3. "tamatar ki kheti" ("tomato farming") ← Correct ✅
4. "Rabi sowing calendar" ← Not relevant

**AgriEmbedding Layer 1** (with domain context and term normalization):
1. "tomato fertilizer requirements Rabi winter India" ← Correct ✅
2. "NPK schedule Solanum lycopersicum Rabi season irrigated" ← Correct ✅
3. "tomato micronutrient deficiency Karnataka" ← Correct ✅
4. "Kolar district tomato crop calendar" ← Correct ✅

---

## Layer 1: AgriEmbeddingWrapper (Sprint 05)

**File**: `ai/rag/agri_embeddings.py`

```python
class AgriEmbeddingWrapper:
    """
    Wraps the base BGE-M3 embedding model with agricultural
    domain context injection. Zero new model downloads required.
    
    Improvements:
    - Domain-specific instruction prefix for queries
    - Hindi/Kannada → normalized English term mapping
    - Agricultural entity enrichment before embedding
    """
    
    # Complete, authoritative instruction prefix
    AGRI_QUERY_INSTRUCTION = (
        "Represent this Indian agricultural query for searching knowledge "
        "about crop cultivation, mandi commodity prices, pest and disease "
        "management, government agricultural schemes, soil and irrigation "
        "science, and Karnataka/Maharashtra farming practices: "
    )
    
    AGRI_DOC_INSTRUCTION = (
        "Represent this Indian agricultural document about farming knowledge, "
        "crop science, market information, or government schemes: "
    )
    
    # Bilingual normalization map (Hindi/Kannada → domain-standard English)
    TERM_MAP: dict[str, str] = {
        # Crops
        "tamatar": "tomato Solanum lycopersicum",
        "pyaj": "onion Allium cepa",
        "aloo": "potato Solanum tuberosum",
        "gehu": "wheat Triticum aestivum",
        "dhan": "paddy rice Oryza sativa",
        "kapas": "cotton Gossypium",
        "makka": "maize corn Zea mays",
        "til": "sesame Sesamum indicum",
        "moong": "green gram mung bean",
        "arhar": "pigeon pea tur dal",
        # Seasons
        "kharif": "kharif summer season June-October rainfed",
        "rabi": "rabi winter season October-March irrigated",
        "zaid": "zaid spring summer March-June cash crops",
        # Markets
        "mandi": "APMC agricultural produce market wholesale",
        "haath": "local weekly market haat bazaar",
        # Organizations
        "kvk": "Krishi Vigyan Kendra farm science center",
        "fpo": "Farmer Producer Organisation collective",
        "atma": "Agricultural Technology Management Agency",
        # Schemes
        "pm-kisan": "PM-KISAN Pradhan Mantri Kisan Samman Nidhi income support",
        "pmfby": "PMFBY Pradhan Mantri Fasal Bima Yojana crop insurance",
        "kcc": "Kisan Credit Card farmer loan",
        # Soil
        "kali mitti": "black cotton soil vertisol Deccan plateau",
        "ret mitti": "sandy loam soil alluvial",
        "lalite mitti": "red laterite soil Karnataka",
    }
    
    def embed_query(self, query: str) -> list[float]:
        """Embed query with domain context."""
        normalized = self._normalize_terms(query)
        prefixed = f"{self.AGRI_QUERY_INSTRUCTION}{normalized}"
        return self.base_model.embed_query(prefixed)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with domain instruction."""
        enriched = [f"{self.AGRI_DOC_INSTRUCTION}{t}" for t in texts]
        return self.base_model.embed_documents(enriched)
    
    def _normalize_terms(self, text: str) -> str:
        """Replace domain shorthand with expanded English terms."""
        text_lower = text.lower()
        for term, expansion in self.TERM_MAP.items():
            if term in text_lower:
                text_lower = text_lower.replace(term, expansion)
        return text_lower
```

**Expected improvement**: +8–12% context precision on CropFresh golden dataset.

---

## Layer 2: Fine-tuned cropfresh-agri-embed-v1 (Phase 4)

### Training Data Pipeline

```
Sources                    Processing              Output
──────                     ──────────              ──────
ICAR Publications ──→      Chunk + QA pairs  ──→   Training pairs
KVK Advisories   ──→       Extract Q+A        ──→  (query, doc, label)
eNAM Price Data  ──→       Synthetic QA       ──→
Platform Logs    ──→       User query + result──→  Fine-tune dataset
AgriEval 2025   ──→       Benchmark examples  ──→
```

### Model Training

Base model: `BAAI/bge-m3` (1024-dim, multilingual)  
Training method: `MultipleNegativesRankingLoss` (contrastive)  
Training pairs format: `(query, positive_doc, [hard_negatives])`

```python
# scripts/train_agri_embeddings.py
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-m3")
train_examples = load_cropfresh_training_data()  # 10,000+ pairs

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=200,
    evaluation_steps=500,
    output_path="models/cropfresh-agri-embed-v1",
    save_best_model=True,
    show_progress_bar=True,
)
```

### Hard Negative Mining for Agriculture

Critical for domain performance — the model must distinguish:
- "tomato late blight treatment" ≠ "potato late blight treatment" (different fungicides)
- "tomato price Hubli" ≠ "tomato cultivation Hubli" (market vs. agronomy)
- "Rabi tomato fertilizer" ≠ "Kharif tomato fertilizer" (different NPK schedules)

---

## Qdrant Collection Schema

When Layer 2 is deployed, a full re-embedding is required:

| Collection | Embedding Model | Dims | Purpose |
|-----------|----------------|------|---------|
| `agri_knowledge` | `cropfresh-agri-embed-v1` | 1024 | Main crop/pest/scheme KB |
| `price_history` | `cropfresh-agri-embed-v1` | 1024 | Historical price context |
| `live_web` | `BGE-M3` (generic) | 1024 | Browser-scraped web docs (temporary) |
| `graph_summaries` | `cropfresh-agri-embed-v1` | 1024 | Neo4j community summaries |

> **Migration note**: On Layer 2 deployment, run `scripts/reembed_qdrant.py` during scheduled maintenance window (estimated: 2-3 hours on 50K documents).

---

## Quality Metrics

| Metric | BGE-M3 baseline | AgriWrapper L1 | Fine-tuned L2 (target) |
|--------|----------------|----------------|------------------------|
| Context Precision | Unknown | +8–12% | +18–25% |
| Kannada query recall | Unknown | +5% | +15% |
| Hindi code-mix query recall | Unknown | +10% | +22% |
| Domain terminology coverage | ~60% | ~80% | ~95% |

---

## Related Documents

- [ADR-009: Embedding Decision](../decisions/ADR-009-agri-embeddings.md)
- Implementation: `ai/rag/agri_embeddings.py` (Sprint 05 — Layer 1)
- Training: `scripts/train_agri_embeddings.py` (Phase 4)
- Evaluation: `scripts/evaluate_agri_embeddings.py`
