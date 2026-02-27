# ADR-009: Agricultural Domain Embedding Fine-tuning

**Date**: 2026-02-27  
**Status**: Proposed  
**Deciders**: CropFresh AI Team  
**Affects**: `ai/rag/embeddings.py` + new `ai/rag/agri_embeddings.py`

---

## Context

The current embedding model (`BAAI/bge-m3`, 1024-dim) is a **general-purpose multilingual model**. While it handles Kannada/Hindi/English text acceptably, it has significant limitations for the agricultural domain:

1. **Vocabulary gap** — Terms like "Rabi sowing", "Kharif harvest", "APMC mandi", "Agri Input Dealer", "PM-KISAN eligibility", "FPO aggregation" are rare or absent from its training corpus.
2. **Semantic drift** — "blight" in generic context vs. "tomato late blight" (Phytophthora infestans) are treated as similar; the domain specialist knows they differ greatly in treatment.
3. **Code-mixing blind spot** — Farmers often write "mere tomato mein kya hua?" (mixed Hindi/Kannada/English) — cross-lingual agri retrieval degrades without domain fine-tuning.
4. **Price terminology** — "AISP", "MSP", "modal price", "arrival quantity in quintals" require domain embedding alignment.

Research shows domain-fine-tuned embedding models achieve **18–25% better retrieval precision** on in-domain benchmarks vs. general models (AgriEval 2025, AgriBench 2024).

---

## Decision

Implement a **two-layer embedding strategy** with a gradual migration path:

### Layer 1: AgriEmbeddingWrapper (Immediate — Sprint 05)
Wrap `bge-m3` with **domain-specific query transformation** — no model retraining required. Prepend structured agricultural context to queries before embedding.

```python
class AgriEmbeddingWrapper:
    """Wraps BGE-M3 with agricultural domain context injection."""
    
    AGRI_INSTRUCTION = (
        "Represent this agricultural query for searching Indian farming "
        "knowledge including crop cultivation, mandi prices, pest management, "
        "government schemes, and Karnataka/India agronomy: "
    )
    
    DOMAIN_TERMS = {
        # Hindi/Kannada → normalized form for embedding
        "tamatar": "tomato", "pyaj": "onion", "gehu": "wheat",
        "dhan": "paddy rice", "kapas": "cotton", "aloo": "potato",
        "kharif": "kharif summer crop season June-October",
        "rabi": "rabi winter crop season October-March",
        "mandi": "agricultural market APMC mandi wholesale market",
        "AISP": "Agri Input Support Price farmer income support",
        "FPO": "Farmer Producer Organisation collective marketing",
        "KVK": "Krishi Vigyan Kendra agricultural extension center",
    }
```

### Layer 2: Fine-tuned AgriEmbedding Model (Phase 4 — 2027)
Fine-tune `bge-m3` on a curated agricultural corpus using **contrastive learning** (sentence-transformers `MultipleNegativesRankingLoss`).

**Training data sources:**
- ICAR crop production guides (~500 docs)
- State KVK advisories — Karnataka + Maharashtra (~1,000)
- APMC price history Q&A pairs (synthetic from eNAM data)
- CropFresh platform query logs (after 6 months of production)
- AgriEval 2025 benchmark dataset

**Training configuration:**
```python
model = SentenceTransformer("BAAI/bge-m3")
train_loss = losses.MultipleNegativesRankingLoss(model)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=100,
    output_path="models/cropfresh-agri-embed-v1",
)
```

**Target improvement:** +18% context precision on CropFresh golden evaluation dataset.

---

## Consequences

**Layer 1 (Wrapper) — Immediate:**
- ✅ Zero infra cost — reuses existing BGE-M3 model
- ✅ Estimated +8–12% retrieval precision from richer query context
- ✅ Ships in 2 days with no model download changes
- ⚠️ Longer query strings → slightly higher embedding inference time (~10ms)

**Layer 2 (Fine-tuned) — Phase 4:**
- ✅ +18–25% domain retrieval precision  
- ✅ Better Kannada/Hindi code-mixing handling
- ⚠️ Requires 1,000+ high-quality training pairs (labor intensive)
- ⚠️ Fine-tuned model hosting (~2GB, needs dedicated model server or HF endpoint)
- ⚠️ Re-embedding entire Qdrant collection on model upgrade (schedule downtime)

---

## Alternatives Considered

| Approach | Reason |
|----------|--------|
| Use OpenAI `text-embedding-3-large` | Better out-of-box quality but $0.13/1M tokens, no Kannada fine-tuning |
| Use `e5-mistral-7b-instruct` | Excellent quality but 7B params = high RAM; not viable on current infra |
| BM25-only (no embeddings) | No semantic similarity; can't handle paraphrase queries |
| Multilingual-E5-large | Decent but inferior to BGE-M3 for cross-lingual agri |

---

## Related

- Architecture: [`agri_embeddings.md`](../architecture/agri_embeddings.md)
- Implementation: `ai/rag/agri_embeddings.py` (Sprint 05 — Layer 1)
- Training script: `scripts/train_agri_embeddings.py` (Phase 4)
- Evaluation: `scripts/evaluate_agri_embeddings.py`
