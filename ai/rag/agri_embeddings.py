"""
Agricultural Domain Embedding Wrapper
======================================
Layer 1 of the two-layer agri-embedding strategy (ADR-009).

Wraps the existing BGE-M3 EmbeddingManager with:
- Agricultural domain instruction prefix for queries
- Bilingual Hindi/Kannada → normalized English term map (60+ entries)
- Expected improvement: +8–12% context precision on agri golden dataset

Layer 2 (fine-tuned cropfresh-agri-embed-v1) is planned for Phase 4 / 2027.

Architecture: docs/architecture/agri_embeddings.md
ADR: docs/decisions/ADR-009-agri-embeddings.md
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from loguru import logger

from ai.rag.embeddings import EmbeddingManager


class AgriEmbeddingWrapper(EmbeddingManager):
    """
    Agricultural domain embedding wrapper over BGE-M3.

    Improves retrieval precision for Indian agricultural queries by:
    1. Adding a domain-specific instruction prefix to queries
    2. Normalizing bilingual shorthand (Hindi/Kannada) to expanded English terms
    3. Enriching document instruction for agricultural content

    Usage:
        wrapper = AgriEmbeddingWrapper()
        vector = wrapper.embed_query("tamatar ki kharif kheti kaise kare")
        # Internally: normalizes → adds instruction → embeds with BGE-M3

    Note: Does NOT change the embedding model or vector dimensions (still 1024-dim).
    """

    # Domain-specific instruction prefix for agricultural queries
    # More specific than generic BGE instruction for better domain retrieval
    AGRI_QUERY_INSTRUCTION = (
        "Represent this Indian agricultural query for searching knowledge "
        "about crop cultivation, mandi commodity prices, pest and disease "
        "management, government agricultural schemes, soil and irrigation "
        "science, and Karnataka/Maharashtra farming practices: "
    )

    # Domain-specific instruction prefix for agricultural documents
    AGRI_DOC_INSTRUCTION = (
        "Represent this Indian agricultural document about farming knowledge, "
        "crop science, market information, or government schemes: "
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Bilingual normalization map: Hindi/Kannada shorthand → expanded English
    # Format: "input term": "expanded English equivalent (domain standard)"
    # ──────────────────────────────────────────────────────────────────────────
    TERM_MAP: dict[str, str] = {
        # ── Crops (Hindi) ──────────────────────────────────────────────────
        "tamatar": "tomato Solanum lycopersicum",
        "pyaj": "onion Allium cepa",
        "aloo": "potato Solanum tuberosum",
        "gehu": "wheat Triticum aestivum",
        "dhan": "paddy rice Oryza sativa",
        "kapas": "cotton Gossypium",
        "makka": "maize corn Zea mays",
        "til": "sesame Sesamum indicum",
        "moong": "green gram mung bean Vigna radiata",
        "urad": "black gram Vigna mungo",
        "arhar": "pigeon pea tur dal Cajanus cajan",
        "masoor": "red lentil Lens culinaris",
        "chana": "chickpea gram Cicer arietinum",
        "sarson": "mustard Brassica juncea",
        "methi": "fenugreek Trigonella foenum-graecum",
        "palak": "spinach Spinacia oleracea",
        "lauki": "bottle gourd Lagenaria siceraria",
        "karela": "bitter gourd Momordica charantia",
        "bhindi": "okra ladyfinger Abelmoschus esculentus",
        "baingan": "brinjal eggplant Solanum melongena",
        # ── Crops (Kannada) ────────────────────────────────────────────────
        "togaribele": "pigeon pea tur dal Cajanus cajan",
        "hesarubele": "green gram mung bean",
        "kadale": "groundnut peanut Arachis hypogaea",
        "ragi": "finger millet Eleusine coracana",
        "jowar": "sorghum Sorghum bicolor",
        "bajra": "pearl millet Pennisetum glaucum",
        "huchellu": "sunflower Helianthus annuus",
        "shengri": "drumstick Moringa oleifera",
        "togari": "pigeon pea Karnataka tur",
        # ── Seasons ────────────────────────────────────────────────────────
        "kharif": "kharif summer season June October rainfed monsoon crops",
        "rabi": "rabi winter season October March irrigated winter crops",
        "zaid": "zaid spring summer season March June cash crops",
        "vasant": "spring season March April planting",
        # ── Markets & Trade ────────────────────────────────────────────────
        "mandi": "APMC agricultural produce market committee wholesale market",
        "haath": "local weekly market haat bazaar periodic market",
        "bhaav": "market price commodity rate prevailing price",
        "tola": "weight unit 11.66 grams precious metals",
        "quintal": "100 kilograms bulk commodity weight",
        "fasal": "crop harvest season produce yield",
        # ── Organizations ──────────────────────────────────────────────────
        "kvk": "Krishi Vigyan Kendra farm science center agricultural extension",
        "fpo": "Farmer Producer Organisation collective farmer group",
        "atma": "Agricultural Technology Management Agency extension",
        "icar": "Indian Council of Agricultural Research national research institute",
        "apmc": "Agricultural Produce Market Committee regulated mandi",
        "nafed": "National Agricultural Cooperative Marketing Federation",
        # ── Government Schemes ─────────────────────────────────────────────
        "pm-kisan": "PM-KISAN Pradhan Mantri Kisan Samman Nidhi income support scheme",
        "pmfby": "PMFBY Pradhan Mantri Fasal Bima Yojana crop insurance scheme",
        "kcc": "Kisan Credit Card farmer loan credit facility",
        "msp": "Minimum Support Price government guaranteed procurement price",
        "mksp": "Mahila Kisan Sashaktikaran Pariyojana women farmer empowerment",
        "pkvy": "Paramparagat Krishi Vikas Yojana organic farming scheme",
        "rkvy": "Rashtriya Krishi Vikas Yojana agricultural development scheme",
        # ── Soil Types ─────────────────────────────────────────────────────
        "kali mitti": "black cotton soil vertisol Deccan plateau deep irrigation",
        "ret mitti": "sandy loam soil alluvial well-drained light soil",
        "lalite mitti": "red laterite soil Karnataka acidic low fertility",
        "chiknai mitti": "clay soil heavy waterlogged paddy cultivation",
        # ── Agricultural Inputs ────────────────────────────────────────────
        "khad": "fertilizer manure nutrient NPK",
        "dawa": "pesticide insecticide fungicide agrochemical spray",
        "beej": "seed planting material variety certified",
        "sinchai": "irrigation water supply drip sprinkler flood",
        "compost": "organic compost manure bio-decomposed plant residue",
        "vermicompost": "vermicompost earthworm compost organic amendment",
        # ── Weather ────────────────────────────────────────────────────────
        "barish": "rainfall precipitation rain monsoon",
        "garmi": "heat summer temperature high temperature stress",
        "sardi": "cold winter frost temperature low temperature",
        "aandhiyan": "storm cyclone wind damage crop loss",
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        enable_term_normalization: bool = True,
    ):
        """
        Initialize the AgriEmbeddingWrapper.

        Args:
            model_name: HuggingFace model name (default: BAAI/bge-m3)
            device: Compute device — "cpu" or "cuda"
            cache_dir: Optional model cache directory
            enable_term_normalization: Set False to disable bilingual normalization (testing)
        """
        super().__init__(model_name=model_name, device=device, cache_dir=cache_dir)
        self.enable_term_normalization = enable_term_normalization
        logger.info(
            f"AgriEmbeddingWrapper initialized | "
            f"model={model_name} | "
            f"term_map_entries={len(self.TERM_MAP)} | "
            f"normalization={'ON' if enable_term_normalization else 'OFF'}"
        )

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query with agricultural domain context.

        Pipeline:
        1. Normalize bilingual terms (Hindi/Kannada → English)
        2. Add agricultural instruction prefix
        3. Encode with BGE-M3 (returns 1024-dim normalized vector)

        Args:
            query: Raw user query (may contain Hindi/Kannada terms)

        Returns:
            1024-dimensional normalized embedding vector
        """
        # Step 1: Normalize bilingual terms
        if self.enable_term_normalization:
            query = self._normalize_terms(query)

        # Step 2: Add domain instruction prefix (replaces parent's generic BGE prefix)
        prefixed = f"{self.AGRI_QUERY_INSTRUCTION}{query}"

        # Step 3: Encode with base BGE-M3 model (skip parent prefix logic)
        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=True,
        )

        logger.debug(f"AgriEmbeddingWrapper.embed_query | original_len={len(query)} | normalized_query={query[:80]}...")
        return embedding.tolist()

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed documents with agricultural domain instruction prefix.

        Args:
            texts: List of document texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            List of 1024-dimensional embedding vectors
        """
        if not texts:
            return []

        # Add agricultural instruction prefix to all documents
        enriched = [f"{self.AGRI_DOC_INSTRUCTION}{t}" for t in texts]

        logger.debug(f"AgriEmbeddingWrapper.embed_documents | n_docs={len(texts)}")
        embeddings = self.model.encode(
            enriched,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_queries(
        self,
        queries: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Embed multiple queries with agricultural domain context.

        Args:
            queries: List of query texts
            batch_size: Batch size for encoding

        Returns:
            List of 1024-dimensional embedding vectors
        """
        if not queries:
            return []

        # Normalize and prefix each query
        processed = []
        for q in queries:
            normalized = self._normalize_terms(q) if self.enable_term_normalization else q
            processed.append(f"{self.AGRI_QUERY_INSTRUCTION}{normalized}")

        embeddings = self.model.encode(
            processed,
            batch_size=batch_size,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def _normalize_terms(self, text: str) -> str:
        """
        Normalize bilingual agricultural terms to expanded English equivalents.

        Uses a single-pass regex replacement to prevent double-substitution
        (e.g., 'dhan' inside an already-expanded 'Pradhan Mantri' string).

        Terms are sorted by length (longest first) so more specific multi-word
        terms like 'kali mitti' are matched before shorter sub-terms.

        Args:
            text: Input text (may contain Hindi/Kannada terms)

        Returns:
            Text with expanded English equivalents

        Example:
            "pm-kisan scheme tamatar apply" →
            "PM-KISAN Pradhan Mantri Kisan Samman Nidhi income support scheme tomato Solanum lycopersicum apply"
        """
        import re

        text_lower = text.lower()

        # Build a single regex that matches all terms at word boundaries
        # Sort longest first to ensure multi-word terms match before sub-terms
        sorted_terms = sorted(self.TERM_MAP.keys(), key=len, reverse=True)

        # Escape special regex characters in term keys
        escaped_terms = [re.escape(t) for t in sorted_terms]

        # Build alternation pattern
        pattern = re.compile(
            r'\b(' + '|'.join(escaped_terms) + r')\b',
            re.IGNORECASE,
        )

        def replace_match(m: re.Match) -> str:
            matched = m.group(0).lower()
            return self.TERM_MAP.get(matched, m.group(0))

        return pattern.sub(replace_match, text_lower)


    def get_domain_stats(self) -> dict:
        """Return stats about the domain wrapper for observability."""
        return {
            "model_name": self.model_name,
            "term_map_size": len(self.TERM_MAP),
            "normalization_enabled": self.enable_term_normalization,
            "query_instruction_len": len(self.AGRI_QUERY_INSTRUCTION),
            "doc_instruction_len": len(self.AGRI_DOC_INSTRUCTION),
            "vector_dimensions": self._dimensions,
        }


@lru_cache(maxsize=1)
def get_agri_embedding_manager(
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
) -> AgriEmbeddingWrapper:
    """
    Get a cached AgriEmbeddingWrapper instance.

    Uses LRU cache to avoid reloading the embedding model on every call.
    Respects EMBEDDING_DEVICE env var for device selection.

    Args:
        model_name: HuggingFace model name
        device: Compute device ("cpu" or "cuda")

    Returns:
        Singleton AgriEmbeddingWrapper instance
    """
    # Allow env override
    device = os.getenv("EMBEDDING_DEVICE", device)

    logger.info(f"Creating AgriEmbeddingWrapper | model={model_name} | device={device}")
    return AgriEmbeddingWrapper(model_name=model_name, device=device)
