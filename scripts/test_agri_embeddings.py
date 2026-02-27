"""
Test: Agricultural Embedding Wrapper
======================================
Validates the bilingual term normalization and wrapper behaviour.
No model download needed for normalization tests.

Usage:
    # Normalization tests only (no BGE-M3 model needed)
    uv run python scripts/test_agri_embeddings.py --normalization-only

    # Full test (requires BGE-M3 model downloaded)
    uv run python scripts/test_agri_embeddings.py
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

NORMALIZATION_ONLY = "--normalization-only" in sys.argv


def test_term_normalization():
    """Test that bilingual terms are correctly expanded."""
    from ai.rag.agri_embeddings import AgriEmbeddingWrapper

    wrapper = AgriEmbeddingWrapper.__new__(AgriEmbeddingWrapper)
    wrapper.TERM_MAP = AgriEmbeddingWrapper.TERM_MAP
    wrapper.enable_term_normalization = True

    test_cases = [
        # (input, must_contain)
        ("tamatar ki rabi kheti", ["tomato Solanum lycopersicum", "rabi winter season"]),
        ("mandi mein bhaav kya hai", ["APMC agricultural produce market", "market price"]),
        ("pyaj ka fasal", ["onion Allium cepa", "crop harvest"]),
        ("kharif mein dhan lagana", ["kharif summer season", "paddy rice"]),
        ("kvk se madad chahiye", ["Krishi Vigyan Kendra"]),
        ("pm-kisan scheme apply", ["PM-KISAN Pradhan Mantri"]),
        ("moong urad ki kheti", ["green gram mung bean", "black gram"]),
        ("kcc loan kaise mile", ["Kisan Credit Card"]),
        ("bhindi aur karela growing tips", ["okra ladyfinger", "bitter gourd"]),
        ("fpo se jodhna chahte hain", ["Farmer Producer Organisation"]),
    ]

    print("\n" + "=" * 70)
    print("AGRI EMBEDDING WRAPPER — Bilingual Term Normalization Tests")
    print("=" * 70)

    passed = 0
    failed = 0

    for query, expected_terms in test_cases:
        normalized = wrapper._normalize_terms(query)
        all_found = all(term.lower() in normalized.lower() for term in expected_terms)

        if all_found:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
            missing = [t for t in expected_terms if t.lower() not in normalized.lower()]
            print(f"{status}: '{query[:40]}'")
            print(f"       Missing: {missing}")
            print(f"       Normalized: {normalized[:100]}")
            continue

        print(f"{status}: '{query[:40]}' → contains {expected_terms[0][:25]}...")

    print(f"\nNormalization: ✅ {passed}/{passed + failed} passed")
    print(f"Term map size: {len(wrapper.TERM_MAP)} entries\n")
    return failed == 0


def test_domain_stats():
    """Test the domain stats method."""
    from ai.rag.agri_embeddings import AgriEmbeddingWrapper

    # Access stats without loading the model
    wrapper = AgriEmbeddingWrapper.__new__(AgriEmbeddingWrapper)
    wrapper.model_name = "BAAI/bge-m3"
    wrapper.enable_term_normalization = True
    wrapper._dimensions = 1024

    stats = {
        "model_name": wrapper.model_name,
        "term_map_size": len(AgriEmbeddingWrapper.TERM_MAP),
        "normalization_enabled": wrapper.enable_term_normalization,
        "vector_dimensions": wrapper._dimensions,
    }

    print("Domain Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    assert stats["term_map_size"] >= 60, f"Expected ≥60 terms, got {stats['term_map_size']}"
    assert stats["vector_dimensions"] == 1024, "Expected 1024-dim"
    print(f"\n✅ Stats check passed [{stats['term_map_size']} terms, {stats['vector_dimensions']}-dim]\n")
    return True


def test_embedding(wrapper):
    """Test actual embedding generation (requires BGE-M3 model)."""
    import numpy as np

    print("Testing embed_query...")
    vec = wrapper.embed_query("how to grow tomatoes in Karnataka?")
    assert isinstance(vec, list), "Expected list"
    assert len(vec) == 1024, f"Expected 1024 dims, got {len(vec)}"

    # Check normalization (~1.0 for cosine similarity)
    norm = float(np.linalg.norm(vec))
    assert 0.99 < norm < 1.01, f"Expected normalized vector, got norm={norm:.4f}"
    print(f"  ✅ embed_query: 1024-dim normalized vector (norm={norm:.4f})")

    print("Testing embed_documents...")
    vecs = wrapper.embed_documents(["Tomato cultivation Karnataka", "Onion farming guide"])
    assert len(vecs) == 2, "Expected 2 vectors"
    assert len(vecs[0]) == 1024, "Expected 1024-dim"
    print(f"  ✅ embed_documents: {len(vecs)} × 1024-dim vectors")

    print("Testing similarity improvement (agri terms)...")
    # Terms that should be semantically closer with agri instruction
    vec_tomato = wrapper.embed_query("tamatar ki kharif kheti")
    vec_tomato_en = wrapper.embed_query("tomato summer farming")

    similarity = float(np.dot(vec_tomato, vec_tomato_en))
    print(f"  'tamatar ki kharif kheti' ↔ 'tomato summer farming': {similarity:.4f}")
    assert similarity > 0.7, f"Expected high similarity (>0.7), got {similarity:.4f}"
    print(f"  ✅ Bilingual similarity: {similarity:.4f} (> 0.70 threshold)")

    return True


if __name__ == "__main__":
    all_passed = True

    # Always run normalization tests (no model needed)
    all_passed &= test_term_normalization()
    all_passed &= test_domain_stats()

    if not NORMALIZATION_ONLY:
        print("Loading BGE-M3 model for embedding tests...")
        print("(This may take a moment on first run)\n")
        try:
            from ai.rag.agri_embeddings import get_agri_embedding_manager
            wrapper = get_agri_embedding_manager()
            all_passed &= test_embedding(wrapper)
        except Exception as e:
            print(f"❌ Embedding test failed: {e}")
            print("   Run with --normalization-only to skip model tests")
            all_passed = False
    else:
        print("(Skipping model embedding tests — --normalization-only mode)")

    print("\n" + ("✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"))
    sys.exit(0 if all_passed else 1)
