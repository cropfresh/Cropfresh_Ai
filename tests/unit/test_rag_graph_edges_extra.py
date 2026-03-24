from __future__ import annotations

from src.rag.graph_runtime.edges import after_grade


def test_after_grade_skips_second_web_search_attempt():
    state = {
        "needs_web_search": True,
        "relevant_documents": [],
        "web_search_attempted": True,
    }

    assert after_grade(state) == "generate"
