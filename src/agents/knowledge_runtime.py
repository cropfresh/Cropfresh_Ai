from __future__ import annotations

import os


def configure_benchmark_embeddings(agent) -> None:
    """Opt into agri embeddings for benchmark/debug runs when requested."""
    if os.getenv("RAG_BENCHMARK_USE_AGRI_EMBEDDINGS", "false").lower() != "true":
        return

    from src.rag.agri_embeddings import get_agri_embedding_manager

    # TODO: Point debug runs at a dedicated benchmark collection after reindexing.
    agent.knowledge_base._embedding_manager = get_agri_embedding_manager()
