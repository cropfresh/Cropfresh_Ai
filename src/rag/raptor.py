"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

! This file is kept for backward compatibility. The actual implementation
! has been split into the `src.rag.raptor_pkg` package.
! Import from `src.rag.raptor_pkg` directly in new code.
"""

from src.rag.raptor_pkg.models import (
    ClusterInfo,
    NodeLevel,
    RAPTORConfig,
    RAPTORNode,
    RAPTORTreeStats,
)
from src.rag.raptor_pkg.index import RAPTORIndex, create_raptor_index

__all__ = [
    "ClusterInfo",
    "NodeLevel",
    "RAPTORConfig",
    "RAPTORIndex",
    "RAPTORNode",
    "RAPTORTreeStats",
    "create_raptor_index",
]
