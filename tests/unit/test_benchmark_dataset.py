from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from ai.rag.evaluation.dataset_loader import BenchmarkDatasetLoader
from ai.rag.evaluation.models import GoldenEntry
from ai.rag.evaluation.reference_resolver import ReferenceResolver
from src.tools.agmarknet import AgmarknetPrice


def test_loader_reads_custom_dataset_directory(tmp_path):
    dataset_path = tmp_path / "mini.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": "custom_1",
                    "query": "What is PM-KISAN?",
                    "category": "scheme",
                    "mode": "static",
                    "ground_truth": "Income support scheme.",
                    "contexts": ["PM-KISAN gives direct income support."],
                }
            ]
        ),
        encoding="utf-8",
    )

    entries = BenchmarkDatasetLoader(datasets_dir=tmp_path).load("mini")

    assert len(entries) == 1
    assert entries[0].id == "custom_1"


@pytest.mark.asyncio
async def test_reference_resolver_returns_static_entry_as_is():
    entry = GoldenEntry(
        id="static_1",
        query="How to apply for PM-KISAN?",
        category="scheme",
        mode="static",
        ground_truth="Apply with Aadhaar and land records.",
        contexts=["PM-KISAN requires Aadhaar and land records."],
    )

    resolved = await ReferenceResolver().resolve(entry)

    assert resolved.ground_truth == entry.ground_truth
    assert resolved.contexts == entry.contexts
    assert resolved.freshness_ok is True


@pytest.mark.asyncio
async def test_reference_resolver_marks_stale_live_prices(monkeypatch):
    stale_price = AgmarknetPrice(
        commodity="Tomato",
        state="Karnataka",
        district="Kolar",
        market="Kolar Main Market",
        date=datetime.now() - timedelta(days=10),
        min_price=1000,
        max_price=2000,
        modal_price=1500,
    )

    async def fake_get_prices(self, commodity: str, state: str, district: str | None = None):
        return [stale_price]

    monkeypatch.setattr("src.tools.agmarknet.AgmarknetTool.get_prices", fake_get_prices)
    entry = GoldenEntry(
        id="live_1",
        query="What is tomato price in Kolar today?",
        category="market",
        mode="live",
        reference_resolver="agmarknet_price",
        resolver_params={"commodity": "Tomato", "district": "Kolar"},
    )

    resolved = await ReferenceResolver().resolve(entry)

    assert resolved.freshness_ok is False
    assert "Kolar" in resolved.ground_truth
