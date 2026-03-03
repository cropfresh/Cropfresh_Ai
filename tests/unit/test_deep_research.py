"""
Unit Tests: Deep Research Tool
================================
Tests for src/tools/deep_research.py

Covers:
- PageContent and ExtractedFact models
- fetch_all_pages concurrent fetching (mocked httpx)
- extract_all_facts map step (mocked LLM)
- synthesise_answer reduce step (mocked LLM)
- DeepResearchTool.research full pipeline (end-to-end mock)
- Tool registry auto-registration
- format_for_llm output format

Author: CropFresh AI Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.deep_research import (
    DeepResearchResult,
    DeepResearchTool,
    ExtractedFact,
    PageContent,
    _deep_research,
    _extract_facts,
    _groq_complete,
    extract_all_facts,
    fetch_all_pages,
    synthesise_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_page(url: str, success: bool = True, content: str = "Sample content") -> PageContent:
    """Create a PageContent for tests."""
    return PageContent(url=url, markdown=content, success=success)


def _make_fact(url: str, facts: str = "Fact data", skipped: bool = False) -> ExtractedFact:
    """Create an ExtractedFact for tests."""
    return ExtractedFact(url=url, facts=facts, skipped=skipped)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestPageContent:
    """Tests for the PageContent data model."""

    def test_default_success_false(self):
        page = PageContent(url="https://example.com")
        assert page.success is False
        assert page.markdown == ""

    def test_successful_page(self):
        page = PageContent(
            url="https://agmarknet.gov.in",
            markdown="# Tomato Price\n₹35/kg",
            success=True,
        )
        assert page.success is True
        assert "₹35" in page.markdown

    def test_failed_page_stores_error(self):
        page = PageContent(url="https://bad.xyz", success=False, error="Timeout")
        assert page.error == "Timeout"


class TestExtractedFact:
    """Tests for the ExtractedFact data model."""

    def test_default_not_skipped(self):
        fact = ExtractedFact(url="https://x.com")
        assert fact.skipped is False
        assert fact.facts == ""

    def test_skipped_flag(self):
        fact = ExtractedFact(url="https://irrelevant.com", skipped=True)
        assert fact.skipped is True


class TestDeepResearchResult:
    """Tests for the DeepResearchResult data model."""

    def test_defaults(self):
        result = DeepResearchResult(query="test", answer="answer text")
        assert result.sources == []
        assert result.pages_fetched == 0
        assert result.pages_useful == 0


# ---------------------------------------------------------------------------
# fetch_all_pages tests
# ---------------------------------------------------------------------------

class TestFetchAllPages:
    """Tests for fetch_all_pages concurrent fetching."""

    @pytest.mark.asyncio
    async def test_fetches_multiple_urls_concurrently(self):
        """fetch_all_pages should return one PageContent per URL."""
        urls = [
            "https://agmarknet.gov.in",
            "https://icar.gov.in",
            "https://krishijagran.com",
        ]

        mock_response = MagicMock()
        mock_response.text = "# Crop price data"
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client = AsyncMock()
        mock_client.get = mock_get

        results = await fetch_all_pages(urls, mock_client)

        assert len(results) == 3
        assert all(isinstance(r, PageContent) for r in results)

    @pytest.mark.asyncio
    async def test_failed_fetch_returns_failed_page(self):
        """A connection error should result in a failed PageContent."""
        async def mock_get(*args, **kwargs):
            raise Exception("Connection refused")

        mock_client = AsyncMock()
        mock_client.get = mock_get

        results = await fetch_all_pages(["https://fail.example.com"], mock_client)
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].markdown == ""

    @pytest.mark.asyncio
    async def test_caps_at_max_pages(self):
        """fetch_all_pages should cap at MAX_PAGES (15)."""
        urls = [f"https://site{i}.com" for i in range(20)]

        mock_response = MagicMock()
        mock_response.text = "content"
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client = AsyncMock()
        mock_client.get = mock_get

        results = await fetch_all_pages(urls, mock_client)
        # MAX_PAGES = 15
        assert len(results) == 15

    @pytest.mark.asyncio
    async def test_content_truncated_to_max_chars(self):
        """Page content must be truncated to MAX_CONTENT_CHARS."""
        from src.tools.deep_research import MAX_CONTENT_CHARS
        long_content = "X" * (MAX_CONTENT_CHARS + 5000)

        mock_response = MagicMock()
        mock_response.text = long_content
        mock_response.raise_for_status = MagicMock()

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_client = AsyncMock()
        mock_client.get = mock_get

        results = await fetch_all_pages(["https://bigpage.com"], mock_client)
        assert len(results[0].markdown) <= MAX_CONTENT_CHARS


# ---------------------------------------------------------------------------
# _extract_facts (Map step) tests
# ---------------------------------------------------------------------------

class TestExtractFacts:
    """Tests for the Map step — extracting facts from single pages."""

    @pytest.mark.asyncio
    async def test_returns_skipped_for_failed_page(self):
        """A failed page should immediately be skipped without LLM call."""
        page = PageContent(url="https://fail.com", success=False)
        client = AsyncMock()

        fact = await _extract_facts(page, "tomato price", client, api_key="test-key")

        assert fact.skipped is True
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_skipped_for_skip_response(self):
        """Pages where LLM responds 'SKIP' should be marked skipped."""
        page = _make_page("https://irrelevant.com", content="Article about politics.")

        async def mock_post(*args, **kwargs):
            mock = MagicMock()
            mock.raise_for_status = MagicMock()
            mock.json = MagicMock(return_value={
                "choices": [{"message": {"content": "SKIP"}}]
            })
            return mock

        mock_client = AsyncMock()
        mock_client.post = mock_post

        fact = await _extract_facts(page, "tomato price Karnataka", mock_client, "key")
        assert fact.skipped is True

    @pytest.mark.asyncio
    async def test_returns_facts_for_relevant_page(self):
        """Relevant pages should return extracted facts, not be skipped."""
        page = _make_page(
            "https://agmarknet.gov.in",
            content="Tomato price in Kolar: ₹32/kg today.",
        )

        async def mock_post(*args, **kwargs):
            mock = MagicMock()
            mock.raise_for_status = MagicMock()
            mock.json = MagicMock(return_value={
                "choices": [{"message": {"content": "Tomato price Kolar: ₹32/kg (2026)"}}]
            })
            return mock

        mock_client = AsyncMock()
        mock_client.post = mock_post

        fact = await _extract_facts(page, "tomato price Kolar", mock_client, "key")
        assert fact.skipped is False
        assert "₹32" in fact.facts


# ---------------------------------------------------------------------------
# extract_all_facts tests
# ---------------------------------------------------------------------------

class TestExtractAllFacts:
    """Tests for the parallel Map step across all pages."""

    @pytest.mark.asyncio
    async def test_processes_all_pages(self):
        """extract_all_facts should return one fact per page."""
        pages = [
            _make_page("https://a.com"),
            _make_page("https://b.com"),
            _make_page("https://c.com", success=False),
        ]

        async def mock_post(*args, **kwargs):
            mock = MagicMock()
            mock.raise_for_status = MagicMock()
            mock.json = MagicMock(return_value={
                "choices": [{"message": {"content": "Some relevant fact"}}]
            })
            return mock

        mock_client = AsyncMock()
        mock_client.post = mock_post

        facts = await extract_all_facts(pages, "query", mock_client, "key")
        assert len(facts) == 3  # One result per page (though c is skipped)
        # Page c (failed) should be skipped
        c_fact = next(f for f in facts if f.url == "https://c.com")
        assert c_fact.skipped is True


# ---------------------------------------------------------------------------
# synthesise_answer tests
# ---------------------------------------------------------------------------

class TestSynthesiseAnswer:
    """Tests for the Reduce step — synthesising the final answer."""

    @pytest.mark.asyncio
    async def test_empty_facts_returns_fallback_message(self):
        """With no useful facts, return a graceful 'could not find' message."""
        mock_client = AsyncMock()
        answer = await synthesise_answer("my query", [], mock_client, "key")
        assert "could not find" in answer.lower()
        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesises_from_multiple_facts(self):
        """synthesise_answer should call the LLM and return its text."""
        facts = [
            _make_fact("https://a.com", facts="Tomato price: ₹35"),
            _make_fact("https://b.com", facts="Tomato wholesale: ₹30"),
        ]

        mock_post_response = MagicMock()
        mock_post_response.raise_for_status = MagicMock()
        mock_post_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "Final synthesised answer here."}}]
        })

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_post_response)

        answer = await synthesise_answer("tomato price", facts, mock_client, "key")
        assert "synthesised" in answer.lower()
        mock_client.post.assert_called_once()


# ---------------------------------------------------------------------------
# DeepResearchTool full pipeline tests
# ---------------------------------------------------------------------------

class TestDeepResearchTool:
    """End-to-end integration tests for DeepResearchTool.research()."""

    @pytest.mark.asyncio
    async def test_research_returns_deep_research_result(self):
        """DeepResearchTool.research() should return a DeepResearchResult."""
        tool = DeepResearchTool(api_key="test-key")

        # Mock the search tool
        from src.tools.web_search import SearchResult, SearchResults
        mock_search = AsyncMock(return_value=SearchResults(
            query="tomato price",
            results=[
                SearchResult(
                    title="Agmarknet",
                    url="https://agmarknet.gov.in",
                    snippet="Tomato ₹35/kg",
                    source="Gov",
                ),
            ],
            total=1,
        ))
        tool._search_tool.search = mock_search

        # Mock httpx responses
        mock_get_response = MagicMock()
        mock_get_response.text = "Tomato price today ₹35 per kg in Kolar."
        mock_get_response.raise_for_status = MagicMock()

        mock_post_response = MagicMock()
        mock_post_response.raise_for_status = MagicMock()
        mock_post_response.json = MagicMock(return_value={
            "choices": [{"message": {"content": "Tomato is ₹35/kg [1]."}}]
        })

        with patch("src.tools.deep_research.httpx.AsyncClient") as mock_client_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.get = AsyncMock(return_value=mock_get_response)
            mock_ctx.post = AsyncMock(return_value=mock_post_response)
            mock_client_cls.return_value = mock_ctx

            result = await tool.research("tomato price Karnataka")

        assert isinstance(result, DeepResearchResult)
        assert result.query == "tomato price Karnataka"

    @pytest.mark.asyncio
    async def test_research_handles_empty_search_results(self):
        """If search returns no URLs, research should return a fallback answer."""
        tool = DeepResearchTool(api_key="test-key")

        from src.tools.web_search import SearchResults
        tool._search_tool.search = AsyncMock(return_value=SearchResults(
            query="unknown query",
            results=[],
            total=0,
        ))

        result = await tool.research("unknown query")
        assert "No search results" in result.answer

    def test_format_for_llm_includes_sources(self):
        """format_for_llm should include source URLs in output."""
        tool = DeepResearchTool()
        result = DeepResearchResult(
            query="test query",
            answer="This is the answer.",
            sources=["https://src1.com", "https://src2.com"],
            pages_fetched=5,
            pages_useful=2,
        )
        formatted = tool.format_for_llm(result)
        assert "https://src1.com" in formatted
        assert "https://src2.com" in formatted
        assert "test query" in formatted
        assert "Pages searched: 5" in formatted

    def test_max_pages_capped_at_15(self):
        """max_pages should be capped at 15 regardless of input."""
        from src.tools.deep_research import MAX_PAGES
        tool = DeepResearchTool(max_pages=100)
        assert tool.max_pages == MAX_PAGES


# ---------------------------------------------------------------------------
# Tool registry tests
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Tests that the deep_research tool is registered correctly."""

    def test_tool_registered_in_registry(self):
        """deep_research should appear in the global tool registry."""
        import importlib
        import src.tools.deep_research  # noqa: F401 (ensures registration runs)
        importlib.import_module("src.tools.deep_research")

        from src.tools.registry import get_tool_registry
        registry = get_tool_registry()
        assert "deep_research" in registry.list_tools()

    def test_tool_definition_category_is_web(self):
        """The registered tool should have category='web'."""
        from src.tools.registry import get_tool_registry
        registry = get_tool_registry()
        defn = registry.get_definition("deep_research")
        assert defn is not None
        assert defn.category == "web"

    def test_tool_is_async(self):
        """The registered deep_research function must be async."""
        from src.tools.registry import get_tool_registry
        registry = get_tool_registry()
        defn = registry.get_definition("deep_research")
        assert defn is not None
        assert defn.is_async is True


# ---------------------------------------------------------------------------
# Entry point wrapper test
# ---------------------------------------------------------------------------

class TestDeepResearchEntrypoint:
    """Tests for the _deep_research tool wrapper function."""

    @pytest.mark.asyncio
    async def test_wrapper_returns_dict_with_expected_keys(self):
        """_deep_research should return a dict matching the registry contract."""
        with patch(
            "src.tools.deep_research.DeepResearchTool.research",
            new_callable=AsyncMock,
        ) as mock_research:
            mock_research.return_value = DeepResearchResult(
                query="banana export policy",
                answer="Policy details here [1].",
                sources=["https://apeda.gov.in"],
                pages_fetched=8,
                pages_useful=3,
            )
            result = await _deep_research("banana export policy", max_pages=8)

        assert "answer" in result
        assert "sources" in result
        assert "pages_fetched" in result
        assert "pages_useful" in result
        assert "formatted" in result
        assert result["query"] == "banana export policy"
