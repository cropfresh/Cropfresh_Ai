"""
LLM Schema Extractor Mixin
==========================
Handles structured data extraction using Pydantic schemas and LLMs.
"""

import json
import re
from datetime import datetime
from typing import Any, Optional, Type

from loguru import logger
from pydantic import BaseModel

from src.tools.web_search import WebSearchTool

from .browser import BrowserScraperMixin
from .models import ScrapingConfig, ScrapingResult


class LLMExtractorMixin(BrowserScraperMixin):
    """Mixin for unstructured-to-structured extraction via LLM."""

    async def scrape_with_schema(
        self,
        url: str,
        schema: Type[BaseModel],
        instruction: str,
        config: Optional[ScrapingConfig] = None,
        fallback_query: Optional[str] = None,
    ) -> ScrapingResult:
        """Extract structured data using LLM and Pydantic schema."""
        if not self.llm_provider:
            return ScrapingResult(
                url=url,
                success=False,
                error="LLM provider not configured for schema extraction",
            )

        markdown_result = await self.scrape_to_markdown(url, config)

        if not markdown_result.success and fallback_query:
            logger.warning(f"URL {url} failed. Initiating search fallback for: {fallback_query}")
            search_tool = WebSearchTool()
            search_results = await search_tool.search(fallback_query, max_results=1)

            if search_results and search_results.results:
                new_url = search_results.results[0].url
                logger.info(f"Fallback Search found new URL: {new_url}. Retrying scrape...")
                url = new_url
                markdown_result = await self.scrape_to_markdown(new_url, config)

        if not markdown_result.success:
            return markdown_result

        try:
            extracted = await self._extract_chunks_with_llm(
                markdown=markdown_result.markdown,
                schema=schema,
                instruction=instruction
            )

            return ScrapingResult(
                url=url,
                success=True,
                markdown=markdown_result.markdown,
                extracted_data=extracted,
                scrape_time_ms=markdown_result.scrape_time_ms,
            )

        except Exception as e:
            logger.error("Schema extraction failed for {}: {}", url, str(e))
            return ScrapingResult(
                url=url,
                success=False,
                markdown=markdown_result.markdown,
                error=f"LLM extraction failed: {str(e)}",
                scrape_time_ms=markdown_result.scrape_time_ms,
            )

    async def extract_with_schema(
        self,
        html_content: str,
        url: str,
        schema: Type[BaseModel],
        instruction: str,
    ) -> ScrapingResult:
        """Extract structured data using LLM and Pydantic schema from raw HTML."""
        if not self.llm_provider:
            return ScrapingResult(
                url=url,
                success=False,
                error="LLM provider not configured for schema extraction",
            )

        start_time = datetime.now()
        markdown = self._html_to_markdown(html_content)

        try:
            extracted = await self._extract_chunks_with_llm(
                markdown=markdown,
                schema=schema,
                instruction=instruction
            )

            return ScrapingResult(
                url=url,
                success=True,
                markdown=markdown,
                extracted_data=extracted,
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

        except Exception as e:
            logger.error("Schema extraction failed for {}: {}", url, str(e))
            return ScrapingResult(
                url=url,
                success=False,
                markdown=markdown,
                error=f"LLM extraction failed: {str(e)}",
                scrape_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )

    async def _extract_chunks_with_llm(
        self,
        markdown: str,
        schema: Type[BaseModel],
        instruction: str,
        chunk_size: int = 15000,
    ) -> Any:
        """Helper to chunk markdown and extract data safely across chunks."""
        chunks = []
        current_chunk = ""

        paragraphs = markdown.split("\n\n")

        for p in paragraphs:
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        if not chunks:
            return []

        all_results = []
        is_list_expected = False

        for chunk in chunks:
            extraction_prompt = f"""Extract data from the following web page content according to the schema.

INSTRUCTION: {instruction}

SCHEMA:
{json.dumps(schema.model_json_schema(), indent=2)}

PAGE CONTENT:
{chunk}

Return a valid JSON object or array matching the schema. Only output JSON, no explanation."""

            llm_response = await self.llm_provider.agenerate(extraction_prompt)

            json_match = re.search(r'[\[\{].*[\]\}]', llm_response, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                if isinstance(extracted, list):
                    is_list_expected = True
                    all_results.extend(extracted)
                elif isinstance(extracted, dict) and extracted:
                    all_results.append(extracted)

        if is_list_expected:
            return all_results

        if all_results:
            return all_results[0]

        return {}
