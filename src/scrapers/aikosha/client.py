"""
AI Kosha Client
===============
Client for India's AI Kosha platform (indiaai.gov.in).
"""

from typing import Any, Optional
import httpx
from loguru import logger

from .models import AIKoshaCategory, AIKoshaDataset, AIKoshaSearchResult
from .catalog import get_agri_catalog


class AIKoshaClient:
    """
    Client for India's AI Kosha platform (indiaai.gov.in).

    Provides access to 10,000+ datasets with focus on agricultural data.

    Usage:
        client = AIKoshaClient(api_key="your_api_key")

        # Search agricultural datasets
        result = await client.search_datasets(
            query="crop prices Karnataka",
            category=AIKoshaCategory.AGRICULTURE,
        )

        # Get specific dataset
        dataset = await client.get_dataset("dataset_id_123")

        # Download dataset data
        data = await client.download_dataset("dataset_id_123")
    """

    BASE_URL = "https://indiaai.gov.in/api/v1"
    KOSHA_URL = "https://indiaai.gov.in/ai-kosha"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = False

        self._agri_catalog = get_agri_catalog()

        logger.info(
            f"🇮🇳 AI Kosha client initialized "
            f"(api_key={'configured' if api_key else 'not set'})"
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.timeout,
            )
        return self._client

    # ── Dataset Discovery ─────────────────────────────────────────────

    async def search_datasets(
        self,
        query: str = "",
        category: Optional[AIKoshaCategory] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> AIKoshaSearchResult:
        """Search AI Kosha datasets."""
        if self.api_key:
            return await self._search_via_api(query, category, page, per_page)
        else:
            return self._search_local_catalog(query, category, page, per_page)

    async def _search_via_api(
        self,
        query: str,
        category: Optional[AIKoshaCategory],
        page: int,
        per_page: int,
    ) -> AIKoshaSearchResult:
        """Search datasets via AI Kosha API."""
        try:
            client = await self._get_client()
            params: dict[str, Any] = {
                "q": query,
                "page": page,
                "per_page": per_page,
            }
            if category:
                params["category"] = category.value

            response = await client.get("/datasets/search", params=params)
            response.raise_for_status()

            data = response.json()
            datasets = [
                AIKoshaDataset(
                    id=d.get("id", ""),
                    title=d.get("title", ""),
                    description=d.get("description", ""),
                    category=d.get("category", ""),
                    source_organization=d.get("source_organization", ""),
                    format=d.get("format", ""),
                    record_count=d.get("record_count"),
                    tags=d.get("tags", []),
                    ai_readiness_score=d.get("ai_readiness_score"),
                )
                for d in data.get("results", [])
            ]

            return AIKoshaSearchResult(
                total_results=data.get("total", 0),
                page=page,
                per_page=per_page,
                datasets=datasets,
                query=query,
            )

        except Exception as e:
            logger.error(f"AI Kosha API search failed: {e}")
            return self._search_local_catalog(query, category, page, per_page)

    def _search_local_catalog(
        self,
        query: str,
        category: Optional[AIKoshaCategory],
        page: int,
        per_page: int,
    ) -> AIKoshaSearchResult:
        """Search the built-in curated agricultural dataset catalog."""
        results = self._agri_catalog

        if category:
            results = [d for d in results if category.value in d.category]

        if query:
            query_lower = query.lower()
            results = [
                d
                for d in results
                if query_lower in d.title.lower()
                or query_lower in d.description.lower()
                or any(query_lower in tag.lower() for tag in d.tags)
            ]

        start = (page - 1) * per_page
        end = start + per_page
        paginated = results[start:end]

        return AIKoshaSearchResult(
            total_results=len(results),
            page=page,
            per_page=per_page,
            datasets=paginated,
            query=query,
        )

    # ── Dataset Access ────────────────────────────────────────────────

    async def get_dataset(self, dataset_id: str) -> Optional[AIKoshaDataset]:
        """Get details of a specific dataset."""
        if self.api_key:
            try:
                client = await self._get_client()
                response = await client.get(f"/datasets/{dataset_id}")
                response.raise_for_status()
                data = response.json()
                return AIKoshaDataset(**data)
            except Exception as e:
                logger.error(f"Failed to get dataset {dataset_id}: {e}")

        for d in self._agri_catalog:
            if d.id == dataset_id:
                return d
        return None

    async def download_dataset(
        self,
        dataset_id: str,
        format: str = "json",
    ) -> Optional[list[dict]]:
        """Download and parse a dataset."""
        dataset = await self.get_dataset(dataset_id)
        if not dataset:
            logger.error(f"Dataset {dataset_id} not found")
            return None

        if not dataset.download_url and not self.api_key:
            logger.warning(
                f"Dataset {dataset_id} requires API key for download. "
                "Register at https://indiaai.gov.in/ai-kosha"
            )
            return None

        try:
            client = await self._get_client()
            url = dataset.download_url or f"/datasets/{dataset_id}/download"
            response = await client.get(url, params={"format": format})
            response.raise_for_status()
            return response.json() if format == "json" else []
        except Exception as e:
            logger.error(f"Dataset download failed: {e}")
            return None

    # ── Agricultural Data Helpers ─────────────────────────────────────

    async def get_crop_price_datasets(self) -> AIKoshaSearchResult:
        """Search for crop price datasets."""
        return await self.search_datasets(
            query="commodity prices mandi",
            category=AIKoshaCategory.AGRICULTURE,
        )

    async def get_weather_datasets(self) -> AIKoshaSearchResult:
        """Search for weather/meteorological datasets."""
        return await self.search_datasets(
            query="weather rainfall temperature",
            category=AIKoshaCategory.METEOROLOGY,
        )

    async def get_soil_datasets(self) -> AIKoshaSearchResult:
        """Search for soil health datasets."""
        return await self.search_datasets(
            query="soil health nutrients",
            category=AIKoshaCategory.AGRICULTURE,
        )

    async def get_satellite_datasets(self) -> AIKoshaSearchResult:
        """Search for satellite/remote sensing datasets for agriculture."""
        return await self.search_datasets(
            query="agriculture satellite NDVI crop monitoring",
            category=AIKoshaCategory.SATELLITE,
        )

    # ── Health ────────────────────────────────────────────────────────

    async def health_check(self) -> dict:
        """Check AI Kosha API availability."""
        status = {
            "name": "ai_kosha",
            "api_key_configured": bool(self.api_key),
            "local_catalog_size": len(self._agri_catalog),
            "status": "available",
        }

        if self.api_key:
            try:
                client = await self._get_client()
                response = await client.get("/health", timeout=5.0)
                status["api_reachable"] = response.status_code == 200
            except Exception:
                status["api_reachable"] = False
                status["status"] = "degraded"

        return status

    # ── Cleanup ───────────────────────────────────────────────────────

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("🇮🇳 AI Kosha client closed")
