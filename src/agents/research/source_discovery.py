"""
Source Discovery Engine
=======================
Multi-source information discovery.

Integrates:
- Web scraping (WebScrapingAgent)
- Knowledge base retrieval (RAG)
- Graph traversal (Neo4j)
- RSS feeds (AgriScrapers)
"""

import asyncio
from typing import Optional, Any

from loguru import logger

from src.agents.research.models import (
    Finding,
    SourceType,
    Citation,
)


class SourceDiscovery:
    """
    Multi-source discovery engine.
    
    Usage:
        discovery = SourceDiscovery()
        await discovery.initialize()
        
        findings = await discovery.search(
            question="What is the current price of tomatoes in Bengaluru?",
            source_types=[SourceType.WEB, SourceType.KNOWLEDGE_BASE],
        )
    """
    
    def __init__(
        self,
        web_scraper=None,
        knowledge_base=None,
        graph_client=None,
        rss_scraper=None,
    ):
        """
        Initialize discovery engine.
        
        Args:
            web_scraper: WebScrapingAgent instance
            knowledge_base: KnowledgeBase instance
            graph_client: Neo4j graph client
            rss_scraper: RSS scraper instance
        """
        self._web_scraper = web_scraper
        self._knowledge_base = knowledge_base
        self._graph_client = graph_client
        self._rss_scraper = rss_scraper
        self._initialized = False
    
    async def initialize(self):
        """Initialize discovery components."""
        # Lazy load scrapers if not provided
        if self._web_scraper is None:
            try:
                from src.agents.web_scraping_agent import WebScrapingAgent
                self._web_scraper = WebScrapingAgent()
                await self._web_scraper.initialize()
            except Exception as e:
                logger.warning("WebScrapingAgent not available: {}", str(e))
        
        if self._rss_scraper is None:
            try:
                from src.tools.agri_scrapers import RSSNewsScraper
                self._rss_scraper = RSSNewsScraper()
            except Exception as e:
                logger.warning("RSSNewsScraper not available: {}", str(e))
        
        self._initialized = True
        logger.info("SourceDiscovery initialized")
    
    async def search(
        self,
        question: str,
        source_types: list[SourceType],
        max_results: int = 5,
    ) -> list[Finding]:
        """
        Search for information across multiple sources.
        
        Args:
            question: Research question
            source_types: Types of sources to search
            max_results: Maximum results per source type
            
        Returns:
            List of findings from all sources
        """
        if not self._initialized:
            await self.initialize()
        
        all_findings = []
        tasks = []
        
        # Create tasks for each source type
        for source_type in source_types:
            if source_type == SourceType.WEB:
                tasks.append(self._search_web(question, max_results))
            elif source_type == SourceType.KNOWLEDGE_BASE:
                tasks.append(self._search_kb(question, max_results))
            elif source_type == SourceType.GRAPH:
                tasks.append(self._search_graph(question, max_results))
            elif source_type == SourceType.RSS:
                tasks.append(self._search_rss(question, max_results))
        
        # Execute in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Source search failed: {}", str(result))
                elif result:
                    all_findings.extend(result)
        
        logger.info("Found {} total findings for: {}", len(all_findings), question[:50])
        return all_findings
    
    async def _search_web(self, question: str, max_results: int) -> list[Finding]:
        """Search web sources."""
        findings = []
        
        if not self._web_scraper:
            return findings
        
        try:
            # Build search URLs based on question
            search_urls = self._build_search_urls(question)
            
            for url in search_urls[:max_results]:
                result = await self._web_scraper.scrape_to_markdown(url)
                
                if result.success and result.markdown:
                    findings.append(Finding(
                        content=result.markdown[:2000],  # Limit content
                        summary=result.markdown[:300],
                        source_url=url,
                        source_title=self._extract_title(result.markdown),
                        source_type=SourceType.WEB,
                        reliability_score=0.6,  # Default for web
                        relevance_score=0.7,
                        citation=Citation(
                            title=self._extract_title(result.markdown),
                            url=url,
                            source_type=SourceType.WEB,
                        ),
                    ))
                    
        except Exception as e:
            logger.error("Web search failed: {}", str(e))
        
        return findings
    
    async def _search_kb(self, question: str, max_results: int) -> list[Finding]:
        """Search knowledge base."""
        findings = []
        
        if not self._knowledge_base:
            try:
                from src.rag.knowledge_base import KnowledgeBase
                self._knowledge_base = KnowledgeBase()
            except Exception as e:
                logger.warning("KnowledgeBase not available: {}", str(e))
                return findings
        
        try:
            # Query knowledge base
            results = await self._knowledge_base.search(
                query=question,
                limit=max_results,
            )
            
            for doc in results:
                findings.append(Finding(
                    content=doc.content if hasattr(doc, 'content') else str(doc),
                    summary=doc.content[:300] if hasattr(doc, 'content') else str(doc)[:300],
                    source_type=SourceType.KNOWLEDGE_BASE,
                    source_title=doc.metadata.get('title', 'KB Document') if hasattr(doc, 'metadata') else 'KB Document',
                    reliability_score=0.85,  # High trust for KB
                    relevance_score=doc.score if hasattr(doc, 'score') else 0.7,
                    citation=Citation(
                        title=doc.metadata.get('title', 'Knowledge Base') if hasattr(doc, 'metadata') else 'KB',
                        source_type=SourceType.KNOWLEDGE_BASE,
                    ),
                ))
                
        except Exception as e:
            logger.error("KB search failed: {}", str(e))
        
        return findings
    
    async def _search_graph(self, question: str, max_results: int) -> list[Finding]:
        """Search graph database."""
        findings = []
        
        if not self._graph_client:
            try:
                from src.rag.graph_constructor import GraphRAGConstructor
                self._graph_client = GraphRAGConstructor()
            except Exception as e:
                logger.warning("Graph client not available: {}", str(e))
                return findings
        
        try:
            # Extract entities from question for graph query
            entities = self._extract_entities(question)
            
            if hasattr(self._graph_client, 'query_relationships'):
                for entity in entities[:3]:
                    rels = await self._graph_client.query_relationships(entity)
                    
                    if rels:
                        findings.append(Finding(
                            content=f"Entity: {entity}. Relationships: {rels}",
                            summary=f"Found {len(rels)} relationships for {entity}",
                            source_type=SourceType.GRAPH,
                            reliability_score=0.9,  # High trust for graph
                            relevance_score=0.8,
                            citation=Citation(
                                title=f"Knowledge Graph: {entity}",
                                source_type=SourceType.GRAPH,
                            ),
                        ))
                        
        except Exception as e:
            logger.error("Graph search failed: {}", str(e))
        
        return findings
    
    async def _search_rss(self, question: str, max_results: int) -> list[Finding]:
        """Search RSS feeds."""
        findings = []
        
        if not self._rss_scraper:
            return findings
        
        try:
            articles = await self._rss_scraper.get_news("rural_voice", limit=max_results)
            
            for article in articles:
                # Check relevance (simple keyword matching)
                question_lower = question.lower()
                article_text = f"{article.title} {article.summary or ''}".lower()
                
                if any(word in article_text for word in question_lower.split()):
                    findings.append(Finding(
                        content=article.summary or article.title,
                        summary=article.title,
                        source_url=article.url,
                        source_title=article.title,
                        source_type=SourceType.RSS,
                        reliability_score=0.7,
                        relevance_score=0.6,
                        citation=Citation(
                            title=article.title,
                            url=article.url,
                            publisher=article.source,
                            source_type=SourceType.RSS,
                        ),
                    ))
                    
        except Exception as e:
            logger.error("RSS search failed: {}", str(e))
        
        return findings
    
    def _build_search_urls(self, question: str) -> list[str]:
        """Build search URLs for agricultural queries."""
        # For agricultural queries, use known portals
        urls = []
        
        question_lower = question.lower()
        
        # Price queries
        if any(kw in question_lower for kw in ["price", "mandi", "₹", "rate", "cost"]):
            urls.append("https://agmarknet.gov.in/")
            urls.append("https://enam.gov.in/web/dashboard/trade-data")
        
        # Weather queries
        if any(kw in question_lower for kw in ["weather", "rain", "forecast", "climate"]):
            urls.append("https://mausam.imd.gov.in/")
        
        # Crop information
        if any(kw in question_lower for kw in ["grow", "plant", "variety", "seed", "cultivation"]):
            urls.append("https://farmer.gov.in/")
            urls.append("https://www.icar.org.in/")
        
        # Schemes
        if any(kw in question_lower for kw in ["scheme", "subsidy", "loan", "pm-kisan", "insurance"]):
            urls.append("https://pmkisan.gov.in/")
            urls.append("https://www.myscheme.gov.in/schemes")
        
        # Default to general agricultural portal
        if not urls:
            urls.append("https://farmer.gov.in/")
        
        return urls
    
    def _extract_title(self, markdown: str) -> str:
        """Extract title from markdown content."""
        lines = markdown.strip().split('\n')
        for line in lines[:5]:
            if line.startswith('#'):
                return line.lstrip('#').strip()
            if line.strip():
                return line.strip()[:100]
        return "Untitled"
    
    def _extract_entities(self, question: str) -> list[str]:
        """Extract key entities from question for graph queries."""
        # Simple extraction - in production, use NER
        entities = []
        
        # Common agricultural entities
        crops = ["tomato", "rice", "wheat", "cotton", "onion", "potato", "mango", "sugarcane"]
        states = ["karnataka", "maharashtra", "punjab", "uttar pradesh", "andhra pradesh"]
        
        question_lower = question.lower()
        
        for crop in crops:
            if crop in question_lower:
                entities.append(crop.title())
        
        for state in states:
            if state in question_lower:
                entities.append(state.title())
        
        return entities or ["agriculture"]
    
    async def close(self):
        """Clean up resources."""
        if self._web_scraper and hasattr(self._web_scraper, 'close'):
            await self._web_scraper.close()
