"""
Research Memory
===============
Persistent memory for research findings and incremental learning.

Features:
- Store and retrieve past research
- Semantic similarity search
- Fact consolidation
- Learning from feedback
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from src.agents.research.models import (
    Finding,
    ResearchMemoryEntry,
    ResearchReport,
)


class ResearchMemory:
    """
    Persistent research memory with incremental learning.
    
    Usage:
        memory = ResearchMemory()
        
        # Store research
        await memory.store(query="tomato prices", findings=[...])
        
        # Retrieve similar past research
        similar = await memory.find_similar("tomato market rates")
        
        # Learn from feedback
        await memory.learn_from_feedback(query, was_helpful=True)
    """
    
    MEMORY_DIR = Path("data/research_memory")
    
    def __init__(self, memory_dir: Optional[Path] = None):
        """
        Initialize research memory.
        
        Args:
            memory_dir: Directory for memory storage
        """
        self.memory_dir = memory_dir or self.MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        self._index_file = self.memory_dir / "index.json"
        self._index: list[dict] = []
        
        self._load_index()
    
    def _load_index(self):
        """Load memory index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
                logger.info("Loaded research memory index: {} entries", len(self._index))
            except Exception as e:
                logger.error("Failed to load memory index: {}", str(e))
                self._index = []
    
    def _save_index(self):
        """Save memory index to disk."""
        try:
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save memory index: {}", str(e))
    
    async def store(
        self,
        query: str,
        findings: list[Finding],
        report: Optional[ResearchReport] = None,
    ) -> str:
        """
        Store research findings in memory.
        
        Args:
            query: Original research query
            findings: List of findings
            report: Optional final report
            
        Returns:
            Memory entry ID
        """
        import hashlib
        
        # Generate entry ID
        entry_id = hashlib.md5(f"{query}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        # Extract key facts from findings
        key_facts = self._extract_key_facts(findings)
        
        # Create entry
        entry = ResearchMemoryEntry(
            query=query,
            findings_summary=self._summarize_findings(findings),
            key_facts=key_facts,
            source_count=len(findings),
            avg_reliability=sum(f.reliability_score for f in findings) / max(len(findings), 1),
        )
        
        # Save entry file
        entry_file = self.memory_dir / f"{entry_id}.json"
        with open(entry_file, 'w', encoding='utf-8') as f:
            json.dump({
                "entry": entry.model_dump(mode='json'),
                "findings": [f.model_dump(mode='json') for f in findings],
                "report": report.model_dump(mode='json') if report else None,
            }, f, indent=2, default=str)
        
        # Update index
        self._index.append({
            "id": entry_id,
            "query": query,
            "key_facts": key_facts[:5],
            "created_at": datetime.now().isoformat(),
            "accessed_count": 0,
        })
        self._save_index()
        
        logger.info("Stored research memory: {} ({} findings)", entry_id, len(findings))
        return entry_id
    
    async def find_similar(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Find similar past research by query.
        
        Args:
            query: Query to match
            limit: Maximum results
            
        Returns:
            List of similar memory entries
        """
        if not self._index:
            return []
        
        # Simple keyword matching (in production, use embeddings)
        query_words = set(query.lower().split())
        
        scored_entries = []
        for entry in self._index:
            entry_words = set(entry["query"].lower().split())
            
            # Add key facts words
            for fact in entry.get("key_facts", []):
                entry_words.update(fact.lower().split())
            
            # Calculate overlap
            overlap = len(query_words & entry_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scored_entries.append((score, entry))
        
        # Sort by score and return top entries
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, entry in scored_entries[:limit]:
            full_entry = await self.get_entry(entry["id"])
            if full_entry:
                full_entry["similarity_score"] = score
                results.append(full_entry)
        
        return results
    
    async def get_entry(self, entry_id: str) -> Optional[dict]:
        """Get a memory entry by ID."""
        entry_file = self.memory_dir / f"{entry_id}.json"
        
        if not entry_file.exists():
            return None
        
        try:
            with open(entry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update access count
            for idx_entry in self._index:
                if idx_entry["id"] == entry_id:
                    idx_entry["accessed_count"] = idx_entry.get("accessed_count", 0) + 1
                    self._save_index()
                    break
            
            return data
            
        except Exception as e:
            logger.error("Failed to load entry {}: {}", entry_id, str(e))
            return None
    
    async def learn_from_feedback(
        self,
        entry_id: str,
        was_helpful: bool,
        feedback_text: Optional[str] = None,
    ):
        """
        Learn from user feedback on research.
        
        Args:
            entry_id: Memory entry ID
            was_helpful: Whether research was helpful
            feedback_text: Optional feedback text
        """
        entry_file = self.memory_dir / f"{entry_id}.json"
        
        if not entry_file.exists():
            return
        
        try:
            with open(entry_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add feedback
            if "feedback" not in data:
                data["feedback"] = []
            
            data["feedback"].append({
                "helpful": was_helpful,
                "text": feedback_text,
                "timestamp": datetime.now().isoformat(),
            })
            
            with open(entry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info("Recorded feedback for {}: helpful={}", entry_id, was_helpful)
            
        except Exception as e:
            logger.error("Failed to record feedback: {}", str(e))
    
    def _extract_key_facts(self, findings: list[Finding]) -> list[str]:
        """Extract key facts from findings."""
        facts = []
        
        for finding in findings:
            # Extract numbers with context
            content = finding.content or finding.summary
            words = content.split()
            
            for i, word in enumerate(words):
                # Look for numeric facts
                if any(c.isdigit() for c in word):
                    context_start = max(0, i - 3)
                    context_end = min(len(words), i + 4)
                    fact = " ".join(words[context_start:context_end])
                    if len(fact) < 100:
                        facts.append(fact)
        
        return facts[:10]  # Limit to 10 facts
    
    def _summarize_findings(self, findings: list[Finding]) -> str:
        """Create summary of findings."""
        summaries = []
        for finding in findings[:5]:  # Top 5 findings
            summary = finding.summary or finding.content[:200]
            summaries.append(summary)
        
        return " | ".join(summaries)
    
    @property
    def entry_count(self) -> int:
        """Number of memory entries."""
        return len(self._index)
