"""
Source Verifier
===============
Verifies source reliability and scores findings.

Checks:
- Domain authority
- Content freshness
- Cross-reference consistency
- Bias indicators
"""

from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse

from loguru import logger

from src.agents.research.models import Finding, SourceType


# Domain authority scores (0-1)
DOMAIN_AUTHORITY = {
    # Government (highest trust)
    "gov.in": 0.95,
    "nic.in": 0.95,
    "icar.org.in": 0.9,
    "imd.gov.in": 0.95,
    "enam.gov.in": 0.9,
    "agmarknet.gov.in": 0.9,
    "farmer.gov.in": 0.9,
    "pmkisan.gov.in": 0.9,
    
    # Research institutions
    "iari.res.in": 0.85,
    "icrisat.org": 0.85,
    "cgiar.org": 0.85,
    
    # Reputable news
    "economictimes.com": 0.75,
    "thehindu.com": 0.75,
    "reuters.com": 0.8,
    
    # Agricultural portals
    "ruralvoice.in": 0.7,
    "agrifarming.in": 0.65,
    "krishijagran.com": 0.65,
    
    # General sites
    "wikipedia.org": 0.6,
    
    # Default for unknown
    "default": 0.5,
}

# Source type base reliability
SOURCE_TYPE_RELIABILITY = {
    SourceType.KNOWLEDGE_BASE: 0.85,
    SourceType.GRAPH: 0.9,
    SourceType.WEB: 0.6,
    SourceType.RSS: 0.65,
    SourceType.API: 0.8,
}


class SourceVerifier:
    """
    Verifies and scores source reliability.
    
    Usage:
        verifier = SourceVerifier()
        scored_finding = await verifier.verify(finding)
        print(f"Reliability: {scored_finding.reliability_score}")
    """
    
    def __init__(self, llm=None):
        """
        Initialize verifier.
        
        Args:
            llm: Optional LLM for advanced verification
        """
        self.llm = llm
    
    async def verify(self, finding: Finding) -> Finding:
        """
        Verify and score a finding.
        
        Args:
            finding: Finding to verify
            
        Returns:
            Finding with updated reliability_score
        """
        scores = []
        
        # 1. Domain authority
        domain_score = self._get_domain_authority(finding.source_url)
        scores.append(domain_score)
        
        # 2. Source type base score
        type_score = SOURCE_TYPE_RELIABILITY.get(finding.source_type, 0.5)
        scores.append(type_score)
        
        # 3. Freshness score
        freshness_score = self._get_freshness_score(finding.timestamp)
        scores.append(freshness_score)
        
        # 4. Content quality score
        quality_score = self._get_content_quality_score(finding.content)
        scores.append(quality_score)
        
        # Calculate weighted average
        weights = [0.35, 0.25, 0.2, 0.2]  # Domain, Type, Freshness, Quality
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        finding.reliability_score = round(final_score, 3)
        
        logger.debug(
            "Verified {} (domain={:.2f}, type={:.2f}, fresh={:.2f}, quality={:.2f}) -> {:.2f}",
            finding.source_url or finding.source_type.value,
            domain_score, type_score, freshness_score, quality_score,
            final_score,
        )
        
        return finding
    
    async def verify_batch(self, findings: list[Finding]) -> list[Finding]:
        """Verify multiple findings."""
        return [await self.verify(f) for f in findings]
    
    async def cross_reference(
        self,
        findings: list[Finding],
        min_agreement: float = 0.6,
    ) -> list[Finding]:
        """
        Cross-reference findings and boost scores for consistent information.
        
        Args:
            findings: List of findings to cross-reference
            min_agreement: Minimum agreement threshold
            
        Returns:
            Findings with adjusted scores
        """
        if len(findings) < 2:
            return findings
        
        # Simple cross-reference: check for overlapping key terms
        for i, finding in enumerate(findings):
            agreement_count = 0
            finding_words = set(finding.content.lower().split())
            
            for j, other in enumerate(findings):
                if i != j:
                    other_words = set(other.content.lower().split())
                    overlap = len(finding_words & other_words) / max(len(finding_words), 1)
                    if overlap > 0.1:  # Some meaningful overlap
                        agreement_count += 1
            
            # Boost score if multiple sources agree
            if agreement_count > 0:
                boost = min(agreement_count * 0.05, 0.15)  # Max 15% boost
                finding.reliability_score = min(1.0, finding.reliability_score + boost)
                finding.metadata["cross_referenced"] = True
                finding.metadata["agreement_count"] = agreement_count
        
        return findings
    
    def _get_domain_authority(self, url: Optional[str]) -> float:
        """Get domain authority score."""
        if not url:
            return 0.5
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check for exact match
            for known_domain, score in DOMAIN_AUTHORITY.items():
                if domain.endswith(known_domain):
                    return score
            
            # Check TLD
            if domain.endswith(".gov.in"):
                return 0.9
            if domain.endswith(".edu") or domain.endswith(".ac.in"):
                return 0.8
            if domain.endswith(".org"):
                return 0.7
            
            return DOMAIN_AUTHORITY["default"]
            
        except Exception:
            return 0.5
    
    def _get_freshness_score(self, timestamp: datetime) -> float:
        """Score based on content freshness."""
        age = datetime.now() - timestamp
        
        if age < timedelta(hours=1):
            return 1.0
        elif age < timedelta(days=1):
            return 0.95
        elif age < timedelta(days=7):
            return 0.85
        elif age < timedelta(days=30):
            return 0.7
        elif age < timedelta(days=365):
            return 0.5
        else:
            return 0.3
    
    def _get_content_quality_score(self, content: str) -> float:
        """Score based on content quality indicators."""
        if not content:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length (longer usually more detailed)
        if len(content) > 500:
            score += 0.1
        if len(content) > 1000:
            score += 0.1
        
        # Contains numbers (data-driven)
        if any(c.isdigit() for c in content):
            score += 0.1
        
        # Contains units (₹, kg, quintal)
        if any(u in content.lower() for u in ["₹", "rs", "kg", "quintal", "hectare", "%"]):
            score += 0.1
        
        # Contains dates (recent context)
        if any(y in content for y in ["2024", "2025", "2026"]):
            score += 0.1
        
        return min(1.0, score)
    
    def filter_low_quality(
        self,
        findings: list[Finding],
        min_score: float = 0.4,
    ) -> list[Finding]:
        """Filter out low-quality findings."""
        return [f for f in findings if f.reliability_score >= min_score]
