"""
Research Agent
==============
Main orchestrator for deep research with multi-step planning and citations.

Features:
- Multi-step research planning
- Multi-source discovery (web, KB, graph)
- Source verification and scoring
- Academic-style citations
- Research memory for learning
- Structured report synthesis
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional, Any

from loguru import logger
from pydantic import BaseModel

from src.agents.base_agent import AgentConfig, AgentResponse, BaseAgent
from src.agents.research.models import (
    Finding,
    ResearchPlan,
    ResearchReport,
    ResearchStep,
    SourceType,
    StepStatus,
)
from src.agents.research.planner import ResearchPlanner
from src.agents.research.source_discovery import SourceDiscovery
from src.agents.research.verifier import SourceVerifier
from src.agents.research.citation_manager import CitationManager
from src.agents.research.memory import ResearchMemory


SYNTHESIS_PROMPT = """You are a research synthesis expert. Create a comprehensive report from the following research findings.

Research Query: {query}
Research Objective: {objective}

Findings:
{findings_text}

Create a well-structured report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points with key insights)
3. Detailed Analysis (organized by topic)
4. Recommendations (if applicable)
5. Limitations and Caveats

Use the citation markers [1], [2], etc. to reference sources.
Be factual, cite sources, and acknowledge uncertainty where appropriate.

Format your response as markdown."""


class ResearchAgent(BaseAgent):
    """
    Deep Research Agent for comprehensive investigations.
    
    Orchestrates:
    - Research planning (query decomposition)
    - Multi-source discovery
    - Verification and scoring
    - Report synthesis with citations
    
    Usage:
        agent = ResearchAgent()
        await agent.initialize()
        
        result = await agent.research("Best tomato varieties for Karnataka climate")
        print(result.content)  # Full report with citations
    """
    
    def __init__(
        self,
        llm=None,
        knowledge_base=None,
        graph_client=None,
    ):
        """
        Initialize Research Agent.
        
        Args:
            llm: LLM provider
            knowledge_base: Knowledge base for retrieval
            graph_client: Neo4j graph client
        """
        config = AgentConfig(
            name="research_agent",
            description="Deep research with multi-step planning, source verification, and citations",
            max_retries=2,
            temperature=0.4,
            max_tokens=4000,
        )
        
        super().__init__(
            config=config,
            llm=llm,
            knowledge_base=knowledge_base,
        )
        
        # Initialize components
        self.planner = ResearchPlanner(llm=llm)
        self.discovery = SourceDiscovery(
            knowledge_base=knowledge_base,
            graph_client=graph_client,
        )
        self.verifier = SourceVerifier(llm=llm)
        self.citations = CitationManager()
        self.memory = ResearchMemory()
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize agent and components."""
        await self.discovery.initialize()
        self._initialized = True
        logger.info("ResearchAgent initialized")
        return True
    
    def _get_system_prompt(self, context: Optional[dict] = None) -> str:
        """Get system prompt for research agent."""
        return """You are a research expert specializing in agricultural and market research.
Your role is to:
- Conduct thorough, multi-step research investigations
- Verify information from multiple sources
- Provide well-cited, accurate reports
- Acknowledge uncertainty when appropriate

Always cite your sources using [1], [2] markers and provide factual, evidence-based answers."""
    
    async def process(
        self,
        query: str,
        context: Optional[dict] = None,
        execution=None,
    ) -> AgentResponse:
        """
        Process a research query.
        
        Args:
            query: Research question
            context: Optional context
            execution: Optional execution state
            
        Returns:
            AgentResponse with research report
        """
        return await self.research(query, context)
    
    async def research(
        self,
        query: str,
        context: Optional[dict] = None,
        max_steps: int = 6,
        min_reliability: float = 0.4,
    ) -> AgentResponse:
        """
        Conduct comprehensive research on a query.
        
        Args:
            query: Research question
            context: Optional context
            max_steps: Maximum research steps
            min_reliability: Minimum source reliability threshold
            
        Returns:
            AgentResponse with full research report
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = datetime.now()
        logger.info("Starting research: {}", query[:50])
        
        # Reset citations for new research
        self.citations.clear()
        
        try:
            # Step 1: Check memory for similar past research
            similar = await self.memory.find_similar(query, limit=3)
            prior_knowledge = ""
            if similar:
                logger.info("Found {} similar past researches", len(similar))
                prior_knowledge = self._extract_prior_knowledge(similar)
            
            # Step 2: Create research plan
            plan = await self.planner.create_plan(query, max_steps=max_steps)
            logger.info("Created plan: {} steps", len(plan.steps))
            
            # Step 3: Execute plan steps
            all_findings = []
            for step in plan.steps:
                step_findings = await self._execute_step(step, plan)
                all_findings.extend(step_findings)
            
            # Step 4: Verify and score findings
            verified_findings = await self.verifier.verify_batch(all_findings)
            verified_findings = await self.verifier.cross_reference(verified_findings)
            
            # Filter low-quality findings
            quality_findings = self.verifier.filter_low_quality(
                verified_findings,
                min_score=min_reliability,
            )
            
            logger.info("Verified {} findings ({} passed quality filter)", 
                       len(verified_findings), len(quality_findings))
            
            # Step 5: Add citations
            for finding in quality_findings:
                self.citations.add_finding(finding)
            
            # Step 6: Synthesize report
            report = await self._synthesize_report(query, plan, quality_findings, prior_knowledge)
            
            # Step 7: Store in memory
            await self.memory.store(query, quality_findings, report)
            
            # Build response
            duration = (datetime.now() - start_time).total_seconds()
            
            return AgentResponse(
                content=report.to_markdown(),
                agent_name="research_agent",
                confidence=report.confidence_score,
                sources=[f.source_url for f in quality_findings if f.source_url],
                reasoning=f"Research completed in {duration:.1f}s with {len(quality_findings)} verified sources",
                tools_used=["planner", "discovery", "verifier", "synthesizer"],
                steps=[f"step_{s.step_id}" for s in plan.steps],
                metadata={
                    "plan_id": plan.plan_id,
                    "total_findings": len(all_findings),
                    "quality_findings": len(quality_findings),
                    "citation_count": self.citations.count,
                    "duration_sec": duration,
                },
            )
            
        except Exception as e:
            logger.error("Research failed: {}", str(e))
            import traceback
            traceback.print_exc()
            
            return AgentResponse(
                content=f"I was unable to complete the research. Error: {str(e)}",
                agent_name="research_agent",
                confidence=0.0,
                error=str(e),
            )
    
    async def _execute_step(
        self,
        step: ResearchStep,
        plan: ResearchPlan,
    ) -> list[Finding]:
        """Execute a single research step."""
        step.status = StepStatus.IN_PROGRESS
        step.started_at = datetime.now()
        
        logger.debug("Executing step {}: {}", step.step_id, step.question[:50])
        
        try:
            # Search sources
            findings = await self.discovery.search(
                question=step.question,
                source_types=step.source_types,
                max_results=3,
            )
            
            step.findings = findings
            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now()
            
            logger.debug("Step {} completed: {} findings", step.step_id, len(findings))
            return findings
            
        except Exception as e:
            logger.error("Step {} failed: {}", step.step_id, str(e))
            step.status = StepStatus.FAILED
            step.error = str(e)
            return []
    
    async def _synthesize_report(
        self,
        query: str,
        plan: ResearchPlan,
        findings: list[Finding],
        prior_knowledge: str = "",
    ) -> ResearchReport:
        """Synthesize findings into a report."""
        
        if not findings:
            return ResearchReport(
                title=f"Research: {query}",
                query=query,
                summary="No reliable sources found for this query.",
                confidence_score=0.0,
            )
        
        # Build findings text for LLM
        findings_text = []
        for i, finding in enumerate(findings, 1):
            ref = f"[{finding.citation.ref_id}]" if finding.citation else ""
            findings_text.append(
                f"{i}. {ref} (reliability: {finding.reliability_score:.2f})\n"
                f"   {finding.summary or finding.content[:300]}"
            )
        
        if self.llm:
            # Use LLM for synthesis
            prompt = SYNTHESIS_PROMPT.format(
                query=query,
                objective=plan.objective,
                findings_text="\n".join(findings_text),
            )
            
            try:
                synthesis = await self.generate_with_llm(
                    [{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_tokens=2000,
                )
                
                report = ResearchReport(
                    title=f"Research Report: {query}",
                    query=query,
                    summary=self._extract_summary(synthesis),
                    sections=[{"title": "Analysis", "content": synthesis}],
                    findings=findings,
                    citations=self.citations.get_all_citations(),
                    confidence_score=self._calculate_confidence(findings),
                    total_sources=len(findings),
                )
                
                return report
                
            except Exception as e:
                logger.error("LLM synthesis failed: {}", str(e))
        
        # Fallback: simple aggregation
        report = ResearchReport(
            title=f"Research: {query}",
            query=query,
            summary=f"Found {len(findings)} sources on this topic.",
            sections=[{
                "title": "Findings",
                "content": "\n".join(f"- {f.summary}" for f in findings[:10]),
            }],
            findings=findings,
            citations=self.citations.get_all_citations(),
            confidence_score=self._calculate_confidence(findings),
            total_sources=len(findings),
        )
        
        return report
    
    def _extract_prior_knowledge(self, similar: list[dict]) -> str:
        """Extract key insights from similar past research."""
        insights = []
        for entry in similar[:3]:
            if "entry" in entry:
                facts = entry["entry"].get("key_facts", [])
                insights.extend(facts[:3])
        return " | ".join(insights) if insights else ""
    
    def _extract_summary(self, synthesis: str) -> str:
        """Extract summary from synthesized report."""
        lines = synthesis.strip().split('\n')
        for i, line in enumerate(lines):
            if "summary" in line.lower():
                # Return next non-empty lines
                summary_lines = []
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('#'):
                        summary_lines.append(lines[j].strip())
                    elif lines[j].startswith('#'):
                        break
                if summary_lines:
                    return " ".join(summary_lines)
        
        # Fallback: first paragraph
        return lines[0] if lines else "Research completed."
    
    def _calculate_confidence(self, findings: list[Finding]) -> float:
        """Calculate overall confidence score."""
        if not findings:
            return 0.0
        
        # Weighted average of reliability scores
        total_weight = 0
        weighted_sum = 0
        
        for finding in findings:
            weight = finding.relevance_score
            weighted_sum += finding.reliability_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return round(weighted_sum / total_weight, 2)
    
    async def close(self):
        """Clean up resources."""
        await self.discovery.close()
