"""
Research Planner
================
Generates multi-step research plans from queries.

Uses LLM to decompose complex queries into:
- Sub-questions
- Source type assignments
- Dependency ordering
"""

import json
import uuid
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from src.agents.research.models import (
    ResearchPlan,
    ResearchStep,
    SourceType,
    StepStatus,
)


PLANNING_PROMPT = """You are a research planning assistant. Given a research query, create a structured plan.

Break down the query into 3-6 sub-questions that together will answer the main query comprehensively.

For each sub-question, specify:
1. The question itself
2. Best source types: "web" (live data), "kb" (knowledge base), "graph" (relationship data)
3. Dependencies (which questions need to be answered first)

Respond with JSON:
{{
    "objective": "Clear statement of research goal",
    "sub_questions": ["q1", "q2", ...],
    "steps": [
        {{
            "step_id": "1",
            "question": "sub-question text",
            "description": "what this step accomplishes",
            "source_types": ["web", "kb"],
            "depends_on": []
        }},
        ...
    ],
    "estimated_time_sec": 120
}}

Research Query: {query}

Respond ONLY with valid JSON."""


class ResearchPlanner:
    """
    Generates multi-step research plans.
    
    Usage:
        planner = ResearchPlanner(llm=groq_llm)
        plan = await planner.create_plan("Best tomato varieties for Karnataka")
    """
    
    def __init__(self, llm=None):
        """
        Initialize planner.
        
        Args:
            llm: LLM provider for plan generation
        """
        self.llm = llm
    
    async def create_plan(
        self,
        query: str,
        max_steps: int = 6,
    ) -> ResearchPlan:
        """
        Create a research plan for a query.
        
        Args:
            query: Research question
            max_steps: Maximum number of steps
            
        Returns:
            ResearchPlan with ordered steps
        """
        plan_id = str(uuid.uuid4())[:8]
        
        if not self.llm:
            # Fallback to simple plan
            return self._create_simple_plan(query, plan_id)
        
        try:
            # Generate plan via LLM
            prompt = PLANNING_PROMPT.format(query=query)
            
            response = await self.llm.agenerate(prompt)
            
            # Parse JSON response
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            plan_data = json.loads(response_text)
            
            # Build steps
            steps = []
            for step_data in plan_data.get("steps", [])[:max_steps]:
                source_types = [
                    SourceType(st) for st in step_data.get("source_types", ["web"])
                    if st in [e.value for e in SourceType]
                ]
                
                steps.append(ResearchStep(
                    step_id=step_data.get("step_id", str(len(steps) + 1)),
                    question=step_data.get("question", ""),
                    description=step_data.get("description", ""),
                    source_types=source_types or [SourceType.WEB],
                    depends_on=step_data.get("depends_on", []),
                ))
            
            plan = ResearchPlan(
                plan_id=plan_id,
                original_query=query,
                refined_query=query,
                objective=plan_data.get("objective", query),
                sub_questions=plan_data.get("sub_questions", []),
                steps=steps,
                estimated_time_sec=plan_data.get("estimated_time_sec", 120),
            )
            
            logger.info("Created research plan: {} steps, ~{}s", len(steps), plan.estimated_time_sec)
            return plan
            
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse plan JSON: {}", str(e))
            return self._create_simple_plan(query, plan_id)
        except Exception as e:
            logger.error("Plan generation failed: {}", str(e))
            return self._create_simple_plan(query, plan_id)
    
    def _create_simple_plan(self, query: str, plan_id: str) -> ResearchPlan:
        """Create a simple fallback plan."""
        steps = [
            ResearchStep(
                step_id="1",
                question=f"Find general information about: {query}",
                description="Initial broad search",
                source_types=[SourceType.KNOWLEDGE_BASE, SourceType.WEB],
                depends_on=[],
            ),
            ResearchStep(
                step_id="2", 
                question=f"Find specific details and data for: {query}",
                description="Detailed data gathering",
                source_types=[SourceType.WEB, SourceType.GRAPH],
                depends_on=["1"],
            ),
            ResearchStep(
                step_id="3",
                question=f"Find expert opinions and recommendations for: {query}",
                description="Expert insights",
                source_types=[SourceType.WEB, SourceType.RSS],
                depends_on=["1"],
            ),
        ]
        
        return ResearchPlan(
            plan_id=plan_id,
            original_query=query,
            objective=f"Research: {query}",
            sub_questions=[s.question for s in steps],
            steps=steps,
            estimated_time_sec=90,
        )
    
    async def refine_plan(
        self,
        plan: ResearchPlan,
        feedback: str,
    ) -> ResearchPlan:
        """
        Refine a plan based on feedback.
        
        Args:
            plan: Existing plan
            feedback: User or system feedback
            
        Returns:
            Refined plan
        """
        # For now, just add a clarification step
        new_step = ResearchStep(
            step_id=str(len(plan.steps) + 1),
            question=feedback,
            description="Clarification based on feedback",
            source_types=[SourceType.WEB],
            depends_on=[plan.steps[-1].step_id if plan.steps else []],
        )
        
        plan.steps.append(new_step)
        plan.sub_questions.append(feedback)
        
        return plan
