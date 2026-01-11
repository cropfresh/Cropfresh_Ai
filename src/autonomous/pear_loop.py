"""
PEAR Loop
=========
Plan-Execute-Act-Reflect pattern for autonomous task completion.

Loop stages:
1. PLAN: Create actionable plan from objective
2. EXECUTE: Run the plan steps
3. ACT: Take actions based on results
4. REFLECT: Analyze outcomes and adjust
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable

from loguru import logger
from pydantic import BaseModel, Field


class PEARStage(str, Enum):
    """Stages of PEAR loop."""
    PLAN = "plan"
    EXECUTE = "execute"
    ACT = "act"
    REFLECT = "reflect"


class PlanStep(BaseModel):
    """A step in a plan."""
    step_id: int
    action: str
    description: str = ""
    expected_outcome: str = ""
    actual_outcome: Optional[str] = None
    success: bool = False
    duration_ms: float = 0


class PEARState(BaseModel):
    """State of a PEAR loop iteration."""
    iteration: int = 0
    stage: PEARStage = PEARStage.PLAN
    objective: str
    plan: list[PlanStep] = Field(default_factory=list)
    current_step: int = 0
    
    # Execution results
    results: list[Any] = Field(default_factory=list)
    
    # Reflection
    reflection: str = ""
    needs_replanning: bool = False
    
    # Timing
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


PLAN_PROMPT = """Create an actionable plan for this objective.

Objective: {objective}
Context: {context}

Create 3-6 concrete steps. For each step, specify:
1. Action to take
2. Expected outcome

Respond with JSON:
{{
    "steps": [
        {{
            "action": "what to do",
            "description": "details",
            "expected_outcome": "what should result"
        }}
    ]
}}"""


REFLECT_PROMPT = """Reflect on the execution results and suggest improvements.

Objective: {objective}

Plan and Results:
{plan_results}

Analyze:
1. What worked well?
2. What failed or was suboptimal?
3. Should we replan or proceed?

Respond with JSON:
{{
    "reflection": "analysis text",
    "success_rate": 0.0-1.0,
    "needs_replanning": true/false,
    "improvements": ["suggestion1", "suggestion2"]
}}"""


class PEARLoop:
    """
    Plan-Execute-Act-Reflect autonomous loop.
    
    Usage:
        pear = PEARLoop(llm=groq)
        
        result = await pear.run(
            objective="Get current tomato prices and recommend action",
            executor=my_execution_function,
            max_iterations=3
        )
    """
    
    def __init__(
        self,
        llm=None,
        on_stage_change: Optional[Callable[[PEARStage, PEARState], None]] = None,
    ):
        """
        Initialize PEAR loop.
        
        Args:
            llm: LLM for planning and reflection
            on_stage_change: Callback when stage changes
        """
        self.llm = llm
        self.on_stage_change = on_stage_change
    
    async def run(
        self,
        objective: str,
        executor: Callable[[str], Any],
        context: str = "",
        max_iterations: int = 3,
        human_checkpoint: Optional[Callable[[PEARState], bool]] = None,
    ) -> PEARState:
        """
        Run PEAR loop until objective is met or max iterations.
        
        Args:
            objective: What to accomplish
            executor: Async function to execute actions
            context: Additional context
            max_iterations: Maximum loop iterations
            human_checkpoint: Optional function for human approval
            
        Returns:
            Final PEARState
        """
        state = PEARState(objective=objective)
        
        for iteration in range(max_iterations):
            state.iteration = iteration + 1
            logger.info("PEAR iteration {} for: {}", iteration + 1, objective[:40])
            
            # PLAN
            state.stage = PEARStage.PLAN
            self._notify_stage(state)
            
            state.plan = await self._plan(objective, context, state)
            
            # Human checkpoint if provided
            if human_checkpoint and not human_checkpoint(state):
                logger.info("Human rejected plan, stopping")
                break
            
            # EXECUTE
            state.stage = PEARStage.EXECUTE
            self._notify_stage(state)
            
            state.results = await self._execute(state.plan, executor)
            
            # ACT
            state.stage = PEARStage.ACT
            self._notify_stage(state)
            
            await self._act(state)
            
            # REFLECT
            state.stage = PEARStage.REFLECT
            self._notify_stage(state)
            
            await self._reflect(state)
            
            # Check if we should continue
            if not state.needs_replanning:
                logger.info("PEAR completed after {} iterations", iteration + 1)
                break
            
            # Add reflection to context for next iteration
            context = f"{context}\n\nPrevious attempt: {state.reflection}"
        
        state.completed_at = datetime.now()
        return state
    
    async def _plan(
        self,
        objective: str,
        context: str,
        state: PEARState,
    ) -> list[PlanStep]:
        """Create a plan."""
        if self.llm:
            try:
                import json
                
                prompt = PLAN_PROMPT.format(objective=objective, context=context)
                response = await self.llm.agenerate(prompt)
                
                # Parse JSON
                response_text = response.strip()
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                
                data = json.loads(response_text)
                
                steps = []
                for i, step_data in enumerate(data.get("steps", []), 1):
                    steps.append(PlanStep(
                        step_id=i,
                        action=step_data.get("action", ""),
                        description=step_data.get("description", ""),
                        expected_outcome=step_data.get("expected_outcome", ""),
                    ))
                
                logger.debug("Created plan with {} steps", len(steps))
                return steps
                
            except Exception as e:
                logger.warning("Planning failed: {}. Using simple plan.", str(e))
        
        # Simple fallback plan
        return [
            PlanStep(step_id=1, action=f"Research: {objective}", expected_outcome="Information gathered"),
            PlanStep(step_id=2, action="Analyze findings", expected_outcome="Insights generated"),
            PlanStep(step_id=3, action="Generate output", expected_outcome="Result produced"),
        ]
    
    async def _execute(
        self,
        plan: list[PlanStep],
        executor: Callable,
    ) -> list[Any]:
        """Execute plan steps."""
        results = []
        
        for step in plan:
            start = datetime.now()
            
            try:
                result = await executor(step.action)
                
                step.actual_outcome = str(result)[:500] if result else "Completed"
                step.success = True
                results.append(result)
                
            except Exception as e:
                step.actual_outcome = f"Error: {str(e)}"
                step.success = False
                results.append(None)
                
            step.duration_ms = (datetime.now() - start).total_seconds() * 1000
            logger.debug("Step {}: {} ({})", step.step_id, "✓" if step.success else "✗", step.duration_ms)
        
        return results
    
    async def _act(self, state: PEARState):
        """Take actions based on execution results."""
        # Aggregate results
        successes = sum(1 for s in state.plan if s.success)
        total = len(state.plan)
        
        state.current_step = total
        
        logger.info("Execution: {}/{} steps succeeded", successes, total)
    
    async def _reflect(self, state: PEARState):
        """Reflect on execution and decide next steps."""
        if self.llm:
            try:
                import json
                
                # Build plan results text
                plan_results = []
                for step in state.plan:
                    plan_results.append(
                        f"Step {step.step_id}: {step.action}\n"
                        f"  Expected: {step.expected_outcome}\n"
                        f"  Actual: {step.actual_outcome}\n"
                        f"  Success: {step.success}"
                    )
                
                prompt = REFLECT_PROMPT.format(
                    objective=state.objective,
                    plan_results="\n".join(plan_results),
                )
                
                response = await self.llm.agenerate(prompt)
                
                # Parse JSON
                response_text = response.strip()
                if response_text.startswith("```"):
                    response_text = response_text.split("```")[1]
                    if response_text.startswith("json"):
                        response_text = response_text[4:]
                
                data = json.loads(response_text)
                
                state.reflection = data.get("reflection", "")
                state.needs_replanning = data.get("needs_replanning", False)
                
                logger.info("Reflection complete: replan={}", state.needs_replanning)
                return
                
            except Exception as e:
                logger.warning("Reflection failed: {}", str(e))
        
        # Simple reflection
        success_rate = sum(1 for s in state.plan if s.success) / max(len(state.plan), 1)
        state.reflection = f"Completed with {success_rate:.0%} success rate"
        state.needs_replanning = success_rate < 0.5
    
    def _notify_stage(self, state: PEARState):
        """Notify stage change."""
        if self.on_stage_change:
            self.on_stage_change(state.stage, state)
