"""
Goal-Directed Agent
===================
Manages objective trees for autonomous task completion.

Features:
- Hierarchical objective decomposition
- Objective prioritization
- Success criteria tracking
- Dynamic goal adjustment
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable

from loguru import logger
from pydantic import BaseModel, Field


class ObjectiveStatus(str, Enum):
    """Status of an objective."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class ObjectivePriority(str, Enum):
    """Priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Objective(BaseModel):
    """A single objective in the tree."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str = ""
    
    # Hierarchy
    parent_id: Optional[str] = None
    sub_objectives: list[str] = Field(default_factory=list)  # Child IDs
    
    # Status
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    priority: ObjectivePriority = ObjectivePriority.MEDIUM
    progress: float = 0.0  # 0-1
    
    # Success criteria
    success_criteria: list[str] = Field(default_factory=list)
    criteria_met: list[bool] = Field(default_factory=list)
    
    # Execution
    assigned_agent: Optional[str] = None
    requires_human: bool = False
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration_sec: int = 60
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None


class ObjectiveTree(BaseModel):
    """Hierarchical tree of objectives."""
    tree_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    root_objective: str  # Root objective ID
    objectives: dict[str, Objective] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def add_objective(self, objective: Objective, parent_id: Optional[str] = None):
        """Add an objective to the tree."""
        if parent_id:
            objective.parent_id = parent_id
            if parent_id in self.objectives:
                self.objectives[parent_id].sub_objectives.append(objective.id)
        
        self.objectives[objective.id] = objective
    
    def get_objective(self, obj_id: str) -> Optional[Objective]:
        """Get objective by ID."""
        return self.objectives.get(obj_id)
    
    def get_ready_objectives(self) -> list[Objective]:
        """Get objectives ready to execute (dependencies met)."""
        ready = []
        
        for obj in self.objectives.values():
            if obj.status != ObjectiveStatus.PENDING:
                continue
            
            # Check if parent is complete (or no parent)
            if obj.parent_id:
                parent = self.objectives.get(obj.parent_id)
                if parent and parent.status != ObjectiveStatus.IN_PROGRESS:
                    continue
            
            # Check if sub-objectives block this
            if obj.sub_objectives:
                # All sub-objectives must be complete
                all_done = all(
                    self.objectives.get(sub_id, Objective(name="")).status == ObjectiveStatus.COMPLETED
                    for sub_id in obj.sub_objectives
                )
                if not all_done:
                    continue
            
            ready.append(obj)
        
        # Sort by priority
        priority_order = {
            ObjectivePriority.CRITICAL: 0,
            ObjectivePriority.HIGH: 1,
            ObjectivePriority.MEDIUM: 2,
            ObjectivePriority.LOW: 3,
        }
        ready.sort(key=lambda o: priority_order.get(o.priority, 2))
        
        return ready
    
    @property
    def progress(self) -> float:
        """Overall tree progress."""
        if not self.objectives:
            return 0.0
        completed = sum(1 for o in self.objectives.values() if o.status == ObjectiveStatus.COMPLETED)
        return completed / len(self.objectives)
    
    @property
    def is_complete(self) -> bool:
        """Check if root objective is complete."""
        root = self.objectives.get(self.root_objective)
        return root and root.status == ObjectiveStatus.COMPLETED


GOAL_DECOMPOSITION_PROMPT = """Decompose this goal into a hierarchy of objectives.

Goal: {goal}

Create a structured breakdown:
1. Main objective (the goal itself)
2. Sub-objectives needed to achieve it
3. Success criteria for each

Respond with JSON:
{{
    "root": {{
        "name": "Main objective",
        "description": "...",
        "success_criteria": ["criterion1", "criterion2"],
        "sub_objectives": [
            {{
                "name": "Sub-objective 1",
                "description": "...",
                "success_criteria": ["..."],
                "priority": "high|medium|low",
                "agent": "agronomy_agent|commerce_agent|research_agent|general_agent"
            }}
        ]
    }}
}}"""


class GoalAgent:
    """
    Manages goal-directed autonomous task completion.
    
    Usage:
        agent = GoalAgent(llm=groq)
        
        tree = await agent.create_objective_tree(
            "Research and recommend best crop for next season"
        )
        
        while not tree.is_complete:
            await agent.execute_next_objective(tree, executors)
    """
    
    def __init__(self, llm=None):
        """
        Initialize goal agent.
        
        Args:
            llm: LLM for goal decomposition
        """
        self.llm = llm
    
    async def create_objective_tree(
        self,
        goal: str,
        max_depth: int = 3,
    ) -> ObjectiveTree:
        """
        Create an objective tree from a goal.
        
        Args:
            goal: High-level goal
            max_depth: Maximum nesting depth
            
        Returns:
            ObjectiveTree with decomposed objectives
        """
        if self.llm:
            return await self._decompose_with_llm(goal, max_depth)
        else:
            return self._simple_decomposition(goal)
    
    async def _decompose_with_llm(
        self,
        goal: str,
        max_depth: int,
    ) -> ObjectiveTree:
        """Decompose using LLM."""
        try:
            import json
            
            prompt = GOAL_DECOMPOSITION_PROMPT.format(goal=goal)
            response = await self.llm.agenerate(prompt)
            
            # Parse JSON
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            data = json.loads(response_text)
            
            # Build tree
            tree = ObjectiveTree(root_objective="")
            root_data = data.get("root", {})
            
            # Create root objective
            root = Objective(
                name=root_data.get("name", goal),
                description=root_data.get("description", ""),
                success_criteria=root_data.get("success_criteria", []),
                priority=ObjectivePriority.HIGH,
            )
            tree.root_objective = root.id
            tree.add_objective(root)
            
            # Create sub-objectives
            for sub_data in root_data.get("sub_objectives", [])[:6]:
                sub = Objective(
                    name=sub_data.get("name", ""),
                    description=sub_data.get("description", ""),
                    success_criteria=sub_data.get("success_criteria", []),
                    priority=self._parse_priority(sub_data.get("priority", "medium")),
                    assigned_agent=sub_data.get("agent", "general_agent"),
                )
                tree.add_objective(sub, parent_id=root.id)
            
            logger.info("Created objective tree: {} objectives", len(tree.objectives))
            return tree
            
        except Exception as e:
            logger.warning("LLM decomposition failed: {}. Using simple.", str(e))
            return self._simple_decomposition(goal)
    
    def _simple_decomposition(self, goal: str) -> ObjectiveTree:
        """Simple decomposition without LLM."""
        tree = ObjectiveTree(root_objective="")
        
        # Root objective
        root = Objective(
            name=goal,
            description="Main objective",
            success_criteria=["Task completed successfully"],
            priority=ObjectivePriority.HIGH,
        )
        tree.root_objective = root.id
        tree.add_objective(root)
        
        # Generic sub-objectives
        sub1 = Objective(
            name="Gather information",
            description="Collect relevant data",
            assigned_agent="research_agent",
        )
        tree.add_objective(sub1, parent_id=root.id)
        
        sub2 = Objective(
            name="Analyze information",
            description="Process and analyze data",
            assigned_agent="general_agent",
        )
        tree.add_objective(sub2, parent_id=root.id)
        
        sub3 = Objective(
            name="Generate output",
            description="Create final result",
            assigned_agent="general_agent",
        )
        tree.add_objective(sub3, parent_id=root.id)
        
        return tree
    
    async def execute_next_objective(
        self,
        tree: ObjectiveTree,
        executors: dict[str, Callable],
    ) -> Optional[Objective]:
        """
        Execute the next ready objective.
        
        Args:
            tree: Objective tree
            executors: Dict of agent_name -> executor function
            
        Returns:
            Executed objective or None if none ready
        """
        ready = tree.get_ready_objectives()
        
        if not ready:
            return None
        
        # Execute first ready objective
        obj = ready[0]
        obj.status = ObjectiveStatus.IN_PROGRESS
        obj.started_at = datetime.now()
        
        executor = executors.get(obj.assigned_agent or "general_agent")
        
        if not executor:
            obj.status = ObjectiveStatus.FAILED
            obj.error = f"No executor for {obj.assigned_agent}"
            return obj
        
        try:
            # Execute
            result = await executor(obj.name, obj.description)
            
            obj.status = ObjectiveStatus.COMPLETED
            obj.result = result
            obj.completed_at = datetime.now()
            obj.progress = 1.0
            
            # Update parent progress
            if obj.parent_id:
                parent = tree.get_objective(obj.parent_id)
                if parent:
                    completed_subs = sum(
                        1 for sub_id in parent.sub_objectives
                        if tree.objectives.get(sub_id, Objective(name="")).status == ObjectiveStatus.COMPLETED
                    )
                    parent.progress = completed_subs / max(len(parent.sub_objectives), 1)
            
            logger.info("Objective completed: {}", obj.name)
            
        except Exception as e:
            obj.status = ObjectiveStatus.FAILED
            obj.error = str(e)
            obj.completed_at = datetime.now()
            
            logger.error("Objective failed: {} - {}", obj.name, str(e))
        
        return obj
    
    def _parse_priority(self, priority_str: str) -> ObjectivePriority:
        """Parse priority string."""
        mapping = {
            "critical": ObjectivePriority.CRITICAL,
            "high": ObjectivePriority.HIGH,
            "medium": ObjectivePriority.MEDIUM,
            "low": ObjectivePriority.LOW,
        }
        return mapping.get(priority_str.lower(), ObjectivePriority.MEDIUM)
