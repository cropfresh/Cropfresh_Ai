"""
Task Decomposer
===============
Break complex tasks into sub-task graphs for parallel execution.

Features:
- Automatic task decomposition
- Dependency graph generation
- Parallel execution planning
- Progress tracking
"""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any

from loguru import logger
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    READY = "ready"      # Dependencies met
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SubTask(BaseModel):
    """A single sub-task in the graph."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    executor: Optional[str] = None  # Agent name


class TaskGraph(BaseModel):
    """Graph of sub-tasks with dependencies."""
    graph_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    original_task: str
    tasks: list[SubTask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def add_task(self, task: SubTask):
        """Add a sub-task to the graph."""
        self.tasks.append(task)
    
    def get_task(self, task_id: str) -> Optional[SubTask]:
        """Get task by ID."""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def get_ready_tasks(self) -> list[SubTask]:
        """Get tasks that are ready to execute."""
        completed_ids = {t.task_id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        
        ready = []
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                deps_met = all(dep in completed_ids for dep in task.depends_on)
                if deps_met:
                    ready.append(task)
        
        return ready
    
    @property
    def progress(self) -> float:
        """Get completion percentage."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return completed / len(self.tasks)
    
    @property
    def is_complete(self) -> bool:
        """Check if all tasks are done."""
        return all(t.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED]
                   for t in self.tasks)


DECOMPOSITION_PROMPT = """Decompose the following task into smaller, executable sub-tasks.

Task: {task}

Rules:
1. Create 2-8 sub-tasks
2. Each sub-task should be a single, focused action
3. Identify dependencies between tasks
4. Assign each task to an executor: agronomy_agent, commerce_agent, web_scraping_agent, research_agent, or general_agent

Respond with JSON:
{{
    "tasks": [
        {{
            "name": "Task name",
            "description": "What this task does",
            "depends_on": [],  // List of task indices this depends on
            "executor": "agent_name"
        }},
        ...
    ]
}}"""


class TaskDecomposer:
    """
    Decomposes complex tasks into sub-task graphs.
    
    Usage:
        decomposer = TaskDecomposer(llm=groq_llm)
        
        graph = await decomposer.decompose(
            "Research tomato prices and recommend best time to sell"
        )
        
        # Execute graph
        await decomposer.execute_graph(graph, executors)
    """
    
    def __init__(self, llm=None):
        """
        Initialize task decomposer.
        
        Args:
            llm: LLM for task decomposition
        """
        self.llm = llm
    
    async def decompose(self, task: str) -> TaskGraph:
        """
        Decompose a task into sub-tasks.
        
        Args:
            task: Complex task description
            
        Returns:
            TaskGraph with sub-tasks
        """
        graph = TaskGraph(original_task=task)
        
        if not self.llm:
            # Fallback: simple decomposition
            return self._simple_decomposition(task)
        
        try:
            import json
            
            prompt = DECOMPOSITION_PROMPT.format(task=task)
            response = await self.llm.agenerate(prompt)
            
            # Parse JSON
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            data = json.loads(response_text)
            
            # Build tasks
            for i, task_data in enumerate(data.get("tasks", [])):
                # Convert dependency indices to IDs
                deps = []
                for dep_idx in task_data.get("depends_on", []):
                    if 0 <= dep_idx < len(graph.tasks):
                        deps.append(graph.tasks[dep_idx].task_id)
                
                sub_task = SubTask(
                    name=task_data.get("name", f"Task {i+1}"),
                    description=task_data.get("description", ""),
                    depends_on=deps,
                    executor=task_data.get("executor", "general_agent"),
                )
                graph.add_task(sub_task)
            
            logger.info("Decomposed task into {} sub-tasks", len(graph.tasks))
            return graph
            
        except Exception as e:
            logger.warning("Decomposition failed: {}. Using simple decomposition.", str(e))
            return self._simple_decomposition(task)
    
    def _simple_decomposition(self, task: str) -> TaskGraph:
        """Simple fallback decomposition."""
        graph = TaskGraph(original_task=task)
        
        # Create basic tasks
        graph.add_task(SubTask(
            name="Understand query",
            description="Analyze and understand the request",
            executor="general_agent",
        ))
        
        graph.add_task(SubTask(
            name="Gather information",
            description="Collect relevant data",
            depends_on=[graph.tasks[0].task_id],
            executor="research_agent",
        ))
        
        graph.add_task(SubTask(
            name="Process and analyze",
            description="Analyze the gathered information",
            depends_on=[graph.tasks[1].task_id],
            executor="general_agent",
        ))
        
        graph.add_task(SubTask(
            name="Generate response",
            description="Create the final response",
            depends_on=[graph.tasks[2].task_id],
            executor="general_agent",
        ))
        
        return graph
    
    async def execute_graph(
        self,
        graph: TaskGraph,
        executors: dict[str, Callable],
        parallel: bool = True,
    ) -> TaskGraph:
        """
        Execute a task graph.
        
        Args:
            graph: Task graph to execute
            executors: Dict of agent_name -> async executor function
            parallel: Execute independent tasks in parallel
            
        Returns:
            Updated TaskGraph with results
        """
        while not graph.is_complete:
            ready_tasks = graph.get_ready_tasks()
            
            if not ready_tasks:
                # No tasks ready but graph not complete - likely failed dependencies
                break
            
            if parallel and len(ready_tasks) > 1:
                # Execute in parallel
                await asyncio.gather(*[
                    self._execute_task(task, executors)
                    for task in ready_tasks
                ])
            else:
                # Execute sequentially
                for task in ready_tasks:
                    await self._execute_task(task, executors)
        
        graph.completed_at = datetime.now()
        logger.info("Graph execution complete: {:.0%} success", graph.progress)
        
        return graph
    
    async def _execute_task(
        self,
        task: SubTask,
        executors: dict[str, Callable],
    ):
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        executor = executors.get(task.executor or "general_agent")
        
        if not executor:
            task.status = TaskStatus.FAILED
            task.error = f"No executor for {task.executor}"
            return
        
        try:
            # Execute
            result = await executor(task.name, task.description)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()
            
            logger.debug("Task '{}' completed", task.name)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            logger.error("Task '{}' failed: {}", task.name, str(e))
