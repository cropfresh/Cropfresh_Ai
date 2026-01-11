"""
Autonomous Task Module
======================
Goal-directed autonomous task completion.

Components:
- Goal Agent: Objective tree management
- PEAR Loop: Plan-Execute-Act-Reflect
- Progress Monitor: Task tracking
- Task Persistence: Save/resume tasks
"""

from src.autonomous.goal_agent import GoalAgent, Objective, ObjectiveTree
from src.autonomous.pear_loop import PEARLoop, PlanStep
from src.autonomous.progress_monitor import ProgressMonitor, TaskProgress
from src.autonomous.persistence import TaskPersistence

__all__ = [
    "GoalAgent",
    "Objective",
    "ObjectiveTree",
    "PEARLoop",
    "PlanStep",
    "ProgressMonitor",
    "TaskProgress",
    "TaskPersistence",
]
