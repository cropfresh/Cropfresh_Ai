"""
Execution Tracker Mixin
=======================
Handles state tracking for single-agent executions.
"""

from datetime import datetime
from typing import Optional
from loguru import logger

from .models import AgentExecutionState
from .base import BaseStateManager


class ExecutionTrackerMixin(BaseStateManager):
    """Mixin for agent execution tracking logic."""

    def create_execution(
        self,
        session_id: str,
        query: str,
    ) -> AgentExecutionState:
        """Create execution state for a single query."""
        execution = AgentExecutionState(
            session_id=session_id,
            original_query=query,
        )

        self._executions[execution.execution_id] = execution
        logger.debug(f"Created execution: {execution.execution_id}")

        return execution

    def update_execution(
        self,
        execution_id: str,
        **updates,
    ) -> Optional[AgentExecutionState]:
        """Update execution state with specific fields."""
        execution = self._executions.get(execution_id)
        if not execution:
            return None

        for key, value in updates.items():
            if hasattr(execution, key):
                setattr(execution, key, value)

        return execution

    def add_step(
        self,
        execution_id: str,
        step: str,
    ) -> None:
        """Add step to execution trace."""
        execution = self._executions.get(execution_id)
        if execution:
            execution.steps_executed.append(step)

    def complete_execution(
        self,
        execution_id: str,
        response: str,
    ) -> Optional[AgentExecutionState]:
        """Mark execution as complete."""
        execution = self._executions.get(execution_id)
        if execution:
            execution.final_response = response
            execution.end_time = datetime.now()
        return execution
