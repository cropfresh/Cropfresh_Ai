"""
Progress Monitor
================
Track and report task progress.

Features:
- Progress tracking for tasks
- ETA estimation
- Progress notifications
- Dashboard data generation
"""

from datetime import datetime, timedelta
from typing import Optional, Callable

from loguru import logger
from pydantic import BaseModel, Field


class TaskProgress(BaseModel):
    """Progress of a single task."""
    task_id: str
    task_name: str
    total_steps: int = 0
    completed_steps: int = 0
    current_step: str = ""
    progress_percent: float = 0.0
    
    started_at: datetime = Field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    status: str = "running"  # running, paused, completed, failed
    error: Optional[str] = None
    
    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()
    
    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        if self.progress_percent <= 0:
            return None
        if self.progress_percent >= 100:
            return 0
        
        elapsed = self.elapsed_seconds
        total_estimated = elapsed / (self.progress_percent / 100)
        return total_estimated - elapsed


class ProgressMonitor:
    """
    Monitor task progress.
    
    Usage:
        monitor = ProgressMonitor()
        
        # Create task
        task = monitor.create_task("research_123", "Research tomatoes", total_steps=5)
        
        # Update progress
        monitor.update_progress("research_123", step=2, current="Analyzing data")
        
        # Get status
        progress = monitor.get_progress("research_123")
    """
    
    def __init__(
        self,
        on_progress: Optional[Callable[[TaskProgress], None]] = None,
        on_complete: Optional[Callable[[TaskProgress], None]] = None,
    ):
        """
        Initialize progress monitor.
        
        Args:
            on_progress: Callback when progress updates
            on_complete: Callback when task completes
        """
        self._tasks: dict[str, TaskProgress] = {}
        self.on_progress = on_progress
        self.on_complete = on_complete
    
    def create_task(
        self,
        task_id: str,
        task_name: str,
        total_steps: int = 0,
    ) -> TaskProgress:
        """
        Create a new task to track.
        
        Args:
            task_id: Unique task identifier
            task_name: Human-readable name
            total_steps: Total number of steps (0 for unknown)
            
        Returns:
            TaskProgress object
        """
        progress = TaskProgress(
            task_id=task_id,
            task_name=task_name,
            total_steps=total_steps,
        )
        
        self._tasks[task_id] = progress
        logger.info("Created task: {} ({})", task_id, task_name)
        
        return progress
    
    def update_progress(
        self,
        task_id: str,
        step: Optional[int] = None,
        percent: Optional[float] = None,
        current: str = "",
    ) -> Optional[TaskProgress]:
        """
        Update task progress.
        
        Args:
            task_id: Task identifier
            step: Current step number
            percent: Direct percentage (0-100)
            current: Description of current activity
            
        Returns:
            Updated TaskProgress or None
        """
        progress = self._tasks.get(task_id)
        if not progress:
            return None
        
        if step is not None:
            progress.completed_steps = step
            if progress.total_steps > 0:
                progress.progress_percent = (step / progress.total_steps) * 100
        
        if percent is not None:
            progress.progress_percent = percent
        
        if current:
            progress.current_step = current
        
        # Estimate completion
        if progress.progress_percent > 0 and progress.eta_seconds:
            progress.estimated_completion = datetime.now() + timedelta(seconds=progress.eta_seconds)
        
        if self.on_progress:
            self.on_progress(progress)
        
        return progress
    
    def complete_task(
        self,
        task_id: str,
        status: str = "completed",
        error: Optional[str] = None,
    ) -> Optional[TaskProgress]:
        """
        Mark task as complete.
        
        Args:
            task_id: Task identifier
            status: Final status (completed/failed)
            error: Error message if failed
            
        Returns:
            Final TaskProgress
        """
        progress = self._tasks.get(task_id)
        if not progress:
            return None
        
        progress.status = status
        progress.completed_at = datetime.now()
        progress.progress_percent = 100 if status == "completed" else progress.progress_percent
        
        if error:
            progress.error = error
        
        if self.on_complete:
            self.on_complete(progress)
        
        logger.info(
            "Task {} {}: {} in {:.1f}s",
            task_id, status, progress.task_name, progress.elapsed_seconds
        )
        
        return progress
    
    def get_progress(self, task_id: str) -> Optional[TaskProgress]:
        """Get progress for a task."""
        return self._tasks.get(task_id)
    
    def get_all_progress(self) -> dict[str, TaskProgress]:
        """Get all task progress."""
        return self._tasks.copy()
    
    def get_running_tasks(self) -> list[TaskProgress]:
        """Get all running tasks."""
        return [t for t in self._tasks.values() if t.status == "running"]
    
    def get_dashboard_data(self) -> dict:
        """Get data for dashboard display."""
        running = [t for t in self._tasks.values() if t.status == "running"]
        completed = [t for t in self._tasks.values() if t.status == "completed"]
        failed = [t for t in self._tasks.values() if t.status == "failed"]
        
        return {
            "total_tasks": len(self._tasks),
            "running": len(running),
            "completed": len(completed),
            "failed": len(failed),
            "tasks": [
                {
                    "id": t.task_id,
                    "name": t.task_name,
                    "progress": t.progress_percent,
                    "status": t.status,
                    "current": t.current_step,
                    "eta": t.eta_seconds,
                }
                for t in self._tasks.values()
            ],
        }
