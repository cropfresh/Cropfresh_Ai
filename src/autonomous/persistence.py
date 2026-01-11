"""
Task Persistence
================
Save and resume autonomous tasks.

Features:
- Save task state to disk
- Resume interrupted tasks
- History tracking
- Cleanup of old tasks
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any

from loguru import logger
from pydantic import BaseModel


class TaskPersistence:
    """
    Persist and resume autonomous tasks.
    
    Usage:
        persistence = TaskPersistence()
        
        # Save task state
        await persistence.save("task_123", state_dict)
        
        # Resume later
        state = await persistence.load("task_123")
        if state:
            # Continue execution
            pass
    """
    
    TASKS_DIR = Path("data/autonomous_tasks")
    
    def __init__(self, tasks_dir: Optional[Path] = None):
        """
        Initialize task persistence.
        
        Args:
            tasks_dir: Directory for task storage
        """
        self.tasks_dir = tasks_dir or self.TASKS_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        
        self._index_file = self.tasks_dir / "index.json"
        self._index: dict = {}
        
        self._load_index()
    
    def _load_index(self):
        """Load task index."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning("Failed to load task index: {}", str(e))
                self._index = {}
    
    def _save_index(self):
        """Save task index."""
        try:
            with open(self._index_file, 'w', encoding='utf-8') as f:
                json.dump(self._index, f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save task index: {}", str(e))
    
    async def save(
        self,
        task_id: str,
        state: dict,
        metadata: Optional[dict] = None,
    ) -> bool:
        """
        Save task state.
        
        Args:
            task_id: Unique task identifier
            state: Task state to save
            metadata: Additional metadata
            
        Returns:
            True if saved successfully
        """
        task_file = self.tasks_dir / f"{task_id}.json"
        
        try:
            data = {
                "task_id": task_id,
                "state": state,
                "metadata": metadata or {},
                "saved_at": datetime.now().isoformat(),
            }
            
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            # Update index
            self._index[task_id] = {
                "saved_at": data["saved_at"],
                "metadata": metadata or {},
            }
            self._save_index()
            
            logger.debug("Saved task: {}", task_id)
            return True
            
        except Exception as e:
            logger.error("Failed to save task {}: {}", task_id, str(e))
            return False
    
    async def load(self, task_id: str) -> Optional[dict]:
        """
        Load task state.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task state dict or None if not found
        """
        task_file = self.tasks_dir / f"{task_id}.json"
        
        if not task_file.exists():
            return None
        
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug("Loaded task: {}", task_id)
            return data.get("state")
            
        except Exception as e:
            logger.error("Failed to load task {}: {}", task_id, str(e))
            return None
    
    async def delete(self, task_id: str) -> bool:
        """
        Delete a saved task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if deleted
        """
        task_file = self.tasks_dir / f"{task_id}.json"
        
        if task_file.exists():
            task_file.unlink()
        
        self._index.pop(task_id, None)
        self._save_index()
        
        logger.debug("Deleted task: {}", task_id)
        return True
    
    async def list_tasks(self) -> list[dict]:
        """
        List all saved tasks.
        
        Returns:
            List of task info dicts
        """
        tasks = []
        
        for task_id, info in self._index.items():
            tasks.append({
                "task_id": task_id,
                "saved_at": info.get("saved_at"),
                "metadata": info.get("metadata", {}),
            })
        
        return tasks
    
    async def list_resumable(self) -> list[dict]:
        """
        List tasks that can be resumed.
        
        Returns:
            List of resumable tasks
        """
        resumable = []
        
        for task_id, info in self._index.items():
            task_data = await self.load(task_id)
            if task_data:
                status = task_data.get("status", "unknown")
                if status in ["running", "paused", "pending"]:
                    resumable.append({
                        "task_id": task_id,
                        "saved_at": info.get("saved_at"),
                        "progress": task_data.get("progress", 0),
                        "objective": task_data.get("objective", "Unknown"),
                    })
        
        return resumable
    
    async def cleanup(self, max_age_days: int = 7) -> int:
        """
        Remove old completed tasks.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of tasks deleted
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        deleted = 0
        
        to_delete = []
        for task_id, info in self._index.items():
            saved_at_str = info.get("saved_at", "")
            try:
                saved_at = datetime.fromisoformat(saved_at_str)
                if saved_at < cutoff:
                    to_delete.append(task_id)
            except:
                pass
        
        for task_id in to_delete:
            await self.delete(task_id)
            deleted += 1
        
        if deleted:
            logger.info("Cleaned up {} old tasks", deleted)
        
        return deleted
    
    async def checkpoint(
        self,
        task_id: str,
        state: dict,
        checkpoint_name: str = "",
    ) -> bool:
        """
        Save a checkpoint for recovery.
        
        Args:
            task_id: Task identifier
            state: Current state
            checkpoint_name: Optional checkpoint name
            
        Returns:
            True if saved
        """
        checkpoint_id = f"{task_id}_cp_{checkpoint_name or datetime.now().strftime('%H%M%S')}"
        
        return await self.save(
            checkpoint_id,
            state,
            metadata={"parent_task": task_id, "is_checkpoint": True},
        )
