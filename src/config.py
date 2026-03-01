# * Re-export from canonical location so `from src.config import get_settings` works
from src.api.config import Settings, get_settings  # noqa: F401

__all__ = ["Settings", "get_settings"]
