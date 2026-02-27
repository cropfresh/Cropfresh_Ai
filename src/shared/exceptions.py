"""Custom exception classes for CropFresh AI."""

class CropFreshError(Exception):
    pass

class AgentError(CropFreshError):
    pass

class ScraperError(CropFreshError):
    pass

class DatabaseError(CropFreshError):
    pass
