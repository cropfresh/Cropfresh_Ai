# Coding Standards — CropFresh AI

## Python Style
- **Version**: Python 3.11+
- **Linter**: Ruff (line-length: 100)
- **Type hints**: Required on all function signatures
- **Docstrings**: Google style

## Naming Conventions
| Element | Convention | Example |
|---------|-----------|---------|
| Files | snake_case | `crop_service.py` |
| Classes | PascalCase | `CropListingAgent` |
| Functions | snake_case | `get_crop_price()` |
| Constants | UPPER_SNAKE | `MAX_RETRIES` |
| Private | _prefix | `_internal_method()` |

## Patterns to Follow
1. **Async-first**: Use `async def` for all I/O operations
2. **Dependency injection**: Via FastAPI `Depends()`
3. **Repository pattern**: DB access through service layer
4. **Agent pattern**: All agents extend `BaseAgent`
5. **Structured logging**: Use `loguru`, never `print()`

## Patterns to Avoid
- ❌ Global mutable state
- ❌ Bare `except:` clauses
- ❌ Direct DB queries in route handlers
- ❌ Hardcoded configuration values
- ❌ Synchronous I/O in async contexts

## Import Order (enforced by Ruff)
1. Standard library
2. Third-party packages
3. Local imports (absolute: `from src.agents...`)

## Testing Standards
- Minimum 80% coverage for new code
- Unit tests in `tests/` or co-located `tests/` folders
- Integration tests in `tests/integration/`
- E2E tests in `tests/e2e/`
- Use `pytest` with `pytest-asyncio`
