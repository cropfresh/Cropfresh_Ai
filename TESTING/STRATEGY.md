# 🧪 TESTING/STRATEGY.md — CropFresh AI Test Strategy

> **Last Updated:** 2026-02-27
> **Owner:** Solo founder + AI agents
> **Philosophy:** Test-as-you-code. Every feature ships with tests.

---

## 1. Testing Philosophy

- **Tests are first-class citizens** — written alongside code, not after
- **AI writes first draft** — ask Claude/Antigravity to generate tests from specs, then review
- **Small, fast, focused** — prefer many small unit tests over large end-to-end tests
- **Document the "why"** — tests serve as living documentation of expected behavior

---

## 2. Test Pyramid

```
        [E2E Tests] (slow, few)
           /    \
      [Integration Tests]
         /          \
  [Unit Tests] ← Majority (fast, many)
```

| Layer | Scope | Tools | Location |
|-------|-------|-------|----------|
| Unit | Single function/class | `pytest` | `tests/unit/` |
| Integration | Multiple components | `pytest` + `httpx` | `tests/integration/` |
| API | Endpoint contracts | `httpx` + `FastAPI TestClient` | `tests/api/` |
| E2E | Full flow (voice → response) | `pytest` + real services | `tests/e2e/` |
| Agent Eval | Agent accuracy & quality | LangSmith + custom scripts | `ai/evaluations/` |

---

## 3. What "Done" Means for Each Feature Type

### API Endpoint
- [ ] Unit test for service/business logic
- [ ] Integration test for DB interaction
- [ ] API test for request/response schema validation
- [ ] Error cases tested (400, 404, 500)
- [ ] Docs updated in `docs/api/`

### AI Agent
- [ ] Unit test for routing logic
- [ ] Prompt tested against at least 5 representative queries
- [ ] Evaluation metrics captured in LangSmith (accuracy, latency, cost)
- [ ] Fallback behavior tested (what happens when LLM fails or low confidence)

### Scraper
- [ ] Test with mocked HTML fixture
- [ ] Integration test against real source with rate-limit safeguards
- [ ] Data validation assertions (field types, non-empty, reasonableness)
- [ ] Retry/failure handling test

### Voice Pipeline
- [ ] Unit test STT transcription with sample audio
- [ ] Unit test TTS synthesis (output is non-empty audio)
- [ ] Integration test: audio in → text out → audio out
- [ ] Language test: Kannada, Hindi, English

---

## 4. Running Tests

```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run all integration tests
uv run pytest tests/integration/ -v

# Run API tests
uv run pytest tests/api/ -v

# Run full suite with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run agent evaluation scripts
uv run python ai/evaluations/eval_agents.py

# Run multi-agent test suite
uv run python scripts/test_multi_agent.py

# Run specific test file
uv run pytest tests/unit/test_supervisor_agent.py -v
```

---

## 5. AI Prompts for Test Generation

### Generate unit tests for a function
```
"Generate unit tests for [FunctionName] in [file_path].
- Follow our test strategy in TESTING/STRATEGY.md
- Cover: happy path, edge cases, error cases
- Use pytest, mock external dependencies
- Add descriptive docstrings to each test"
```

### Analyze a diff for missing tests
```
"Analyze this git diff and identify:
- Functions/endpoints not covered by tests
- Missing error handling cases
- Risky changes that need integration tests
Reference our TESTING/STRATEGY.md checklist."
```

### Generate E2E scenario
```
"Based on this feature spec and PLAN.md, propose an end-to-end test scenario
for [feature name]. Include: preconditions, steps, expected outcomes,
and any edge cases that might break the happy path."
```

---

## 6. Agent Evaluation Framework

### Metrics to Track per Agent
| Metric | Description | Target |
|--------|-------------|--------|
| Routing Accuracy | % queries routed to correct agent | > 90% |
| Response Relevance | LLM judge: 1-5 scale | > 4.0 |
| Latency P95 | End-to-end response time | < 3s |
| Cost per Query | Avg Groq API cost | < ₹0.50 |
| Tool Use Rate | % queries using tools correctly | > 80% |

### Evaluation Sets
- `ai/evaluations/datasets/agronomy_queries.json` — 50 sample farming questions
- `ai/evaluations/datasets/commerce_queries.json` — 50 market/price questions
- `ai/evaluations/datasets/platform_queries.json` — 30 app/support questions
- `ai/evaluations/datasets/voice_samples/` — Audio samples for STT eval

---

## 7. Test Data & Fixtures

- Use `tests/fixtures/` for shared test data (mock API responses, sample documents, audio)
- Never use production data in tests
- Audio test fixtures stored as `.wav` files: `tests/fixtures/audio/`
- Mock Qdrant with `tests/mocks/qdrant_mock.py`

---

## 8. Checklists per Sprint

Run these before closing a sprint:

- [ ] All new files have corresponding unit tests
- [ ] Integration tests pass locally
- [ ] `uv run pytest --cov=src` coverage not decreased from last sprint
- [ ] New API endpoints tested via FastAPI TestClient
- [ ] Agent evaluation scores recorded in sprint file
- [ ] Known failing tests documented with issue references
