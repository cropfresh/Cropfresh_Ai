# ✅ TESTING/CHECKLISTS.md — CropFresh AI Feature Checklists

> **Last Updated:** 2026-02-27
> Reference: Complete the relevant checklist before marking any task as done.

---

## API Endpoint Checklist

Before marking any endpoint as complete:

```
[ ] Endpoint returns correct HTTP status codes
[ ] Request body validated (Pydantic schema)
[ ] Response schema matches docs/api/ spec
[ ] 400 Bad Request tested (invalid input)
[ ] 404 Not Found tested (missing resource)
[ ] 500 Internal Server Error handled gracefully
[ ] Authentication/authorization checked (if applicable)
[ ] Rate limiting in place (if public endpoint)
[ ] Unit test for service layer
[ ] API test via FastAPI TestClient
[ ] Docs updated in docs/api/
[ ] Added to WORKFLOW_STATUS.md file changes log
```

---

## AI Agent Checklist

Before marking any agent change as complete:

```
[ ] Agent routes correctly on at least 5 test queries
[ ] Fallback behavior works when LLM returns low confidence
[ ] Tool usage is correct (correct tool called, params valid)
[ ] Response quality tested (no hallucinations on factual queries)
[ ] Latency measured (target < 3s P95)
[ ] Cost per query estimated
[ ] LangSmith trace reviewed for at least one session
[ ] Agent prompt version bumped in docs/agents/
[ ] Unit test for routing logic added
```

---

## Scraper Checklist

Before marking any scraper as complete:

```
[ ] Tested with saved HTML fixture (offline)
[ ] Data fields validated (not null, correct types)
[ ] Rate limiting + retry logic implemented
[ ] Caching enabled (Redis or file cache)
[ ] Scheduler integration tested (if applicable)
[ ] Error alerting in place (log + optional webhook)
[ ] Data freshness checked (staleness threshold set)
[ ] Integration test with real source (with care, use staging target)
```

---

## Voice Pipeline Checklist

Before marking any voice feature as complete:

```
[ ] STT tested with sample audio in Kannada
[ ] STT tested with sample audio in Hindi
[ ] STT tested with sample audio in English
[ ] TTS output is non-empty and plays back correctly
[ ] Round-trip latency measured (audio in → audio out)
[ ] WebSocket stream handles disconnect gracefully
[ ] Silence / low-audio edge case handled
[ ] Audio format validated (WAV, MP3, sample rate)
```

---

## Sprint Close Checklist

At the end of every sprint:

```
[ ] Sprint outcome section filled in sprint-XXX.md
[ ] PROJECT_STATUS.md updated with new truths
[ ] WORKFLOW_STATUS.md file changes log updated
[ ] New ADRs created for any significant decisions
[ ] PLAN.md updated if strategy/architecture changed
[ ] Git tag applied for milestone releases
[ ] Unfinished tasks captured in next sprint file
[ ] Team/AI agent context refreshed (AGENTS.md current?)
[ ] Test coverage not decreased from last sprint
[ ] Daily log for last day of sprint written
```

---

## New ADR Checklist

When making a significant tech/architecture decision:

```
[ ] ADR file created in docs/decisions/ADR-XXX-title.md
[ ] Status set (Proposed / Accepted / Deprecated / Superseded)
[ ] Context clearly explains WHY a decision was needed
[ ] Decision states WHAT was chosen
[ ] Consequences lists pros AND cons
[ ] Linked from relevant sprint or PLAN.md
[ ] Committed in same PR as code change
```
