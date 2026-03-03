# Changelog

All notable changes to CropFresh AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

## [0.9.2] - 2026-03-03

### Added

- **Cloud databases fully connected** — all three verified with live connectivity tests:
  - **Qdrant Cloud** (`cropfresh-vectors` cluster, EU-Central) — vector search active
  - **Neo4j AuraDB** (`93ac2928.databases.neo4j.io`) — graph DB, CONNECT ✅ CREATE 20ms ✅
  - **Redis Labs Cloud** (`redis-13641.crce179.ap-south-1-1.ec2.cloud.redislabs.com:13641`) — PING ✅ SET/GET ✅
- `src/db/neo4j_client.py` — updated `get_neo4j()` to read from `NEO4J_*` env vars directly (no settings circular import)
- `.env` — `REDIS_URL`, `REDIS_HOST/PORT/USER/PASS`, `NEO4J_URI/USERNAME/PASSWORD/DATABASE` all set

### Changed

- `docs/architecture/tech-stack.md` — updated Databases section to reflect cloud stack
- `tracking/PROJECT_STATUS.md` — v0.9.2, updated tech stack table with status flags, updated blockers, added cloud DB milestone

---

### Added

- Advanced folder structure for production-scale project organization
- Complete documentation hierarchy (planning, architecture, decisions, features, agents, api)
- Development tracking system (goals, milestones, sprints, daily logs, retros)
- AI infrastructure (data pipeline, model registry, evaluation framework, RAG)
- Infrastructure configs (Docker, GCP, monitoring)
- Database configs (Supabase migrations, Qdrant, Neo4j, Firebase, n8n)
- CI/CD pipeline templates (.github/workflows)
- Test infrastructure (e2e, integration, load testing)

### Changed

- Restructured `src/` into organized modules (api, agents, scrapers, pipelines, shared)
- Moved RAG pipeline from `src/rag/` to `ai/rag/`
- Moved notebooks and models into `ai/` directory
- Reorganized utility modules into `src/shared/`

---

## [0.1.0] - 2026-02-27

### Added

- Initial project setup with FastAPI backend
- Multi-agent system with LangGraph
- RAG pipeline with Qdrant vector database
- Neo4j graph knowledge base
- Voice processing (STT/TTS with Kannada support)
- APMC market data scraping tools
- WebSocket real-time communication
