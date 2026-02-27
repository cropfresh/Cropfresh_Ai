# Changelog

All notable changes to CropFresh AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

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
