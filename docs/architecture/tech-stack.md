# Tech Stack — CropFresh AI

> **Last Updated:** 2026-03-03

## Backend

| Technology       | Purpose              | Why Chosen                                    |
| ---------------- | -------------------- | --------------------------------------------- |
| **FastAPI**      | REST + WebSocket API | Async-native, auto-docs, Pydantic integration |
| **Python 3.11+** | Runtime              | Best AI/ML ecosystem, type hints              |
| **uv**           | Package manager      | 10-100x faster than pip, lockfile support     |

## AI & LLM

| Technology                | Purpose                     | Why Chosen                                      |
| ------------------------- | --------------------------- | ----------------------------------------------- |
| **LangGraph**             | Multi-agent orchestration   | Stateful agent graphs, tool use, streaming      |
| **LangChain**             | LLM tooling                 | Tool/chain compositions, memory                 |
| **AWS Bedrock**           | LLM inference (production)  | Claude Sonnet 4 — quality, IAM auth, AWS-native |
| **Groq**                  | LLM inference (dev/routing) | Fast inference (~80ms), Llama-3.1 for routing   |
| **Sentence Transformers** | Embeddings                  | Local, fast, multilingual (BGE-M3)              |

## Databases — All Cloud ✅

| Technology           | Purpose                   | Connection                                                                           |
| -------------------- | ------------------------- | ------------------------------------------------------------------------------------ |
| **Qdrant Cloud**     | Vector database (primary) | `283826ed-...eu-central-1-0.aws.cloud.qdrant.io:6333` — cluster: `cropfresh-vectors` |
| **Neo4j AuraDB**     | Graph database            | `neo4j+s://93ac2928.databases.neo4j.io` — buyer-seller relations, supply chain       |
| **Redis Labs Cloud** | Caching + sessions        | `redis-13641.crce179.ap-south-1-1.ec2.cloud.redislabs.com:13641`                     |
| ~~RDS PostgreSQL~~   | ~~Primary relational DB~~ | Blocked by AWS Free Tier — local PostgreSQL used for dev                             |
| ~~Supabase~~         | ~~Primary DB~~            | Superseded — see [ADR-012](../decisions/ADR-012-aurora-pgvector-consolidation.md)    |

## Voice & NLP

| Technology          | Purpose                         | Why Chosen                                            |
| ------------------- | ------------------------------- | ----------------------------------------------------- |
| **Edge-TTS**        | Text-to-speech (primary)        | Free, 11 Indian languages, no model download          |
| **Faster-Whisper**  | Speech-to-text (primary)        | CPU-friendly, small model, offline                    |
| **Groq Whisper**    | Speech-to-text (cloud fallback) | `whisper-large-v3-turbo`, fast, uses existing API key |
| **WebRTC + aiortc** | Real-time audio streaming       | Low latency voice transport                           |
| **Silero VAD**      | Voice activity detection        | ONNX model, 1.8MB, 30ms chunks, ~1ms inference        |

## DevOps

| Technology         | Purpose             | Why Chosen                     |
| ------------------ | ------------------- | ------------------------------ |
| **Docker**         | Containerization    | Consistent environments        |
| **AWS**            | Cloud platform      | Bedrock, IAM — unified billing |
| **GitHub Actions** | CI/CD               | Free, integrated with repo     |
| **n8n**            | Workflow automation | Visual workflows, self-hosted  |

## Mobile (Planned)

| Technology | Purpose | Why Chosen |
| ---------- | ------- | ---------- |

## Databases

| Technology                    | Purpose                    | Why Chosen                                                                                                       |
| ----------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **RDS PostgreSQL + pgvector** | Primary DB + vector search | Single instance for relational + embeddings, IAM auth, ~$14/month                                                |
| **Neo4j**                     | Graph database             | Buyer-seller matching, relationship scoring                                                                      |
| **Redis**                     | Caching                    | Session cache, rate limiting                                                                                     |
| ~~Supabase~~                  | ~~Primary DB~~             | Superseded by RDS PostgreSQL ([ADR-012](../decisions/ADR-012-aurora-pgvector-consolidation.md))                  |
| ~~Qdrant~~                    | ~~Vector DB~~              | Superseded by pgvector — kept as dev fallback ([ADR-012](../decisions/ADR-012-aurora-pgvector-consolidation.md)) |

## Voice & NLP

| Technology   | Purpose         | Why Chosen                         |
| ------------ | --------------- | ---------------------------------- |
| **Edge-TTS** | Text-to-speech  | Free, Kannada support, low latency |
| **Whisper**  | Speech-to-text  | Multi-language, Kannada support    |
| **WebRTC**   | Real-time audio | Low latency voice streaming        |

## DevOps

| Technology         | Purpose             | Why Chosen                          |
| ------------------ | ------------------- | ----------------------------------- |
| **Docker**         | Containerization    | Consistent environments             |
| **AWS**            | Cloud platform      | Bedrock, RDS, IAM — unified billing |
| **GitHub Actions** | CI/CD               | Free, integrated with repo          |
| **n8n**            | Workflow automation | Visual workflows, self-hosted       |

## Mobile (Planned)

| Technology   | Purpose              | Why Chosen                     |
| ------------ | -------------------- | ------------------------------ |
| **Flutter**  | Mobile app           | Cross-platform, Dart, fast dev |
| **Firebase** | Auth + notifications | Google integration, free tier  |
