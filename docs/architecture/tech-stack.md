# Tech Stack — CropFresh AI

> **Last Updated:** 2026-03-01

## Backend
| Technology | Purpose | Why Chosen |
|-----------|---------|------------|
| **FastAPI** | REST + WebSocket API | Async-native, auto-docs, Pydantic integration |
| **Python 3.11+** | Runtime | Best AI/ML ecosystem, type hints |
| **uv** | Package manager | 10-100x faster than pip, lockfile support |

## AI & LLM
| Technology | Purpose | Why Chosen |
|-----------|---------|------------|
| **LangGraph** | Multi-agent orchestration | Stateful agent graphs, tool use, streaming |
| **LangChain** | LLM tooling | Tool/chain compositions, memory |
| **AWS Bedrock** | LLM inference (production) | Claude Sonnet 4 — quality, IAM auth, AWS-native |
| **Groq** | LLM inference (dev/routing) | Fast inference (~80ms), Llama-3.1 for routing |
| **Sentence Transformers** | Embeddings | Local, fast, multilingual (BGE-M3) |

## Databases
| Technology | Purpose | Why Chosen |
|-----------|---------|------------|
| **RDS PostgreSQL + pgvector** | Primary DB + vector search | Single instance for relational + embeddings, IAM auth, ~$14/month |
| **Neo4j** | Graph database | Buyer-seller matching, relationship scoring |
| **Redis** | Caching | Session cache, rate limiting |
| ~~Supabase~~ | ~~Primary DB~~ | Superseded by RDS PostgreSQL ([ADR-012](../decisions/ADR-012-aurora-pgvector-consolidation.md)) |
| ~~Qdrant~~ | ~~Vector DB~~ | Superseded by pgvector — kept as dev fallback ([ADR-012](../decisions/ADR-012-aurora-pgvector-consolidation.md)) |

## Voice & NLP
| Technology | Purpose | Why Chosen |
|-----------|---------|------------|
| **Edge-TTS** | Text-to-speech | Free, Kannada support, low latency |
| **Whisper** | Speech-to-text | Multi-language, Kannada support |
| **WebRTC** | Real-time audio | Low latency voice streaming |

## DevOps
| Technology | Purpose | Why Chosen |
|-----------|---------|------------|
| **Docker** | Containerization | Consistent environments |
| **AWS** | Cloud platform | Bedrock, RDS, IAM — unified billing |
| **GitHub Actions** | CI/CD | Free, integrated with repo |
| **n8n** | Workflow automation | Visual workflows, self-hosted |

## Mobile (Planned)
| Technology | Purpose | Why Chosen |
|-----------|---------|------------|
| **Flutter** | Mobile app | Cross-platform, Dart, fast dev |
| **Firebase** | Auth + notifications | Google integration, free tier |
