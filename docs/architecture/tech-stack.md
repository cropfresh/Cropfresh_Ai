# Tech Stack — CropFresh AI

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
| **Groq** | LLM inference | Fast inference, Llama/Mixtral access |
| **Sentence Transformers** | Embeddings | Local, fast, multilingual |

## Databases
| Technology | Purpose | Why Chosen |
|-----------|---------|------------|
| **Supabase** | Primary DB (PostgreSQL) | Auth, realtime, edge functions, free tier |
| **Qdrant** | Vector database | RAG retrieval, semantic search |
| **Neo4j** | Graph database | Buyer-seller matching, relationship scoring |
| **Redis** | Caching | Session cache, rate limiting |

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
| **GCP Cloud Run** | Deployment | Serverless, auto-scaling, pay-per-use |
| **GitHub Actions** | CI/CD | Free, integrated with repo |
| **n8n** | Workflow automation | Visual workflows, self-hosted |

## Mobile (Planned)
| Technology | Purpose | Why Chosen |
|-----------|---------|------------|
| **Flutter** | Mobile app | Cross-platform, Dart, fast dev |
| **Firebase** | Auth + notifications | Google integration, free tier |
