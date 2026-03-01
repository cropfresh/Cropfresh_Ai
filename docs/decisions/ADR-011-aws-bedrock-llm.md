# ADR-011: AWS Bedrock as Primary LLM Provider

**Date**: 2026-03-01  |  **Status**: Accepted  
**Supersedes**: N/A (Groq was previously default, now dev-only fallback)

### Context
CropFresh AI initially used Groq (Llama-3.3-70B) as sole LLM provider. As we migrate to AWS infrastructure, we need AWS-native LLM integration for production. Groq remains useful for speed-critical dev work.

### Decision
Use **AWS Bedrock** (Claude Sonnet 4) as the primary production LLM provider, with Groq as a speed-optimized fallback for development and router tasks.

**Dual-provider strategy:**
| Task Type | Provider | Model | Rationale |
|-----------|----------|-------|-----------|
| RAG generation, complex chat | Bedrock | Claude Sonnet 4 | Quality, depth |
| Query routing, drafting | Groq | Llama-3.1-8B | Speed (~80ms) |
| Agent planning, reasoning | Bedrock | Claude Sonnet 4 | Accuracy |

### Consequences
- ✅ AWS-native: IAM auth, CloudWatch metrics, VPC endpoints
- ✅ Better quality on complex agricultural queries (Claude vs Llama)
- ✅ No API key rotation — uses IAM roles in production
- ✅ Groq kept for fast routing (~80ms vs ~1.5s Bedrock)
- ⚠️ Higher latency for Bedrock (~1.5s first-token)
- ⚠️ Bedrock pricing is higher than Groq free tier

### Implementation
- `src/shared/orchestrator/llm_provider.py` — `BedrockProvider` class added
- `src/config/settings.py` — `llm_provider` defaults to `bedrock`
- Provider-agnostic guards via `has_llm_configured` property
- `.env.example` — both provider sections documented
