# CropFresh AI — Agent Registry

> **Last Updated:** 2026-03-11
> **Total Agents:** 15 (1 Supervisor + 14 Domain Agents)
> **Source:** `src/agents/agent_registry.py`

---

## How Agents Work

Every user query goes through `SupervisorAgent` which routes to the best domain agent:

```
User Query → SupervisorAgent.route_query() → TargetAgent.process() → AgentResponse
```

All agents are wired at startup by `agent_registry.create_agent_system()`, which:
1. Creates shared infrastructure (StateManager, ToolRegistry)
2. Instantiates agents in 6 groups (core, pricing, marketplace, web, wrapper, knowledge)
3. Registers each with `SupervisorAgent.register_agent()`
4. Sets `general_agent` as the fallback

---

## Agent Summary Table

| # | Agent | Module | Inherits BaseAgent | Group | Status |
|---|-------|--------|-------------------|-------|--------|
| 0 | **SupervisorAgent** | `src/agents/supervisor_agent.py` | ✅ | — | Stable |
| 1 | AgronomyAgent | `src/agents/agronomy_agent.py` | ✅ | Core | Stable |
| 2 | CommerceAgent | `src/agents/commerce_agent.py` | ✅ | Core | Stable |
| 3 | PlatformAgent | `src/agents/platform_agent.py` | ✅ | Core | Stable |
| 4 | GeneralAgent | `src/agents/general_agent.py` | ✅ | Core | Stable (fallback) |
| 5 | PricingAgent | `src/agents/pricing_agent.py` | ❌ | Pricing | Partial |
| 6 | PricePredictionAgent | `src/agents/price_prediction/agent.py` | ❌ | Pricing | Partial |
| 7 | BuyerMatchingAgent | `src/agents/buyer_matching/agent.py` | ❌ | Marketplace | Partial |
| 8 | QualityAssessmentAgent | `src/agents/quality_assessment/agent.py` | ❌ | Marketplace | Partial |
| 9 | CropListingAgent | `src/agents/crop_listing/agent.py` | ❌ | Marketplace | Partial |
| 10 | WebScrapingAgent | `src/agents/web_scraping_agent.py` | ❌ | Web | Stable |
| 11 | BrowserAgent | `src/agents/browser_agent.py` | ❌ | Web | Stable |
| 12 | ResearchAgent | `src/agents/research/research_agent.py` | ✅ | Web | Stable |
| 13 | ADCLWrapperAgent | `src/agents/adcl_wrapper_agent.py` | ✅ | Wrapper | TODO |
| 14 | LogisticsWrapperAgent | `src/agents/logistics_wrapper_agent.py` | ❌ | Wrapper | TODO |
| 15 | KnowledgeAgent | `src/agents/knowledge_agent.py` | ❌ | Knowledge | Stable |

---

## Detailed Agent Descriptions

### 0. SupervisorAgent (Orchestrator)

| Property | Value |
|----------|-------|
| **File** | `src/agents/supervisor_agent.py` (656 lines) |
| **Purpose** | Central orchestrator — routes queries to specialized agents |
| **Temperature** | 0.3 (routing), 0.1 (LLM routing calls) |
| **Routing** | LLM-based (JSON output) with rule-based fallback |
| **Multi-agent** | Supports `requires_multiple` + `secondary_agents` |
| **Session** | `process_with_session()` maintains conversation context |

**Key Methods:**
- `route_query(query, context)` → `RoutingDecision`
- `process(query, context, execution)` → `AgentResponse`
- `process_with_session(query, session_id)` → `AgentResponse`
- `register_agent(name, agent)` / `set_fallback_agent(agent)`

### 1. AgronomyAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/agronomy_agent.py` (9,733 bytes) |
| **Purpose** | Crop cultivation, pest management, soil health, irrigation advice |
| **KB Categories** | `agronomy`, `crops` |
| **Tools** | `get_weather`, `imd_weather` |
| **Keywords** | grow, plant, cultivate, harvest, pest, disease, fertilizer, soil, seed |

### 2. CommerceAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/commerce_agent.py` (12,836 bytes) |
| **Purpose** | Market prices, AISP calculations, sell/hold recommendations |
| **KB Categories** | `commerce`, `prices` |
| **Tools** | `agmarknet`, `ml_forecaster`, `news_sentiment` |
| **Keywords** | price, sell, buy, mandi, market, rate, cost, profit, AISP |

### 3. PlatformAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/platform_agent.py` (8,930 bytes) |
| **Purpose** | CropFresh app features, registration, account support |
| **Keywords** | register, login, app, feature, account, order, payment |

### 4. GeneralAgent (Fallback)

| Property | Value |
|----------|-------|
| **File** | `src/agents/general_agent.py` (9,216 bytes) |
| **Purpose** | Greetings, general questions, unclear intents |
| **Keywords** | hello, hi, thanks, help, who are you |
| **Note** | Used as fallback when no other agent matches |

### 5. PricingAgent (DPLE Engine)

| Property | Value |
|----------|-------|
| **File** | `src/agents/pricing_agent.py` (18,640 bytes) |
| **Purpose** | AISP calculation, mandi price aggregation |
| **Formula** | AISP = Farmer Ask + Logistics + Margin(4-8%) + Risk Buffer(2%) |
| **Note** | Does NOT inherit BaseAgent — uses own LLM interface |
| **Related** | `src/agents/aisp_calculator.py` |

### 6. PricePredictionAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/price_prediction/agent.py` |
| **Purpose** | Price forecasting, trend analysis, sell/hold timing |
| **Keywords** | predict, forecast, trend, future price, hold or sell |

### 7. BuyerMatchingAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/buyer_matching/agent.py` |
| **Purpose** | Match farmers with buyers using grade, price, and distance |
| **Keywords** | find buyer, match buyer, sell my produce |
| **Related** | `src/agents/buyer_matching/` (subdirectory with models/logic) |

### 8. QualityAssessmentAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/quality_assessment/agent.py` |
| **Purpose** | Produce grading (A+/A/B/C), defect detection, shelf life, HITL |
| **Keywords** | quality check, grade produce, defects, shelf life |
| **Auto-route** | Triggered when `context.image_b64` is present (photo input) |

### 9. CropListingAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/crop_listing/agent.py` |
| **Purpose** | Creating/managing produce listings for sale |
| **Keywords** | list my crop, create listing, my listings, cancel listing |

### 10. WebScrapingAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/web_scraping_agent.py` (22,648 bytes) |
| **Purpose** | Fetch LIVE data from websites — mandi prices, weather, news |
| **Keywords** | live, current, today, real-time, fetch, scrape, Agmarknet |
| **Note** | Does NOT inherit BaseAgent — uses Scrapling directly |

### 11. BrowserAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/browser_agent.py` (15,089 bytes) |
| **Purpose** | Interactive web tasks requiring login, form submission, navigation |
| **Keywords** | login to, submit, navigate, download, dashboard |
| **Note** | Does NOT inherit BaseAgent — uses Playwright |

### 12. ResearchAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/research/research_agent.py` |
| **Purpose** | Deep research with multiple sources, citations, reports |
| **Keywords** | research, investigate, comprehensive, compare, analysis |
| **Tools** | `deep_research`, `web_search` |

### 13. ADCLWrapperAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/adcl_wrapper_agent.py` (7,189 bytes) |
| **Purpose** | Assured Demand Crop List — weekly crop recommendations |
| **Keywords** | recommend, sow, what to grow, demand, which crop |
| **Related** | `src/agents/adcl/` (ADCL engine modules) |

### 14. LogisticsWrapperAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/logistics_wrapper_agent.py` (6,445 bytes) |
| **Purpose** | Delivery routing, transport cost, vehicle assignment |
| **Keywords** | delivery, transport, route, vehicle, logistics cost |
| **Related** | `src/agents/logistics_router/` (routing engine) |

### 15. KnowledgeAgent

| Property | Value |
|----------|-------|
| **File** | `src/agents/knowledge_agent.py` (6,811 bytes) |
| **Purpose** | Deep knowledge retrieval from agricultural knowledge base |
| **Keywords** | explain, tell me about, information, what is, how does |
| **Backend** | Qdrant (agri_knowledge collection) |
| **Note** | Does NOT inherit BaseAgent — has own Qdrant connection |

---

## BaseAgent Contract

All agents inheriting `BaseAgent` (from `src/agents/base_agent.py`) must implement:

```python
class MyAgent(BaseAgent):
    async def process(self, query, context=None, execution=None) -> AgentResponse:
        """Main agent logic — MUST be implemented."""
        ...

    def _get_system_prompt(self, context=None) -> str:
        """Domain-specific system prompt — MUST be implemented."""
        ...
```

**Available in BaseAgent:**
- `self.llm` — LLM provider instance
- `self.tools` — ToolRegistry for executing tools
- `self.state_manager` — AgentStateManager for session/memory
- `self.knowledge_base` — KnowledgeBase for RAG retrieval
- `await self.retrieve_context(query, top_k, categories)` — KB search
- `await self.use_tool(tool_name, **kwargs)` — Execute tool safely
- `await self.generate_with_llm(messages, context=context)` — LLM call with memory injection
- `self.format_context(documents)` — Format retrieved docs for prompt
- `await self._retry_operation(func, ...)` — Retry with exponential backoff

---

## Tool Registry

Tools available to agents via `ToolRegistry` (from `src/agents/agent_registry.py`):

| Tool | Category | Module | Description |
|------|----------|--------|-------------|
| `agmarknet` | commerce | `src/tools/agmarknet.py` | APMC mandi price data |
| `ml_forecaster` | commerce | `src/tools/ml_forecaster.py` | ML-based price prediction |
| `news_sentiment` | commerce | `src/tools/news_sentiment.py` | Agri news sentiment analysis |
| `imd_weather` | agronomy | `src/tools/imd_weather.py` | IMD weather forecasts |
| `get_weather` | agronomy | `src/tools/weather.py` | General weather tool |
| `deep_research` | research | `src/tools/deep_research.py` | Multi-source deep research |
| `web_search` | general | `src/tools/web_search.py` | Web search tool |

---

## Related Documentation

| Document | Path |
|----------|------|
| Agent Design Guide | [`docs/agents/agent-design-guide.md`](agent-design-guide.md) |
| System Architecture | [`docs/architecture/system-architecture.md`](../architecture/system-architecture.md) |
| Data Flow Diagrams | [`docs/architecture/data-flow.md`](../architecture/data-flow.md) |
