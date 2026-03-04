# Sprint Retrospective — AI Agents Subsystem

## 🟢 What Went Well

- **Supervisor Orchestration**: `SupervisorAgent` successfully implements a robust dual-layer routing system. It uses an LLM to dynamically parse intents and route queries to 9 specialized agents, while seamlessly falling back to a weighted keyword heuristic engine if the LLM fails or is unavailable.
- **Dynamic Pricing Engine (DPLE)**: `PricingAgent` serves as a masterclass in data aggregation. It concurrently fetches live Agmarknet prices, ML forecasts (7-day ahead), weather warnings, and news sentiment, feeding them into a contextualized Prompt for the LLM to provide holistic "sell/hold/wait" recommendations.
- **Extensible Base Architecture**: `BaseAgent` provides solid operational abstractions for State Management (memory limits), Tool execution tracking, and context formulation, making adding new agents (e.g. `AgronomyAgent`) trivial.

## 🟡 What Could Improve

- **Fragile LLM Routing Parsing**: In `SupervisorAgent.route_query()`, extracting the JSON routing decision uses basic string splitting (`split("```")`). This is prone to breaking if the LLM adds conversational wrapping or malformed markdown. It does not utilize native structured output configurations.
- **Lack of Adaptive Context**: The `BaseAgent.retrieve_context` defaults to a static `top_k=5` instead of adjusting based on query complexity or token limits, which can blow up the context window for broad queries.
- **Tool Error Handling**: In `BaseAgent.use_tool`, if a tool fails, it silently records `"success": False` and returns. The agent isn't automatically prompted to retry a different strategy or fix tool arguments.

## 🔴 Action Items

- [ ] Upgrade `SupervisorAgent` to use formal `ResponseFormat` (e.g. Pydantic schema enforcement on OpenAI/Groq) to guarantee valid JSON routing outputs natively.
- [ ] Implement an 'auto-retry tool' loop within `BaseAgent.process()` that detects tool failures and gives the LLM one chance to self-correct its parameters.
- [ ] Add dynamic `top_k` calculation in context retrieval based on the token count of the incoming user query and available window size.
