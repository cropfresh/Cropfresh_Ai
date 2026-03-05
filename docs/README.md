# � CRITICAL DIRECTIVE: CropFresh AI Documentation Hub 🚨

**Attention AI Coding Assistant:** You are standing at the gates of the `docs/` repository. Before you write, modify, or delete a **single line of functional code** in this project, you MUST read and acknowledge this document.

This is not a suggestion; this is a **strict project law**.

**"THE CODE + DOCS CODEPENDENCY RULE"**: In CropFresh AI, a feature is **NOT DONE** until the `docs/` folder reflects the change. If you change a small calculation in the ADCL agent, you must update the ADCL docs. If you add a new API route, you must update the API specs. **Code and documentation must mutate simultaneously.**

---

## 🗂️ The Documentation Directory Layout

To quickly find or update the context you need, use this directory guide:

| Folder / File    | Purpose & AI Mandate                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `/architecture/` | Core system blueprints (RAG, data flow). **Mandate:** Update if you change core DB schemas or routing logic.                          |
| `/agents/`       | Documentation for the 6 Core Agents. **Mandate:** If you touch `src/agents/*.py`, you MUST update the corresponding file here.        |
| `/features/`     | Specs (F001 to F009). **Mandate:** Update these if business requirements or constraints are altered.                                  |
| `/runbooks/`     | **[NEW]** Operational guides, common bugs, and debug steps. **Mandate:** If you fix a repetitive bug, document the fix here.          |
| `/guides/`       | **[NEW]** Step-by-step how-to tutorials (e.g., adding a RAG pipeline). **Mandate:** Create a guide if you build a repeatable process. |
| `/api-specs/`    | **[NEW]** Postman/OpenAPI schemas. **Mandate:** Update immediately if an endpoint payload changes.                                    |
| `/decisions/`    | Architecture Decision Records (ADRs). **Mandate:** Create an ADR if you make a major tech choice (e.g., swapping a DB).               |
| `/api/`          | REST Endpoints overview.                                                                                                              |
| `/planning/`     | PRDs, Roadmaps, User Journeys.                                                                                                        |

---

## 🔄 Core Agent Data Flow (Executive Summary)

For a deep dive, read `architecture/data-flow.md`. Here is the 10,000-foot view:

1. **Farmer Listing:** Farmer (Voice App) ➔ Voice Agent parses intent ➔ Quality Assessment (CV-QG) verifies ➔ Listing Active in Supabase.
2. **Pricing & Matchmaking:** Matchmaking engine clusters farms/buyers ➔ Pricing Agent calculates AISP (Mandi price + logistics + margin) ➔ DPLE routes transport.
3. **Escrow & Settlement:** Buyer pays Escrow ➔ Hauler picks up/delivers ➔ Digital Twin verifies quality ➔ Funds released.
4. **Advisory (RAG):** Farmer queries ➔ Adaptive Query Router selects best source (Vector, Graph, Live API) ➔ Edge TTS responds.
5. **Market Intelligence (ADCL):** Analyzes buyer demand patterns & seasonality ➔ Generates green-label "what to sow" recommendations.

---

## 🤖 YOUR AGGRESSIVE INITIALIZATION & EXIT PROMPT

As an AI Assistant, you must obey the following workflow to bridge the gap between `tracking/` (planning) and `docs/` (execution).

### When Starting a Task:

Whenever you begin, initialize your context using this mantra:

> "Hello AI. I am modifying a component in the CropFresh AI repository.
>
> 1. I will read `docs/README.md` to identify where my current feature's documentation lives.
> 2. I will read the specific feature/architecture/agent docs BEFORE I write code to understand the constraints.
> 3. I will proceed to `tracking/README.md` to plan my execution."

### Before Finishing a Task (THE STRICT GATE):

Before you output a final summary or tell the user a task is complete, you **MUST** run this internal checklist:

> ❗ **DID I CHANGE THE LOGIC?**
>
> - If YES: Did I update the corresponding file in `docs/agents/`, `docs/api/`, or `docs/architecture/`?
> - If NO: **STOP.** I am not allowed to finish this task until the documentation is updated to match the code.

By adhering to this, you ensure that the next AI Assistant that works on this codebase will not hallucinate based on outdated information.
