# 🎯 CropFresh AI: Coding Agent Tracking Guide

**Hello AI Coding Assistant.** If you are reading this, you have been assigned to develop, refactor, or test a component within the CropFresh AI repository.

Before you write a single line of code, you **MUST** read this entire document. This repository does not just write code; it follows a strict, data-driven **Continuous Upgrade Loop**. This `tracking/` folder is the brain of that loop.

Your primary directive is: **We do not just build blindly; we plan, we build, we review the metrics, and we UPGRADE.**

---

## 🔄 The 4-Step Development & Upgrade Loop

Every single feature, bug fix, or new agent you work on **MUST** pass through these four strictly tracked phases. You are expected to update the corresponding files in this folder at every step.

### 1️⃣ STEP 1: PLAN (What are we doing and why?)

Before coding, define the intent. Does it align with the North Star (e.g., lower logistics to <₹2.5/kg, hit 90% routing accuracy)?

- **Action:**
  - Create a detailed markdown file in `tracking/tasks/` (e.g., `task35-new-feature.md`).
  - Add the task to the current active sprint in `tracking/sprints/`.
  - Verify goals against `tracking/GOALS.md` and `tracking/okrs/`.

### 2️⃣ STEP 2: DEVELOP (Writing the Code)

Execute structural changes to the codebase. Adhere strictly to project constraints (e.g., Max 500 lines per Python file).

- **Action:**
  - Log your day-to-day granular progress in `tracking/daily/` or `tracking/weekly/`.
  - Update the master dashboards: `tracking/PROJECT_STATUS.md` and `tracking/WORKFLOW_STATUS.md` to reflect active development.

### 3️⃣ STEP 3: REVIEW (What went well & Results)

The code is written and tests pass. Now, analyze the output. Did it actually solve the problem? What is the latency? Did the LLM hallucinate?

- **Action:**
  - Write a brutal, honest retrospective in `tracking/retros/` (e.g., `feature_retro.md`).
  - Log specific quantitative metrics (latency, accuracy) in `tracking/agent-performance/`.
  - Update `tracking/OUTCOMES.md` with the final results.

### 4️⃣ STEP 4: UPGRADE (The Next Iteration)

Take the failures, bottlenecks, or missing features identified in **Step 3**, and immediately plan how to fix them.

- **Action:**
  - Loop back to **Step 1 (PLAN)**. Create a new task in `tracking/tasks/` dedicated solely to upgrading and fixing the issues found in the retro.

---

## 🗂️ The Tracking Directory Layout (Where things go)

Do not put files in the wrong place. Use this exact structure:

| Folder / File          | Purpose                                                                           | AI Agent Action Required                                                           |
| ---------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `PROJECT_STATUS.md`    | The High-Level Dashboard. Shows exactly which milestone we are on (e.g., v0.9.4). | Update status to ✅ when a major component finishes.                               |
| `WORKFLOW_STATUS.md`   | The granular Engineering Log.                                                     | Append rows for every file you CREATE, UPDATE, or DELETE during a session.         |
| `/tasks/`              | Individual feature/bug tickets.                                                   | Create `taskXX.md` for new work. Mark completed when AC is met.                    |
| `/sprints/`            | Grouped task logic.                                                               | Move tasks from planned ➔ active ➔ done.                                           |
| `/retros/`             | **CRITICAL:** Post-mortems.                                                       | After finishing a system, write what worked, what failed, and _how to upgrade it_. |
| `/agent-performance/`  | Quantitative metrics.                                                             | Log benchmark speeds, token usage, and latency data here.                          |
| `/daily/` & `/weekly/` | Brain dumps.                                                                      | Log your exact thought process and small file tweaks here.                         |
| `/okrs/`               | Objectives & Key Results.                                                         | Ensure your task aligns with business goals (e.g., reducing dispute rates).        |

---

## 🤖 Context: The 6 Core Agents & Architecture

To know _where_ you are working, you must understand the architecture:

1. **Supervisor Routing Layer:** User queries hit the Supervisor. It uses high-confidence LLM routing to direct data.
2. **The 6 Isolated App Agents:**
   - **Voice Agent:** Whisper STT ➔ LLM ➔ Cartesia TTS.
   - **Pricing Agent:** Calculates real-time AISP (All-Inclusive Sourcing Price) capping at mandi + 5%.
   - **Buyer Matching:** 5-factor weighted algorithm connecting farmers to buyers.
   - **Vision (Quality) Agent:** DINOv2 classification + YOLOv26 defect detection.
   - **Digital Twin:** ResNet50 hashing to verify produce condition at departure vs. arrival.
   - **ADCL (Market Intelligence) Agent:** Analyzes order volume, price trends, and planting seasons to issue weekly crop recommendations.
3. **Data/RAG:** Deep Research tools, Qdrant/pgvector, Neo4j Graph DB.

---

## � YOUR INITIALIZATION PROMPT

**User:** Whenever you start a new conversation with me (the AI Coding Assistant), copy and paste this prompt to ensure I am perfectly synced with the tracking system:

> "Hello AI. You are working in the CropFresh AI repository.
>
> 1. Read `tracking/README.md` immediately to understand the **4-Step Upgrade Loop**.
> 2. Initialize your context by reading `tracking/PROJECT_STATUS.md` to see our current Sprint and Version.
> 3. For this session, we are working on: **[Insert Task/Goal here]**.
> 4. Tell me if this is Step 1 (Plan), Step 2 (Develop), Step 3 (Review), or Step 4 (Upgrade).
> 5. Log all your file changes in `tracking/WORKFLOW_STATUS.md` as we go.
>    Let's begin."
