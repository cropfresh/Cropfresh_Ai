---
description: How to perform a comprehensive documentation update for CropFresh AI
---

# Update Docs Workflow

Use this when performing a comprehensive documentation update.

## Steps

1. **Read all source files** in the affected area:
   ```
   src/agents/  → docs/agents/
   src/api/     → docs/api/
   src/voice/   → docs/features/voice-pipeline.md
   src/rag/     → docs/features/rag-pipeline.md
   src/scrapers/ → docs/features/scraping-system.md
   ```

2. **Update docs to match current code:**
   - Compare module counts, function signatures, and routing keywords
   - Add any new agents/endpoints/tools to their respective docs
   - Update Mermaid diagrams if architecture changed

3. **Update tracking files:**
   - `tracking/PROJECT_STATUS.md` — component status table
   - Active sprint file if tasks completed

4. **Update YAML configs:**
   - `config/project-context.yaml` — if new doc files created
   - `docs/agents-index.yaml` — if agents added/removed

5. **Verify cross-references:**
   - All `[link text](path)` references point to existing files
   - No broken links in the documentation

6. **Log all changes in `WORKFLOW_STATUS.md`**
