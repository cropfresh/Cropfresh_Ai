---
description: How to generate documentation from existing JSON workflow definitions
---

# Workflow Doc from JSON

CropFresh has workflow definitions in `src/workflows/*.json`. Use this workflow to document them.

## Steps

1. **List workflow JSON files:**
   ```bash
   ls src/workflows/*.json
   ```

2. **Read each JSON file** and extract:
   - Workflow name and trigger
   - Steps/nodes and their connections
   - Input/output schemas
   - External service calls

3. **Create a Mermaid diagram** showing the workflow flow:
   ```mermaid
   graph LR
       trigger --> step1 --> step2 --> output
   ```

4. **Create documentation file:**
   - Create `docs/workflows/<workflow-name>.md`
   - Include: purpose, trigger, steps, Mermaid diagram, example payload

5. **Log in `WORKFLOW_STATUS.md`**
