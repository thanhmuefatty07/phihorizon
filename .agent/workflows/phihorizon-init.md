---
description: Load project memory - MUST run at start of every PhiHorizon conversation
---

# /phihorizon-init Workflow

## Purpose
Loads the permanent memory file to ensure continuity across conversations.

## Steps

// turbo
1. Read the project memory file:
```
view_file: C:\Users\ADMIN\PhiHorizon\.agent\MEMORY.md
```

2. Check current project status by testing imports:
```powershell
cd C:\Users\ADMIN\PhiHorizon
python -c "from src import WalkForwardOptimizer, PhiCalculator, IITCore; print('All imports OK')"
```

3. Verify current phase from task tracker if exists in brain artifacts.

## Expected Output
- Memory context loaded
- Import status confirmed
- Ready to continue work on PhiHorizon

## Notes
- This workflow should be run at the START of any PhiHorizon conversation
- If imports fail, check src/consciousness module first
- Current phase: Phase 2 (Data Infrastructure) - 95%
