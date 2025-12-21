---
description: Protocol for deferred execution of heavy tasks on Cloud (Kaggle/Colab)
---

# Hybrid Cloud Execution Protocol
**Created:** Dec 22, 2025
**Status:** ACTIVE - PERMANENT RULE

## 1. The Rule
**"Heavy tasks must be deferred to Cloud, but must not block local workflow."**

## 2. Triggers
ANY task requiring:
- Training (ML/DL models)
- Large-scale Backtesting (>1 year, >100 parameter combinations)
- Heavy Optimization (Hyperopt, Genetic Algorithms)
- Complex Data Processing (>1GB data)

## 3. The Workflow
When the Agent encounters a Trigger:

1.  **ACKNOWLEDGE but DEFER:**
    -   *Do not* run the heavy command locally.
    -   *Do not* ask the user to switch context immediately.
    -   Log the task in `task.md` under a "☁️ CLOUD QUEUE" section.

2.  **CONTINUE LOCAL WORK:**
    -   Proceed with coding, refactoring, or lightweight tests (unit tests).
    -   Keep the user's local flow uninterrupted.

3.  **BATCH & DISPATCH (At Task Boundary):**
    -   At the end of the current major task/session:
    -   **Generate** the necessary script or Notebook Cell.
    -   **Notify** the user: "Hybrid Dispatch: Please run these tasks on Kaggle/Colab now."

## 4. Implementation Details
-   **Kaggle:** Use `colab/phihorizon_complete_training.ipynb` as the execution engine.
-   **Local:** Keep it light (Unit tests, Syntax checks, Logic verification).

