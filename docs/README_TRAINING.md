# PhiHorizon - ML Training Guide (Hybrid Protocol)

## ☁️ Hybrid Cloud Execution Protocol

> **CRITICAL RULE:** Do NOT run heavy training or full backtests locally.
> Use the **Hybrid Workflow**: Develop locally → Execute on Cloud.

---

## 1. Local Development (Lightweight)

Use for: Unit tests, syntax checks, small logic verification.

```bash
# Verify imports and basic logic
pytest tests/ -v
```

## 2. Cloud Execution (Heavy)

Use for: Full training, Walk-Forward Optimization (>1yr), Hyperparameter tuning.

**Google Colab / Kaggle:**
1.  **Notebook:** `colab/phihorizon_complete_training.ipynb`
2.  **Upload:** Upload this notebook to your preferred cloud platform.
3.  **Run:** Execute all cells. This notebook is self-contained.
4.  **Download:** Get `benchmark_report.json` and model files.

## 3. Why Hybrid?
-   **Local:** Limited hardware (CPU), good for coding.
-   **Cloud:** Powerful GPUs (T4/P100), good for computation.
-   **Result:** Faster iteration without freezing your local machine.

---

**Last Updated:** Dec 22, 2025
**Protocol:** Active

