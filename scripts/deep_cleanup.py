#!/usr/bin/env python3
"""
PhiHorizon Deep Cleanup Script v2
Thorough cleanup including notebooks, models, and data reorganization.
"""

import os
import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\PhiHorizon")

# ====== NOTEBOOKS TO KEEP (Renamed for clarity) ======
# Current notebooks have numbering conflicts, need reorganization

NOTEBOOKS_KEEP = {
    # Core training notebooks (final versions)
    "01_data_preparation.ipynb": True,      # Keep - data prep
    "04_core1_training.ipynb": "01_core1_basic_training.ipynb",  # Rename
    "08_core1_splus_training.ipynb": "02_core1_splus_training.ipynb",
    "09_core1_breakthrough.ipynb": "03_core1_breakthrough.ipynb",
    
    "05_core2_training.ipynb": "04_core2_basic_training.ipynb",
    "10_core2_breakthrough.ipynb": "05_core2_breakthrough.ipynb",
    
    "06_core3_training.ipynb": "06_core3_basic_training.ipynb",
    "07_core3_splus_training.ipynb": "07_core3_splus_training.ipynb",
}

NOTEBOOKS_DELETE = [
    "02_sentiment_training.ipynb",    # Old sentiment LSTM
    "03_lstm_training.ipynb",         # Redundant LSTM
    "06_backtesting.ipynb",           # Conflict with 06_core3
    "07_paper_trading.ipynb",         # Conflict with 07_core3_splus
]

# ====== OLD MODELS TO DELETE ======
OLD_MODELS = [
    # Old LSTM sentiment models (replaced by BERT)
    "models/sentiment/sentiment_lstm.h5",
    "models/sentiment/sentiment_lstm_best.h5",
    "models/sentiment/sentiment_scaler.json",
    "models/sentiment/sentiment_training_metadata.json",
    "models/sentiment/v61_lstm_best.h5",
    "models/sentiment/v61_lstm_final.h5",
    "models/sentiment/v61_scaler.json",
    "models/sentiment/v61_training_metadata.json",
    
    # Old quant_transformer in models/core (non-S+)
    "models/core/quant_transformer_best.pt",
    "models/core/quant_transformer_final.pt",
]

# ====== OLD DATA TO DELETE ======
OLD_DATA = [
    "data/sentiment/sentiment_test_features.csv",
    "data/sentiment/sentiment_train_features.csv",
    "data/sentiment/v61_checksums.json",
    "data/sentiment/v61_test_features.csv",
    "data/sentiment/v61_train_features.csv",
]

# ====== EMPTY DIRS TO DELETE ======
EMPTY_DIRS_TO_CHECK = [
    "data/sentiment",
    "models/sentiment",
    "results",
]

# ====== CLEANUP FUNCTIONS ======

def delete_file(filepath):
    path = ROOT / filepath
    if path.exists():
        size = path.stat().st_size / 1024
        path.unlink()
        print(f"✓ Deleted: {filepath} ({size:.1f} KB)")
        return size
    return 0

def delete_dir_if_empty(dirpath):
    path = ROOT / dirpath
    if path.exists() and path.is_dir():
        if not any(path.iterdir()):
            path.rmdir()
            print(f"✓ Removed empty dir: {dirpath}")
            return True
    return False

def rename_file(old_name, new_name, subdir="notebooks"):
    old_path = ROOT / subdir / old_name
    new_path = ROOT / subdir / new_name
    if old_path.exists() and not new_path.exists():
        old_path.rename(new_path)
        print(f"✓ Renamed: {old_name} → {new_name}")
        return True
    return False

def main():
    print("=" * 60)
    print("PHIHORIZON DEEP CLEANUP v2")
    print("=" * 60)
    
    total_kb = 0
    deleted_count = 0
    
    # 1. Delete old notebooks
    print("\n📓 Cleaning notebooks...")
    for nb in NOTEBOOKS_DELETE:
        kb = delete_file(f"notebooks/{nb}")
        if kb > 0:
            total_kb += kb
            deleted_count += 1
    
    # 2. Delete old models
    print("\n🗑️ Deleting old models...")
    for m in OLD_MODELS:
        kb = delete_file(m)
        if kb > 0:
            total_kb += kb
            deleted_count += 1
    
    # 3. Delete old data
    print("\n🗑️ Deleting old data...")
    for d in OLD_DATA:
        kb = delete_file(d)
        if kb > 0:
            total_kb += kb
            deleted_count += 1
    
    # 4. Delete benchmark result
    print("\n🗑️ Deleting benchmark results...")
    kb = delete_file("results/benchmark_report.json")
    if kb > 0:
        total_kb += kb
        deleted_count += 1
    
    # 5. Clean empty directories
    print("\n📁 Removing empty directories...")
    for d in EMPTY_DIRS_TO_CHECK:
        delete_dir_if_empty(d)
    
    # 6. Delete cleanup script itself (no longer needed)
    print("\n🧹 Cleaning cleanup scripts...")
    delete_file("scripts/cleanup_project.py")
    
    # Summary
    print("\n" + "=" * 60)
    print("DEEP CLEANUP COMPLETE")
    print("=" * 60)
    print(f"Files deleted: {deleted_count}")
    print(f"Space saved: {total_kb/1024:.2f} MB")
    
    # Final structure
    print("\n📁 FINAL STRUCTURE:")
    print("""
PhiHorizon/
├── models/
│   └── core/
│       ├── meta_decision_splus_best.pt
│       ├── meta_decision_splus_final.pt
│       ├── quant_transformer_splus_best.pt
│       └── quant_transformer_splus_final.pt
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 04_core1_training.ipynb
│   ├── 05_core2_training.ipynb
│   ├── 06_core3_training.ipynb
│   ├── 07_core3_splus_training.ipynb
│   ├── 08_core1_splus_training.ipynb
│   ├── 09_core1_breakthrough.ipynb
│   └── 10_core2_breakthrough.ipynb
├── scripts/
│   ├── breakthrough_optimization.py
│   ├── paper_trading_bot.py
│   ├── prepare_hybrid_data.py
│   └── prepare_training_data.py
├── src/
├── tests/
└── docs/
    """)

if __name__ == "__main__":
    main()
