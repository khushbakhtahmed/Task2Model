#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collapse labels in Filtered_models_min10rows.csv:
- Keep only the part after the first "/"
- Save new CSV with collapsed labels
- Print analysis stats
"""

import pandas as pd
from pathlib import Path

# --------------------
# Paths
# --------------------
THIS_DIR = Path(__file__).resolve().parent
DATA_IN = THIS_DIR / "Filtered_models_min10rows.csv"
DATA_OUT = THIS_DIR / "Filtered_models_min10rows_collapsed.csv"

# --------------------
# Load data
# --------------------
df = pd.read_csv(DATA_IN)

if "label" not in df.columns:
    raise ValueError("Expected 'label' column in CSV")

# --------------------
# Collapse labels
# --------------------
df["collapsed_label"] = df["label"].astype(str).apply(lambda x: x.split("/", 1)[-1])

# Save new CSV
df.to_csv(DATA_OUT, index=False)
print(f"[INFO] Saved collapsed dataset: {DATA_OUT}")

# --------------------
# Analysis
# --------------------
n_rows = len(df)
n_labels_before = df["label"].nunique()
n_labels_after = df["collapsed_label"].nunique()

print("\n=== Collapse Analysis ===")
print(f"Total rows:              {n_rows:,}")
print(f"Unique labels before:    {n_labels_before:,}")
print(f"Unique labels after:     {n_labels_after:,}")
print(f"Merged away:             {n_labels_before - n_labels_after:,}")

# Show examples of mappings
merged_counts = (
    df.groupby(["collapsed_label", "label"])
      .size()
      .reset_index(name="count")
      .sort_values(["collapsed_label", "count"], ascending=[True, False])
)

print("\nExamples of collapsed mappings:")
print(merged_counts.head(20).to_string(index=False))
