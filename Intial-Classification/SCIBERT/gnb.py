#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gaussian Naive Bayes (OvR) on cached SciBERT embeddings (single-label classification)

Assumes the following exist (from your previous SciBERT run):
  - Filtered_models_min10rows.csv (columns: input_text, label)
  - scibert_embeddings.npy (rows aligned 1:1 with CSV)
  - outputs_scibert_baselines/label_encoder_scibert.sav
  - outputs_scibert_baselines/split_indices.csv

Outputs (written to outputs_scibert_baselines/GaussianNB/):
  - model.sav
  - metrics.json
  - classification_report.txt
  - confusion_matrix.csv
  - predictions.csv
  - y_test.npy, y_pred.npy
"""

import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# --------------------
# Paths / config
# --------------------
DATA_PATH   = "Filtered_models_min10rows_collapsed.csv"           # sanity check for labels length
EMB_PATH    = "scibert_embeddings.npy"
BASE_DIR    = Path("outputs_scibert_baselines")
LE_PATH     = BASE_DIR / "label_encoder_scibert.sav"
SPLIT_CSV   = BASE_DIR / "split_indices.csv"
OUT_DIR     = BASE_DIR / "GaussianNB"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# Helpers
# --------------------
def quick_scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

# --------------------
# Load embeddings & artifacts
# --------------------
print("==== Loading SciBERT embeddings ====")
X = np.load(EMB_PATH, mmap_mode="r")
n_rows_emb = X.shape[0]

print("==== Loading label encoder & split indices ====")
with open(LE_PATH, "rb") as f:
    le: LabelEncoder = pickle.load(f)
classes = list(le.classes_)

split_df = pd.read_csv(SPLIT_CSV)
idx_train = split_df["train_index"].dropna().astype(int).to_numpy()
idx_test  = split_df["test_index"].dropna().astype(int).to_numpy()

# Optionally sanity-check rows match CSV length
try:
    n_rows_csv = len(pd.read_csv(DATA_PATH))
    assert n_rows_csv == n_rows_emb, (
        f"Row mismatch: CSV={n_rows_csv} vs embeddings={n_rows_emb}"
    )
except Exception as e:
    print("[WARN] Could not fully validate CSV vs embeddings:", e)

# --------------------
# Rebuild y from CSV + encoder (ensures same mapping)
# --------------------
df_lbl = pd.read_csv(DATA_PATH, usecols=["label"])
y_all = le.transform(df_lbl["label"].astype(str).values)

X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y_all[idx_train], y_all[idx_test]

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Dim: {X.shape[1]} | Classes: {len(classes)}")

# --------------------
# Train GNB (OvR)
# --------------------
print("==== Training Gaussian Naive Bayes (One-vs-Rest) ====")
gnb = OneVsRestClassifier(GaussianNB(var_smoothing=1e-9))
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# --------------------
# Save artifacts & metrics
# --------------------
with open(OUT_DIR / "model.sav", "wb") as f:
    pickle.dump(gnb, f)

np.save(OUT_DIR / "y_test.npy", y_test)
np.save(OUT_DIR / "y_pred.npy", y_pred)

metrics = quick_scores(y_test, y_pred)
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

(OUT_DIR / "classification_report.txt").write_text(
    classification_report(y_test, y_pred, target_names=classes, digits=4, zero_division=0),
    encoding="utf-8"
)

cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
pd.DataFrame(cm, index=classes, columns=classes).to_csv(OUT_DIR / "confusion_matrix.csv")

pd.DataFrame({
    "csv_row_index": idx_test,
    "true_label":   [classes[i] for i in y_test],
    "pred_label":   [classes[i] for i in y_pred],
}).to_csv(OUT_DIR / "predictions.csv", index=False)

print("==== GaussianNB on SciBERT embeddings — metrics ====")
print(metrics)
print("Done. Outputs in:", OUT_DIR.resolve())
