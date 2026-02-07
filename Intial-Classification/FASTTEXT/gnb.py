#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supervised FastText embeddings -> Gaussian Naive Bayes (single-label classification)

Pipeline:
1) Load train/test FastText embeddings + labels (reusing previous outputs)
2) Train GaussianNB (wrapped in One-vs-Rest for multi-class)
3) Evaluate on test set
4) Save artifacts: model, metrics, classification report, confusion matrix, predictions
"""

import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier

# --------------------
# Config
# --------------------
BASE_DIR = Path("outputs_fasttext_supervised_embeddings")
EMB_TRAIN = BASE_DIR / "train_embeddings.npy"
EMB_TEST  = BASE_DIR / "test_embeddings.npy"
LE_PATH   = BASE_DIR / "label_encoder.sav"
Y_TRAIN   = BASE_DIR / "y_train.npy"
Y_TEST    = BASE_DIR / "y_test.npy"
OUT_DIR   = BASE_DIR / "GaussianNB"
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
# Load data
# --------------------
print("==== Loading embeddings & labels ====")
X_train = np.load(EMB_TRAIN, mmap_mode="r")
X_test  = np.load(EMB_TEST,  mmap_mode="r")

# Load the same label encoder used before
with open(LE_PATH, "rb") as f:
    le: LabelEncoder = pickle.load(f)
classes = list(le.classes_)

# Recreate y_train and y_test from CSV + split indices
DATA_PATH = "Filtered_models_min10rows_collapsed.csv"
df_labels = pd.read_csv(DATA_PATH, usecols=["label"])
all_y = le.transform(df_labels["label"].astype(str).values)

split_df = pd.read_csv(BASE_DIR / "split_indices.csv")
idx_train = split_df["train_index"].dropna().astype(int).to_numpy()
idx_test  = split_df["test_index"].dropna().astype(int).to_numpy()

y_train = all_y[idx_train]
y_test  = all_y[idx_test]

print(f"Train embeddings: {X_train.shape}, Test embeddings: {X_test.shape}, Classes: {len(classes)}")


# --------------------
# Train Gaussian Naive Bayes (OVR)
# --------------------
print("==== Training Gaussian Naive Bayes (One-vs-Rest) ====")
gnb = OneVsRestClassifier(GaussianNB(var_smoothing=1e-9))
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

# --------------------
# Save artifacts
# --------------------
with open(OUT_DIR / "model_fasttext_supervised_embeddings.sav", "wb") as f:
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
    "true_label": [classes[i] for i in y_test],
    "pred_label": [classes[i] for i in y_pred],
}).to_csv(OUT_DIR / "predictions.csv", index=False)

print("==== GaussianNB metrics ====")
print(metrics)
print("Done. Outputs in:", OUT_DIR.resolve())
