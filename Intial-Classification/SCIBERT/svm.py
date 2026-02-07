"""
SciBERT embeddings + Linear SVM (single-label multiclass), saving outputs.

Inputs:
- Filtered_models_min10rows.csv (columns: input_text, label)
- scibert_embeddings.npy (shape: [N, D], aligned row-by-row with CSV)

Outputs:
- outputs_scibert_svm/
    label_encoder.sav
    classes.csv
    summary_metrics.csv
    LinearSVM/
        model.sav
        metrics.json
        classification_report.txt
        confusion_matrix.csv
        predictions.csv
        y_test.npy
        y_pred.npy
"""

import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

# --------------------
# Config
# --------------------
DATA_PATH = "Filtered_models_min10rows.csv"
EMB_PATH  = "scibert_embeddings.npy"
OUT_DIR   = Path("outputs_scibert_svm")
RANDOM_STATE = 42
TEST_SIZE = 0.2

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def save_text(path: Path, text: str): path.write_text(text, encoding="utf-8")
def save_json(path: Path, obj): path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def quick_scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

print("==== Loading CSV ====")
df = pd.read_csv(DATA_PATH)
if not {"input_text", "label"}.issubset(df.columns):
    raise ValueError("CSV must contain columns: input_text, label")

labels = df["label"].astype(str).tolist()
N_csv = len(labels)
print(f"Rows in CSV: {N_csv:,}")

print("==== Loading SciBERT embeddings ====")
X = np.load(EMB_PATH)  # expected shape [N, D]
if X.ndim != 2:
    raise ValueError(f"Embeddings must be 2D (N,D). Got shape {X.shape}.")
N_emb, D = X.shape
print(f"Embeddings shape: {X.shape}, dtype={X.dtype}")

if N_emb != N_csv:
    raise ValueError(
        f"Row mismatch: CSV has {N_csv} rows but embeddings have {N_emb}. "
        "Make sure scibert_embeddings.npy aligns 1:1 with the CSV rows."
    )

# Optional: scale dense embeddings (can help linear models a bit)
print("==== Scaling embeddings (StandardScaler) ====")
scaler = StandardScaler(with_mean=True, with_std=True)
X = scaler.fit_transform(X)

# --------------------
# Labels + dirs
# --------------------
le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)

ensure_dir(OUT_DIR)
svm_dir = OUT_DIR / "LinearSVM"
ensure_dir(svm_dir)

with open(OUT_DIR / "label_encoder.sav", "wb") as f:
    pickle.dump(le, f)
pd.Series(classes, name="class").to_csv(OUT_DIR / "classes.csv", index=False)
print(f"Classes: {len(classes):,}")

# --------------------
# Split
# --------------------
print("==== Train/test split (stratified) ====")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | Dim: {X.shape[1]}")

# --------------------
# Train SVM
# --------------------
print("==== Training Linear SVM ====")
t0 = time.time()
svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    tol=1e-3,                # relaxed tolerance for speed
    dual=False,              # faster when n_samples > n_features (often true here)
    random_state=RANDOM_STATE,
    verbose=1,
)
svm.fit(X_train, y_train)
train_time = time.time() - t0
print(f"SVM fit time: {train_time:.1f}s")

# --------------------
# Evaluate + save
# --------------------
print("==== Evaluating ====")
y_pred = svm.predict(X_test)
report = classification_report(y_test, y_pred, target_names=classes, digits=4)
cm = confusion_matrix(y_test, y_pred)
scores = quick_scores(y_test, y_pred)

# Save core artifacts
with open(svm_dir / "model.sav", "wb") as f:
    pickle.dump(svm, f)

np.save(svm_dir / "y_test.npy", y_test)
np.save(svm_dir / "y_pred.npy", y_pred)

# Human-readable files
save_text(svm_dir / "classification_report.txt", report)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(
    svm_dir / "confusion_matrix.csv", index=True
)

# Prediction CSV with string labels
pred_df = pd.DataFrame({
    "true_label_id": y_test,
    "pred_label_id": y_pred,
    "true_label": [classes[i] for i in y_test],
    "pred_label": [classes[i] for i in y_pred],
})
pred_df.to_csv(svm_dir / "predictions.csv", index=False)

# Metrics JSON + summary CSV
scores_out = dict(scores)
scores_out["train_time_sec"] = round(train_time, 2)
save_json(svm_dir / "metrics.json", scores_out)
pd.DataFrame([{"model": "LinearSVM_SciBERT", **scores_out}]).to_csv(
    OUT_DIR / "summary_metrics.csv", index=False
)

print("\n==== Summary ====")
print(f"Accuracy:    {scores['accuracy']:.4f}")
print(f"Macro-F1:    {scores['macro_f1']:.4f}")
print(f"Weighted-F1: {scores['weighted_f1']:.4f}")
print(f"\nAll done. Outputs in: {OUT_DIR.resolve()}\n")
