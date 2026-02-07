#I USED THIS TILL LR AND THEN TERMINATED IT

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# --------------------
# Config
# --------------------
DATA_PATH = "Filtered_models_min10rows_collapsed.csv"
EMB_PATH = "scibert_embeddings.npy"
OUT_DIR = Path("outputs_scibert_baselines")
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --------------------
# Helpers
# --------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")

def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def save_cm_csv(path: Path, cm, classes):
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(path, index=True)

def quick_scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

# --------------------
# Load inputs
# --------------------
print("==== Loading CSV ====")
df = pd.read_csv(DATA_PATH)
if not {"input_text", "label"}.issubset(df.columns):
    raise ValueError("CSV must contain columns: input_text, label")

labels = df["label"].astype(str).tolist()
n_rows = len(df)
print(f"Rows in CSV: {n_rows:,}")

print("==== Loading cached SciBERT embeddings ====")
X = np.load(EMB_PATH)
if X.shape[0] != n_rows:
    raise ValueError(
        f"Row mismatch: embeddings {X.shape[0]} vs CSV {n_rows}. "
        "They must align 1:1 in the same order."
    )
print(f"Embeddings shape: {X.shape}")

# --------------------
# Encode labels
# --------------------
print("==== Encoding labels (single-label) ====")
le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)
print(f"Classes: {len(classes):,}")

# --------------------
# Prepare output dirs
# --------------------
ensure_dir(OUT_DIR)
for m in ["LogisticRegression", "LinearSVM", "RandomForest"]:
    (OUT_DIR / m).mkdir(parents=True, exist_ok=True)

pd.Series(classes, name="class").to_csv(OUT_DIR / "classes.csv", index=False)
with open(OUT_DIR / "label_encoder_scibert.sav", "wb") as f:
    pickle.dump(le, f)

# --------------------
# Train/test split
# --------------------
print("==== Train/test split (stratified) ====")
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)),
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
pd.DataFrame({
    "train_index": pd.Series(idx_train, dtype=int),
    "test_index": pd.Series(idx_test, dtype=int)
}).to_csv(OUT_DIR / "split_indices.csv", index=False)
print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# --------------------
# Scaling for linear models
# --------------------
print("==== Fitting StandardScaler for linear models ====")
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_lin = scaler.fit_transform(X_train)
X_test_lin = scaler.transform(X_test)
with open(OUT_DIR / "scaler_scibert.sav", "wb") as f:
    pickle.dump(scaler, f)

# --------------------
# 1) Logistic Regression
# --------------------
print("==== Training Logistic Regression ====")
lr = LogisticRegression(
    solver="saga",
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    max_iter=200,            # per reference
    n_jobs=-1,
    random_state=RANDOM_STATE,
    multi_class="multinomial",
    verbose=1,               # <-- progress info
)
lr.fit(X_train_lin, y_train)
y_pred_lr = lr.predict(X_test_lin)

# save outputs
lr_dir = OUT_DIR / "LogisticRegression"
with open(lr_dir / "model.sav", "wb") as f: pickle.dump(lr, f)
np.save(lr_dir / "y_test.npy", y_test); np.save(lr_dir / "y_pred.npy", y_pred_lr)
save_json(lr_dir / "metrics.json", quick_scores(y_test, y_pred_lr))
save_text(lr_dir / "classification_report.txt",
          classification_report(y_test, y_pred_lr, target_names=classes, digits=4))
save_cm_csv(lr_dir / "confusion_matrix.csv", confusion_matrix(y_test, y_pred_lr), classes)
pd.DataFrame({
    "csv_row_index": idx_test,
    "true_label": [classes[i] for i in y_test],
    "pred_label": [classes[i] for i in y_pred_lr],
}).to_csv(lr_dir / "predictions.csv", index=False)

# --------------------
# 2) Linear SVM
# --------------------
print("==== Training Linear SVM ====")
svm = LinearSVC(
    C=1.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    verbose=1,   # <-- progress info
)
svm.fit(X_train_lin, y_train)
y_pred_svm = svm.predict(X_test_lin)

svm_dir = OUT_DIR / "LinearSVM"
with open(svm_dir / "model.sav", "wb") as f: pickle.dump(svm, f)
np.save(svm_dir / "y_test.npy", y_test); np.save(svm_dir / "y_pred.npy", y_pred_svm)
save_json(svm_dir / "metrics.json", quick_scores(y_test, y_pred_svm))
save_text(svm_dir / "classification_report.txt",
          classification_report(y_test, y_pred_svm, target_names=classes, digits=4))
save_cm_csv(svm_dir / "confusion_matrix.csv", confusion_matrix(y_test, y_pred_svm), classes)
pd.DataFrame({
    "csv_row_index": idx_test,
    "true_label": [classes[i] for i in y_test],
    "pred_label": [classes[i] for i in y_pred_svm],
}).to_csv(svm_dir / "predictions.csv", index=False)

# --------------------
# 3) Random Forest
# --------------------
print("==== Training Random Forest ====")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=100,
    n_jobs=-1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    verbose=1,   # <-- progress info
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_dir = OUT_DIR / "RandomForest"
with open(rf_dir / "model.sav", "wb") as f: pickle.dump(rf, f)
np.save(rf_dir / "y_test.npy", y_test); np.save(rf_dir / "y_pred.npy", y_pred_rf)
save_json(rf_dir / "metrics.json", quick_scores(y_test, y_pred_rf))
save_text(rf_dir / "classification_report.txt",
          classification_report(y_test, y_pred_rf, target_names=classes, digits=4))
save_cm_csv(rf_dir / "confusion_matrix.csv", confusion_matrix(y_test, y_pred_rf), classes)
pd.DataFrame({
    "csv_row_index": idx_test,
    "true_label": [classes[i] for i in y_test],
    "pred_label": [classes[i] for i in y_pred_rf],
}).to_csv(rf_dir / "predictions.csv", index=False)

# --------------------
# Top-level summary
# --------------------
print("==== Writing top-level summary ====")
summary = pd.DataFrame([
    {"model": "LogisticRegression", **quick_scores(y_test, y_pred_lr)},
    {"model": "LinearSVM",         **quick_scores(y_test, y_pred_svm)},
    {"model": "RandomForest",      **quick_scores(y_test, y_pred_rf)},
]).sort_values("macro_f1", ascending=False)
summary.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
print(summary.to_string(index=False))

print(f"\nAll done. Outputs in: {OUT_DIR.resolve()}\n")
