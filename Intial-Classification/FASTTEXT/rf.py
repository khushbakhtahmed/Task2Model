#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supervised FastText embeddings -> Random Forest (single-label)

Pipeline:
1) Load CSV (input_text, label), stratified train/test split
2) Create FastText supervised TRAIN file: "__label__<y> <text>"
3) Train FastText supervised model on TRAIN only
4) Build sentence embeddings for TRAIN & TEST via model.get_sentence_vector()
5) Train Random Forest on TRAIN embeddings, evaluate on TEST
6) Save artifacts: model, metrics, classification report, confusion matrix, predictions
"""

import os, json, pickle, re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import fasttext

# --------------------
# Config
# --------------------
DATA_PATH = "Filtered_models_min10rows_collapsed.csv"   # columns: input_text, label
OUT_DIR   = Path("outputs_fasttext_supervised_embeddings")
RF_DIR    = OUT_DIR / "RandomForest"
FT_DIR    = OUT_DIR / "fasttext_model"
FT_TXT    = FT_DIR / "train_supervised.txt"   # TRAIN ONLY
FT_MODEL  = FT_DIR / "fasttext_supervised.bin"
EMB_TRAIN = OUT_DIR / "train_embeddings.npy"
EMB_TEST  = OUT_DIR / "test_embeddings.npy"

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# FastText supervised knobs
FT_DIM     = 300
FT_EPOCH   = 25
FT_WNG     = 2        # wordNgrams
FT_LR      = 0.1
FT_THREADS = min(8, os.cpu_count() or 4)

# RF knobs tuned to avoid OOM
RF_N_ESTIMATORS   = 150
RF_MAX_DEPTH      = 40
RF_MIN_SPLIT      = 10
RF_MIN_LEAF       = 2
RF_MAX_FEATURES   = "sqrt"
RF_MAX_SAMPLES    = 0.7   # subsample rows per tree -> big RAM saver
RF_N_JOBS         = 4     # keep below cpus-per-task to limit memory

OUT_DIR.mkdir(parents=True, exist_ok=True)
RF_DIR.mkdir(parents=True, exist_ok=True)
FT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# Helpers
# --------------------
def clean_text(s: str) -> str:
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", s).strip()

def quick_scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

# --------------------
# Load data
# --------------------
print("==== Loading CSV ====")
df = pd.read_csv(DATA_PATH)
if not {"input_text", "label"}.issubset(df.columns):
    raise ValueError("CSV must contain columns: input_text, label")

texts  = df["input_text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()
n_rows = len(df)
print(f"Rows: {n_rows:,}")

# --------------------
# Encode labels + stratified split
# --------------------
le = LabelEncoder()
y_all = le.fit_transform(labels)
classes = list(le.classes_)
with open(OUT_DIR / "label_encoder.sav", "wb") as f:
    pickle.dump(le, f)

idx_train, idx_test = train_test_split(
    np.arange(n_rows),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_all
)
pd.DataFrame({
    "train_index": pd.Series(idx_train, dtype=int),
    "test_index":  pd.Series(idx_test,  dtype=int),
}).to_csv(OUT_DIR / "split_indices.csv", index=False)

train_texts = [texts[i] for i in idx_train]
train_labels = [labels[i] for i in idx_train]
test_texts  = [texts[i] for i in idx_test]
y_train = y_all[idx_train]
y_test  = y_all[idx_test]

print(f"Train: {len(train_texts):,} | Test: {len(test_texts):,}")
print(f"Unique labels — train: {len(set(train_labels))}, test: {len(set(labels[i] for i in idx_test))}")

# --------------------
# Write supervised FastText TRAIN file
# --------------------
if not FT_TXT.exists():
    print("==== Writing FastText supervised TRAIN file ====")
    with open(FT_TXT, "w", encoding="utf-8") as f:
        for t, l in zip(train_texts, train_labels):
            f.write(f"__label__{l} {clean_text(t)}\n")
else:
    print("==== Using existing TRAIN file:", FT_TXT)

# --------------------
# Train / load FastText SUPERVISED model (TRAIN ONLY)
# --------------------
if FT_MODEL.exists():
    print("==== Loading cached FastText supervised model ====")
    ft = fasttext.load_model(str(FT_MODEL))
else:
    print("==== Training FastText supervised model on TRAIN ====")
    ft = fasttext.train_supervised(
        input=str(FT_TXT),
        lr=FT_LR,
        epoch=FT_EPOCH,
        wordNgrams=FT_WNG,
        dim=FT_DIM,
        loss="softmax",
        thread=FT_THREADS
    )
    ft.save_model(str(FT_MODEL))

# --------------------
# Build sentence embeddings from the supervised model
# --------------------
def sent_vec(text: str) -> np.ndarray:
    return ft.get_sentence_vector(clean_text(text))

if not (EMB_TRAIN.exists() and EMB_TEST.exists()):
    print("==== Building sentence embeddings (supervised) ====")
    X_train = np.vstack([sent_vec(t) for t in train_texts]).astype("float32")
    X_test  = np.vstack([sent_vec(t) for t in test_texts]).astype("float32")
    np.save(EMB_TRAIN, X_train)
    np.save(EMB_TEST, X_test)
else:
    print("==== Loading cached supervised embeddings ====")
    # use mmap to reduce RAM when training RF
    X_train = np.load(EMB_TRAIN, mmap_mode="r")
    X_test  = np.load(EMB_TEST,  mmap_mode="r")

# if we just created them above, also switch to mmap to be safe
if isinstance(X_train, np.ndarray) and X_train.flags['WRITEABLE']:
    # re-open with mmap to avoid full copies in workers
    X_train = np.load(EMB_TRAIN, mmap_mode="r")
    X_test  = np.load(EMB_TEST,  mmap_mode="r")

print(f"Train embeddings: {X_train.shape} | Test embeddings: {X_test.shape}")

# --------------------
# Train Random Forest on supervised embeddings (OOM-safe settings)
# --------------------
print("==== Training Random Forest (memory-safe config) ====")
rf = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    min_samples_split=RF_MIN_SPLIT,
    min_samples_leaf=RF_MIN_LEAF,
    max_features=RF_MAX_FEATURES,
    max_samples=RF_MAX_SAMPLES,            # subsample rows per tree
    bootstrap=True,
    class_weight="balanced_subsample",
    n_jobs=RF_N_JOBS,                       # limit parallelism
    random_state=RANDOM_STATE,
    verbose=1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# --------------------
# Save artifacts & metrics
# --------------------
with open(RF_DIR / "model_fasttext_supervised_embeddings.sav", "wb") as f:
    pickle.dump(rf, f)
np.save(RF_DIR / "y_test.npy", y_test)
np.save(RF_DIR / "y_pred.npy", y_pred)

metrics = quick_scores(y_test, y_pred)
(RF_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

(RF_DIR / "classification_report.txt").write_text(
    classification_report(y_test, y_pred, target_names=classes, digits=4),
    encoding="utf-8"
)

cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
pd.DataFrame(cm, index=classes, columns=classes).to_csv(RF_DIR / "confusion_matrix.csv")

pd.DataFrame({
    "csv_row_index": idx_test,
    "true_label": [classes[i] for i in y_test],
    "pred_label": [classes[i] for i in y_pred],
}).to_csv(RF_DIR / "predictions.csv", index=False)

print("==== RF on supervised FastText embeddings — metrics ====")
print(metrics)
print("Done. Outputs in:", RF_DIR.resolve())
