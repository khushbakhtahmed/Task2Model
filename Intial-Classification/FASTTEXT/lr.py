#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supervised FastText embeddings -> Logistic Regression (single-label)

Pipeline:
1) Load CSV (input_text, label)
2) Stratified train/test split
3) Create FastText supervised training file from TRAIN ONLY:
     __label__<label> <cleaned text>
4) Train FastText supervised model on TRAIN
5) Build sentence embeddings for TRAIN and TEST via model.get_sentence_vector(text)
6) Scale embeddings and train Logistic Regression on TRAIN, evaluate on TEST
7) Save artifacts (model, scaler, metrics, reports, predictions)

Outputs: outputs_fasttext_supervised_embeddings/LogisticRegression/...
"""

import os, json, pickle, re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import fasttext

# --------------------
# Config
# --------------------
DATA_PATH = "Filtered_models_min10rows_collapsed.csv"   # must have columns: input_text, label
OUT_DIR   = Path("outputs_fasttext_supervised_embeddings")
LR_DIR    = OUT_DIR / "LogisticRegression"
FT_DIR    = OUT_DIR / "fasttext_model"
FT_TXT    = FT_DIR / "train_supervised.txt"   # supervised training file (TRAIN ONLY)
FT_MODEL  = FT_DIR / "fasttext_supervised.bin"
EMB_TRAIN = OUT_DIR / "train_embeddings.npy"
EMB_TEST  = OUT_DIR / "test_embeddings.npy"

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# FastText supervised knobs (tune for speed/quality)
FT_DIM     = 300
FT_EPOCH   = 25
FT_WNG     = 2        # wordNgrams
FT_LR      = 0.1
FT_THREADS = min(8, os.cpu_count() or 4)

OUT_DIR.mkdir(parents=True, exist_ok=True)
LR_DIR.mkdir(parents=True, exist_ok=True)
FT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------
# Helpers
# --------------------
def quick_scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

def clean_text(s: str) -> str:
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", s).strip()

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
#   IMPORTANT: use TRAINED model (supervised) to get vectors for both train & test
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
    X_train = np.load(EMB_TRAIN)
    X_test  = np.load(EMB_TEST)

print(f"Train embeddings: {X_train.shape} | Test embeddings: {X_test.shape}")

# --------------------
# Scale + train Logistic Regression on supervised embeddings
# --------------------
print("==== Scaling embeddings ====")
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
with open(LR_DIR / "scaler_fasttext_supervised.sav", "wb") as f:
    pickle.dump(scaler, f)

print("==== Training Logistic Regression (supervised embeddings) ====")
lr = LogisticRegression(
    solver="saga",
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    max_iter=200,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    multi_class="multinomial",
    verbose=1
)
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_test_s)

# --------------------
# Save artifacts & metrics
# --------------------
with open(LR_DIR / "model_fasttext_supervised_embeddings.sav", "wb") as f:
    pickle.dump(lr, f)
np.save(LR_DIR / "y_test.npy", y_test)
np.save(LR_DIR / "y_pred.npy", y_pred)

def quick_scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

metrics = quick_scores(y_test, y_pred)
(LR_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

(LR_DIR / "classification_report.txt").write_text(
    classification_report(y_test, y_pred, target_names=classes, digits=4),
    encoding="utf-8"
)

cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
pd.DataFrame(cm, index=classes, columns=classes).to_csv(LR_DIR / "confusion_matrix.csv")

pd.DataFrame({
    "csv_row_index": idx_test,
    "true_label": [classes[i] for i in y_test],
    "pred_label": [classes[i] for i in y_pred],
}).to_csv(LR_DIR / "predictions.csv", index=False)

print("==== LR on supervised FastText embeddings — metrics ====")
print(metrics)
print("Done. Outputs in:", LR_DIR.resolve())
