#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TF-IDF  ➜  TruncatedSVD (LSA, dense)  ➜  StandardScaler  ➜  GaussianNB (OvR)

Inputs:
  - CSV: Filtered_models_min10rows.csv  (columns: input_text, label)
  - Optional: outputs_scibert_baselines/{label_encoder_scibert.sav, split_indices.csv}

Outputs:
  - outputs_tfidf_gnb/
      - model_gnb_tfidf_lsa.sav
      - metrics.json
      - classification_report.txt
      - confusion_matrix.csv
      - predictions.csv
      - vectorizer_tfidf.sav
      - lsa_svd.sav
      - scaler_lsa.sav
      - (if created locally) label_encoder_tfidf.sav, split_indices_tfidf.csv
"""

import os, json, pickle, re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# --------------------
# Config
# --------------------
DATA_PATH   = "Filtered_models_min10rows_collapsed.csv"   # must contain: input_text, label
OUT_DIR     = Path("outputs_tfidf_gnb")

# If present, we reuse SciBERT split + encoder for fair comparison
SCIBERT_OUT   = Path("outputs_scibert_baselines")
SCIBERT_LE    = SCIBERT_OUT / "label_encoder_scibert.sav"
SCIBERT_SPLIT = SCIBERT_OUT / "split_indices.csv"

# TF-IDF + SVD knobs (tune for speed/quality vs memory)
TF_MAX_FEATS   = 50000   # vocab cap; lower if RAM is tight (e.g., 20k)
TF_NGRAMS      = (1, 2)  # word 1-2 grams
SVD_DIM        = 300     # dense dimension for GNB
SVD_ITER       = 7
SVD_RANDOM     = 42

RANDOM_STATE   = 42
TEST_SIZE      = 0.20

OUT_DIR.mkdir(parents=True, exist_ok=True)

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

texts  = [clean_text(t) for t in df["input_text"].astype(str).tolist()]
labels = df["label"].astype(str).tolist()
n_rows = len(df)
print(f"Rows: {n_rows:,}")

# --------------------
# Encoder + split (reuse SciBERT if available)
# --------------------
reuse_sci = SCIBERT_LE.exists() and SCIBERT_SPLIT.exists()
if reuse_sci:
    print("==== Reusing SciBERT label encoder and split ====")
    with open(SCIBERT_LE, "rb") as f:
        le: LabelEncoder = pickle.load(f)
    classes = list(le.classes_)
    y_all = le.transform(labels)

    split_df = pd.read_csv(SCIBERT_SPLIT)
    idx_train = split_df["train_index"].dropna().astype(int).to_numpy()
    idx_test  = split_df["test_index"].dropna().astype(int).to_numpy()
else:
    print("==== Creating local encoder + split (no SciBERT artifacts found) ====")
    le = LabelEncoder()
    y_all = le.fit_transform(labels)
    classes = list(le.classes_)
    with open(OUT_DIR / "label_encoder_tfidf.sav", "wb") as f:
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
    }).to_csv(OUT_DIR / "split_indices_tfidf.csv", index=False)

X_text_train = [texts[i] for i in idx_train]
X_text_test  = [texts[i] for i in idx_test]
y_train      = y_all[idx_train]
y_test       = y_all[idx_test]

print(f"Train: {len(X_text_train):,} | Test: {len(X_text_test):,} | Classes: {len(classes)}")

# --------------------
# TF-IDF (sparse) -> SVD (dense) -> Scale (mean=0,std=1)
# --------------------
print("==== Vectorizing with TF-IDF ====")
tfidf = TfidfVectorizer(
    max_features=TF_MAX_FEATS,
    ngram_range=TF_NGRAMS,
    lowercase=True,
    strip_accents="unicode",
    analyzer="word",
    dtype=np.float32
)
X_tf_train = tfidf.fit_transform(X_text_train)
X_tf_test  = tfidf.transform(X_text_test)
with open(OUT_DIR / "vectorizer_tfidf.sav", "wb") as f:
    pickle.dump(tfidf, f)
print(f"TF-IDF shapes: train={X_tf_train.shape}, test={X_tf_test.shape}")

print("==== Dimensionality reduction with TruncatedSVD (LSA) ====")
svd = TruncatedSVD(n_components=SVD_DIM, n_iter=SVD_ITER, random_state=SVD_RANDOM)
X_lsa_train = svd.fit_transform(X_tf_train)
X_lsa_test  = svd.transform(X_tf_test)
with open(OUT_DIR / "lsa_svd.sav", "wb") as f:
    pickle.dump(svd, f)
print(f"LSA shapes: train={X_lsa_train.shape}, test={X_lsa_test.shape}")

print("==== Standardizing LSA features ====")
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_dense = scaler.fit_transform(X_lsa_train)
X_test_dense  = scaler.transform(X_lsa_test)
with open(OUT_DIR / "scaler_lsa.sav", "wb") as f:
    pickle.dump(scaler, f)

# --------------------
# GNB (OvR) on dense features
# --------------------
print("==== Training Gaussian Naive Bayes (OvR) ====")
gnb = OneVsRestClassifier(GaussianNB(var_smoothing=1e-9))
gnb.fit(X_train_dense, y_train)
y_pred = gnb.predict(X_test_dense)

# --------------------
# Save artifacts & metrics
# --------------------
with open(OUT_DIR / "model_gnb_tfidf_lsa.sav", "wb") as f:
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

print("==== GaussianNB on TF-IDF/LSA — metrics ====")
print(metrics)
print("Done. Outputs in:", OUT_DIR.resolve())
