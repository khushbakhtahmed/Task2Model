"""
TF-IDF + Random Forest (single-label multiclass) with structured outputs.

Inputs:
- Filtered_models_min10rows.csv  (columns: input_text, label)

Outputs:
- outputs_tfidf_rf/
    tfidf_vectorizer.sav
    svd_transformer.sav
    label_encoder.sav
    classes.csv
    summary_metrics.csv
    RandomForest/
        model.sav
        metrics.json
        classification_report.txt
        confusion_matrix.csv
        predictions.csv
        y_test.npy
        y_pred.npy
"""

import re
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# --------------------
# Config
# --------------------
DATA_PATH = "Filtered_models_min10rows_collapsed.csv"
OUT_DIR = Path("outputs_tfidf_rf")
RANDOM_STATE = 42
TEST_SIZE = 0.2
SVD_COMPONENTS = 512   # compact dense representation for RF (fast & memory-safe)

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

texts = df["input_text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()
print(f"Rows: {len(texts):,}")

# --------------------
# Preprocess (same as your TF-IDF LR)
# --------------------
sw = set(stopwords.words("english"))
def prep(t):
    t = t.lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    toks = [w for w in t.split() if w not in sw and len(w) > 2]
    return " ".join(toks)

print("==== Preprocessing text ====")
texts_clean = [prep(t) for t in texts]

# --------------------
# TF-IDF (sparse, float32)
# --------------------
print("==== Building TF-IDF ====")
tfidf = TfidfVectorizer(
    ngram_range=(1, 1),
    max_df=0.95,
    min_df=3,
    max_features=40000,
    dtype=np.float32,
)
X_sparse = tfidf.fit_transform(texts_clean)
print(f"TF-IDF shape: {X_sparse.shape}, dtype={X_sparse.dtype}")

# --------------------
# Dimensionality reduction for RF (dense)
# --------------------
print(f"==== TruncatedSVD to {SVD_COMPONENTS} components ====")
svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
X = svd.fit_transform(X_sparse).astype(np.float32, copy=False)
print(f"SVD output shape: {X.shape}, dtype={X.dtype}")

# --------------------
# Labels + dirs
# --------------------
le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)
ensure_dir(OUT_DIR)
rf_dir = OUT_DIR / "RandomForest"
ensure_dir(rf_dir)

with open(OUT_DIR / "tfidf_vectorizer.sav", "wb") as f: pickle.dump(tfidf, f)
with open(OUT_DIR / "svd_transformer.sav", "wb") as f: pickle.dump(svd, f)
with open(OUT_DIR / "label_encoder.sav", "wb") as f: pickle.dump(le, f)
pd.Series(classes, name="class").to_csv(OUT_DIR / "classes.csv", index=False)
print(f"Classes: {len(classes):,}")

# --------------------
# Split
# --------------------
print("==== Train/test split (stratified) ====")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# --------------------
# Random Forest (fast, fixed params like reference spirit)
# --------------------
print("==== Training Random Forest ====")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)
rf.fit(X_train, y_train)
train_time = time.time() - t0
print(f"RF fit time: {train_time:.1f}s")

# --------------------
# Evaluate + save
# --------------------
print("==== Evaluating ====")
y_pred = rf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=classes, digits=4)
cm = confusion_matrix(y_test, y_pred)
scores = quick_scores(y_test, y_pred)

with open(rf_dir / "model.sav", "wb") as f: pickle.dump(rf, f)
np.save(rf_dir / "y_test.npy", y_test)
np.save(rf_dir / "y_pred.npy", y_pred)

save_text(rf_dir / "classification_report.txt", report)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(rf_dir / "confusion_matrix.csv", index=True)

scores_out = dict(scores)
scores_out["train_time_sec"] = round(train_time, 2)
save_json(rf_dir / "metrics.json", scores_out)
pd.DataFrame([{"model": "RandomForest_TFIDF", **scores_out}]).to_csv(
    OUT_DIR / "summary_metrics.csv", index=False
)

print("\n==== Summary ====")
print(f"Accuracy:   {scores['accuracy']:.4f}")
print(f"Macro-F1:   {scores['macro_f1']:.4f}")
print(f"Weighted-F1:{scores['weighted_f1']:.4f}")
print(f"\nAll done. Outputs in: {OUT_DIR.resolve()}\n")
