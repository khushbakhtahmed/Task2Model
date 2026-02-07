
import re
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# --------------------
# Config
# --------------------
DATA_PATH = "Filtered_models_min10rows_collapsed.csv"
OUT_DIR = Path("outputs_tfidf_lr")
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

texts = df["input_text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()
print(f"Rows: {len(texts):,}")

# --------------------
# Preprocessing (your original)
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
# TF-IDF
# --------------------
print("==== Building TF-IDF ====")
tfidf = TfidfVectorizer(
    ngram_range=(1, 1),   # unigrams only (faster)
    max_df=0.95,
    min_df=3,
    max_features=40000,
    dtype=np.float32,     # small speed/memory win; safe
)
X = tfidf.fit_transform(texts_clean)
print(f"TF-IDF shape: {X.shape}, dtype={X.dtype}")

# --------------------
# Labels + outputs root
# --------------------
le = LabelEncoder()
y = le.fit_transform(labels)
classes = list(le.classes_)
ensure_dir(OUT_DIR)
lr_dir = OUT_DIR / "LogisticRegression"
ensure_dir(lr_dir)

with open(OUT_DIR / "tfidf_vectorizer.sav", "wb") as f: pickle.dump(tfidf, f)
with open(OUT_DIR / "label_encoder.sav", "wb") as f: pickle.dump(le, f)
pd.Series(classes, name="class").to_csv(OUT_DIR / "classes.csv", index=False)
print(f"Classes: {len(classes):,}")

# --------------------
# Split (no indices saved to mimic reference)
# --------------------
print("==== Train/test split (stratified) ====")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

# (Optional) If you DO want indices later, uncomment:
# import numpy as np
# idx = np.arange(len(y))
# X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
#     X, y, idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
# )
# pd.DataFrame({"train_index": pd.Series(idx_train, dtype=int),
#               "test_index":  pd.Series(idx_test,  dtype=int)}).to_csv(OUT_DIR / "split_indices.csv", index=False)

# --------------------
# Logistic Regression (your settings)
# --------------------
print("==== Training Logistic Regression (single-label multiclass) ====")
t0 = time.time()
base_lr = LogisticRegression(
    solver="saga",            # good for sparse TF-IDF
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    max_iter=200,
    tol=1e-3,
    random_state=RANDOM_STATE,
    verbose=1,
)
best_lr = base_lr.fit(X_train, y_train)
print(f"LR fit time: {time.time() - t0:.1f}s")

# --------------------
# Evaluate + save
# --------------------
print("==== Evaluating ====")
y_pred = best_lr.predict(X_test)
report = classification_report(y_test, y_pred, target_names=classes, digits=4)
cm = confusion_matrix(y_test, y_pred)
scores = quick_scores(y_test, y_pred)

with open(lr_dir / "model.sav", "wb") as f: pickle.dump(best_lr, f)
np.save(lr_dir / "y_test.npy", y_test)
np.save(lr_dir / "y_pred.npy", y_pred)

save_text(lr_dir / "classification_report.txt", report)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(lr_dir / "confusion_matrix.csv", index=True)

scores_out = dict(scores)
save_json(lr_dir / "metrics.json", scores_out)
pd.DataFrame([{"model": "LogisticRegression_TFIDF", **scores_out}]).to_csv(
    OUT_DIR / "summary_metrics.csv", index=False
)

print("\n==== Summary ====")
print(f"Accuracy:   {scores['accuracy']:.4f}")
print(f"Macro-F1:   {scores['macro_f1']:.4f}")
print(f"Weighted-F1:{scores['weighted_f1']:.4f}")
print(f"\nAll done. Outputs in: {OUT_DIR.resolve()}\n")
