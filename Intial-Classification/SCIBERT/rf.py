import os, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# --------------------
# Config
# --------------------
DATA_PATH = "Filtered_models_min10rows_collapsed.csv"
EMB_PATH = "scibert_embeddings.npy"
OUT_DIR = Path("outputs_scibert_baselines")
rf_dir = OUT_DIR / "RandomForest"
rf_dir.mkdir(parents=True, exist_ok=True)

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
# Load CSV and embeddings
# --------------------
print("==== Loading inputs ====")
df = pd.read_csv(DATA_PATH)
X = np.load(EMB_PATH)
assert len(df) == X.shape[0], "Embeddings and CSV must align 1:1"
labels = df["label"].astype(str).tolist()

# --------------------
# Reuse label encoder from LR run
# --------------------
with open(OUT_DIR / "label_encoder_scibert.sav", "rb") as f:
    le = pickle.load(f)
classes = list(le.classes_)
y = le.transform(labels)

# --------------------
# Reuse same split as LR
# --------------------
split_df = pd.read_csv(OUT_DIR / "split_indices.csv")
idx_train = split_df["train_index"].dropna().astype(int).to_numpy()
idx_test  = split_df["test_index"].dropna().astype(int).to_numpy()

X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

# --------------------
# Train Random Forest
# --------------------
print("==== Training Random Forest ====")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=100,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42,
    verbose=1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# --------------------
# Save outputs
# --------------------
with open(rf_dir / "model.sav", "wb") as f:
    pickle.dump(rf, f)

np.save(rf_dir / "y_test.npy", y_test)
np.save(rf_dir / "y_pred.npy", y_pred)

metrics = quick_scores(y_test, y_pred)
(rf_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

(rf_dir / "classification_report.txt").write_text(
    classification_report(y_test, y_pred, target_names=classes, digits=4),
    encoding="utf-8"
)

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=classes, columns=classes).to_csv(rf_dir / "confusion_matrix.csv")

pd.DataFrame({
    "csv_row_index": idx_test,
    "true_label": [classes[i] for i in y_test],
    "pred_label": [classes[i] for i in y_pred],
}).to_csv(rf_dir / "predictions.csv", index=False)

print("==== Random Forest metrics ====")
print(metrics)
print("Done. Outputs in:", rf_dir.resolve())
