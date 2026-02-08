import re
import json
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords


# --------------------
# Paths & config
# --------------------
HERE = Path(__file__).resolve().parent
TRAIN_CSV = HERE / "Task2Model.csv"
TEST_CSV  = HERE / "manualevalreferencefromllm.csv"

OUT_DIR = HERE / "train_group"  # avoid overwrite
RANDOM_STATE = 42
K_VALUES = [1, 3, 5, 10]

TFIDF_CFG = dict(
    ngram_range=(1, 1),
    max_df=0.95,
    min_df=3,
    max_features=40000,
    dtype=np.float32,
)

SVM_CFG = dict(
    C=50,
    class_weight="balanced",
    tol=1e-4,
    dual=False,  # faster when n_samples > n_features
    random_state=RANDOM_STATE,
    verbose=1,
)


# --------------------
# IO helpers
# --------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------
# Text preprocessing
# --------------------
_SW = set(stopwords.words("english"))


def clean_text_for_tfidf(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    toks = [w for w in t.split() if w not in _SW and len(w) > 2]
    return " ".join(toks)


# --------------------
# Label parsing + matching
# --------------------
def clean_label(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"^\d+[\.\)]\s*", "", s)      # 1. or 1)
    s = re.sub(r"^[\-\*\•]\s*", "", s)      # -, *, •
    return s.strip()


def parse_gpt_cell(cell) -> list:
    if pd.isna(cell):
        return []
    s = str(cell).replace("\r", "\n").replace("\n", "|")
    parts = []
    for chunk in s.split("|"):
        for sub in chunk.split(","):
            lab = clean_label(sub)
            if lab:
                parts.append(lab)
    return parts


def build_label_matcher(train_classes: list):
    """
    GPT label -> training class index.
    Priority:
      1) exact match (normalized)
      2) suffix-after-slash exact match
      3) substring either way (BM25-style), choose shortest matching training label
    """
    norm_train = [clean_label(c) for c in train_classes]
    exact_map = {c: i for i, c in enumerate(norm_train)}

    suffix_map = {}
    for i, c in enumerate(norm_train):
        suf = c.split("/")[-1]
        suffix_map.setdefault(suf, i)

    norm_pairs = list(enumerate(norm_train))

    stats = {
        "gpt_labels_total": 0,
        "gpt_labels_mapped": 0,
        "gpt_labels_unmapped": 0,
        "queries_empty_after_mapping": 0,
    }

    def map_one(g: str):
        g = clean_label(g)
        if not g:
            return None
        if g in exact_map:
            return exact_map[g]
        if g in suffix_map:
            return suffix_map[g]

        hits = []
        for i, c in norm_pairs:
            if (g in c) or (c in g):
                hits.append((len(c), i))
        if hits:
            hits.sort()
            return hits[0][1]
        return None

    def map_labels_to_indices(gpt_labels: list) -> list:
        idxs = []
        seen = set()
        for g in gpt_labels:
            stats["gpt_labels_total"] += 1
            mi = map_one(g)
            if mi is None:
                stats["gpt_labels_unmapped"] += 1
                continue
            stats["gpt_labels_mapped"] += 1
            if mi not in seen:
                idxs.append(mi)
                seen.add(mi)
        return idxs

    return map_labels_to_indices, stats


# --------------------
# Scores helper
# --------------------
def _as_2d_scores(scores: np.ndarray) -> np.ndarray:
    if scores.ndim == 1:
        return np.column_stack([-scores, scores])
    return scores


# --------------------
# Metrics: Hit@k + nDCG@k (multi-label)
# --------------------
def _dcg_binary(rels: np.ndarray) -> float:
    if rels.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, rels.size + 2))
    return float((rels / denom).sum())


def _idcg_binary(num_rel: int, k: int) -> float:
    m = int(min(num_rel, k))
    if m <= 0:
        return 0.0
    denom = np.log2(np.arange(2, m + 2))
    return float((np.ones(m, dtype=float) / denom).sum())


def compute_hit_macro_from_hits(hits_per_query: dict, y_true_multi: list, n_classes: int):
    classes = np.arange(n_classes)
    contains_class = []
    for c in classes:
        contains_class.append(np.array([(c in labs) for labs in y_true_multi], dtype=bool))

    hit_micro = {}
    hit_macro = {}
    per_class_tables = {}

    for k, hit_arr in hits_per_query.items():
        hit_micro[k] = float(hit_arr.mean())

        per_class_vals = np.full(n_classes, np.nan, dtype=float)
        for c in classes:
            mask = contains_class[c]
            supp = int(mask.sum())
            if supp > 0:
                per_class_vals[c] = float(hit_arr[mask].mean())

        hit_macro[k] = float(np.nanmean(per_class_vals)) if np.any(~np.isnan(per_class_vals)) else float("nan")
        per_class_tables[k] = per_class_vals

    return hit_micro, hit_macro, per_class_tables


def topk_dict_to_rows(topk: dict) -> pd.DataFrame:
    rows = []
    for metric, kv in topk.items():
        for k, v in kv.items():
            rows.append({"metric": metric, "k": int(k), "value": float(v)})
    return pd.DataFrame(rows).sort_values(["metric", "k"]).reset_index(drop=True)


# --------------------
# Family mapping
# --------------------
def find_family_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "family label", "family_label", "family", "Family Label",
        "family_label_mapped", "family_label_clean"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "family" in c.lower():
            return c
    return None


# --------------------
# Main
# --------------------
def main():
    print("==== Loading CSVs ====")
    print(f"Train: {TRAIN_CSV}")
    print(f"Test : {TEST_CSV}")

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    if "collapsed_label" in train_df.columns:
        label_col = "collapsed_label"
    elif "label" in train_df.columns:
        label_col = "label"
    else:
        raise ValueError("Train CSV must contain 'collapsed_label' or 'label' column.")

    if "input_text" not in train_df.columns:
        raise ValueError("Train CSV must contain column: input_text")
    if label_col not in train_df.columns:
        raise ValueError(f"Train CSV must contain column: {label_col}")
    if ("Queries" not in test_df.columns) or ("gpt" not in test_df.columns):
        raise ValueError("Test CSV must contain columns: Queries, gpt")

    family_col = find_family_column(train_df)
    if family_col is None:
        raise ValueError(
            "Could not find family column in TRAIN_CSV. "
            "Expected something like 'family label' or 'family_label'."
        )
    print(f"Using family column: {family_col}")

    label_to_family = (
        train_df[[label_col, family_col]]
        .dropna()
        .astype(str)
        .assign(**{
            label_col: lambda d: d[label_col].map(clean_label),
            family_col: lambda d: d[family_col].map(clean_label),
        })
        .drop_duplicates(subset=[label_col])
        .set_index(label_col)[family_col]
        .to_dict()
    )

    train_texts  = train_df["input_text"].astype(str).tolist()
    train_labels = train_df[label_col].astype(str).tolist()

    test_texts     = test_df["Queries"].astype(str).tolist()
    test_gpt_lists = [parse_gpt_cell(x) for x in test_df["gpt"].tolist()]

    print(f"Using label column: {label_col}")
    print(f"Train rows: {len(train_texts):,}")
    print(f"Test rows:  {len(test_texts):,}")

    print("==== Preprocessing text ====")
    train_texts_clean = [clean_text_for_tfidf(t) for t in train_texts]
    test_texts_clean  = [clean_text_for_tfidf(t) for t in test_texts]

    print("==== Building TF-IDF ====")
    tfidf = TfidfVectorizer(**TFIDF_CFG)
    X_train = tfidf.fit_transform(train_texts_clean)
    X_test  = tfidf.transform(test_texts_clean)
    print(f"TF-IDF train shape: {X_train.shape}, dtype={X_train.dtype}")
    print(f"TF-IDF test  shape: {X_test.shape}, dtype={X_test.dtype}")

    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    classes = list(le.classes_)
    n_classes = len(classes)
    print(f"Classes: {n_classes:,}")

    map_labels_to_indices, map_stats = build_label_matcher(classes)
    test_labels_multi_idx = [map_labels_to_indices(labs) for labs in test_gpt_lists]
    map_stats["queries_empty_after_mapping"] = int(sum(1 for x in test_labels_multi_idx if len(x) == 0))

    class_idx_to_family = {}
    for i, lbl in enumerate(classes):
        fam = label_to_family.get(clean_label(lbl), None)
        class_idx_to_family[i] = fam if fam is not None else "unknown"

    test_families_multi = []
    for idxs in test_labels_multi_idx:
        fams = []
        for c in idxs:
            f = class_idx_to_family.get(int(c), "unknown")
            if f and f != "unknown":
                fams.append(f)
        test_families_multi.append(sorted(set(fams)))

    ensure_dir(OUT_DIR)
    svm_dir = OUT_DIR / "LinearSVM"
    ensure_dir(svm_dir)

    with open(OUT_DIR / "tfidf_vectorizer.sav", "wb") as f:
        pickle.dump(tfidf, f)
    with open(OUT_DIR / "label_encoder.sav", "wb") as f:
        pickle.dump(le, f)
    pd.Series(classes, name="class").to_csv(OUT_DIR / "classes.csv", index=False)

    print("==== Training Linear SVM ====")
    t0 = time.time()
    svm = LinearSVC(**SVM_CFG)
    svm.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"SVM fit time: {train_time:.1f}s")

    print("==== Evaluating (Top-k Hit + nDCG + Family) ====")
    raw_scores = svm.decision_function(X_test)
    scores_2d = _as_2d_scores(raw_scores)
    ranks = np.argsort(-scores_2d, axis=1)

    # ---------- LABEL-LEVEL ----------
    hits_per_query = {}
    ndcg_label = {}

    for k in K_VALUES:
        k_eff = int(min(k, n_classes))
        topk = ranks[:, :k_eff]

        hit_arr = np.zeros(len(test_labels_multi_idx), dtype=float)
        ndcg_arr = np.zeros(len(test_labels_multi_idx), dtype=float)

        for i in range(len(test_labels_multi_idx)):
            true_set = set(test_labels_multi_idx[i])
            if not true_set:
                continue

            rels = np.array([1.0 if int(lbl) in true_set else 0.0 for lbl in topk[i]], dtype=float)
            hit_arr[i] = 1.0 if rels.sum() > 0 else 0.0

            dcg = _dcg_binary(rels)
            idcg = _idcg_binary(num_rel=len(true_set), k=k_eff)
            ndcg_arr[i] = (dcg / idcg) if idcg > 0 else 0.0

        hits_per_query[k_eff] = hit_arr
        ndcg_label[k_eff] = float(ndcg_arr.mean())

    hit_micro, hit_macro, per_class_tables = compute_hit_macro_from_hits(
        hits_per_query=hits_per_query,
        y_true_multi=test_labels_multi_idx,
        n_classes=n_classes,
    )

    # per-class table (label level)
    class_support = np.zeros(n_classes, dtype=int)
    for labs in test_labels_multi_idx:
        for c in labs:
            if 0 <= int(c) < n_classes:
                class_support[int(c)] += 1

    per_class_df = pd.DataFrame({
        "class": np.arange(n_classes, dtype=int),
        "label": classes,
        "support": class_support.astype(int),
        "family": [class_idx_to_family[i] for i in range(n_classes)],
    })
    for k_eff, vals in per_class_tables.items():
        per_class_df[f"hit_at_{k_eff}"] = vals

    # ---------- FAMILY-LEVEL ----------
    fam_hit_micro = {}
    fam_ndcg = {}

    for k in K_VALUES:
        k_eff = int(min(k, n_classes))
        topk = ranks[:, :k_eff]

        hit_arr = np.zeros(len(test_families_multi), dtype=float)
        ndcg_arr = np.zeros(len(test_families_multi), dtype=float)

        for i in range(len(test_families_multi)):
            true_fams = set(test_families_multi[i])
            if not true_fams:
                continue

            pred_fams = [class_idx_to_family[int(lbl)] for lbl in topk[i]]
            rels = np.array([1.0 if pf in true_fams else 0.0 for pf in pred_fams], dtype=float)

            hit_arr[i] = 1.0 if rels.sum() > 0 else 0.0
            dcg = _dcg_binary(rels)
            idcg = _idcg_binary(num_rel=len(true_fams), k=k_eff)
            ndcg_arr[i] = (dcg / idcg) if idcg > 0 else 0.0

        fam_hit_micro[k_eff] = float(hit_arr.mean())
        fam_ndcg[k_eff] = float(ndcg_arr.mean())

    # ---------- SAVE ----------
    topk = {
        "hit_at_k_micro": hit_micro,
        "hit_at_k_macro": hit_macro,
        "ndcg_at_k": ndcg_label,
        "family_hit_at_k_micro": fam_hit_micro,
        "family_ndcg_at_k": fam_ndcg,
    }

    topk_df = topk_dict_to_rows(topk)
    topk_df.to_csv(svm_dir / "topk_metrics.csv", index=False)
    per_class_df.to_csv(svm_dir / "topk_per_class_hit.csv", index=False)

    with open(svm_dir / "model.sav", "wb") as f:
        pickle.dump(svm, f)

    np.save(svm_dir / "y_test_multi.npy", np.array(test_labels_multi_idx, dtype=object))
    np.save(svm_dir / "decision_scores.npy", scores_2d)

    metrics = {
        "train_time_sec": round(float(train_time), 2),
        "label_column": label_col,
        "family_column": family_col,
        "num_classes": int(n_classes),
        "topk": topk,
        "gpt_label_mapping_stats": map_stats,
    }
    save_json(svm_dir / "metrics.json", metrics)

    pd.DataFrame([{
        "model": "LinearSVM_TFIDF",
        "train_time_sec": metrics["train_time_sec"],
        "num_classes": metrics["num_classes"],
        "gpt_labels_total": map_stats["gpt_labels_total"],
        "gpt_labels_mapped": map_stats["gpt_labels_mapped"],
        "gpt_labels_unmapped": map_stats["gpt_labels_unmapped"],
        "queries_empty_after_mapping": map_stats["queries_empty_after_mapping"],
    }]).to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    print("\n==== Summary ====")
    print(f"Label column:  {label_col}")
    print(f"Family column: {family_col}")
    print(f"Num classes:   {n_classes}")
    print(f"Train time (sec): {metrics['train_time_sec']}")

    print("\nLabel-level Hit@k (micro):")
    for k in sorted(hit_micro.keys()):
        print(f"  @ {k}: {hit_micro[k]:.4f}")
    print("Label-level Hit@k (macro):")
    for k in sorted(hit_macro.keys()):
        print(f"  @ {k}: {hit_macro[k]:.4f}")
    print("Label-level nDCG@k:")
    for k in sorted(ndcg_label.keys()):
        print(f"  @ {k}: {ndcg_label[k]:.4f}")

    print("\nFamily-level Hit@k (micro):")
    for k in sorted(fam_hit_micro.keys()):
        print(f"  @ {k}: {fam_hit_micro[k]:.4f}")
    print("Family-level nDCG@k:")
    for k in sorted(fam_ndcg.keys()):
        print(f"  @ {k}: {fam_ndcg[k]:.4f}")

    print(f"\nSaved: {svm_dir / 'topk_metrics.csv'}")
    print(f"Saved: {svm_dir / 'topk_per_class_hit.csv'}")
    print(f"Saved: {svm_dir / 'metrics.json'}")
    print(f"\nAll done. Outputs in: {OUT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
