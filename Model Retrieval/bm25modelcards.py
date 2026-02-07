#!/usr/bin/env python3
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

# -------------------------
# CONFIG
# -------------------------
MODELCARD_CSV = "filtered_top100_label_and_modelcard_with_collapsed.csv"
TEST_ROWS_CSV  = "test_group.csv"

# Columns for Model Card CSV
MC_TEXT_COL = "Cleaned Model Card"
MC_LABEL_COL = "collapsed_label"

# Columns for Test CSV
TEST_TEXT_COL = "input_text"
TEST_LABEL_COL = "collapsed_label"
TEST_FAMILY_COL = "family label"

Ks = [1, 3, 5, 10]

# -------------------------
# Preprocessing
# -------------------------
STOPWORDS = {"the","a","an","and","or","but","if","while","with","without","of","at","by","for",
             "to","from","in","on","onto","is","are","was","were","be","been","being","this",
             "that","these","those","it","its","as","not","no","so","such","can","will","would",
             "should","could","may","might","do","does","did","doing","have","has","had","having",
             "i","you","we","they","he","she","them","his","her","our","your","my","me"}

def prep(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"[^a-z\s]", " ", t)
    toks = [w for w in t.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(toks)

# -------------------------
# Load Data
# -------------------------
mc_df = pd.read_csv(MODELCARD_CSV)
test_df = pd.read_csv(TEST_ROWS_CSV)

# Drop rows where Model Card is missing
mc_df = mc_df.dropna(subset=[MC_TEXT_COL])

# Build Family Map from Test Set (assuming family_label is there)
# Note: Ensure all labels in mc_df exist in your family_map
family_map = test_df.set_index(TEST_LABEL_COL)[TEST_FAMILY_COL].to_dict()

# -------------------------
# Step 1: Build BM25 Index from Model Cards
# -------------------------
labels_order = mc_df[MC_LABEL_COL].tolist()
corpus_clean = [prep(doc) for doc in mc_df[MC_TEXT_COL]]
tokenized_docs = [doc.split() for doc in corpus_clean]

# Using b=0.75 as model cards vary in length naturally
bm25 = BM25Okapi(tokenized_docs, k1=1.5, b=0.75)
print(f"✅ BM25 Index built using {len(labels_order)} Hugging Face Model Cards.")

# -------------------------
# Step 4: Evaluation Metrics
# -------------------------
def calculate_metrics(true_val, ranked_list, k_list):
    res = {}
    for k in k_list:
        top_k = ranked_list[:k]
        hit = int(true_val in top_k)
        mrr = 1.0 / (top_k.index(true_val) + 1) if true_val in top_k else 0.0
        ndcg = 1.0 / np.log2(top_k.index(true_val) + 2) if true_val in top_k else 0.0
        res[k] = (hit, mrr, ndcg)
    return res

results = {
    "Specific": {k: {"hit": [], "mrr": [], "ndcg": []} for k in Ks},
    "Family":   {k: {"hit": [], "mrr": [], "ndcg": []} for k in Ks}
}

for _, row in test_df.iterrows():
    query = row[TEST_TEXT_COL]
    true_label = row[TEST_LABEL_COL]
    true_family = row[TEST_FAMILY_COL]
    
    q_tok = prep(query).split()
    if not q_tok: continue
    
    scores = bm25.get_scores(q_tok)
    ranked_indices = np.argsort(scores)[::-1]
    retrieved_specs = [labels_order[i] for i in ranked_indices if scores[i] > 0]
    
    # Specific Level
    spec_res = calculate_metrics(true_label, retrieved_specs, Ks)
    
    # Family Level (Deduplicated)
    retrieved_fams = []
    for s in retrieved_specs:
        f = family_map.get(s, "Unknown")
        if f not in retrieved_fams: retrieved_fams.append(f)
    fam_res = calculate_metrics(true_family, retrieved_fams, Ks)
    
    for k in Ks:
        for lvl, res_src in [("Specific", spec_res), ("Family", fam_res)]:
            results[lvl][k]["hit"].append(res_src[k][0])
            results[lvl][k]["mrr"].append(res_src[k][1])
            results[lvl][k]["ndcg"].append(res_src[k][2])

# -------------------------
# FINAL OUTPUT
# -------------------------
header = f"{'K':<3} | {'Spec Hit':<10} | {'Fam Hit':<10} | {'Spec MRR':<10} | {'Fam MRR':<10} | {'Spec nDCG':<10} | {'Fam nDCG':<10}"
print("\n" + header)
print("-" * len(header))
for k in Ks:
    print(f"{k:<3} | "
          f"{np.mean(results['Specific'][k]['hit']):<10.4f} | "
          f"{np.mean(results['Family'][k]['hit']):<10.4f} | "
          f"{np.mean(results['Specific'][k]['mrr']):<10.4f} | "
          f"{np.mean(results['Family'][k]['mrr']):<10.4f} | "
          f"{np.mean(results['Specific'][k]['ndcg']):<10.4f} | "
          f"{np.mean(results['Family'][k]['ndcg']):<10.4f}")