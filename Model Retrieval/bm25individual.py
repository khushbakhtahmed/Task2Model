#!/usr/bin/env python3
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from collections import Counter

# -------------------------
# CONFIG
# -------------------------
#train_group.csv and test_group.csv are the group-aware splits, random splits can be used as well which are train_stratified.csv and test_stratified.csv respectively. The code will work with either as long as the LABEL_COL and FAMILY_COL are present.
TRAIN_ROWS_CSV = "train_group.csv"
TEST_ROWS_CSV  = "test_group.csv"

TEXT_COL = "input_text"
LABEL_COL = "collapsed_label"
FAMILY_COL = "family label"

Ks = [1, 3, 5, 10]
CAP_LIMITS = [100, 500, 1000, 1500] 

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
# Evaluation Logic
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

train_df = pd.read_csv(TRAIN_ROWS_CSV)
test_df = pd.read_csv(TEST_ROWS_CSV)

# Map specific label -> family
family_map = train_df.set_index(LABEL_COL)[FAMILY_COL].to_dict()

all_run_results = []

for cap in CAP_LIMITS:
    print(f"Running Individual Row Evaluation for Cap: {cap}...")
    
    # 1. Create a balanced set of individual rows (Fixed-Prefix)
    balanced_train = train_df.groupby(LABEL_COL).head(cap).copy()
    
    # 2. Tokenize every single row as its own document
    tokenized_corpus = [prep(text).split() for text in balanced_train[TEXT_COL]]
    
    # 3. Store the labels corresponding to each document
    row_labels = balanced_train[LABEL_COL].tolist()
    
    bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75) # Standard b=0.75 works better for small individual rows
    
    run_metrics = {"Specific": {k: {"hit":[], "mrr":[], "ndcg":[]} for k in Ks},
                   "Family":   {k: {"hit":[], "mrr":[], "ndcg":[]} for k in Ks}}
    
    for _, row in test_df.iterrows():
        q_tok = prep(row[TEXT_COL]).split()
        if not q_tok: continue
        
        # Get scores for ALL individual rows
        scores = bm25.get_scores(q_tok)
        
        # Sort rows by similarity
        # We need more results than just 'k' because multiple rows might have the same label
        top_indices = np.argsort(scores)[::-1][:50] 
        
        # Get the ordered labels of the most similar rows
        # We use a "Ordered Set" logic: the first time a label appears, that is its rank.
        retrieved_specs = []
        for idx in top_indices:
            lbl = row_labels[idx]
            if scores[idx] > 0 and lbl not in retrieved_specs:
                retrieved_specs.append(lbl)
        
        # Specific Eval
        spec_res = calculate_metrics(row[LABEL_COL], retrieved_specs, Ks)
        
        # Family Eval
        retrieved_fams = []
        for s in retrieved_specs:
            f = family_map.get(s, "Unknown")
            if f not in retrieved_fams: retrieved_fams.append(f)
        fam_res = calculate_metrics(row[FAMILY_COL], retrieved_fams, Ks)
        
        for k in Ks:
            for lvl, res_src in [("Specific", spec_res), ("Family", fam_res)]:
                run_metrics[lvl][k]["hit"].append(res_src[k][0])
                run_metrics[lvl][k]["mrr"].append(res_src[k][1])
                run_metrics[lvl][k]["ndcg"].append(res_src[k][2])
    
    summary = {lvl: {k: {m: np.mean(run_metrics[lvl][k][m]) for m in ["hit", "mrr", "ndcg"]} for k in Ks} for lvl in ["Specific", "Family"]}
    all_run_results.append(summary)

# -------------------------
# FINAL AVERAGING & OUTPUT
# -------------------------
print("\n" + "="*85)
print(f"{'K':<3} | {'Spec Hit':<9} | {'Fam Hit':<9} | {'Spec MRR':<9} | {'Fam MRR':<9} | {'Spec nDCG':<9} | {'Fam nDCG':<9}")
print("-" * 85)

for k in Ks:
    avg_s_h = np.mean([run["Specific"][k]["hit"] for run in all_run_results])
    avg_f_h = np.mean([run["Family"][k]["hit"] for run in all_run_results])
    avg_s_m = np.mean([run["Specific"][k]["mrr"] for run in all_run_results])
    avg_f_m = np.mean([run["Family"][k]["mrr"] for run in all_run_results])
    avg_s_n = np.mean([run["Specific"][k]["ndcg"] for run in all_run_results])
    avg_f_n = np.mean([run["Family"][k]["ndcg"] for run in all_run_results])
    
    print(f"{k:<3} | {avg_s_h:<9.4f} | {avg_f_h:<9.4f} | {avg_s_m:<9.4f} | {avg_f_m:<9.4f} | {avg_s_n:<9.4f} | {avg_f_n:<9.4f}")