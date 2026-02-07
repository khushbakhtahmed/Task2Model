# save as make_training_data.py
import pandas as pd
import re
import csv

INPUT = "models_with10papers_exploded.csv"
OUTPUT = "models_with10papers_training_data.csv"

REPLACEMENT = "«MODEL»"

# ---------- regex helpers ----------

def _strict_base_pattern_not_in_wrappers(base: str) -> re.Pattern:
    """
    Bare base token only when:
      - not part of a longer token (no letters/digits/_/- around it), and
      - NOT immediately touching common wrappers (so we don't hit cases like «base»-aug)
    """
    return re.compile(
        rf"(?<![\w-])(?<![«\"“'(\[{{]]){re.escape(base)}(?![\w-])(?![»\"”'\)\]}}])",
        re.IGNORECASE
    )

def _quoted_base_pattern_with_outer_boundaries(base: str) -> re.Pattern:
    """Match « base » as a standalone token; blocks m«base» and «base»-suffix/prefix."""
    return re.compile(
        rf"(?<![\w-])«\s*{re.escape(base)}\s*»(?![\w-])",
        re.IGNORECASE
    )

def _strict_repo_pattern_not_in_wrappers(repo: str) -> re.Pattern:
    """
    Bare org/name only when:
      - not part of a longer token (no letters/digits/_/- around it), and
      - NOT immediately inside/adjacent to wrappers (avoid replacing inside « … »).
    """
    return re.compile(
        rf"(?<![\w-])(?<![«\"“'(\[{{]]){re.escape(repo)}(?![\w-])(?![»\"”'\)\]}}])",
        re.IGNORECASE
    )

def _quoted_repo_pattern_with_outer_boundaries(repo: str) -> re.Pattern:
    """Match « org/name » as a standalone token; blocks «repo»-suffix/prefix."""
    return re.compile(
        rf"(?<![\w-])«\s*{re.escape(repo)}\s*»(?![\w-])",
        re.IGNORECASE
    )

def _full_url_pattern_with_path_boundary(repo: str) -> re.Pattern:
    """
    Match https://huggingface.co/<repo> only if followed by end-of-string
    or a path boundary (/ ? #). Prevents matching prefixes of longer paths
    like .../<repo>-Instruct.
    """
    return re.compile(
        rf"https://huggingface\.co/{re.escape(repo)}(?=$|[/?#])",
        re.IGNORECASE
    )

# ---------- core cleaner ----------

def clean_snippet(snippet: str, model_id: str) -> tuple[str, bool]:
    """
    Replace occurrences of the model in priority order:
      1) full HF URL
      2) repo id (org/name)
      3) base model name

    NOTE:
    In later stages of the pipeline, we additionally search for and mask
    generic model aliases (e.g., 'bert', 't5', etc.) to further prevent
    information leakage during training. This step focuses only on masking
    the explicitly matched model identifier.
    """
    if pd.isna(snippet) or snippet is None:
        return "", False
    if pd.isna(model_id) or model_id is None:
        return str(snippet).strip(), False

    text = str(snippet)
    repo = str(model_id)
    base = repo.split("/")[-1]

    matched = False

    # 1) Full URL
    full_url_pat = _full_url_pattern_with_path_boundary(repo)
    if full_url_pat.search(text):
        text = full_url_pat.sub(REPLACEMENT, text)
        matched = True

    # 2) Repo id — quoted
    quoted_repo_pat = _quoted_repo_pattern_with_outer_boundaries(repo)
    if quoted_repo_pat.search(text):
        text = quoted_repo_pat.sub(REPLACEMENT, text)
        matched = True

    # 2b) Repo id — bare
    strict_repo_pat = _strict_repo_pattern_not_in_wrappers(repo)
    if strict_repo_pat.search(text):
        text = strict_repo_pat.sub(REPLACEMENT, text)
        matched = True

    # 3) Base name — quoted
    quoted_base_pat = _quoted_base_pattern_with_outer_boundaries(base)
    if quoted_base_pat.search(text):
        text = quoted_base_pat.sub(REPLACEMENT, text)
        matched = True

    # 3b) Base name — bare
    strict_base_pat = _strict_base_pattern_not_in_wrappers(base)
    if strict_base_pat.search(text):
        text = strict_base_pat.sub(REPLACEMENT, text)
        matched = True

    return text.strip(), matched

# ---------- main I/O ----------

def main():
    df = pd.read_csv(INPUT, dtype=str, quoting=csv.QUOTE_MINIMAL)

    required = {"model_id", "paper_id", "matched_snippet"}
    if not required.issubset(df.columns):
        raise ValueError("Input CSV must contain 'model_id', 'paper_id', and 'matched_snippet' columns.")

    matched_rows = []
    total = len(df)
    unmatched_count = 0

    for _, row in df.iterrows():
        model_id = row.get("model_id", "")
        snippet = row.get("matched_snippet", "")
        paper_id = row.get("paper_id", "")

        cleaned, matched = clean_snippet(snippet, model_id)
        if matched:
            matched_rows.append({
                "paper_id": paper_id,
                "input_text": cleaned,
                "label": model_id
            })
        else:
            unmatched_count += 1

    df_out = pd.DataFrame(matched_rows, columns=["paper_id", "input_text", "label"])
    df_out.to_csv(OUTPUT, index=False)

    written = len(df_out)
    print(f"Done. Wrote {written:,} matched rows to {OUTPUT}")
    print(f"Total rows processed: {total:,}")
    print(f"Rows excluded (no match): {unmatched_count:,} ({(unmatched_count/total if total else 0):.2%})")

if __name__ == "__main__":
    main()
