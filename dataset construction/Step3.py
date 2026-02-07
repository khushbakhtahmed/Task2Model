import pandas as pd
import numpy as np
import csv
import sys

csv.field_size_limit(sys.maxsize)

INPUT = "matches_phase1_exclusive_with_abstracts.csv"
OUTPUT = "models_with10papers_exploded.csv"

def split_field(text: str):
    if pd.isna(text):
        return []
    parts = [p.strip() for p in str(text).split("|||")]
    return [p for p in parts if p]

def main():
    df = pd.read_csv(
        INPUT,
        dtype=str,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
    )

    if "matched_snippets" not in df.columns:
        raise ValueError("Expected a 'matched_snippets' column in the input CSV.")

    # NOTE:
    # The input CSV is assumed to be pre-filtered upstream to models
    # appearing in at least N distinct papers.
    #
    # If filtering were to be enforced here, it could be done as follows.
    # The threshold (10 below) is configurable and can be changed
    # to any value (e.g., 20, 50) depending on dataset size and analysis needs.
    #
    # paper_counts = df.groupby("model_id")["paper_id"].nunique()
    # keep_models = paper_counts[paper_counts >= 10].index  # e.g., 10, 20, 50, ...
    # df = df[df["model_id"].isin(keep_models)]

    df["__row_id"] = np.arange(len(df))

    df_exploded = (
        df.assign(matched_snippet=df["matched_snippets"].apply(split_field))
          .explode("matched_snippet", ignore_index=True)
    )

    df_exploded = df_exploded[
        ~df_exploded["matched_snippet"].isna() &
        (df_exploded["matched_snippet"] != "")
    ]

    df_exploded = df_exploded.drop(columns=["matched_snippets"])

    if len(df_exploded):
        per_row_kept = df_exploded.groupby("__row_id").size().sort_index()
        print(
            f"Kept between {int(per_row_kept.min())} and "
            f"{int(per_row_kept.max())} snippets per original row."
        )
    else:
        print("No snippets found.")

    df_exploded.to_csv(OUTPUT, index=False)
    print(f"Done. Wrote {len(df_exploded):,} rows to {OUTPUT}")

if __name__ == "__main__":
    main()
