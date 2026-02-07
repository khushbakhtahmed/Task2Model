import re
import time
import pandas as pd
import arxiv

INPUT_CSV  = "matches_phase1_exclusive.csv"
OUTPUT_CSV = "matches_phase1_exclusive_with_abstracts.csv"

# --- Helpers ---
_id_version_re = re.compile(r"v\d+$", re.IGNORECASE)

def normalize_arxiv_id(pid: str) -> str:
    if pid is None:
        return ""
    pid = str(pid).strip()
    # Treat common null-like strings as missing
    if not pid or pid.lower() in {"nan", "none"}:
        return ""
    if pid.lower().startswith("arxiv:"):
        pid = pid.split(":", 1)[1]
    pid = _id_version_re.sub("", pid)
    return pid

def fetch_abstract_for_id(pid: str, client: arxiv.Client, retries: int = 3, pause: float = 0.5) -> str:
    pid_norm = normalize_arxiv_id(pid)
    if not pid_norm:
        return ""

    for attempt in range(1, retries + 1):
        try:
            search = arxiv.Search(id_list=[pid_norm])
            results = list(client.results(search))
            if results:
                return results[0].summary or ""
            return ""
        except Exception:
            if attempt == retries:
                return ""
            time.sleep(pause * attempt)

def main():
    df = pd.read_csv(INPUT_CSV)

    if "paper_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'paper_id' column.")

    # Convert missing values to empty string first so `astype(str)` doesn't produce "nan"
    df["__pid_norm"] = df["paper_id"].fillna("").astype(str).map(normalize_arxiv_id)
    unique_ids = sorted(set(df["__pid_norm"]) - {""})

    client = arxiv.Client(page_size=1, delay_seconds=0.3)

    pid_to_abstract = {}
    for i, pid_norm in enumerate(unique_ids, 1):
        abstract = fetch_abstract_for_id(pid_norm, client)
        pid_to_abstract[pid_norm] = abstract
        if i % 50 == 0 or i == len(unique_ids):
            print(f"[{i}/{len(unique_ids)}] {pid_norm} -> {'OK' if abstract else 'NOT FOUND'}")

    df["paper_abstract"] = df["__pid_norm"].map(pid_to_abstract).fillna("")
    df.drop(columns=["__pid_norm"], inplace=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Full file saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
