import csv
import json
import os
import re
import bisect
from hashlib import blake2b

# ==================== CONFIG ====================
input_csv  = "CMA-PROJ_Dataset.csv"
output_csv = "matches_phase1_exclusive.csv"

# Root directory where unarXiv 2024 JSONL files are stored
unarxiv_root = "/path/to/unarxiv_2024/"

PRE  = 500
POST = 500
# =================================================


def sanitize_label(label: str) -> str:
    label = re.sub(r"[\r\n\t]+", " ", label)
    label = re.sub(r"\s{2,}", " ", label).strip()
    return label


def iter_text_segments(obj, path=""):
    def make_label(parent_path, key=None):
        parts = [p for p in parent_path.split(".") if p]
        if key is not None:
            parts.append(str(key))
        if "bib_entries" in parts and len(parts) >= 2:
            parts = ["bib_entries", parts[-1]]
        if "body_text" in parts and "text" in parts:
            parts = ["body_text", "text"]
        if "abstract" in parts and "text" in parts:
            parts = ["abstract", "text"]
        if "metadata" in parts and "title" in parts:
            parts = ["metadata", "title"]
        if "sections" in parts and parts[-1] == "text" and len(parts) >= 3:
            parts = ["sections", parts[-2], "text"]
        return sanitize_label(".".join(parts) if parts else "(root)")

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                yield (make_label(path, k), f"{k}: {v}")
            else:
                yield from iter_text_segments(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, it in enumerate(obj):
            yield from iter_text_segments(it, f"{path}.{i}" if path else str(i))


def build_unified_text(segments):
    parts, spans, cur = [], [], 0
    for label, txt in segments:
        m = f"[[SECTION:{label}]]\n"
        parts.append(m); cur += len(m)
        s = cur
        parts.append(txt); cur += len(txt)
        spans.append((s, cur, label))
        parts.append("\n\n"); cur += 2
    unified = "".join(parts)
    return unified, unified.lower(), spans


def strip_ws_with_map(s):
    out, idx = [], []
    for i, ch in enumerate(s):
        if not ch.isspace():
            out.append(ch); idx.append(i)
    return "".join(out), idx


def find_all_positions(hay, needle):
    res, L, i = [], len(needle), 0
    while True:
        j = hay.find(needle, i)
        if j == -1:
            break
        res.append((j, j + L))
        i = j + L
    return res


def overlaps(a, b):
    return not (a[1] <= b[0] or b[1] <= a[0])


def exclude_overlaps(spans, occupied):
    return [s for s in spans if not any(overlaps(s, o) for o in occupied)]


def section_for_index(spans, idx):
    starts = [s for s, _, _ in spans]
    pos = bisect.bisect_right(starts, idx) - 1
    if pos >= 0:
        s, e, lbl = spans[pos]
        return lbl, s, e
    return "(unknown)", 0, 0


def compute_window(idx, s, e):
    left = min(PRE, max(0, idx - s))
    right = min(POST + (PRE - left), max(0, e - idx))
    return idx - left, idx + right


def dedupe_snippets(snips):
    seen, out = set(), []
    for s in snips:
        h = blake2b(s.encode("utf-8"), digest_size=16).hexdigest()
        if h not in seen:
            seen.add(h); out.append(s)
    return out


def search_plain(unified_lower, spans, needle):
    res, L = [], len(needle)
    for s, e, _ in spans:
        sub = unified_lower[s:e]
        i = 0
        while True:
            j = sub.find(needle, i)
            if j == -1:
                break
            res.append((s + j, s + j + L))
            i = j + L
    return res


def process_target_paper(model_id, paper_id, paper, source_file):
    try:
        segs = list(iter_text_segments(paper))
        unified, unified_lower, spans = build_unified_text(segs)
    except Exception:
        return None

    mid = model_id.lower().strip()
    short = mid.split("/")[-1]
    link = f"https://huggingface.co/{mid}"

    link_spans = search_plain(unified_lower, spans, link)
    full_spans = exclude_overlaps(search_plain(unified_lower, spans, mid), link_spans)
    short_spans = exclude_overlaps(
        search_plain(unified_lower, spans, short),
        link_spans + full_spans
    )

    events = (
        [(s, e, "hf_link") for s, e in link_spans] +
        [(s, e, "full_id") for s, e in full_spans] +
        [(s, e, "short_id") for s, e in short_spans]
    )
    if not events:
        return None

    events.sort(key=lambda x: (x[0], ["hf_link", "full_id", "short_id"].index(x[2])))

    snippets, types, cover = [], [], -1
    for s, e, t in events:
        if s < cover:
            continue
        lbl, ss, se = section_for_index(spans, s)
        ws, we = compute_window(s, ss, se)
        body = unified[ws:we]
        hs, he = s - ws, e - ws
        body = body[:hs] + "«" + body[hs:he] + "»" + body[he:]
        snippets.append(f"[TYPE:{t}] [SECTION:{lbl}] — {body}")
        types.append(t)
        cover = we

    snippets = dedupe_snippets(snippets)

    return (
        model_id,
        paper_id,
        min(types, key=["hf_link", "full_id", "short_id"].index),
        " ||| ".join(snippets),
        len(snippets),
        len(full_spans),
        len(link_spans),
        len(short_spans),
        source_file
    )


def main():
    with open(input_csv, encoding="utf-8") as f:
        models = [r["Model ID"].strip() for r in csv.DictReader(f) if r.get("Model ID")]

    out = []

    for root, _, files in os.walk(unarxiv_root):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    pid = obj.get("paper_id")
                    for m in models:
                        row = process_target_paper(m, pid, obj, path)
                        if row:
                            out.append(row)
                            break

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "model_id","paper_id","overall_match_type","matched_snippets",
            "snippet_count","full_id_count","hf_link_count","short_id_count","source_file"
        ])
        w.writerows(out)

    print(f"Done. Wrote {len(out)} rows to {output_csv}")


if __name__ == "__main__":
    main()
