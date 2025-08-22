from pathlib import Path
import json
import re
import pandas as pd

# ---- 1) Set your root folder here ----
# Example from your screenshot (adjust to your actual root):
root = Path("merchant_documents/3546551650869084011")

# ---- 2) Helpers ----
def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def smart_merge(a, b):
    """
    Recursively merge two JSON-like dicts.
    - Strings for keys like 'text'/'body'/'content' are concatenated.
    - Lists are concatenated.
    - Dicts are merged recursively.
    - Otherwise b overwrites a.
    """
    out = dict(a)
    for k, v in b.items():
        if k in out:
            av = out[k]
            if isinstance(av, str) and isinstance(v, str) and k.lower() in {"text", "body", "content"}:
                out[k] = (av + " " + v).strip()
            elif isinstance(av, list) and isinstance(v, list):
                out[k] = av + v
            elif isinstance(av, dict) and isinstance(v, dict):
                out[k] = smart_merge(av, v)
            else:
                out[k] = v
        else:
            out[k] = v
    return out

# ---- 3) Discover pairs like D-XXXXXX-0.json and D-XXXXXX-1.json per subfolder ----
pairs = {}  # key: (subfolder_name, base_stem) -> {'0': Path, '1': Path}
for sub in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: int(p.name) if p.name.isdigit() else p.name):
    for f in sub.glob("*.json"):
        m = re.match(r"(.+)-([01])\.json$", f.name)
        if not m:
            continue
        base_stem, idx = m.group(1), m.group(2)   # e.g., base_stem='D-25799858', idx='0' or '1'
        pairs.setdefault((sub.name, base_stem), {})[idx] = f

# ---- 4) Build merged records ----
records = []
skipped = []
for (sub_name, base_stem), files in pairs.items():
    j0 = load_json(files["0"]) if "0" in files else None
    j1 = load_json(files["1"]) if "1" in files else None

    if j0 is None and j1 is None:
        skipped.append((sub_name, base_stem, "missing both"))
        continue
    if j0 is None or j1 is None:
        # If one part is missing, keep the one that exists (you can also choose to skip)
        only = j0 if j0 is not None else j1
        merged = only
        note = "only-0" if j1 is None else "only-1"
    else:
        merged = smart_merge(j0, j1)
        note = "merged-0-1"

    # add metadata columns
    merged["_subfolder"] = sub_name
    merged["_doc_id"] = base_stem
    merged["_merge_status"] = note
    records.append(merged)

# ---- 5) Convert to DataFrame (flatten nested keys) ----
df = pd.json_normalize(records, sep=".")

# (Optional) If there are very long text blobs you don’t want fully shown:
# df["text"] = df["text"].str.slice(0, 500)

# ---- 6) Save / inspect ----
print(f"Merged rows: {len(df)}  |  Skipped: {len(skipped)}")
# print("Skipped examples:", skipped[:5])
df.to_csv("merged_documents.csv", index=False)
df.head()
