import json
import pandas as pd
import re

# ---------- helpers ----------
def ensure_json(val):
    """Ensure 'pages' is a list[dict] (parse JSON string if needed)."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return val if isinstance(val, list) else []

def extract_text_from_page(page):
    """Extract human-readable text from a single page dict."""
    # If page accidentally still a JSON string:
    if isinstance(page, str):
        try:
            page = json.loads(page)
        except Exception:
            return ""
    if not isinstance(page, dict):
        return ""
    out = []

    # Most common: blocks[].layout.text
    for block in page.get("blocks", []) or []:
        t = block.get("layout", {}).get("text", "")
        if isinstance(t, str) and t:
            out.append(t)

    # Fallbacks for other schemas
    if not out:
        for block in page.get("blocks", []) or []:
            t = block.get("text", "")
            if isinstance(t, str) and t:
                out.append(t)
    if not out:
        for para in page.get("paragraphs", []) or []:
            t = para.get("layout", {}).get("text", "")
            if isinstance(t, str) and t:
                out.append(t)
    if not out:
        for line in page.get("lines", []) or []:
            t = line.get("text", "")
            if isinstance(t, str) and t:
                out.append(t)

    return "\n".join(out)

def normalize(s: str) -> str:
    """Normalize text for comparison (lower, collapse whitespace)."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

# ---------- 0) normalize pages & keep doc-level text ----------
df["pages"] = df["pages"].apply(ensure_json)
df["_doc_text"] = df["text"].fillna("").astype(str)   # keep original full-doc text

# ---------- 1) explode to one row per page ----------
df["_page_index_tmp"] = df["pages"].apply(lambda p: list(range(len(p))))
df_pages = df.explode(["pages", "_page_index_tmp"], ignore_index=True)
df_pages = df_pages.rename(columns={"pages": "page", "_page_index_tmp": "page_index"})

# ---------- 2) create per-page text from JSON ----------
df_pages["page_text"] = df_pages["page"].apply(extract_text_from_page)

# ---------- 3) keep all original columns + new ones ----------
orig_cols = [c for c in df.columns if c not in ["pages", "_page_index_tmp"]]
df_pages = df_pages[orig_cols + ["page_index", "page_text", "page"]]

# ---------- 4) (optional) verify page_text concatenation ≈ doc-level text ----------
# Pick an id column you have; we'll autodetect a common one:
id_col = next((c for c in ["_doc_id", "doc_id", "id", "uri"] if c in df_pages.columns), None)

if id_col:
    # concat per-page text back to doc-level
    per_doc_joined = (
        df_pages
        .sort_values([id_col, "page_index"])
        .groupby(id_col)["page_text"].apply(lambda s: "\n".join(s))
        .rename("_joined_from_pages")
    )
    check = (
        df_pages[[id_col, "_doc_text"]].drop_duplicates().set_index(id_col)
        .join(per_doc_joined)
        .assign(match=lambda x: normalize(x["_doc_text"]) == normalize(x["_joined_from_pages"]))
    )
    # You can inspect mismatches like this:
    # display(check[~check["match"]].head())

# ---------- 5) (optional) sort for readability ----------
sort_keys = ([id_col] if id_col else []) + ["page_index"]
df_pages = df_pages.sort_values(sort_keys).reset_index(drop=True)

# Result: df_pages has one row per page, your original full-doc text in `_doc_text`,
# and per-page text in `page_text` (plus the raw per-page JSON in `page`).
df_pages.head(10)
