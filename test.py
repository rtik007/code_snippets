import json
import pandas as pd

# =========================
# Helpers
# =========================
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
    if isinstance(page, str):
        try:
            page = json.loads(page)
        except Exception:
            return ""
    if not isinstance(page, dict):
        return ""
    out = []

    # Common: blocks[].layout.text
    for block in page.get("blocks", []) or []:
        t = (block.get("layout", {}) or {}).get("text", "")
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
            t = (para.get("layout", {}) or {}).get("text", "")
            if isinstance(t, str) and t:
                out.append(t)
    if not out:
        for line in page.get("lines", []) or []:
            t = line.get("text", "")
            if isinstance(t, str) and t:
                out.append(t)

    return "\n".join(out)

def normalize_series(s: pd.Series) -> pd.Series:
    """Lowercase + collapse whitespace; safe on NaNs."""
    return (
        s.fillna("")
         .astype(str)
         .str.lower()
         .str.replace(r"\s+", " ", regex=True)
         .str.strip()
    )

# =========================
# Pipeline
# =========================

# 0) Normalize 'pages' and keep the full doc text
df["pages"] = df["pages"].apply(ensure_json)
df["_doc_text"] = df["text"].fillna("").astype(str)  # keep original full-doc string

# 1) Explode to one row per page
df["_page_index_tmp"] = df["pages"].apply(lambda p: list(range(len(p))))
df_pages = df.explode(["pages", "_page_index_tmp"], ignore_index=True)
df_pages = df_pages.rename(columns={"pages": "page", "_page_index_tmp": "page_index"})

# 2) Per-page text from JSON
df_pages["page_text"] = df_pages["page"].apply(extract_text_from_page)

# 3) Keep all original columns + new page fields
orig_cols = [c for c in df.columns if c not in ["pages", "_page_index_tmp"]]
df_pages = df_pages[orig_cols + ["page_index", "page_text", "page"]]

# 4) Verification: does joined page_text ≈ full doc text?
id_col = next((c for c in ["_doc_id", "doc_id", "id", "uri"] if c in df_pages.columns), None)

if id_col:
    per_doc_joined = (
        df_pages.sort_values([id_col, "page_index"])
                .groupby(id_col, as_index=True)["page_text"]
                .apply(lambda s: "\n".join(s))
                .rename("_joined_from_pages")
    )
    check = (
        df_pages[[id_col, "_doc_text"]]
            .drop_duplicates()
            .set_index(id_col)
            .join(per_doc_joined)
    )
    # Vectorized normalization to avoid "truth value is ambiguous"
    check["_doc_text_norm"] = normalize_series(check["_doc_text"])
    check["_joined_from_pages_norm"] = normalize_series(check["_joined_from_pages"])
    check["match"] = check["_doc_text_norm"].eq(check["_joined_from_pages_norm"])

    # Peek at mismatches (if any)
    display(check[~check["match"]].head(10))

# 5) Optional: sort for readability
sort_keys = ([id_col] if id_col else []) + ["page_index"]
df_pages = df_pages.sort_values(sort_keys).reset_index(drop=True)

# Final preview
df_pages.head(10)
