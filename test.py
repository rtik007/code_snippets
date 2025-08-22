import json
import pandas as pd

# ---------------- helpers ----------------
def ensure_json(val):
    """Ensure 'pages' cell is a list[dict] (parse JSON string if needed)."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return val if isinstance(val, list) else []

def extract_text_from_page(page):
    """Flatten text for ONE page (joins all block.layout.text)."""
    out = []
    for block in page.get("blocks", []):
        t = block.get("layout", {}).get("text", "")
        if isinstance(t, str) and t:
            out.append(t)
    return "\n".join(out)

def extract_text_from_pages(pages):
    """Flatten text for a list of pages (document-level)."""
    if isinstance(pages, str):
        try:
            pages = json.loads(pages)
        except Exception:
            return ""
    if not isinstance(pages, (list, tuple)):
        return ""
    return "\n".join(extract_text_from_page(p) for p in pages)

# --------------- pipeline ----------------
# 0) Normalize pages
df["pages"] = df["pages"].apply(ensure_json)

# 1) Ensure we have a document-level text column
if "text" not in df.columns:
    df["text"] = df["pages"].apply(extract_text_from_pages)
else:
    # fill empties from pages if needed
    mask = df["text"].isna() | (df["text"].astype(str).str.len() == 0)
    df.loc[mask, "text"] = df.loc[mask, "pages"].apply(extract_text_from_pages)

# 2) Keep a copy of all original columns order
orig_cols = df.columns.tolist()

# 3) Add page index list, then explode (keeps every other column)
df["_page_index"] = df["pages"].apply(lambda p: list(range(len(p))))
df_pages = df.explode(["pages", "_page_index"], ignore_index=True)

# 4) Rename exploded columns & compute per-page text
df_pages = df_pages.rename(columns={"pages": "page", "_page_index": "page_index"})
df_pages["page_text"] = df_pages["page"].apply(extract_text_from_page)

# 5) Reorder: all original columns first, then page fields
#    If you don't want the big per-page JSON in the table, drop 'page' here.
cols = orig_cols + ["page_index", "page_text", "page"]
df_pages = df_pages[cols]

# 6) (Optional) sort by your id columns + page_index
id_cols = [c for c in ["_doc_id", "uri", "subfolder"] if c in df_pages.columns]
if id_cols:
    df_pages = df_pages.sort_values(id_cols + ["page_index"]).reset_index(drop=True)

# Done: df_pages has ALL your original columns + page_index/page/page_text
df_pages.head(10)
