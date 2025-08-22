import json
import pandas as pd

# ---------- helpers ----------
def ensure_json(val):
    """Make sure 'pages' is a list[dict], not a JSON string."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return val if isinstance(val, list) else []

def extract_text_from_page(page):
    """Get plain text from a single page dict (joins all block texts)."""
    texts = []
    for block in page.get("blocks", []):
        layout = block.get("layout", {})
        t = layout.get("text", "")
        if isinstance(t, str) and t:
            texts.append(t)
    return "\n".join(texts)

# ---------- 1) ensure pages are JSON lists ----------
df["pages"] = df["pages"].apply(ensure_json)

# ---------- 2) build page index list so explode keeps order ----------
df["_page_idx_list"] = df["pages"].apply(lambda p: list(range(len(p))))

# ---------- 3) explode to one row per page ----------
df_pages = df.explode(["pages", "_page_idx_list"], ignore_index=True)
df_pages.rename(columns={"pages": "page", "_page_idx_list": "page_index"}, inplace=True)

# Drop rows where there was no page
df_pages = df_pages[df_pages["page"].notna()]

# ---------- 4) per-page text ----------
df_pages["page_text"] = df_pages["page"].apply(extract_text_from_page)

# ---------- 5) pick the columns you want to keep ----------
# adjust '_doc_id' to your actual id column name if different
keep_cols = [c for c in ["_doc_id", "uri", "subfolder"] if c in df_pages.columns]
df_pages = df_pages[keep_cols + ["page_index", "page_text", "page"]]

# ---------- 6) sorting & quick look ----------
df_pages = df_pages.sort_values(keep_cols + ["page_index"]).reset_index(drop=True)

# Preview
df_pages.head(10)


# Replace '_doc_id' with your id column if needed
for doc_id, g in df_pages.groupby("_doc_id"):
    print(f"\n=== Document {doc_id} ===")
    for _, r in g.sort_values("page_index").iterrows():
        print(f"\n--- Page {r.page_index} ---")
        print(r.page_text)
