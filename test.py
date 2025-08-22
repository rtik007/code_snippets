import json
import pandas as pd

# ------------------------------------------------------
# 1) Keep 'pages' as raw JSON (convert any string to dict/list)
# ------------------------------------------------------
def ensure_json(val):
    if isinstance(val, str):
        try:
            return json.loads(val)   # parse JSON string
        except Exception:
            return val               # keep raw string if not valid JSON
    return val

df["pages"] = df["pages"].apply(ensure_json)


# ------------------------------------------------------
# 2) Extract flattened human-readable text from 'pages'
# ------------------------------------------------------
def extract_text_from_pages(pages):
    texts = []
    try:
        for page in pages if isinstance(pages, (list, tuple)) else []:
            for block in page.get("blocks", []):
                layout = block.get("layout", {})
                t = layout.get("text", "")
                if isinstance(t, str) and t:
                    texts.append(t)
    except Exception:
        return ""
    return "\n".join(texts)


# ------------------------------------------------------
# 3) Store both 'pages' (raw JSON) and 'text' (flattened)
# ------------------------------------------------------
df["text"] = df["pages"].apply(extract_text_from_pages)


# ------------------------------------------------------
# 4) Verify by showing both in the same DataFrame
# ------------------------------------------------------
# Compact preview of JSON for table display
def compact_json(val, max_len=200):
    try:
        s = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(val)
    return (s[:max_len] + "…") if len(s) > max_len else s

df["pages_preview"] = df["pages"].apply(compact_json)

# Show sample
df[["pages_preview", "text"]].head()
