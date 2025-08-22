doc_id = check[~check["match"]].index[0]   # pick the first mismatch
print("Original doc_text:")
print(check.loc[doc_id, "_doc_text"][:500])

print("\nRebuilt from page_text:")
print(check.loc[doc_id, "_joined_from_pages"][:500])
