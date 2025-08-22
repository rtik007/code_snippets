import json

def pretty_json(val):
    """Convert JSON-like or dict-like value to pretty string."""
    if isinstance(val, str):
        try:
            val = json.loads(val)   # if it's a stringified JSON
        except:
            return val              # if it's just a normal string
    try:
        return json.dumps(val, indent=2)  # pretty format
    except:
        return str(val)

# Create new column with pretty-printed JSON
df["pages_pretty"] = df["pages"].apply(pretty_json)

# Now you can see it in head()
df.head()
