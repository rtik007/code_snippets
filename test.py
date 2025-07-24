import pandas as pd

# Example: Assume your column is 'extracted_text'
# Step-by-step cleaning
df['extracted_text'] = (
    df['extracted_text']
    .str.replace(r'\\\\[a-zA-Z]+', '', regex=True)        # remove \\commands like \\prime, \\circ
    .str.replace(r'~', ' ', regex=True)                   # replace tildes with space
    .str.replace(r'\\', '', regex=True)                   # remove any remaining backslashes
    .str.replace(r'\s+', ' ', regex=True)                 # normalize whitespace
    .str.strip()                                          # trim leading/trailing spaces
)




replacements = {
    r'\\prime': '′',
    r'\\circ': '°',
    r'\\otimes': '×',
    r'\\ast': '*',
    r'~': ' ',
}

for pattern, replacement in replacements.items():
    df['extracted_text'] = df['extracted_text'].str.replace(pattern, replacement, regex=True)
