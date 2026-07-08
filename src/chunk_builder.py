import json

with open(
    "data/processed/extracted_text.json",
    "r",
    encoding="utf-8"
) as f:

    text = json.load(f)["content"]

chunk_size = 1500

chunks = []

for i in range(0, len(text), chunk_size):

    chunk = text[i:i + chunk_size]

    if len(chunk.strip()) > 300:
        chunks.append(chunk)

with open(
    "data/processed/chunks.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunks,
        f,
        indent=2,
        ensure_ascii=False
    )

print("Chunks:", len(chunks))