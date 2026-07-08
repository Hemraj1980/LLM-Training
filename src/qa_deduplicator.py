import json

INPUT_FILE = "data/processed/qa_raw.json"
OUTPUT_FILE = "data/processed/qa_unique.json"

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)

seen = set()

unique = []

for row in data:

    question = (
        row["question"]
        .strip()
        .lower()
    )

    if question not in seen:

        seen.add(question)

        unique.append(row)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        unique,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    "Original QA:",
    len(data)
)

print(
    "Unique QA:",
    len(unique)
)