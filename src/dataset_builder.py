import json
import os

INPUT_FILE = "data/processed/qa_unique_150.json"
OUTPUT_FILE = "data/processed/train_150.jsonl"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Cannot find {INPUT_FILE}"
    )

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)

count = 0

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as out:

    for item in data:

        text = f"""<start_of_turn>user
{item['question']}
<end_of_turn>
<start_of_turn>model
{item['answer']}
<end_of_turn>
"""

        out.write(
            json.dumps(
                {"text": text},
                ensure_ascii=False
            )
        )

        out.write("\n")

        count += 1

print("=" * 50)
print("Dataset Created Successfully")
print("Samples:", count)
print("Output:", OUTPUT_FILE)
print("=" * 50)