import json
import re
import os
import ollama

CHUNKS_FILE = "data/processed/chunks.json"
OUTPUT_FILE = "data/processed/qa_raw.json"

def extract_json(content):
    """
    Extract JSON array from model output
    """

    content = content.strip()

    content = content.replace(
        "```json",
        ""
    )

    content = content.replace(
        "```",
        ""
    )

    match = re.search(
        r"\[.*\]",
        content,
        re.DOTALL
    )

    if match:

        json_text = match.group()

        return json.loads(json_text)

    raise ValueError(
        "No valid JSON found"
    )


if not os.path.exists(CHUNKS_FILE):

    raise FileNotFoundError(
        f"{CHUNKS_FILE} not found"
    )

with open(
    CHUNKS_FILE,
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)

print()
print("=" * 60)
print(f"Loaded {len(chunks)} chunks")
print("=" * 60)

dataset = []

for idx, chunk in enumerate(chunks):

    print(
        f"\nProcessing Chunk {idx+1}/{len(chunks)}"
    )

    prompt = f"""
Generate EXACTLY 15 question-answer pairs.

Rules:

1. Use ONLY the provided text.
2. Questions must be UNIQUE.
3. Include:
   - What
   - Why
   - How
   - Risks
   - Benefits
   - Examples
4. Return ONLY JSON.
5. Do NOT explain anything.
6. Do NOT use markdown.
7. Output must start with [
8. Output must end with ]

Format:

[
 {{
   "question":"What is KLP?",
   "answer":"KLP is..."
 }},
 {{
   "question":"How does KLP work?",
   "answer":"..."
 }}
]

TEXT:

{chunk}
"""

    success = False

    for retry in range(3):

        try:

            response = ollama.chat(
                model="gemma4:e2b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            content = response["message"]["content"]

            qa_pairs = extract_json(
                content
            )

            dataset.extend(
                qa_pairs
            )

            print(
                f"✅ Chunk {idx+1} completed "
                f"({len(qa_pairs)} QA)"
            )

            success = True

            break

        except Exception as e:

            print(
                f"Retry {retry+1}/3 failed:"
            )

            print(str(e))

    if not success:

        print(
            f"❌ Chunk {idx+1} skipped"
        )

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dataset,
        f,
        indent=2,
        ensure_ascii=False
    )

print()
print("=" * 60)
print("QA GENERATION COMPLETE")
print("=" * 60)
print(
    "Total QA Generated:",
    len(dataset)
)
print(
    "Saved File:",
    OUTPUT_FILE
)
print("=" * 60)