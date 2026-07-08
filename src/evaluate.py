from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import torch
import json

MODEL_PATH = "models/merged_model"

QUESTIONS = [
    "What is KLP?",
    "What is KTC?",
    "Explain esKTC.",
    "What is the maximum leverage?",
    "How does liquidation work?",
    "What are multiplier points?",
    "How does KLP earn fees?",
    "What is a borrowing fee?",
    "How does staking work?",
    "What is KTX Finance?"
]

print("Loading Model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="cpu"
)

results = []

for idx, question in enumerate(QUESTIONS):

    print(
        f"Testing {idx+1}/{len(QUESTIONS)}"
    )

    inputs = tokenizer(
        question,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    results.append(
        {
            "question": question,
            "answer": answer
        }
    )

with open(
    "evaluation/eval_results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )

print()
print("Evaluation Complete")
print(
    "Results saved to evaluation/eval_results.json"
)