from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import torch

MODEL_PATH = "models/merged_model"

print("Loading Model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    device_map="cpu"
)

while True:

    question = input("\nQuestion: ")

    if question.lower() == "exit":
        break

    prompt = f"""Question: {question}
Answer:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    print("\nAnswer:")
    print(answer)