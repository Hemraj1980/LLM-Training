import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import PeftModel

BASE_MODEL = "google/gemma-2-2b-it"

LORA_PATH = "models/lora"

OUTPUT_PATH = "models/merged_model"

print("=" * 60)
print("Loading Base Model")
print("=" * 60)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("=" * 60)
print("Loading Tokenizer")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

print("=" * 60)
print("Loading LoRA Adapter")
print("=" * 60)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH
)

print("=" * 60)
print("Merging LoRA Weights")
print("=" * 60)

merged_model = model.merge_and_unload()

print("=" * 60)
print("Saving Merged Model")
print("=" * 60)

merged_model.save_pretrained(
    OUTPUT_PATH,
    safe_serialization=True
)

tokenizer.save_pretrained(
    OUTPUT_PATH
)

print("=" * 60)
print("Merge Completed Successfully")
print("=" * 60)

print(f"Merged Model Saved To: {OUTPUT_PATH}")