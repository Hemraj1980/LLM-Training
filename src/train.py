import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model
)

MODEL_NAME = "google/gemma-2-2b-it"

print("Loading Dataset...")

dataset = load_dataset(
    "json",
    data_files="data/processed/train_1000.jsonl"
)

print("Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

print("Loading Model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Applying LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    lora_config
)

def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

print("Tokenizing Dataset...")

tokenized_dataset = dataset.map(
    tokenize,
    batched=True
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="models/lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=5,
    save_steps=25,
    save_total_limit=2,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

print("Starting Training...")

trainer.train()

print("Saving LoRA Adapter...")

model.save_pretrained(
    "models/lora"
)

tokenizer.save_pretrained(
    "models/lora"
)

print("Training Complete")