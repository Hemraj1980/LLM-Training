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

# ----------------------------------------------------
# DEVICE DETECTION
# ----------------------------------------------------
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

# ----------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------
print("Loading Dataset...")

dataset = load_dataset(
    "json",
    data_files="data/processed/train_1000.jsonl"
)

# ----------------------------------------------------
# TOKENIZER
# ----------------------------------------------------
print("Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------------------------
# MODEL
# ----------------------------------------------------
print("Loading Model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.to(device)

# ----------------------------------------------------
# LORA
# ----------------------------------------------------
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

model.print_trainable_parameters()

# ----------------------------------------------------
# TOKENIZATION
# ----------------------------------------------------
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

print("Tokenizing Dataset...")

tokenized_dataset = dataset.map(
    tokenize,
    batched=True
)

# ----------------------------------------------------
# DATA COLLATOR
# ----------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ----------------------------------------------------
# TRAINING ARGUMENTS
# ----------------------------------------------------
training_args = TrainingArguments(
    output_dir="models/lora",
    overwrite_output_dir=True,

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    weight_decay=0.01,

    logging_steps=5,

    save_steps=25,

    save_total_limit=2,

    report_to="none",

    remove_unused_columns=False,

    dataloader_pin_memory=False
)

# ----------------------------------------------------
# TRAINER
# ----------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

# ----------------------------------------------------
# TRAIN
# ----------------------------------------------------
print("Starting Training...")

trainer.train()

# ----------------------------------------------------
# SAVE
# ----------------------------------------------------
print("Saving Adapter...")

model.save_pretrained(
    "models/lora"
)

tokenizer.save_pretrained(
    "models/lora"
)

print("Training Complete!")