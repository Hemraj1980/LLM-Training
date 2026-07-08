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

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "google/gemma-2-2b-it"
DATASET_PATH = "data/processed/train_1000.jsonl"
OUTPUT_DIR = "models/lora"

# =====================================================
# DEVICE DETECTION
# =====================================================

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("=" * 50)
print("Using Device:", device)

if device == "mps":
    print("Apple Silicon GPU Enabled")
elif device == "cuda":
    print("NVIDIA GPU Enabled")
else:
    print("CPU Mode")

print("=" * 50)

# =====================================================
# LOAD DATASET
# =====================================================

print("Loading Dataset...")

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH
)

# =====================================================
# TOKENIZER
# =====================================================

print("Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# MODEL
# =====================================================

print("Loading Gemma Model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

model.to(device)

# =====================================================
# APPLY LORA
# =====================================================

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

# =====================================================
# TOKENIZATION
# =====================================================

def tokenize(example):
    return tokenizer(
        example["text"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )

print("Tokenizing Dataset...")

tokenized_dataset = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# =====================================================
# DATA COLLATOR
# =====================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# =====================================================
# TRAINING ARGUMENTS
# =====================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=5,

    learning_rate=2e-4,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    logging_steps=5,

    save_steps=25,

    save_total_limit=2,

    weight_decay=0.01,

    remove_unused_columns=False,

    report_to="none",

    dataloader_pin_memory=False,

    optim="adamw_torch"
)

# =====================================================
# TRAINER
# =====================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

# =====================================================
# TRAIN
# =====================================================

print("Starting Training...")

trainer.train()

# =====================================================
# SAVE MODEL
# =====================================================

print("Saving LoRA Adapter...")

model.save_pretrained(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("Training Complete!")