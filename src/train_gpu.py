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

# --------------------------------------------------
# GPU CHECK
# --------------------------------------------------
print("Checking GPU...")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not detected. Install CUDA-enabled PyTorch."
    )

print("GPU Found:", torch.cuda.get_device_name(0))
print("CUDA Version:", torch.version.cuda)

# --------------------------------------------------
# DATASET
# --------------------------------------------------
print("Loading Dataset...")

dataset = load_dataset(
    "json",
    data_files="data/processed/train_1000.jsonl"
)

# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------
print("Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.pad_token = tokenizer.eos_token

# --------------------------------------------------
# MODEL
# --------------------------------------------------
print("Loading Model on GPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# --------------------------------------------------
# LORA CONFIG
# --------------------------------------------------
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

# --------------------------------------------------
# TOKENIZATION
# --------------------------------------------------
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
    batched=True,
    remove_columns=dataset["train"].column_names
)

# --------------------------------------------------
# DATA COLLATOR
# --------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# --------------------------------------------------
# TRAINING ARGS
# --------------------------------------------------
training_args = TrainingArguments(
    output_dir="models/lora",

    num_train_epochs=3,

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    weight_decay=0.01,

    logging_steps=5,

    save_steps=50,

    save_total_limit=2,

    fp16=True,

    report_to="none",

    dataloader_pin_memory=True,

    remove_unused_columns=False
)

# --------------------------------------------------
# TRAINER
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------
print("Starting Training...")

trainer.train()

# --------------------------------------------------
# SAVE
# --------------------------------------------------
print("Saving LoRA Adapter...")

model.save_pretrained(
    "models/lora"
)

tokenizer.save_pretrained(
    "models/lora"
)

print("Training Complete")