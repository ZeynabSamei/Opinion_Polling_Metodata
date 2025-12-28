import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Fast fine-tune LLM on OpinionQA/Subpop dataset")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True, help="JSON test dataset")
parser.add_argument("--fine_tune_data", type=str, default=None, help="JSONL fine-tune dataset")
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--ft_epochs", type=int, default=1)
parser.add_argument("--ft_batch_size", type=int, default=4)
parser.add_argument("--subset_size", type=int, default=1000, help="Subset for fast primary run")
parser.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
parser.add_argument("--max_seq_len", type=int, default=256, help="Truncate sequences for speed")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------
# Load datasets
# -----------------------------
with open(args.data_path, "r") as f:
    test_data = json.load(f)
print(f"Loaded {len(test_data)} test samples")

ft_data = None
if args.fine_tune_data is not None:
    ft_data_lines = open(args.fine_tune_data).readlines()
    ft_data = [json.loads(l) for l in ft_data_lines]
    print(f"Loaded {len(ft_data)} fine-tune samples")
    if args.subset_size and len(ft_data) > args.subset_size:
        ft_data = random.sample(ft_data, args.subset_size)
        print(f"Using subset of {len(ft_data)} for fast run")

# -----------------------------
# Load model & tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.eos_token_id
model.eval()
device = model.device if hasattr(model, "device") else next(model.parameters()).device

# -----------------------------
# Optional LoRA
# -----------------------------
if args.lora:
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        print("Applying LoRA...")
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
    except ImportError:
        print("peft not installed, skipping LoRA")
        args.lora = False

# -----------------------------
# Fine-tuning (fast primary run)
# -----------------------------
if ft_data is not None:
    print("Starting fast fine-tuning ...")
    ft_texts = []
    for item in ft_data:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in item.get("messages", []))
        target = ""
        for m in item.get("messages", []):
            if m["role"] == "assistant":
                target = m["content"]
        ft_texts.append({"text": prompt + tokenizer.eos_token + target + tokenizer.eos_token})

    dataset = Dataset.from_list(ft_texts)
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=args.max_seq_len)
    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "ft_model"),
        per_device_train_batch_size=args.ft_batch_size,
        num_train_epochs=args.ft_epochs,
        logging_steps=50,
        save_strategy="no",
        fp16=True,
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    print("Fine-tuning completed.")

print("Ready for fast inference or evaluation!")
