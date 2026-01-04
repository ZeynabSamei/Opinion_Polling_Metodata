import os
import json
import pickle
import random
import argparse
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# ===============================
# Arguments
# ===============================
parser = argparse.ArgumentParser(
    description="Fine-tune (optional) + Evaluate LLM on ANES interview prompts"
)

parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--fine_tune_data", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--election_year", type=int, choices=[2020, 2024], required=True)
parser.add_argument("--ft_epochs", type=int, default=1)
parser.add_argument("--ft_batch_size", type=int, default=4)
parser.add_argument("--max_seq_length", type=int, default=192)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_lora", action="store_true")

args = parser.parse_args()

# ===============================
# Setup
# ===============================
os.makedirs(args.out_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Allowed answers (UNCHANGED)
# ===============================
ALLOWED_ANSWERS = {
    "gender": ["male", "female"],
    "race": ["white", "black", "hispanic", "asian", "native american", "mixed race"],
    "church_attendance": ["yes", "no"],
    "pol_interest": [
        "very interested",
        "somewhat interested",
        "not very interested",
        "not at all interested"
    ],
    "vote_choice": [
        "kamala harris",
        "donald trump",
        "someone else"
    ],
    "ideology": [
        "extremely liberal",
        "liberal",
        "slightly liberal",
        "moderate",
        "slightly conservative",
        "conservative",
        "extremely conservative"
    ]
}

# ===============================
# Load Interview Data
# ===============================
with open(args.data_path) as f:
    interviews = json.load(f)

print(f"Loaded {len(interviews)} interview prompts")

# ===============================
# Load model
# ===============================
print(f"Loading model: {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)
model.eval()

# ===============================
# LoRA (optional)
# ===============================
if args.use_lora:
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA enabled")

# ===============================
# Fine-tuning (ONCE)
# ===============================
if args.fine_tune_data:

    print("Starting fine-tuning...")

    def build_ft_text(item):
        # Case 1: chat format
        if "messages" in item:
            return "\n".join(
                f"{m['role']}: {m['content']}"
                for m in item["messages"]
            )

        # Case 2: completion format
        if "prompt" in item and "completion" in item:
            return item["prompt"] + item["completion"]

        return None

    ft_samples = []
    with open(args.fine_tune_data) as f:
        for line in f:
            item = json.loads(line)
            text = build_ft_text(item)
            if text:
                ft_samples.append({"text": text + tokenizer.eos_token})

    dataset = Dataset.from_list(ft_samples)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length
        )

    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(args.out_dir, "ft_model"),
            per_device_train_batch_size=args.ft_batch_size,
            num_train_epochs=args.ft_epochs,
            learning_rate=1e-4,
            logging_steps=50,
            save_strategy="no",
            seed=args.seed,
            bf16=True if DEVICE == "cuda" else False
        ),
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )

    trainer.train()
    print("Fine-tuning completed")

# ===============================
# Inference (UNCHANGED LOGIC)
# ===============================
results = []

for item in tqdm(interviews):

    system_msg = item["messages"][0]["content"]
    user_msg = item["messages"][1]["content"]
    target = item["omitted_feature"]

    raw_value = item["features_raw"][target]

    if target in ALLOWED_ANSWERS:
        try:
            ground_truth = ALLOWED_ANSWERS[target][int(raw_value) - 1]
        except Exception:
            ground_truth = str(raw_value)
    else:
        ground_truth = str(raw_value)

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    pred_raw = decoded.strip()
    pred_norm = pred_raw.lower().strip()

    valid = True
    if target in ALLOWED_ANSWERS:
        valid = pred_norm in ALLOWED_ANSWERS[target]

    results.append({
        "election_year": args.election_year,
        "model": args.model_name,
        "omitted_feature": target,
        "ground_truth": ground_truth,
        "prompt": prompt,
        "prediction_raw": pred_raw,
        "prediction_norm": pred_norm,
        "valid": valid,
        "features_raw": item["features_raw"]
    })

# ===============================
# Save output (IDENTICAL)
# ===============================
# out_pkl = Path(args.out_dir) / f"anes_{args.election_year}_{args.model_name.replace('/', '_')}_interview.pkl"
out_pkl = os.path.join(
    args.out_dir,
    f"{safe_name(args.model_name)}_"
    f"{args.election_year}_"
    f"{safe_name(args.fine_tune_data)}_results.pkl"
)

df = pd.DataFrame(results)
df.to_pickle(out_pkl)

print(f"Saved results to: {out_pkl}")
print(f"Total rows: {len(df)}")
