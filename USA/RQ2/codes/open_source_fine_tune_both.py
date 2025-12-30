
import os
import json
import random
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import argparse
from tqdm import tqdm

# Optional PEFT / LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    peft_available = True
except ImportError:
    peft_available = False


# =============================
# Arguments
# =============================
parser = argparse.ArgumentParser(description="Fine-tune once, evaluate on ANES 2020 + 2024")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--anes_2020_path", type=str, required=True)
parser.add_argument("--anes_2024_path", type=str, required=True)
parser.add_argument("--fine_tune_data", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--ft_epochs", type=int, default=1)
parser.add_argument("--ft_batch_size", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_lora", action="store_true")
args = parser.parse_args()


# =============================
# Setup
# =============================
os.makedirs(args.out_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# =============================
# Load test datasets
# =============================
with open(args.anes_2020_path, "r") as f:
    anes_2020 = json.load(f)

with open(args.anes_2024_path, "r") as f:
    anes_2024 = json.load(f)

test_sets = {
    2020: anes_2020,
    2024: anes_2024
}

print(f"Loaded {len(anes_2020)} ANES 2020 samples")
print(f"Loaded {len(anes_2024)} ANES 2024 samples")


# =============================
# Load fine-tuning data
# =============================
ft_data = None
if args.fine_tune_data:
    with open(args.fine_tune_data, "r") as f:
        ft_data = [json.loads(line) for line in f]
    print(f"Loaded {len(ft_data)} fine-tuning samples")


# =============================
# Load model
# =============================
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
device = next(model.parameters()).device


# =============================
# Apply LoRA (optional)
# =============================
if args.use_lora:
    if not peft_available:
        raise ImportError("Install peft to use LoRA")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_cfg)
    print("LoRA adapters applied")


# =============================
# Helper functions
# =============================
def get_candidates(year):
    if year == 2020:
        return ["Donald Trump", "Joe Biden"]
    return ["Donald Trump", "Kamala Harris"]


def normalize_vote(text, candidates):
    if text is None:
        return None
    t = text.lower()
    for c in candidates:
        if c.lower() in t:
            return c
    return None


def extract_ground_truth(messages, candidates):
    for m in messages:
        if m["role"] == "assistant":
            return normalize_vote(m["content"], candidates)
    return None


def get_vote_probs(messages, candidates):
    clean = [m for m in messages if m["role"] != "assistant"]
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean)
    prompt += f"\nVote choice ({' or '.join(candidates)}):"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    probs = {}

    for cand in candidates:
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        p = 1.0
        cur = input_ids.clone()

        with torch.no_grad():
            for tok in cand_ids:
                out = model(input_ids=cur)
                logits = out.logits[:, -1, :]
                token_probs = torch.softmax(logits, dim=-1)
                p *= token_probs[0, tok].item()
                cur = torch.cat([cur, torch.tensor([[tok]]).to(device)], dim=1)

        probs[cand] = p

    Z = sum(probs.values())
    return {c: p / Z for c, p in probs.items()} if Z > 0 else {c: 1/len(candidates) for c in candidates}


def accuracy_from_probs(probs, gt):
    return int(max(probs, key=probs.get) == gt)


def mutual_information(probs, gt, eps=1e-12):
    return -np.log2(max(probs.get(gt, eps), eps))


def safe_name(s):
    return str(s).replace("/", "_").replace(" ", "_")


# =============================
# Fine-tuning (ONCE)
# =============================
if ft_data:
    ft_texts = []
    for item in ft_data:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in item["messages"])
        target = next(m["content"] for m in item["messages"] if m["role"] == "assistant")
        ft_texts.append({"text": prompt + tokenizer.eos_token + target + tokenizer.eos_token})

    dataset = Dataset.from_list(ft_texts)

    def tokenize_fn(ex):
        return tokenizer(ex["text"], truncation=True, padding="max_length", max_length=192)

    tokenized = dataset.map(tokenize_fn, batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "ft_model"),
        per_device_train_batch_size=args.ft_batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=args.ft_epochs,
        save_strategy="no",
        logging_steps=50,
        bf16=True,
        seed=args.seed,
        learning_rate=1e-4,
        warmup_ratio=0.1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=collator
    )

    trainer.train()
    print("Fine-tuning completed")


# =============================
# Evaluation (SEPARATE .pkl PER DATASET)
# =============================
for year, test_data in test_sets.items():
    print(f"\nEvaluating ANES {year}")

    CANDIDATES = get_candidates(year)
    CANDIDATES_NORM = [c.lower() for c in CANDIDATES]

    results = []

    for idx, entry in tqdm(enumerate(test_data), total=len(test_data)):
        messages = entry["messages"]
        gt = extract_ground_truth(messages, CANDIDATES)
        if gt is None or gt.lower() not in CANDIDATES_NORM:
            continue

        probs = get_vote_probs(messages, CANDIDATES)
        pred = max(probs, key=probs.get)

        results.append({
            "idx": idx,
            "messages": messages,
            "ground_truth": gt,
            "predicted_vote": pred,
            "probs": probs,
            "accuracy": accuracy_from_probs(probs, gt),
            "mutual_inf": mutual_information(probs, gt)
        })

    df_final = pd.DataFrame(results)

    base = (
        f"{safe_name(args.model_name)}"
        f"_FT_tested_on_ANES_{year}"
    )

    pkl_path = os.path.join(args.out_dir, base + ".pkl")
    csv_path = os.path.join(args.out_dir, base + ".csv")

    df_final.to_pickle(pkl_path)
    df_final.to_csv(csv_path, index=False)

    print(f"Saved results:\n  {pkl_path}")

print("\nAll evaluations completed successfully.")
