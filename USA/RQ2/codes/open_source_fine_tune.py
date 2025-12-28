import os
import json
import random
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score
import argparse
from tqdm import tqdm

# Optional PEFT / LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    peft_available = True
except ImportError:
    peft_available = False

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Fast fine-tune + inference with optional LoRA")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--fine_tune_data", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--election_year", type=int, choices=[2020, 2024], required=True)
parser.add_argument("--n_samples", type=int, default=10)
parser.add_argument("--ft_epochs", type=int, default=1)
parser.add_argument("--ft_batch_size", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
args = parser.parse_args()

# -----------------------------
# Setup
# -----------------------------
os.makedirs(args.out_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# -----------------------------
# Election year → candidates
# -----------------------------
if args.election_year == 2020:
    CANDIDATES = ["Donald Trump", "Joe Biden"]
elif args.election_year == 2024:
    CANDIDATES = ["Donald Trump", "Kamala Harris"]
CANDIDATES_NORM = [c.lower() for c in CANDIDATES]

# -----------------------------
# Load datasets
# -----------------------------
with open(args.data_path, "r") as f:
    test_data = json.load(f)
print(f"Loaded {len(test_data)} test samples")

ft_data = None
if args.fine_tune_data:
    with open(args.fine_tune_data, "r") as f:
        ft_data = [json.loads(line) for line in f]  # JSONL
    print(f"Loaded {len(ft_data)} fine-tune samples")

# -----------------------------
# Load model
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
# Apply LoRA if requested
# -----------------------------
if args.use_lora:
    if not peft_available:
        raise ImportError("PEFT is not installed. Install via `pip install peft` to use LoRA.")
    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA applied!")

# -----------------------------
# Helper functions
# -----------------------------
def normalize_vote(text):
    if text is None: return None
    t = text.lower()
    for c in CANDIDATES:
        if c.lower() in t:
            return c
    return None

def extract_ground_truth(messages):
    for m in messages:
        if m["role"] == "assistant":
            return normalize_vote(m["content"])
    return None

def get_vote_probs(messages, max_new_tokens=10):
    clean_msgs = [m for m in messages if m["role"] != "assistant"]
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
    prompt += f"\nVote choice ({' or '.join(CANDIDATES)}):"

    inputs_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    candidate_probs = {}
    for candidate in CANDIDATES:
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
        prob = 1.0
        current_input_ids = inputs_ids.clone()
        with torch.no_grad():
            for token_id in candidate_ids:
                outputs = model(input_ids=current_input_ids)
                logits = outputs.logits[:, -1, :]
                token_probs = torch.softmax(logits, dim=-1)
                prob *= token_probs[0, token_id].item()
                current_input_ids = torch.cat([current_input_ids, torch.tensor([[token_id]]).to(device)], dim=1)
        candidate_probs[candidate] = prob

    total = sum(candidate_probs.values())
    if total > 0:
        candidate_probs = {c: p / total for c, p in candidate_probs.items()}
    else:
        candidate_probs = {c: 1/len(CANDIDATES) for c in CANDIDATES}

    return candidate_probs

def accuracy_from_probs(probs, ground_truth):
    return int(max(probs, key=probs.get) == ground_truth)

# -----------------------------
# Fine-tuning (optional)
# -----------------------------
if ft_data:
    print("Starting fine-tuning ...")
    ft_texts = []
    for item in ft_data[:5000]:  # small subset for speed
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in item.get("messages", []))
        target = next((m["content"] for m in item.get("messages", []) if m["role"]=="assistant"), "")
        ft_texts.append({"text": prompt + tokenizer.eos_token + target + tokenizer.eos_token})

    dataset = Dataset.from_list(ft_texts)
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokenized_ds = dataset.map(tokenize_fn, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "ft_model"),
        per_device_train_batch_size=args.ft_batch_size,
        num_train_epochs=args.ft_epochs,
        logging_steps=50,
        save_strategy="no",
        fp16=True,
        seed=args.seed
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

# -----------------------------
# Inference
# -----------------------------
results = []
for idx, entry in tqdm(enumerate(test_data), total=len(test_data)):
    messages = entry.get("messages", [])
    gt = extract_ground_truth(messages)
    if gt is None or gt.lower() not in CANDIDATES_NORM:
        continue

    probs = get_vote_probs(messages)
    pred = max(probs, key=probs.get)
    acc = accuracy_from_probs(probs, gt)

    results.append({
        "idx": idx,
        "messages": messages,
        "ground_truth": gt,
        "predicted_vote": pred,
        "probs": probs,
        "accuracy": acc
    })

# -----------------------------
# Save results
# -----------------------------
out_file = os.path.join(args.out_dir, f"{args.model_name.replace('/', '_')}_{args.election_year}_results.pkl")
pd.DataFrame(results).to_pickle(out_file)
pd.DataFrame(results).to_csv(out_file.replace(".pkl",".csv"), index=False)
print(f"Saved results to {out_file}")
print("Finished!")
