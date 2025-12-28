import os
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.metrics import cohen_kappa_score
from datasets import Dataset

# Optional ICC
try:
    import pingouin as pg
    icc_available = True
except ImportError:
    print("pingouin not installed, ICC will be skipped")
    icc_available = False

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Vote prediction with LLMs (full-text generation + optional fine-tuning)")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--fine_tune_data", type=str, default=None, help="JSONL file with fine-tune dataset")
parser.add_argument("--out_dir", type=str, default="./output")
parser.add_argument("--election_year", type=int, choices=[2020, 2024], required=True)
parser.add_argument("--n_samples", type=int, default=10, help="Number of generations per prompt for probability estimation")
parser.add_argument("--sleep", type=float, default=0.1)
parser.add_argument("--save_every", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ft_epochs", type=int, default=1, help="Number of epochs for fine-tuning")
parser.add_argument("--ft_batch_size", type=int, default=4, help="Batch size for fine-tuning")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# -----------------------------
# Load datasets
# -----------------------------
with open(args.data_path, "r") as f:
    test_data = json.load(f)
print(f"Loaded {len(test_data)} test samples")

ft_data = None
if args.fine_tune_data is not None:
    # Support JSONL or JSON
    if args.fine_tune_data.endswith(".jsonl"):
        ft_data = []
        with open(args.fine_tune_data, "r") as f:
            for line in f:
                ft_data.append(json.loads(line))
    else:
        with open(args.fine_tune_data, "r") as f:
            ft_data = json.load(f)
    print(f"Loaded {len(ft_data)} fine-tune samples")

# -----------------------------
# Load model
# -----------------------------
print(f"Loading model {args.model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# pad token fix
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.eos_token_id

model.eval()
device = model.device if hasattr(model, "device") else next(model.parameters()).device

# -----------------------------
# Election year → candidates
# -----------------------------
if args.election_year == 2020:
    CANDIDATES = ["Donald Trump", "Joe Biden"]
elif args.election_year == 2024:
    CANDIDATES = ["Donald Trump", "Kamala Harris"]
else:
    raise ValueError(f"Unsupported election_year: {args.election_year}")
CANDIDATES_NORM = [c.lower() for c in CANDIDATES]

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
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    candidate_probs = {}
    for candidate in CANDIDATES:
        candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
        prob = 1.0
        current_input_ids = input_ids.clone()
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

def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)

def vote_to_numeric(vote):
    return 0 if vote.lower() == CANDIDATES[1].lower() else 1

# -----------------------------
# Fine-tuning step (optional)
# -----------------------------
if ft_data is not None:
    print("Starting fine-tuning ...")
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
        seed=args.seed,
        report_to=[]  # <-- disable wandb
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
# Inference loop
# -----------------------------
results = []
for idx, entry in tqdm(enumerate(test_data), total=len(test_data)):
    messages = entry.get("messages", [])
    gt = extract_ground_truth(messages)
    if gt is None or gt.lower() not in CANDIDATES_NORM:
        continue
    probs = get_vote_probs(messages)
    pred = max(probs, key=probs.get)
    mi = mutual_information(probs, gt)
    acc = accuracy_from_probs(probs, gt)
    results.append({
        "idx": idx,
        "messages": messages,
        "ground_truth": gt,
        "predicted_vote": pred,
        "probs": probs,
        "accuracy": acc,
        "mutual_inf": mi
    })
    if (idx+1) % args.save_every == 0:
        df_tmp = pd.DataFrame(results)
        save_path = os.path.join(args.out_dir, f"{args.model_name.replace('/', '_')}_{args.election_year}_partial.pkl")
        df_tmp.to_pickle(save_path)
        print(f"Saved intermediate results at index {idx} to {save_path}")
    time.sleep(args.sleep)

df_final = pd.DataFrame(results)
print("Inference completed.")
