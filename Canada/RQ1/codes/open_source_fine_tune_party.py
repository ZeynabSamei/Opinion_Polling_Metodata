# =====================================================
# Canada 2021 Party Choice Prediction with LLMs
# Deterministic likelihood-based evaluation
# Optional fine-tuning
# =====================================================

import os
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.metrics import cohen_kappa_score
from datasets import Dataset

# Optional ICC
try:
    import pingouin as pg
    ICC_AVAILABLE = True
except ImportError:
    ICC_AVAILABLE = False

# =====================================================
# Arguments
# =====================================================
parser = argparse.ArgumentParser(
    description="Canada 2021 party choice prediction using LLM likelihoods, with optional fine-tuning"
)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--fine_tune_data", type=str, default=None, help="JSON file with fine-tune dataset")
parser.add_argument("--out_dir", type=str, default="./output")
parser.add_argument("--election_year", type=str, default="2021")
parser.add_argument("--save_every", type=int, default=100)
parser.add_argument("--sleep", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--ft_epochs", type=int, default=1, help="Fine-tuning epochs")
parser.add_argument("--ft_batch_size", type=int, default=4, help="Fine-tuning batch size")

args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# =====================================================
# Party labels
# =====================================================
PARTIES = [
    "Liberal Party",
    "Conservative Party",
    "NDP",
    "Bloc Québécois Party",
    "Green Party",
    "People's Party of Canada"
]

PARTIES_LOWER = [p.lower() for p in PARTIES]
PARTY2ID = {p: i for i, p in enumerate(PARTIES)}
ID2PARTY = {i: p for p, i in PARTY2ID.items()}

# =====================================================
# Load dataset
# =====================================================
with open(args.data_path, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

# =====================================================
# Load model
# =====================================================
print(f"Loading model: {args.model_name}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

# Pad token fix
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
model.eval()

device = next(model.parameters()).device

# =====================================================
# Helper functions
# =====================================================
def normalize_party(text):
    if text is None:
        return None
    text = text.lower()
    for p in PARTIES:
        if p.lower() in text:
            return p
    return None

def extract_ground_truth(messages):
    for m in messages:
        if m["role"] == "assistant":
            return normalize_party(m["content"])
    return None

def get_party_probs(messages):
    clean_msgs = [m for m in messages if m["role"] != "assistant"]
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
    prompt += f"\nParty choice ({' or '.join(PARTIES)}):"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    probs = {}
    with torch.no_grad():
        for party in PARTIES:
            party_ids = tokenizer.encode(party, add_special_tokens=False)
            prob = 1.0
            cur_ids = input_ids.clone()
            for tok in party_ids:
                outputs = model(input_ids=cur_ids)
                logits = outputs.logits[:, -1, :]
                token_probs = torch.softmax(logits, dim=-1)
                prob *= token_probs[0, tok].item()
                cur_ids = torch.cat([cur_ids, torch.tensor([[tok]], device=device)], dim=1)
            probs[party] = prob

    total = sum(probs.values())
    if total > 0:
        probs = {p: v / total for p, v in probs.items()}
    else:
        probs = {p: 1 / len(PARTIES) for p in PARTIES}

    return probs

def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)

def party_to_numeric(party):
    return PARTY2ID[party]

# =====================================================
# Fine-tuning (optional)
# =====================================================
if args.fine_tune_data is not None:
    print("Starting fine-tuning ...")
    with open(args.fine_tune_data, "r") as f:
        ft_data = json.load(f)
    print(f"Loaded {len(ft_data)} fine-tune samples")

    ft_texts = []
    for item in ft_data:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in item.get("messages", []))
        target = ""
        for m in item.get("messages", []):
            if m["role"]
