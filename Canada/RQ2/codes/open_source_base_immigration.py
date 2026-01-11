# =====================================================
# Canada 2021 Immigration Attitude Prediction with LLMs
# Deterministic likelihood-based evaluation
# =====================================================

import os
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import cohen_kappa_score

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
    description="Canada 2021 immigration attitude prediction using LLM likelihoods"
)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="./output")
parser.add_argument("--election_year", type=str, default="2021")
parser.add_argument("--save_every", type=int, default=100)
parser.add_argument("--sleep", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


# =====================================================
# Immigration response labels (MUST match dataset)
# =====================================================
IMMIGRATION_CHOICES = [
    "More immigrants",
    "Fewer immigrants",
    "About the same number of immigrants as now"
]

IMMIGRATION_LOWER = [c.lower() for c in IMMIGRATION_CHOICES]
IMM2ID = {c: i for i, c in enumerate(IMMIGRATION_CHOICES)}
ID2IMM = {i: c for c, i in IMM2ID.items()}


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
def normalize_immigration(text):
    """Extract immigration stance from assistant text."""
    if text is None:
        return None
    text = text.lower()
    for c in IMMIGRATION_CHOICES:
        if c.lower() in text:
            return c
    return None


def extract_ground_truth(messages):
    """Ground truth is the assistant message."""
    for m in messages:
        if m["role"] == "assistant":
            return normalize_immigration(m["content"])
    return None


def get_immigration_probs(messages):
    """
    Deterministic token-level probability for each immigration option.
    """
    clean_msgs = [m for m in messages if m["role"] != "assistant"]

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
    prompt += (
        "\nImmigration stance "
        f"({' or '.join(IMMIGRATION_CHOICES)}):"
    )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    probs = {}

    with torch.no_grad():
        for choice in IMMIGRATION_CHOICES:
            choice_ids = tokenizer.encode(choice, add_special_tokens=False)
            prob = 1.0
            cur_ids = input_ids.clone()

            for tok in choice_ids:
                outputs = model(input_ids=cur_ids)
                logits = outputs.logits[:, -1, :]
                token_probs = torch.softmax(logits, dim=-1)
                prob *= token_probs[0, tok].item()
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[tok]], device=device)], dim=1
                )

            probs[choice] = prob

    total = sum(probs.values())
    if total > 0:
        probs = {c: v / total for c, v in probs.items()}
    else:
        probs = {c: 1 / len(IMMIGRATION_CHOICES) for c in IMMIGRATION_CHOICES}

    return probs


def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)


def imm_to_numeric(choice):
    return IMM2ID[choice]


# =====================================================
# Inference loop
# =====================================================
results = []

for idx, entry in tqdm(enumerate(data), total=len(data)):
    messages = entry.get("messages", [])
    gt = extract_ground_truth(messages)

    if gt is None or gt not in IMMIGRATION_CHOICES:
        continue

    probs = get_immigration_probs(messages)
    pred = max(probs, key=probs.get)

    results.append({
        "idx": idx,
        "ground_truth": gt,
        "predicted_immigration": pred,
        "accuracy": int(pred == gt),
        "mutual_information": mutual_information(probs, gt),
        "probs": probs,
        "messages": messages,
    })

    if (idx + 1) % args.save_every == 0:
        pd.DataFrame(results).to_pickle(
            os.path.join(
                args.out_dir,
                f"{args.model_name.replace('/', '_')}_{args.election_year}_immigration_partial.pkl"
            )
        )

    time.sleep(args.sleep)

df = pd.DataFrame(results)


# =====================================================
# Metrics
# =====================================================
human = df["ground_truth"].map(imm_to_numeric).to_numpy()
model_preds = df["predicted_immigration"].map(imm_to_numeric).to_numpy()

metrics = {
    "accuracy": df["accuracy"].mean(),
    "cohen_kappa": cohen_kappa_score(human, model_preds),
    "proportion_agreement": np.mean(human == model_preds),
    "mean_mutual_information": df["mutual_information"].mean(),
}

if ICC_AVAILABLE:
    try:
        df_long = (
            pd.DataFrame({"human": human, "model": model_preds})
            .reset_index()
            .melt(id_vars="index", var_name="rater", value_name="rating")
        )
        icc = pg.intraclass_corr(
            data=df_long,
            targets="index",
            raters="rater",
            ratings="rating"
        )
        metrics["ICC"] = icc.loc[icc["Type"] == "ICC2k", "ICC"].values[0]
    except Exception:
        metrics["ICC"] = None
else:
    metrics["ICC"] = None

for k, v in metrics.items():
    df[k] = v


# =====================================================
# Save outputs
# =====================================================
out_base = f"{args.model_name.replace('/', '_')}_{args.election_year}_immigration_final"
out_pkl = os.path.join(args.out_dir, out_base + ".pkl")
out_csv = os.path.join(args.out_dir, out_base + ".csv")

df.to_pickle(out_pkl)
df.to_csv(out_csv, index=False)

print("\n=== Immigration Attitude Prediction Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

print(f"\nSaved results to:\n{out_pkl}\n{out_csv}")
