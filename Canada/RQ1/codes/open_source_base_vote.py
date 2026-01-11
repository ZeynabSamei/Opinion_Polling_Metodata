# =====================================================
# Canada 2021 Election Vote Prediction with LLMs
# Deterministic token-level probability evaluation
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
    description="Canada 2021 vote prediction using LLM likelihoods"
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
# Candidates (Canada 2021 leaders)
# =====================================================
CANDIDATES = [
    "Justin Trudeau",
    "Erin O'Toole",
    "Jagmeet Singh",
    "Yves-François Blanchet",
    "Annamie Paul",
    "Maxime Bernier",
]

CANDIDATES_LOWER = [c.lower() for c in CANDIDATES]
VOTE2ID = {c: i for i, c in enumerate(CANDIDATES)}
ID2VOTE = {i: c for c, i in VOTE2ID.items()}


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
def normalize_vote(text):
    """Extract candidate name from generated text."""
    if text is None:
        return None
    text = text.lower()
    for c in CANDIDATES:
        if c.lower() in text:
            return c
    return None


def extract_ground_truth(messages):
    """Ground truth is stored in assistant turn."""
    for m in messages:
        if m["role"] == "assistant":
            return normalize_vote(m["content"])
    return None


def get_vote_probs(messages):
    """
    Deterministic token-level probability of each candidate.
    Matches likelihood-based evaluation in the paper.
    """
    # Remove assistant messages
    clean_msgs = [m for m in messages if m["role"] != "assistant"]

    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
    prompt += f"\nVote choice ({' or '.join(CANDIDATES)}):"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    probs = {}

    with torch.no_grad():
        for candidate in CANDIDATES:
            cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
            prob = 1.0
            cur_ids = input_ids.clone()

            for tok in cand_ids:
                outputs = model(input_ids=cur_ids)
                logits = outputs.logits[:, -1, :]
                token_probs = torch.softmax(logits, dim=-1)
                prob *= token_probs[0, tok].item()
                cur_ids = torch.cat(
                    [cur_ids, torch.tensor([[tok]], device=device)], dim=1
                )

            probs[candidate] = prob

    total = sum(probs.values())
    if total > 0:
        probs = {c: p / total for c, p in probs.items()}
    else:
        probs = {c: 1 / len(CANDIDATES) for c in CANDIDATES}

    return probs


def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)


def vote_to_numeric(vote):
    return VOTE2ID[vote]


# =====================================================
# Inference loop
# =====================================================
results = []

for idx, entry in tqdm(enumerate(data), total=len(data)):
    messages = entry.get("messages", [])
    gt = extract_ground_truth(messages)

    if gt is None or gt not in CANDIDATES:
        continue

    probs = get_vote_probs(messages)
    pred = max(probs, key=probs.get)

    results.append({
        "idx": idx,
        "ground_truth": gt,
        "predicted_vote": pred,
        "accuracy": int(pred == gt),
        "mutual_information": mutual_information(probs, gt),
        "probs": probs,
        "messages": messages,
    })

    if (idx + 1) % args.save_every == 0:
        pd.DataFrame(results).to_pickle(
            os.path.join(
                args.out_dir,
                f"{args.model_name.replace('/', '_')}_{args.election_year}_partial.pkl"
            )
        )

    time.sleep(args.sleep)

df = pd.DataFrame(results)


# =====================================================
# Metrics
# =====================================================
anes_votes = df["ground_truth"].map(vote_to_numeric).to_numpy()
gpt_votes = df["predicted_vote"].map(vote_to_numeric).to_numpy()

metrics = {
    "accuracy": df["accuracy"].mean(),
    "cohen_kappa": cohen_kappa_score(anes_votes, gpt_votes),
    "proportion_agreement": np.mean(anes_votes == gpt_votes),
    "mean_mutual_information": df["mutual_information"].mean(),
}

if ICC_AVAILABLE:
    try:
        df_long = (
            pd.DataFrame({"anes": anes_votes, "gpt": gpt_votes})
            .reset_index()
            .melt(id_vars="index", var_name="rater", value_name="vote")
        )
        icc = pg.intraclass_corr(
            data=df_long,
            targets="index",
            raters="rater",
            ratings="vote"
        )
        metrics["ICC"] = icc.loc[icc["Type"] == "ICC2k", "ICC"].values[0]
    except Exception:
        metrics["ICC"] = None
else:
    metrics["ICC"] = None

for k, v in metrics.items():
    df[k] = v


# =====================================================
# Save final outputs
# =====================================================
out_base = f"{args.model_name.replace('/', '_')}_{args.election_year}_final"
out_pkl = os.path.join(args.out_dir, out_base + ".pkl")
out_csv = os.path.join(args.out_dir, out_base + ".csv")

df.to_pickle(out_pkl)
df.to_csv(out_csv, index=False)

print("\n=== Final Metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

print(f"\nSaved results to:\n{out_pkl}\n{out_csv}")
