import os
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import wasserstein_distance

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Vote prediction with LLMs (2-class)")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="./output")
parser.add_argument("--election_year", type=int, choices=[2020, 2024], required=True)
parser.add_argument("--sleep", type=float, default=0.1)
parser.add_argument("--save_every", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# -----------------------------
# Load dataset
# -----------------------------
with open(args.data_path, "r") as f:
    data = json.load(f)
print(f"Loaded {len(data)} samples")

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

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.eos_token_id

model.eval()
device = model.device if hasattr(model, "device") else next(model.parameters()).device

# -----------------------------
# Election year → candidates (BINARY ONLY)
# -----------------------------
if args.election_year == 2020:
    CANDIDATES = ["Donald Trump", "Joe Biden"]
elif args.election_year == 2024:
    CANDIDATES = ["Donald Trump", "Kamala Harris"]
else:
    raise ValueError("Unsupported election year")

VOTE2IDX = {c: i for i, c in enumerate(CANDIDATES)}

# -----------------------------
# Helper functions
# -----------------------------
def normalize_vote(text):
    if text is None:
        return None
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


def get_vote_probs(messages):
    clean_msgs = [m for m in messages if m["role"] != "assistant"]
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
    prompt += f"\nVote choice ({' or '.join(CANDIDATES)}):"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    probs = {}

    with torch.no_grad():
        for candidate in CANDIDATES:
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)
            p = 1.0
            curr = input_ids.clone()

            for tid in token_ids:
                logits = model(curr).logits[:, -1, :]
                token_probs = torch.softmax(logits, dim=-1)
                p *= token_probs[0, tid].item()
                curr = torch.cat([curr, torch.tensor([[tid]], device=device)], dim=1)

            probs[candidate] = p

    # Binary normalization
    Z = sum(probs.values())
    if Z > 0:
        probs = {k: v / Z for k, v in probs.items()}
    else:
        probs = {c: 0.5 for c in CANDIDATES}

    return probs


def accuracy_from_probs(probs, ground_truth):
    return int(max(probs, key=probs.get) == ground_truth)


def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)

# -----------------------------
# Inference loop
# -----------------------------
results = []

for idx, entry in tqdm(enumerate(data), total=len(data)):
    messages = entry.get("messages", [])
    gt = extract_ground_truth(messages)
    if gt is None:
        continue

    probs = get_vote_probs(messages)
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

    if (idx + 1) % args.save_every == 0:
        pd.DataFrame(results).to_pickle(
            os.path.join(
                args.out_dir,
                f"{args.model_name.replace('/', '_')}_{args.election_year}_partial.pkl"
            )
        )

    time.sleep(args.sleep)

df_final = pd.DataFrame(results)

# -----------------------------
# Wasserstein distance (BINARY)
# -----------------------------
def sample_wasserstein(probs, ground_truth):
    gt_vec = np.zeros(2)
    gt_vec[VOTE2IDX[ground_truth]] = 1.0
    pred_vec = np.array([probs[c] for c in CANDIDATES])
    return wasserstein_distance(gt_vec, pred_vec)

df_final["wasserstein"] = df_final.apply(
    lambda r: sample_wasserstein(r["probs"], r["ground_truth"]),
    axis=1
)

# -----------------------------
# Summary
# -----------------------------
summary = {
    "n_samples": len(df_final),
    "Avg_Accuracy": df_final["accuracy"].mean(),
    "Avg_Mutual_Info": df_final["mutual_inf"].mean(),
    "Avg_Wasserstein": df_final["wasserstein"].mean(),
}

for c in CANDIDATES:
    real_pct = np.mean(df_final["ground_truth"] == c)
    llm_pct = np.mean([p[c] for p in df_final["probs"]])
    summary[f"RealPct_{c}"] = real_pct
    summary[f"LLMPct_{c}"] = llm_pct
    summary[f"Bias_{c}"] = llm_pct - real_pct

df_summary = pd.DataFrame([summary])

# -----------------------------
# Save outputs
# -----------------------------
final_path = os.path.join(
    args.out_dir,
    f"{args.model_name.replace('/', '_')}_{args.election_year}_final.pkl"
)
summary_path = final_path.replace(".pkl", "_summary.csv")

df_final.to_pickle(final_path)
df_summary.to_csv(summary_path, index=False)

# -----------------------------
# Print summary
# -----------------------------
print("\nSummary:")
print(df_summary.to_string(index=False))
