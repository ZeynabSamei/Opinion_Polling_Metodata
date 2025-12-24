import os
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import cohen_kappa_score

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Vote prediction with LLMs (3-class)")
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
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.eos_token_id

model.eval()

device = (
    model.device
    if hasattr(model, "device")
    else next(model.parameters()).device
)

# -----------------------------
# Election year → candidates
# -----------------------------
if args.election_year == 2020:
    CANDIDATES = ["Donald Trump", "Joe Biden"]
elif args.election_year == 2024:
    CANDIDATES = ["Donald Trump", "Kamala Harris"]
else:
    raise ValueError(f"Unsupported election_year: {args.election_year}")

# -----------------------------
# Helper functions
# -----------------------------
def normalize_vote(text):
    if text is None:
        return "Other"
    t = text.lower()
    for c in CANDIDATES:
        if c.lower() in t:
            return c
    return "Other"


def extract_ground_truth(messages):
    for m in messages:
        if m["role"] == "assistant":
            return normalize_vote(m["content"])
    return "Other"


def get_vote_probs(messages):
    clean_msgs = [m for m in messages if m["role"] != "assistant"]
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
    prompt += f"\nVote choice ({' or '.join(CANDIDATES)}):"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    candidate_probs = {}

    with torch.no_grad():
        for candidate in CANDIDATES:
            cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
            prob = 1.0
            curr_ids = input_ids.clone()

            for token_id in cand_ids:
                outputs = model(input_ids=curr_ids)
                logits = outputs.logits[:, -1, :]
                token_probs = torch.softmax(logits, dim=-1)
                prob *= token_probs[0, token_id].item()

                curr_ids = torch.cat(
                    [curr_ids, torch.tensor([[token_id]], device=device)], dim=1
                )

            candidate_probs[candidate] = prob

    total = sum(candidate_probs.values())
    if total > 0:
        candidate_probs = {k: v / total for k, v in candidate_probs.items()}
    else:
        candidate_probs = {k: 1 / len(CANDIDATES) for k in CANDIDATES}

    others_prob = max(0.0, 1.0 - sum(candidate_probs.values()))
    candidate_probs["Other"] = others_prob

    total = sum(candidate_probs.values())
    candidate_probs = {k: v / total for k, v in candidate_probs.items()}

    return candidate_probs


def accuracy_from_probs(probs, ground_truth):
    return int(max(probs, key=probs.get) == ground_truth)


def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)


def vote_to_numeric(vote):
    if vote == CANDIDATES[0]:
        return 1
    elif vote == CANDIDATES[1]:
        return 2
    else:
        return 3

# -----------------------------
# Inference loop
# -----------------------------
results = []

for idx, entry in tqdm(enumerate(data), total=len(data)):
    messages = entry.get("messages", [])
    gt = extract_ground_truth(messages)

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
# Binary encoding (for tetra + bias)
# -----------------------------
df_final["llm_binary"] = df_final["predicted_vote"]
df_final["anes_binary"] = df_final["ground_truth"]

df_final["llm_num"] = df_final["llm_binary"].apply(
    lambda x: 0 if x == CANDIDATES[1] else 1
)
df_final["anes_num"] = df_final["anes_binary"].apply(
    lambda x: 0 if x == CANDIDATES[1] else 1
)

# -----------------------------
# Tetrachoric correlation
# -----------------------------
def tetrachoric_corr_safe(vec1, vec2):
    A = np.sum((vec1 == 0) & (vec2 == 0))
    B = np.sum((vec1 == 0) & (vec2 == 1))
    C = np.sum((vec1 == 1) & (vec2 == 0))
    D = np.sum((vec1 == 1) & (vec2 == 1))
    if (A+B)==0 or (C+D)==0 or (A+C)==0 or (B+D)==0:
        return np.nan
    try:
        return np.cos(np.pi / (1 + np.sqrt((A*D)/(B*C))))
    except:
        return np.nan

tetra = tetrachoric_corr_safe(
    df_final["llm_num"].values,
    df_final["anes_num"].values
)

# -----------------------------
# Bias computation
# -----------------------------
summary_rows = []
row = {
    "Variable": "Wholesample",
    "n_samples": len(df_final),
    "Tetra": tetra,
    "Prop.Agree": np.mean(df_final["llm_binary"] == df_final["anes_binary"])
}

for c in CANDIDATES:
    real_pct = np.mean(df_final["anes_binary"] == c)
    llm_pct = np.mean([p[c] for p in df_final["probs"]])
    row[f"RealPct_{c}"] = real_pct
    row[f"LLMPct_{c}"] = llm_pct
    row[f"Bias_{c}"] = llm_pct - real_pct

summary_rows.append(row)
df_summary = pd.DataFrame(summary_rows)

# -----------------------------
# Save outputs
# -----------------------------
final_path = os.path.join(
    args.out_dir,
    f"{args.model_name.replace('/', '_')}_{args.election_year}_final.pkl"
)
summary_path = final_path.replace(".pkl", "_summary.csv")

df_final.to_pickle(final_path)
df_final.to_csv(final_path.replace(".pkl", ".csv"), index=False)
df_summary.to_csv(summary_path, index=False)

# -----------------------------
# Print summary
# -----------------------------
print("\nSummary:")
print(df_summary)
print("\nAverage accuracy:", df_final["accuracy"].mean())
print("Average mutual information:", df_final["mutual_inf"].mean())
print("Tetrachoric correlation:", tetra)
print(f"\nSaved df_summary to: {summary_path}")
