import os
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from openai import OpenAI

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Vote prediction with LLMs (3-class)")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--api_key", type=str, required=True)
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

# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI(api_key=args.api_key)

# -----------------------------
# Load dataset
# -----------------------------
with open(args.data_path, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

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

    prompt = (
        "You are an expert political analyst. "
        "Based on the demographic profile below, predict the vote choice.\n\n"
    )
    for m in clean_msgs:
        prompt += f"{m['role']}: {m['content']}\n"
    prompt += f"\nVote choice ({' or '.join(CANDIDATES)} or Other):"

    response = client.chat.completions.create(
        model=args.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        logprobs=True,
        top_logprobs=5
    )

    logprobs = response.choices[0].logprobs.content[0].top_logprobs

    probs = {c: 0.0 for c in CANDIDATES}
    probs["Other"] = 0.0

    for lp in logprobs:
        token = lp.token.lower()
        prob = np.exp(lp.logprob)

        for c in CANDIDATES:
            if c.lower() in token:
                probs[c] += prob
        if "other" in token:
            probs["Other"] += prob

    total = sum(probs.values())
    if total == 0:
        probs = {k: 1 / 3 for k in probs}
    else:
        probs = {k: v / total for k, v in probs.items()}

    return probs


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
print("\nAverage accuracy:", df_final["accuracy"].mean())
print("Tetrachoric correlation:", tetra)
print("Bias on Trump:", df_summary["Bias_Donald Trump"][0])
print("Average mutual information:", df_final["mutual_inf"].mean())
print(f"\nSaved df_summary to: {summary_path}")
