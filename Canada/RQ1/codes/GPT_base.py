import os
import time
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
import openai

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
parser = argparse.ArgumentParser(description="Vote prediction with GPT models")
parser.add_argument("--model_name", type=str, required=True, help="OpenAI model name (e.g., gpt-3.5-turbo, gpt-4)")
parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="./output")
parser.add_argument("--election_year", type=int, choices=[2020, 2024], required=True)
parser.add_argument("--n_samples", type=int, default=1, help="Number of generations per prompt for probability estimation")
parser.add_argument("--sleep", type=float, default=0.1)
parser.add_argument("--save_every", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# -----------------------------
# Setup
# -----------------------------
os.makedirs(args.out_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
openai.api_key = args.api_key

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
CANDIDATES_NORM = [c.lower() for c in CANDIDATES]

# -----------------------------
# Helper functions
# -----------------------------
def strip_assistant_messages(messages):
    return [m for m in messages if m["role"] != "assistant"]

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

def get_vote_probs(messages, n_samples=1):
    """
    Estimate candidate probabilities using GPT completions.
    Deterministic if n_samples=1, stochastic if >1.
    """
    clean_msgs = strip_assistant_messages(messages)
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
    prompt += f"\nVote choice ({' or '.join(CANDIDATES)}):"

    counts = {c: 0 for c in CANDIDATES}

    for _ in range(n_samples):
        try:
            response = openai.ChatCompletion.create(
                model=args.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0 if n_samples==1 else 0.7,
                max_tokens=10
            )
            output_text = response['choices'][0]['message']['content'].strip()
            vote = normalize_vote(output_text)
            if vote:
                counts[vote] += 1
        except Exception as e:
            print(f"Error during API call: {e}")

    # normalize to probabilities with Laplace smoothing
    total_counts = sum(counts.values())
    if total_counts == 0:
        probs = {c: 1/len(CANDIDATES) for c in CANDIDATES}
    else:
        probs = {c: (counts[c] + 1)/(total_counts + len(CANDIDATES)) for c in CANDIDATES}

    return probs

def accuracy_from_probs(probs, ground_truth):
    return int(max(probs, key=probs.get) == ground_truth)

def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)

def vote_to_numeric(vote):
    return 0 if vote.lower() == CANDIDATES[1].lower() else 1

# -----------------------------
# Inference loop
# -----------------------------
results = []
for idx, entry in tqdm(enumerate(data), total=len(data)):
    messages = entry.get("messages", [])
    gt = extract_ground_truth(messages)
    if gt is None or gt.lower() not in CANDIDATES_NORM:
        continue

    probs = get_vote_probs(messages, n_samples=args.n_samples)
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

# -----------------------------
# Compute metrics
# -----------------------------
anes_votes = df_final['ground_truth'].map(vote_to_numeric).to_numpy()
gpt_votes = df_final['predicted_vote'].map(vote_to_numeric).to_numpy()

vote_metrics = {}
vote_metrics['cohen_kappa'] = cohen_kappa_score(anes_votes, gpt_votes)

if icc_available:
    try:
        df_temp = pd.DataFrame({'anes': anes_votes, 'gpt': gpt_votes})
        df_long = df_temp.reset_index().melt(id_vars='index', value_vars=['anes','gpt'],
                                            var_name='rater', value_name='vote')
        icc_df = pg.intraclass_corr(data=df_long, targets='index', raters='rater', ratings='vote')
        vote_metrics['ICC'] = icc_df.loc[icc_df['Type']=='ICC2k','ICC'].values[0]
    except Exception as e:
        print(f"Could not compute ICC: {e}")
        vote_metrics['ICC'] = None
else:
    vote_metrics['ICC'] = None

vote_metrics['proportion_agreement'] = np.mean(anes_votes == gpt_votes)

for k,v in vote_metrics.items():
    df_final[k] = v

# -----------------------------
# Merge original dataset
# -----------------------------
df_input = pd.DataFrame(data)
input_cols = [c for c in df_input.columns if c not in df_final.columns]
df_final = pd.concat([df_final.reset_index(drop=True), df_input[input_cols].reset_index(drop=True)], axis=1)

# -----------------------------
# Save final results
# -----------------------------
out_file = os.path.join(args.out_dir, f"{args.model_name.replace('/', '_')}_{args.election_year}_final.pkl")
df_final.to_pickle(out_file)
df_final.to_csv(out_file.replace(".pkl",".csv"), index=False)
print(f"Saved final results to {out_file}")

# -----------------------------
# Summary
# -----------------------------
print("\nSummary:")
MI = np.mean([r["mutual_inf"] for r in results])
print("Average accuracy:", df_final["accuracy"].mean())
print(f"Average mutual information: {MI:.3f}")
for k,v in vote_metrics.items():
    print(f"{k}: {v}")
