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

# Pad token fix
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

CANDIDATES_NORM = [c.lower() for c in CANDIDATES]

# -----------------------------
# 3-class definition
# -----------------------------
CLASS_MAP = {
    1: CANDIDATES[0],
    2: CANDIDATES[1],
    3: "Other"
}

# -----------------------------
# Helper functions
# -----------------------------
def normalize_vote(text):
    """
    Returns:
      - candidate name if mentioned
      - 'others' otherwise
    """
    if text is None:
        return "other"

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
    """
    Deterministic single-pass probability extraction.
    Produces probabilities for:
      - Candidate A
      - Candidate B
      - Other (residual)
    """

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

    # Normalize candidate probs
    total = sum(candidate_probs.values())
    if total > 0:
        candidate_probs = {k: v / total for k, v in candidate_probs.items()}
    else:
        candidate_probs = {k: 1 / len(CANDIDATES) for k in CANDIDATES}

    # Add "Other" as residual class
    others_prob = max(0.0, 1.0 - sum(candidate_probs.values()))
    candidate_probs["Other"] = others_prob

    # Final renormalization
    total = sum(candidate_probs.values())
    candidate_probs = {k: v / total for k, v in candidate_probs.items()}

    return candidate_probs


def accuracy_from_probs(probs, ground_truth):
    return int(max(probs, key=probs.get) == ground_truth)


def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)


def vote_to_numeric(vote):
    """
    3-class numeric encoding:
      1 = Candidate A
      2 = Candidate B
      3 = Others
    """
    if vote is None:
        return 3

    v = vote.lower()
    if v == CANDIDATES[0].lower():
        return 1
    elif v == CANDIDATES[1].lower():
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
        df_tmp = pd.DataFrame(results)
        save_path = os.path.join(
            args.out_dir,
            f"{args.model_name.replace('/', '_')}_{args.election_year}_partial.pkl"
        )
        df_tmp.to_pickle(save_path)
        print(f"Saved intermediate results at index {idx}")

    time.sleep(args.sleep)

df_final = pd.DataFrame(results)

# -----------------------------
# Metrics
# -----------------------------
anes_votes = df_final["ground_truth"].map(vote_to_numeric).to_numpy()
gpt_votes = df_final["predicted_vote"].map(vote_to_numeric).to_numpy()

vote_metrics = {
    "cohen_kappa": cohen_kappa_score(anes_votes, gpt_votes),
    "proportion_agreement": np.mean(anes_votes == gpt_votes)
}

if icc_available:
    try:
        df_temp = pd.DataFrame({"anes": anes_votes, "gpt": gpt_votes})
        df_long = df_temp.reset_index().melt(
            id_vars="index",
            value_vars=["anes", "gpt"],
            var_name="rater",
            value_name="vote"
        )
        icc_df = pg.intraclass_corr(
            data=df_long,
            targets="index",
            raters="rater",
            ratings="vote"
        )
        vote_metrics["ICC"] = icc_df.loc[
            icc_df["Type"] == "ICC2k", "ICC"
        ].values[0]
    except Exception as e:
        print(f"ICC computation failed: {e}")
        vote_metrics["ICC"] = None
else:
    vote_metrics["ICC"] = None

for k, v in vote_metrics.items():
    df_final[k] = v

# -----------------------------
# Merge original input
# -----------------------------
df_input = pd.DataFrame(data)
extra_cols = [c for c in df_input.columns if c not in df_final.columns]
df_final = pd.concat(
    [df_final.reset_index(drop=True), df_input[extra_cols].reset_index(drop=True)],
    axis=1
)

# -----------------------------
# Save results
# -----------------------------
out_file = os.path.join(
    args.out_dir,
    f"{args.model_name.replace('/', '_')}_{args.election_year}_final.pkl"
)

df_final.to_pickle(out_file)
df_final.to_csv(out_file.replace(".pkl", ".csv"), index=False)

print(f"Saved final results to {out_file}")

# -----------------------------
# Summary
# -----------------------------
print("\nSummary:")
print("Average accuracy:", df_final["accuracy"].mean())
print("Average mutual information:", df_final["mutual_inf"].mean())
for k, v in vote_metrics.items():
    print(f"{k}: {v}")

print("\n⚠️ Note: ICC is not theoretically ideal for 3-class nominal outcomes.")
