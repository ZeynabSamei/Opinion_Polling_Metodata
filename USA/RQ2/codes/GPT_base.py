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
    ICC_AVAILABLE = True
except ImportError:
    print("pingouin not installed, ICC will be skipped")
    ICC_AVAILABLE = False


# -----------------------------
# Arguments
# -----------------------------
def parse_args():
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
    return parser.parse_args()


# -----------------------------
# Setup
# -----------------------------
def setup_environment(args):
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    openai.api_key = args.api_key


# -----------------------------
# Load dataset
# -----------------------------
def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    return data


# -----------------------------
# Candidates setup
# -----------------------------
def get_candidates(year):
    if year == 2020:
        candidates = ["Donald Trump", "Joe Biden"]
    elif year == 2024:
        candidates = ["Donald Trump", "Kamala Harris"]
    else:
        raise ValueError(f"Unsupported election_year: {year}")
    return candidates, [c.lower() for c in candidates]


# -----------------------------
# Helper functions
# -----------------------------
def normalize_vote(text, candidates):
    if text is None:
        return None
    t = text.lower()
    for c in candidates:
        if c.lower() in t:
            return c
    return None


def extract_ground_truth(messages, candidates):
    for m in messages:
        if m["role"] == "assistant":
            return normalize_vote(m["content"], candidates)
    return None


def get_vote_probs(messages, candidates, client, model_name, n_samples=1, temperature=0.7):
    """Estimate candidate probabilities using GPT completions."""
    clean_msgs = [m for m in messages if m["role"] != "assistant"]
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
    prompt += f"\nVote choice ({' or '.join(candidates)}):"

    counts = {c: 0 for c in candidates}

    for _ in range(n_samples):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0 if n_samples == 1 else 0.7,
                max_tokens=20
            )
            vote = normalize_vote(response.choices[0].message.content.strip(), candidates)
            if vote:
                counts[vote] += 1
        except Exception as e:
            print(f"Error during API call: {e}")

    total = sum(counts.values())
    if total == 0:
        return {c: 1/len(candidates) for c in candidates}  # fallback uniform
    return {c: counts[c]/total for c in candidates}


def accuracy_from_probs(probs, ground_truth):
    return int(max(probs, key=probs.get) == ground_truth)


def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)


def vote_to_numeric(vote, candidates):
    return 0 if vote.lower() == candidates[1].lower() else 1


# -----------------------------
# Main inference loop
# -----------------------------
def run_inference(data, candidates, candidates_norm, client, model_name, args):
    results = []
    for idx, entry in tqdm(enumerate(data), total=len(data)):
        messages = entry.get("messages", [])
        gt = extract_ground_truth(messages, candidates)
        if gt is None or gt.lower() not in candidates_norm:
            continue

        probs = get_vote_probs(messages, candidates, client, model_name, n_samples=args.n_samples)
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

        if (idx + 1) % args.save_every == 0:
            df_tmp = pd.DataFrame(results)
            save_path = os.path.join(args.out_dir, f"{model_name.replace('/', '_')}_{args.election_year}_partial.pkl")
            df_tmp.to_pickle(save_path)
            print(f"Saved intermediate results at index {idx} to {save_path}")

        time.sleep(args.sleep)
    return results


# -----------------------------
# Compute metrics
# -----------------------------
def compute_metrics(results, candidates):
    df_final = pd.DataFrame(results)
    anes_votes = df_final['ground_truth'].map(lambda x: vote_to_numeric(x, candidates)).to_numpy()
    gpt_votes = df_final['predicted_vote'].map(lambda x: vote_to_numeric(x, candidates)).to_numpy()

    metrics = {}
    metrics['cohen_kappa'] = cohen_kappa_score(anes_votes, gpt_votes)

    if ICC_AVAILABLE:
        try:
            df_temp = pd.DataFrame({'anes': anes_votes, 'gpt': gpt_votes})
            df_long = df_temp.reset_index().melt(id_vars='index', value_vars=['anes','gpt'],
                                                var_name='rater', value_name='vote')
            icc_df = pg.intraclass_corr(data=df_long, targets='index', raters='rater', ratings='vote')
            metrics['ICC'] = icc_df.loc[icc_df['Type']=='ICC2k','ICC'].values[0]
        except Exception as e:
            print(f"Could not compute ICC: {e}")
            metrics['ICC'] = None
    else:
        metrics['ICC'] = None

    metrics['proportion_agreement'] = np.mean(anes_votes == gpt_votes)

    for k, v in metrics.items():
        df_final[k] = v

    return df_final, metrics


# -----------------------------
# Save results
# -----------------------------
def save_results(df_final, data, out_dir, model_name, election_year):
    df_input = pd.DataFrame(data)
    input_cols = [c for c in df_input.columns if c not in df_final.columns]
    df_final = pd.concat([df_final.reset_index(drop=True), df_input[input_cols].reset_index(drop=True)], axis=1)

    out_file = os.path.join(out_dir, f"{model_name.replace('/', '_')}_{election_year}_final.pkl")
    df_final.to_pickle(out_file)
    df_final.to_csv(out_file.replace(".pkl", ".csv"), index=False)
    print(f"Saved final results to {out_file}")
    return df_final


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    setup_environment(args)
    data = load_data(args.data_path)
    candidates, candidates_norm = get_candidates(args.election_year)

    client = openai.OpenAI(api_key=args.api_key)
    results = run_inference(data, candidates, candidates_norm, client, args.model_name, args)

    df_final, metrics = compute_metrics(results, candidates)
    df_final = save_results(df_final, data, args.out_dir, args.model_name, args.election_year)

    print("\nSummary:")
    MI = np.mean([r["mutual_inf"] for r in results])
    print("Average accuracy:", df_final["accuracy"].mean())
    print(f"Average mutual information: {MI:.3f}")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
