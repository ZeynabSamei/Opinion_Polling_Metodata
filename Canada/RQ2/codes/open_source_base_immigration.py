# =====================================================
# Canada 2021 Election Vote Prediction with LLMs
# Direct next-token candidate scoring
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, f1_score

try:
    import pingouin as pg
    ICC_AVAILABLE = True
except ImportError:
    ICC_AVAILABLE = False


# =====================================================
# Arguments
# =====================================================
parser = argparse.ArgumentParser(
    description="Canada 2021 Immigration choices using LLM likelihoods"
)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="./output")
parser.add_argument("--election_year", type=str, default="2021")
parser.add_argument("--save_every", type=int, default=100)
parser.add_argument("--sleep", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_batch_size", type=int, default=24)
parser.add_argument("--max_prompt_length", type=int, default=512)

args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


# =====================================================
# Models
# =====================================================
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]

# =====================================================
# Labels
# =====================================================
IMMIGRATION_CHOICES = [
    "More immigrants",
    "Fewer immigrants",
    "About the same number of immigrants as now"
]

VOTE2ID = {c: i for i, c in enumerate(IMMIGRATION_CHOICES)}


# =====================================================
# Load dataset
# =====================================================
with open(args.data_path, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")


# =====================================================
# Helpers
# =====================================================
def normalize_vote(text):
    if text is None:
        return None
    text = str(text).strip().lower()
    for c in CANDIDATES:
        if c.lower() == text:
            return c
    for c in CANDIDATES:
        if c.lower() in text:
            return c
    return None


def extract_ground_truth(messages):
    for m in messages:
        if m["role"] == "assistant":
            return normalize_vote(m["content"])
    return None


def build_prompt_messages(messages):
    user_text = None
    for m in messages:
        if m["role"] == "user":
            user_text = m["content"]
            break

    if user_text is None:
        return None

    system_text = (
        "You are a political analyst with expertise in Canadian public opinion and social issues. "
        "Using the demographic information provided, predict the respondent's stance on immigration. "
        "Respond with exactly one option from the following labels:\n"
        "More immigrants\n"
        "Fewer immigrants\n"
        "About the same number of immigrants as now\n\n"
    )

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def render_prompts(chat_messages, tokenizer):
    prompts = []
    for msgs in chat_messages:
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
    return prompts


def get_candidate_probs_batched(
    prompts,
    model,
    tokenizer,
    device,
    candidate_first_token_ids,
    max_prompt_length,
):
    """
    Score candidates from the next-token distribution only, using the
    first token of each candidate label.
    """
    if not prompts:
        return []

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    )

    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    prompt_lengths = attention_mask.sum(dim=1)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    out = []

    for i in range(input_ids.size(0)):
        last_pos = int(prompt_lengths[i].item()) - 1
        next_token_logits = logits[i, last_pos]

        cand_logits = torch.stack(
            [next_token_logits[tok_id] for tok_id in candidate_first_token_ids]
        )
        cand_probs = torch.softmax(cand_logits, dim=0).detach().float().cpu().numpy()

        probs = {
            CANDIDATES[j]: float(cand_probs[j])
            for j in range(len(CANDIDATES))
        }
        out.append(probs)

    return out


def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)


def vote_to_numeric(vote):
    return VOTE2ID[vote]


# =====================================================
# Prepare valid entries
# =====================================================
valid_entries = []

for idx, entry in enumerate(data):
    messages = entry.get("messages", [])
    gt = extract_ground_truth(messages)
    prompt_messages = build_prompt_messages(messages)

    if gt is None or gt not in CANDIDATES or prompt_messages is None:
        continue

    valid_entries.append((idx, messages, gt, prompt_messages))


# =====================================================
# Inference
# =====================================================
for model_name in MODELS:
    print(f"\n=== Loading model: {model_name} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model.eval()

    device = next(model.parameters()).device

    candidate_first_token_ids = []
    for candidate in CANDIDATES:
        tok_ids = tokenizer.encode(" " + candidate, add_special_tokens=False)
        print(f"{model_name} candidate {candidate} tokenization: {tok_ids}")
        if len(tok_ids) < 1:
            raise ValueError(f"Candidate '{candidate}' produced no tokens for {model_name}.")
        candidate_first_token_ids.append(tok_ids[0])

    results = []
    n_batches = (len(valid_entries) + args.eval_batch_size - 1) // args.eval_batch_size

    for batch_idx in tqdm(range(n_batches), total=n_batches, desc=f"Inference {model_name}"):
        start = batch_idx * args.eval_batch_size
        chunk = valid_entries[start : start + args.eval_batch_size]

        chat_messages = [x[3] for x in chunk]
        prompts = render_prompts(chat_messages, tokenizer)

        prob_list = get_candidate_probs_batched(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            device=device,
            candidate_first_token_ids=candidate_first_token_ids,
            max_prompt_length=args.max_prompt_length,
        )

        for (idx, messages, gt, _prompt_msgs), probs in zip(chunk, prob_list):
            pred = max(probs, key=probs.get)

            results.append(
                {
                    "idx": idx,
                    "ground_truth": gt,
                    "predicted_vote": pred,
                    "accuracy": int(pred == gt),
                    "mutual_information": mutual_information(probs, gt),
                    "probs": probs,
                    "messages": messages,
                }
            )

            if (idx + 1) % args.save_every == 0:
                pd.DataFrame(results).to_pickle(
                    os.path.join(
                        args.out_dir,
                        f"{model_name.replace('/', '_')}_{args.election_year}_partial.pkl",
                    )
                )

            if args.sleep > 0:
                time.sleep(args.sleep)

    df = pd.DataFrame(results)
    if df.empty:
        print(f"No valid rows for model {model_name}; skipping.")
        continue

    y_true = df["ground_truth"].map(vote_to_numeric).to_numpy()
    y_pred = df["predicted_vote"].map(vote_to_numeric).to_numpy()

    metrics = {
        "accuracy": df["accuracy"].mean(),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "proportion_agreement": np.mean(y_true == y_pred),
        "mean_mutual_information": df["mutual_information"].mean(),
    }

    if ICC_AVAILABLE:
        try:
            df_long = (
                pd.DataFrame({"anes": y_true, "gpt": y_pred})
                .reset_index()
                .melt(id_vars="index", var_name="rater", value_name="vote")
            )
            icc = pg.intraclass_corr(
                data=df_long,
                targets="index",
                raters="rater",
                ratings="vote",
            )
            metrics["ICC"] = icc.loc[icc["Type"] == "ICC2k", "ICC"].values[0]
        except Exception:
            metrics["ICC"] = None
    else:
        metrics["ICC"] = None

    for k, v in metrics.items():
        df[k] = v

    out_base = f"{model_name.replace('/', '_')}_{args.election_year}_final_immigration"
    out_pkl = os.path.join(args.out_dir, out_base + ".pkl")
    out_csv = os.path.join(args.out_dir, out_base + ".csv")

    df.to_pickle(out_pkl)
    df.to_csv(out_csv, index=False)

    print("\n=== Final Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\nSaved results to:\n{out_pkl}\n{out_csv}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



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


# # =====================================================
# # Immigration response labels (MUST match dataset)
# # =====================================================
# IMMIGRATION_CHOICES = [
#     "More immigrants",
#     "Fewer immigrants",
#     "About the same number of immigrants as now"
# ]

# IMMIGRATION_LOWER = [c.lower() for c in IMMIGRATION_CHOICES]
# IMM2ID = {c: i for i, c in enumerate(IMMIGRATION_CHOICES)}
# ID2IMM = {i: c for c, i in IMM2ID.items()}


# # =====================================================
# # Load dataset
# # =====================================================
# with open(args.data_path, "r") as f:
#     data = json.load(f)

# print(f"Loaded {len(data)} samples")


# # =====================================================
# # Load model
# # =====================================================
# print(f"Loading model: {args.model_name}")

# tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# model = AutoModelForCausalLM.from_pretrained(
#     args.model_name,
#     device_map="auto",
#     torch_dtype=torch.float16
# )

# # Pad token fix
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id
# model.config.pad_token_id = tokenizer.eos_token_id
# model.eval()

# device = next(model.parameters()).device


# # =====================================================
# # Helper functions
# # =====================================================
# def normalize_immigration(text):
#     """Extract immigration stance from assistant text."""
#     if text is None:
#         return None
#     text = text.lower()
#     for c in IMMIGRATION_CHOICES:
#         if c.lower() in text:
#             return c
#     return None


# def extract_ground_truth(messages):
#     """Ground truth is the assistant message."""
#     for m in messages:
#         if m["role"] == "assistant":
#             return normalize_immigration(m["content"])
#     return None


# def get_immigration_probs(messages):
#     """
#     Deterministic token-level probability for each immigration option.
#     """
#     clean_msgs = [m for m in messages if m["role"] != "assistant"]

#     prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
#     prompt += (
#         "\nImmigration stance "
#         f"({' or '.join(IMMIGRATION_CHOICES)}):"
#     )

#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

#     probs = {}

#     with torch.no_grad():
#         for choice in IMMIGRATION_CHOICES:
#             choice_ids = tokenizer.encode(choice, add_special_tokens=False)
#             prob = 1.0
#             cur_ids = input_ids.clone()

#             for tok in choice_ids:
#                 outputs = model(input_ids=cur_ids)
#                 logits = outputs.logits[:, -1, :]
#                 token_probs = torch.softmax(logits, dim=-1)
#                 prob *= token_probs[0, tok].item()
#                 cur_ids = torch.cat(
#                     [cur_ids, torch.tensor([[tok]], device=device)], dim=1
#                 )

#             probs[choice] = prob

#     total = sum(probs.values())
#     if total > 0:
#         probs = {c: v / total for c, v in probs.items()}
#     else:
#         probs = {c: 1 / len(IMMIGRATION_CHOICES) for c in IMMIGRATION_CHOICES}

#     return probs


# def mutual_information(probs, ground_truth, eps=1e-12):
#     p = max(probs.get(ground_truth, eps), eps)
#     return -np.log2(p)


# def imm_to_numeric(choice):
#     return IMM2ID[choice]


# # =====================================================
# # Inference loop
# # =====================================================
# results = []

# for idx, entry in tqdm(enumerate(data), total=len(data)):
#     messages = entry.get("messages", [])
#     gt = extract_ground_truth(messages)

#     if gt is None or gt not in IMMIGRATION_CHOICES:
#         continue

#     probs = get_immigration_probs(messages)
#     pred = max(probs, key=probs.get)

#     results.append({
#         "idx": idx,
#         "ground_truth": gt,
#         "predicted_immigration": pred,
#         "accuracy": int(pred == gt),
#         "mutual_information": mutual_information(probs, gt),
#         "probs": probs,
#         "messages": messages,
#     })

#     if (idx + 1) % args.save_every == 0:
#         pd.DataFrame(results).to_pickle(
#             os.path.join(
#                 args.out_dir,
#                 f"{args.model_name.replace('/', '_')}_{args.election_year}_immigration_partial.pkl"
#             )
#         )

#     time.sleep(args.sleep)

# df = pd.DataFrame(results)


# # =====================================================
# # Metrics
# # =====================================================
# human = df["ground_truth"].map(imm_to_numeric).to_numpy()
# model_preds = df["predicted_immigration"].map(imm_to_numeric).to_numpy()

# metrics = {
#     "accuracy": df["accuracy"].mean(),
#     "cohen_kappa": cohen_kappa_score(human, model_preds),
#     "proportion_agreement": np.mean(human == model_preds),
#     "mean_mutual_information": df["mutual_information"].mean(),
# }

# if ICC_AVAILABLE:
#     try:
#         df_long = (
#             pd.DataFrame({"human": human, "model": model_preds})
#             .reset_index()
#             .melt(id_vars="index", var_name="rater", value_name="rating")
#         )
#         icc = pg.intraclass_corr(
#             data=df_long,
#             targets="index",
#             raters="rater",
#             ratings="rating"
#         )
#         metrics["ICC"] = icc.loc[icc["Type"] == "ICC2k", "ICC"].values[0]
#     except Exception:
#         metrics["ICC"] = None
# else:
#     metrics["ICC"] = None

# for k, v in metrics.items():
#     df[k] = v


# # =====================================================
# # Save outputs
# # =====================================================
# out_base = f"{args.model_name.replace('/', '_')}_{args.election_year}_immigration_final"
# out_pkl = os.path.join(args.out_dir, out_base + ".pkl")
# out_csv = os.path.join(args.out_dir, out_base + ".csv")

# df.to_pickle(out_pkl)
# df.to_csv(out_csv, index=False)

# print("\n=== Immigration Attitude Prediction Metrics ===")
# for k, v in metrics.items():
#     print(f"{k}: {v}")

# print(f"\nSaved results to:\n{out_pkl}\n{out_csv}")
