# =====================================================
# LLM Voting Behavior Evaluation (Publication Grade)
# Canada Federal Election 2021
# =====================================================

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, f1_score


# =====================================================
# Configuration
# =====================================================
SEED = 42
BATCH_SIZE = 16
MAX_LENGTH = 512
OUT_DIR = "./results"
DATA_PATH = "dataset_test.json"

# =====================================================
# Models
# =====================================================
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
]


CANDIDATES = [
    "Justin Trudeau",
    "Erin O'Toole",
    "Others",
]

os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a political behavior model that predicts voting choice based on demographic profiles.\n\n"

    "Task:\n"
    "Given a person's demographic and political attributes, predict their MOST LIKELY vote choice "
    "in the 2021 Canadian federal election.\n\n"

    "Rules:\n"
    "- You must choose ONLY ONE label.\n"
    "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
    "Justin Trudeau\n"
    "Erin O'Toole\n"
    "Others\n\n"

    "Definition:\n"
    "'Others' includes Jagmeet Singh, Yves-François Blanchet, Annamie Paul, and Maxime Bernier.\n\n"

    "Important:\n"
    "- Base your decision on typical voting patterns, demographics, and political alignment.\n"
    "- Do NOT explain your reasoning.\n"
    "- Do NOT repeat the input.\n"
    "- Output ONLY the label."
    
)

os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================
# Reproducibility
# =====================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =====================================================
# Data
# =====================================================
def load_data(path):
    with open(path, "r") as f:
        return json.load(f)


def extract_label(messages):
    for m in messages:
        if m["role"] == "assistant":
            return m["content"].strip()
    return None


def build_prompt(tokenizer, messages):
    user_text = None

    for m in messages:
        if m["role"] == "user":
            user_text = m["content"]
            break

    if user_text is None:
        return None

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )


# =====================================================
# Sequence log-prob scoring (CORRECT METHOD)
# =====================================================
@torch.no_grad()
def score_candidates(model, tokenizer, prompts, device):
    results = []

    for prompt in prompts:
        scores = {}

        for cand in CANDIDATES:
            text = prompt + cand

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
            )

            input_ids = enc.input_ids.to(device)

            out = model(input_ids=input_ids)
            logits = out.logits

            log_probs = F.log_softmax(logits, dim=-1)

            seq_logprob = 0.0

            # teacher-forced scoring
            for t in range(input_ids.shape[1] - 1):
                next_token = input_ids[0, t + 1]
                seq_logprob += log_probs[0, t, next_token].item()

            scores[cand] = seq_logprob

        # stable softmax normalization
        max_lp = max(scores.values())
        probs = {k: np.exp(v - max_lp) for k, v in scores.items()}
        norm = sum(probs.values())
        probs = {k: v / norm for k, v in probs.items()}

        results.append(probs)

    return results


# =====================================================
# Metrics
# =====================================================
def compute_metrics(df):
    label_map = {c: i for i, c in enumerate(CANDIDATES)}

    y_true = df["ground_truth"].map(label_map).values
    y_pred = df["prediction"].map(label_map).values

    return {
        "accuracy": (y_true == y_pred).mean(),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


# =====================================================
# Main loop
# =====================================================
data = load_data(DATA_PATH)

for model_name in MODELS:
    print(f"\n=== Loading model: {model_name} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model.eval()
    device = next(model.parameters()).device

    # -------------------------------------------------
    # Preprocess dataset
    # -------------------------------------------------
    samples = []

    for i, item in enumerate(data):
        messages = item.get("messages", [])
        gt = extract_label(messages)

        if gt not in CANDIDATES:
            continue

        prompt = build_prompt(tokenizer, messages)
        if prompt is None:
            continue

        samples.append((i, prompt, gt))

    # -------------------------------------------------
    # Inference
    # -------------------------------------------------
    results = []

    for start in tqdm(range(0, len(samples), BATCH_SIZE)):
        batch = samples[start : start + BATCH_SIZE]
        idxs, prompts, gts = zip(*batch)

        probs_list = score_candidates(model, tokenizer, prompts, device)

        for idx, gt, probs in zip(idxs, gts, probs_list):
            pred = max(probs, key=probs.get)

            results.append({
                "idx": idx,
                "ground_truth": gt,
                "prediction": pred,
                "correct": int(pred == gt),
                "probs": probs,
            })

    df = pd.DataFrame(results)

    metrics = compute_metrics(df)

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    model_tag = model_name.replace("/", "_")
    out_path = os.path.join(OUT_DIR, f"{model_tag}_results.csv")
    df.to_csv(out_path, index=False)

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nSaved: {out_path}")

    del model
    torch.cuda.empty_cache()


# # =====================================================
# # Canada 2021 Election Vote Prediction with LLMs
# # Direct next-token candidate scoring
# # =====================================================

# import os
# import time
# import json
# import argparse
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F

# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, f1_score

# try:
#     import pingouin as pg
#     ICC_AVAILABLE = True
# except ImportError:
#     ICC_AVAILABLE = False


# # =====================================================
# # Arguments
# # =====================================================
# parser = argparse.ArgumentParser(
#     description="Canada 2021 vote prediction using LLM likelihoods"
# )
# parser.add_argument("--data_path", type=str, required=True)
# parser.add_argument("--out_dir", type=str, default="./output")
# parser.add_argument("--election_year", type=str, default="2021")
# parser.add_argument("--save_every", type=int, default=100)
# parser.add_argument("--sleep", type=float, default=0.0)
# parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--eval_batch_size", type=int, default=24)
# parser.add_argument("--max_prompt_length", type=int, default=512)

# args = parser.parse_args()
# os.makedirs(args.out_dir, exist_ok=True)

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     torch.set_float32_matmul_precision("high")


# # =====================================================
# # Models
# # =====================================================
# MODELS = [
#     "meta-llama/Llama-3.1-8B-Instruct",
#     "meta-llama/Llama-3.1-70B-Instruct",
#     "Qwen/Qwen2.5-7B-Instruct",
#     # "Qwen/Qwen2.5-14B-Instruct",
# ]


# # =====================================================
# # Labels
# # =====================================================
# CANDIDATES = [
#     "Justin Trudeau",
#     "Erin O'Toole",
#     "Others",
# ]

# VOTE2ID = {c: i for i, c in enumerate(CANDIDATES)}


# # =====================================================
# # Load dataset
# # =====================================================
# with open(args.data_path, "r") as f:
#     data = json.load(f)

# print(f"Loaded {len(data)} samples")


# # =====================================================
# # Helpers
# # =====================================================
# def normalize_vote(text):
#     if text is None:
#         return None
#     text = str(text).strip().lower()
#     for c in CANDIDATES:
#         if c.lower() == text:
#             return c
#     for c in CANDIDATES:
#         if c.lower() in text:
#             return c
#     return None


# def extract_ground_truth(messages):
#     for m in messages:
#         if m["role"] == "assistant":
#             return normalize_vote(m["content"])
#     return None


# def build_prompt_messages(messages):
#     user_text = None
#     for m in messages:
#         if m["role"] == "user":
#             user_text = m["content"]
#             break

#     if user_text is None:
#         return None


#     system_text = (
#         "You are a political behavior model that predicts voting choice based on demographic profiles.\n\n"
    
#         "Task:\n"
#         "Given a person's demographic and political attributes, predict their MOST LIKELY vote choice "
#         "in the 2021 Canadian federal election.\n\n"
    
#         "Rules:\n"
#         "- You must choose ONLY ONE label.\n"
#         "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
#         "Justin Trudeau\n"
#         "Erin O'Toole\n"
#         "Others\n\n"
    
#         "Definition:\n"
#         "'Others' includes Jagmeet Singh, Yves-François Blanchet, Annamie Paul, and Maxime Bernier.\n\n"
    
#         "Important:\n"
#         "- Base your decision on typical voting patterns, demographics, and political alignment.\n"
#         "- Do NOT explain your reasoning.\n"
#         "- Do NOT repeat the input.\n"
#         "- Output ONLY the label."
#     )
        

#     # system_text = (
#     #     "You are an expert political analyst specializing in Canadian elections and voting behavior. "
#     #     "Predict the individual's vote choice in the 2021 Canadian federal election based on provided demographic features.\n\n"
#     #     "Choose exactly one of the following labels:\n"
#     #     "Justin Trudeau\n"
#     #     "Erin O'Toole\n"
#     #     "Others\n\n"
#     #     "Answer with only one label exactly as written above."
#     # )

#     # system_text = (
#     #     "You are a classifier that predicts vote choice in the 2021 Canadian federal election.\n\n"
#     #     "Output exactly one label:\n"
#     #     "- Justin Trudeau\n"
#     #     "- Erin O'Toole\n"
#     #     "- Others\n\n"
#     #     "Do not output anything else."
#     # )

#     return [
#         {"role": "system", "content": system_text},
#         {"role": "user", "content": user_text},
#     ]


# def render_prompts(chat_messages, tokenizer):
#     prompts = []
#     for msgs in chat_messages:
#         prompt = tokenizer.apply_chat_template(
#             msgs,
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#         prompts.append(prompt)
#     return prompts


# def get_candidate_probs_batched(
#     prompts,
#     model,
#     tokenizer,
#     device,
#     candidate_first_token_ids,
#     max_prompt_length,
# ):
#     """
#     Score candidates from the next-token distribution only, using the
#     first token of each candidate label.
#     """
#     if not prompts:
#         return []

#     enc = tokenizer(
#         prompts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=max_prompt_length,
#         add_special_tokens=False,
#     )

#     input_ids = enc.input_ids.to(device)
#     attention_mask = enc.attention_mask.to(device)
#     prompt_lengths = attention_mask.sum(dim=1)

#     with torch.inference_mode():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits

#     out = []

#     for i in range(input_ids.size(0)):
#         last_pos = int(prompt_lengths[i].item()) - 1
#         next_token_logits = logits[i, last_pos]

#         cand_logits = torch.stack(
#             [next_token_logits[tok_id] for tok_id in candidate_first_token_ids]
#         )
#         cand_probs = torch.softmax(cand_logits, dim=0).detach().float().cpu().numpy()

#         probs = {
#             CANDIDATES[j]: float(cand_probs[j])
#             for j in range(len(CANDIDATES))
#         }
#         out.append(probs)

#     return out


# def mutual_information(probs, ground_truth, eps=1e-12):
#     p = max(probs.get(ground_truth, eps), eps)
#     return -np.log2(p)


# def vote_to_numeric(vote):
#     return VOTE2ID[vote]


# # =====================================================
# # Prepare valid entries
# # =====================================================
# valid_entries = []

# for idx, entry in enumerate(data):
#     messages = entry.get("messages", [])
#     gt = extract_ground_truth(messages)
#     prompt_messages = build_prompt_messages(messages)

#     if gt is None or gt not in CANDIDATES or prompt_messages is None:
#         continue

#     valid_entries.append((idx, messages, gt, prompt_messages))


# # =====================================================
# # Inference
# # =====================================================
# for model_name in MODELS:
#     print(f"\n=== Loading model: {model_name} ===")

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id

#     try:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#             attn_implementation="flash_attention_2",
#         )
#     except Exception:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#         )

#     model.config.pad_token_id = tokenizer.pad_token_id
#     model.config.use_cache = True
#     model.eval()

#     device = next(model.parameters()).device

#     candidate_first_token_ids = []
#     for candidate in CANDIDATES:
#         tok_ids = tokenizer.encode(" " + candidate, add_special_tokens=False)
#         print(f"{model_name} candidate {candidate} tokenization: {tok_ids}")
#         if len(tok_ids) < 1:
#             raise ValueError(f"Candidate '{candidate}' produced no tokens for {model_name}.")
#         candidate_first_token_ids.append(tok_ids[0])

#     results = []
#     n_batches = (len(valid_entries) + args.eval_batch_size - 1) // args.eval_batch_size

#     for batch_idx in tqdm(range(n_batches), total=n_batches, desc=f"Inference {model_name}"):
#         start = batch_idx * args.eval_batch_size
#         chunk = valid_entries[start : start + args.eval_batch_size]

#         chat_messages = [x[3] for x in chunk]
#         prompts = render_prompts(chat_messages, tokenizer)

#         prob_list = get_candidate_probs_batched(
#             prompts=prompts,
#             model=model,
#             tokenizer=tokenizer,
#             device=device,
#             candidate_first_token_ids=candidate_first_token_ids,
#             max_prompt_length=args.max_prompt_length,
#         )

#         for (idx, messages, gt, _prompt_msgs), probs in zip(chunk, prob_list):
#             pred = max(probs, key=probs.get)

#             results.append(
#                 {
#                     "idx": idx,
#                     "ground_truth": gt,
#                     "predicted_vote": pred,
#                     "accuracy": int(pred == gt),
#                     "mutual_information": mutual_information(probs, gt),
#                     "probs": probs,
#                     "messages": messages,
#                 }
#             )

#             if (idx + 1) % args.save_every == 0:
#                 pd.DataFrame(results).to_pickle(
#                     os.path.join(
#                         args.out_dir,
#                         f"{model_name.replace('/', '_')}_{args.election_year}_partial.pkl",
#                     )
#                 )

#             if args.sleep > 0:
#                 time.sleep(args.sleep)

#     df = pd.DataFrame(results)
#     if df.empty:
#         print(f"No valid rows for model {model_name}; skipping.")
#         continue

#     y_true = df["ground_truth"].map(vote_to_numeric).to_numpy()
#     y_pred = df["predicted_vote"].map(vote_to_numeric).to_numpy()

#     metrics = {
#         "accuracy": df["accuracy"].mean(),
#         "cohen_kappa": cohen_kappa_score(y_true, y_pred),
#         "mcc": matthews_corrcoef(y_true, y_pred),
#         "macro_f1": f1_score(y_true, y_pred, average="macro"),
#         "proportion_agreement": np.mean(y_true == y_pred),
#         "mean_mutual_information": df["mutual_information"].mean(),
#     }

#     if ICC_AVAILABLE:
#         try:
#             df_long = (
#                 pd.DataFrame({"anes": y_true, "gpt": y_pred})
#                 .reset_index()
#                 .melt(id_vars="index", var_name="rater", value_name="vote")
#             )
#             icc = pg.intraclass_corr(
#                 data=df_long,
#                 targets="index",
#                 raters="rater",
#                 ratings="vote",
#             )
#             metrics["ICC"] = icc.loc[icc["Type"] == "ICC2k", "ICC"].values[0]
#         except Exception:
#             metrics["ICC"] = None
#     else:
#         metrics["ICC"] = None

#     for k, v in metrics.items():
#         df[k] = v

#     out_base = f"{model_name.replace('/', '_')}_{args.election_year}_final_vote"
#     out_pkl = os.path.join(args.out_dir, out_base + ".pkl")
#     out_csv = os.path.join(args.out_dir, out_base + ".csv")

#     df.to_pickle(out_pkl)
#     df.to_csv(out_csv, index=False)

#     print("\n=== Final Metrics ===")
#     for k, v in metrics.items():
#         print(f"{k}: {v}")

#     print(f"\nSaved results to:\n{out_pkl}\n{out_csv}")

#     del model
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()



# # =====================================================
# # Canada 2021 Election Vote Prediction with LLMs
# # Direct next-token option scoring
# # =====================================================

# import os
# import time
# import json
# import argparse
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F

# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, f1_score

# try:
#     import pingouin as pg
#     ICC_AVAILABLE = True
# except ImportError:
#     ICC_AVAILABLE = False


# # =====================================================
# # Arguments
# # =====================================================
# parser = argparse.ArgumentParser(
#     description="Canada 2021 vote prediction using LLM likelihoods"
# )
# parser.add_argument("--data_path", type=str, required=True)
# parser.add_argument("--out_dir", type=str, default="./output")
# parser.add_argument("--election_year", type=str, default="2021")
# parser.add_argument("--save_every", type=int, default=100)
# parser.add_argument("--sleep", type=float, default=0.0)
# parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--eval_batch_size", type=int, default=24)
# parser.add_argument("--max_prompt_length", type=int, default=512)

# args = parser.parse_args()
# os.makedirs(args.out_dir, exist_ok=True)

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     torch.set_float32_matmul_precision("high")


# # =====================================================
# # Models
# # =====================================================
# MODELS = [
#     "meta-llama/Llama-3.1-8B-Instruct",
#     "Qwen/Qwen2.5-7B-Instruct",
#     "Qwen/Qwen2.5-14B-Instruct",
# ]


# # =====================================================
# # Labels
# # =====================================================
# CANDIDATES = [
#     "Justin Trudeau",
#     "Erin O'Toole",
#     "Others",
# ]

# OPTION_TO_VOTE = {
#     "A": "Justin Trudeau",
#     "B": "Erin O'Toole",
#     "C": "Others",
# }
# VOTE2ID = {c: i for i, c in enumerate(CANDIDATES)}


# # =====================================================
# # Load dataset
# # =====================================================
# with open(args.data_path, "r") as f:
#     data = json.load(f)

# print(f"Loaded {len(data)} samples")


# # =====================================================
# # Helpers
# # =====================================================
# def normalize_vote(text):
#     if text is None:
#         return None
#     text = str(text).strip().lower()
#     for c in CANDIDATES:
#         if c.lower() == text:
#             return c
#     for c in CANDIDATES:
#         if c.lower() in text:
#             return c
#     return None


# def extract_ground_truth(messages):
#     for m in messages:
#         if m["role"] == "assistant":
#             return normalize_vote(m["content"])
#     return None


# def build_prompt_messages(messages):
#     user_text = None
#     for m in messages:
#         if m["role"] == "user":
#             user_text = m["content"]
#             break

#     if user_text is None:
#         return None

#     system_text = (
#         "You are an expert political analyst specializing in Canadian elections and voting behavior. "
#         "Predict the individual's vote choice in the 2021 Canadian federal election.\n\n"
#         "Choose exactly one option:\n"
#         "A. Justin Trudeau\n"
#         "B. Erin O'Toole\n"
#         "C. Others\n\n"
#         "Answer with only A, B, or C."
#     )

#     return [
#         {"role": "system", "content": system_text},
#         {"role": "user", "content": user_text},
#     ]


# def render_prompts(chat_messages, tokenizer):
#     prompts = []
#     for msgs in chat_messages:
#         prompt = tokenizer.apply_chat_template(
#             msgs,
#             tokenize=False,
#             add_generation_prompt=True,
#         )
#         prompts.append(prompt)
#     return prompts


# def get_option_probs_batched(
#     prompts,
#     model,
#     tokenizer,
#     device,
#     option_token_ids,
#     max_prompt_length,
# ):
#     """
#     Score A/B/C from the next-token distribution only.
#     This is preferred when each option is represented by a single token.
#     """
#     if not prompts:
#         return []

#     enc = tokenizer(
#         prompts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=max_prompt_length,
#         add_special_tokens=False,
#     )

#     input_ids = enc.input_ids.to(device)
#     attention_mask = enc.attention_mask.to(device)
#     prompt_lengths = attention_mask.sum(dim=1)

#     with torch.inference_mode():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         logits = outputs.logits

#     out = []
#     option_keys = list(OPTION_TO_VOTE.keys())

#     for i in range(input_ids.size(0)):
#         last_pos = int(prompt_lengths[i].item()) - 1
#         next_token_logits = logits[i, last_pos]

#         opt_logits = torch.stack([next_token_logits[tok_id] for tok_id in option_token_ids])
#         # opt_probs = torch.softmax(opt_logits, dim=0).detach().cpu().numpy()
#         opt_probs = torch.softmax(opt_logits, dim=0).detach().float().cpu().numpy()

#         probs = {
#             OPTION_TO_VOTE[option_keys[j]]: float(opt_probs[j])
#             for j in range(len(option_keys))
#         }
#         out.append(probs)

#     return out


# def mutual_information(probs, ground_truth, eps=1e-12):
#     p = max(probs.get(ground_truth, eps), eps)
#     return -np.log2(p)


# def vote_to_numeric(vote):
#     return VOTE2ID[vote]


# # =====================================================
# # Prepare valid entries
# # =====================================================
# valid_entries = []

# for idx, entry in enumerate(data):
#     messages = entry.get("messages", [])
#     gt = extract_ground_truth(messages)
#     prompt_messages = build_prompt_messages(messages)

#     if gt is None or gt not in CANDIDATES or prompt_messages is None:
#         continue

#     valid_entries.append((idx, messages, gt, prompt_messages))


# # =====================================================
# # Inference
# # =====================================================
# for model_name in MODELS:
#     print(f"\n=== Loading model: {model_name} ===")

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id

#     try:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#             attn_implementation="flash_attention_2",
#         )
#     except Exception:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#         )

#     model.config.pad_token_id = tokenizer.pad_token_id
#     model.config.use_cache = True
#     model.eval()

#     device = next(model.parameters()).device

#     # Direct next-token scoring requires each option to be one token.
#     # Prefer " A"/" B"/" C" first; if not, test alternatives like "1"/"2"/"3".
#     option_keys = ["A", "B", "C"]
#     option_token_ids = []

#     for opt in option_keys:
#         tok_ids = tokenizer.encode(" " + opt, add_special_tokens=False)
#         print(f"{model_name} option {opt} tokenization: {tok_ids}")
#         if len(tok_ids) != 1:
#             raise ValueError(
#                 f"Option '{opt}' is not a single token for {model_name}. "
#                 "Use a different option scheme such as 1/2/3."
#             )
#         option_token_ids.append(tok_ids[0])

#     results = []
#     n_batches = (len(valid_entries) + args.eval_batch_size - 1) // args.eval_batch_size

#     for batch_idx in tqdm(range(n_batches), total=n_batches, desc=f"Inference {model_name}"):
#         start = batch_idx * args.eval_batch_size
#         chunk = valid_entries[start : start + args.eval_batch_size]

#         chat_messages = [x[3] for x in chunk]
#         prompts = render_prompts(chat_messages, tokenizer)

#         prob_list = get_option_probs_batched(
#             prompts=prompts,
#             model=model,
#             tokenizer=tokenizer,
#             device=device,
#             option_token_ids=option_token_ids,
#             max_prompt_length=args.max_prompt_length,
#         )

#         for (idx, messages, gt, _prompt_msgs), probs in zip(chunk, prob_list):
#             pred = max(probs, key=probs.get)

#             results.append(
#                 {
#                     "idx": idx,
#                     "ground_truth": gt,
#                     "predicted_vote": pred,
#                     "accuracy": int(pred == gt),
#                     "mutual_information": mutual_information(probs, gt),
#                     "probs": probs,
#                     "messages": messages,
#                 }
#             )

#             if (idx + 1) % args.save_every == 0:
#                 pd.DataFrame(results).to_pickle(
#                     os.path.join(
#                         args.out_dir,
#                         f"{model_name.replace('/', '_')}_{args.election_year}_partial.pkl",
#                     )
#                 )

#             if args.sleep > 0:
#                 time.sleep(args.sleep)

#     df = pd.DataFrame(results)
#     if df.empty:
#         print(f"No valid rows for model {model_name}; skipping.")
#         continue

#     y_true = df["ground_truth"].map(vote_to_numeric).to_numpy()
#     y_pred = df["predicted_vote"].map(vote_to_numeric).to_numpy()

#     metrics = {
#         "accuracy": df["accuracy"].mean(),
#         "cohen_kappa": cohen_kappa_score(y_true, y_pred),
#         "mcc": matthews_corrcoef(y_true, y_pred),
#         "macro_f1": f1_score(y_true, y_pred, average="macro"),
#         "proportion_agreement": np.mean(y_true == y_pred),
#         "mean_mutual_information": df["mutual_information"].mean(),
#     }

#     if ICC_AVAILABLE:
#         try:
#             df_long = (
#                 pd.DataFrame({"anes": y_true, "gpt": y_pred})
#                 .reset_index()
#                 .melt(id_vars="index", var_name="rater", value_name="vote")
#             )
#             icc = pg.intraclass_corr(
#                 data=df_long,
#                 targets="index",
#                 raters="rater",
#                 ratings="vote",
#             )
#             metrics["ICC"] = icc.loc[icc["Type"] == "ICC2k", "ICC"].values[0]
#         except Exception:
#             metrics["ICC"] = None
#     else:
#         metrics["ICC"] = None

#     for k, v in metrics.items():
#         df[k] = v

#     out_base = f"{model_name.replace('/', '_')}_{args.election_year}_final"
#     out_pkl = os.path.join(args.out_dir, out_base + ".pkl")
#     out_csv = os.path.join(args.out_dir, out_base + ".csv")

#     df.to_pickle(out_pkl)
#     df.to_csv(out_csv, index=False)

#     print("\n=== Final Metrics ===")
#     for k, v in metrics.items():
#         print(f"{k}: {v}")

#     print(f"\nSaved results to:\n{out_pkl}\n{out_csv}")

#     del model
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# # =====================================================
# # Canada 2021 Election Vote Prediction with LLMs
# # Deterministic token-level probability evaluation (batched + fast)
# # =====================================================

# import os
# import time
# import json
# import argparse
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F

# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sklearn.metrics import cohen_kappa_score

# # Optional ICC
# try:
#     import pingouin as pg

#     ICC_AVAILABLE = True
# except ImportError:
#     ICC_AVAILABLE = False


# # =====================================================
# # Arguments
# # =====================================================
# parser = argparse.ArgumentParser(
#     description="Canada 2021 vote prediction using LLM likelihoods"
# )
# parser.add_argument("--data_path", type=str, required=True)
# parser.add_argument("--out_dir", type=str, default="./output")
# parser.add_argument("--election_year", type=str, default="2021")
# parser.add_argument("--save_every", type=int, default=100)
# parser.add_argument("--sleep", type=float, default=0.0)
# parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--eval_batch_size", type=int, default=24)
# parser.add_argument("--max_prompt_length", type=int, default=512)

# args = parser.parse_args()
# os.makedirs(args.out_dir, exist_ok=True)

# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     torch.set_float32_matmul_precision("high")


# # =====================================================
# # Models
# # =====================================================
# MODELS = [
#     "meta-llama/Llama-3.1-8B-Instruct",
#     "Qwen/Qwen2.5-7B-Instruct",
#     "Qwen/Qwen2.5-14B-Instruct",
# ]


# # =====================================================
# # Candidates (Canada 2021 leaders)
# # =====================================================
# CANDIDATES = [
#     "Justin Trudeau",
#     "Erin O'Toole",
#     "Others",
# ]
# CANDIDATES_LOWER = [c.lower() for c in CANDIDATES]
# VOTE2ID = {c: i for i, c in enumerate(CANDIDATES)}
# ID2VOTE = {i: c for c, i in VOTE2ID.items()}


# # =====================================================
# # Load dataset
# # =====================================================
# with open(args.data_path, "r") as f:
#     data = json.load(f)

# print(f"Loaded {len(data)} samples")


# # =====================================================
# # Helper functions
# # =====================================================
# def normalize_vote(text):
#     """Extract candidate name from generated text."""
#     if text is None:
#         return None
#     text = text.lower()
#     for c in CANDIDATES:
#         if c.lower() in text:
#             return c
#     return None


# def extract_ground_truth(messages):
#     """Ground truth is stored in assistant turn."""
#     for m in messages:
#         if m["role"] == "assistant":
#             return normalize_vote(m["content"])
#     return None


# def build_prompt(messages):
#     clean_msgs = [m for m in messages if m["role"] != "assistant"]
#     prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
#     prompt += f"\nVote choice ({' or '.join(CANDIDATES)}):"
#     return prompt


# def get_vote_probs_batched(
#     prompts,
#     model,
#     tokenizer,
#     device,
#     candidate_token_ids,
#     max_prompt_length,
# ):
#     """
#     Deterministic token-level probability of each candidate, computed in batch.
#     Produces candidate-normalized probabilities equivalent to the original method.
#     """
#     if not prompts:
#         return []

#     enc = tokenizer(
#         prompts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=max_prompt_length,
#         add_special_tokens=False,
#     )
#     prompt_input_ids = enc.input_ids.to(device)
#     prompt_attention_mask = enc.attention_mask.to(device)
#     prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()

#     seqs = []
#     seq_prompt_lens = []
#     seq_cand_lens = []

#     for i, p_len in enumerate(prompt_lengths):
#         prompt_tokens = prompt_input_ids[i, :p_len]
#         for cand_ids in candidate_token_ids:
#             cand_tensor = torch.tensor(cand_ids, dtype=torch.long, device=device)
#             seq = torch.cat([prompt_tokens, cand_tensor], dim=0)
#             seqs.append(seq)
#             seq_prompt_lens.append(int(p_len))
#             seq_cand_lens.append(len(cand_ids))

#     max_len = max(s.size(0) for s in seqs)
#     n_seq = len(seqs)

#     input_ids = torch.full(
#         (n_seq, max_len),
#         fill_value=tokenizer.pad_token_id,
#         dtype=torch.long,
#         device=device,
#     )
#     attention_mask = torch.zeros((n_seq, max_len), dtype=torch.long, device=device)

#     for i, seq in enumerate(seqs):
#         l = seq.size(0)
#         input_ids[i, :l] = seq
#         attention_mask[i, :l] = 1

#     with torch.inference_mode():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         log_probs = F.log_softmax(outputs.logits, dim=-1)

#     seq_logps = []
#     for i in range(n_seq):
#         p_len = seq_prompt_lens[i]
#         c_len = seq_cand_lens[i]
#         lp = torch.tensor(0.0, device=device)

#         for k in range(c_len):
#             logit_pos = p_len - 1 + k
#             token_id = input_ids[i, p_len + k]
#             lp = lp + log_probs[i, logit_pos, token_id]
#         seq_logps.append(lp)

#     seq_logps = torch.stack(seq_logps).view(len(prompts), len(CANDIDATES))
#     cand_probs = torch.softmax(seq_logps, dim=1).detach().cpu().numpy()

#     out = []
#     for row in cand_probs:
#         out.append({c: float(row[j]) for j, c in enumerate(CANDIDATES)})
#     return out


# def mutual_information(probs, ground_truth, eps=1e-12):
#     p = max(probs.get(ground_truth, eps), eps)
#     return -np.log2(p)


# def vote_to_numeric(vote):
#     return VOTE2ID[vote]


# # =====================================================
# # Inference loop (batched)
# # =====================================================
# valid_entries = []

# for idx, entry in enumerate(data):
#     messages = entry.get("messages", [])
#     gt = extract_ground_truth(messages)

#     if gt is None or gt not in CANDIDATES:
#         continue

#     valid_entries.append((idx, messages, gt, build_prompt(messages)))


# for model_name in MODELS:
#     print(f"\n=== Loading model: {model_name} ===")

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id

#     try:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#             attn_implementation="flash_attention_2",
#         )
#     except Exception:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#         )

#     model.config.pad_token_id = tokenizer.pad_token_id
#     model.config.use_cache = True
#     model.eval()

#     device = next(model.parameters()).device
#     candidate_token_ids = [
#         tokenizer.encode(candidate, add_special_tokens=False) for candidate in CANDIDATES
#     ]

#     results = []
#     n_batches = (len(valid_entries) + args.eval_batch_size - 1) // args.eval_batch_size
#     for batch_idx in tqdm(
#         range(n_batches),
#         total=n_batches,
#         desc=f"Inference {model_name}",
#     ):
#         start = batch_idx * args.eval_batch_size
#         chunk = valid_entries[start : start + args.eval_batch_size]

#         prompts = [x[3] for x in chunk]
#         prob_list = get_vote_probs_batched(
#             prompts=prompts,
#             model=model,
#             tokenizer=tokenizer,
#             device=device,
#             candidate_token_ids=candidate_token_ids,
#             max_prompt_length=args.max_prompt_length,
#         )

#         for (idx, messages, gt, _prompt), probs in zip(chunk, prob_list):
#             pred = max(probs, key=probs.get)

#             results.append(
#                 {
#                     "idx": idx,
#                     "ground_truth": gt,
#                     "predicted_vote": pred,
#                     "accuracy": int(pred == gt),
#                     "mutual_information": mutual_information(probs, gt),
#                     "probs": probs,
#                     "messages": messages,
#                 }
#             )

#             if (idx + 1) % args.save_every == 0:
#                 pd.DataFrame(results).to_pickle(
#                     os.path.join(
#                         args.out_dir,
#                         f"{model_name.replace('/', '_')}_{args.election_year}_partial.pkl",
#                     )
#                 )

#             if args.sleep > 0:
#                 time.sleep(args.sleep)

#     df = pd.DataFrame(results)
#     if df.empty:
#         print(f"No valid rows for model {model_name}; skipping metrics and save.")
#         continue

#     # =====================================================
#     # Metrics
#     # =====================================================
#     anes_votes = df["ground_truth"].map(vote_to_numeric).to_numpy()
#     gpt_votes = df["predicted_vote"].map(vote_to_numeric).to_numpy()

#     metrics = {
#         "accuracy": df["accuracy"].mean(),
#         "cohen_kappa": cohen_kappa_score(anes_votes, gpt_votes),
#         "proportion_agreement": np.mean(anes_votes == gpt_votes),
#         "mean_mutual_information": df["mutual_information"].mean(),
#     }

#     if ICC_AVAILABLE:
#         try:
#             df_long = (
#                 pd.DataFrame({"anes": anes_votes, "gpt": gpt_votes})
#                 .reset_index()
#                 .melt(id_vars="index", var_name="rater", value_name="vote")
#             )
#             icc = pg.intraclass_corr(
#                 data=df_long,
#                 targets="index",
#                 raters="rater",
#                 ratings="vote",
#             )
#             metrics["ICC"] = icc.loc[icc["Type"] == "ICC2k", "ICC"].values[0]
#         except Exception:
#             metrics["ICC"] = None
#     else:
#         metrics["ICC"] = None

#     for k, v in metrics.items():
#         df[k] = v

#     # =====================================================
#     # Save final outputs
#     # =====================================================
#     out_base = f"{model_name.replace('/', '_')}_{args.election_year}_final"
#     out_pkl = os.path.join(args.out_dir, out_base + ".pkl")
#     out_csv = os.path.join(args.out_dir, out_base + ".csv")

#     df.to_pickle(out_pkl)
#     df.to_csv(out_csv, index=False)

#     print("\n=== Final Metrics ===")
#     for k, v in metrics.items():
#         print(f"{k}: {v}")

#     print(f"\nSaved results to:\n{out_pkl}\n{out_csv}")

#     del model
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()


# # # =====================================================
# # # Canada 2021 Election Vote Prediction with LLMs
# # # Deterministic token-level probability evaluation
# # # =====================================================

# # import os
# # import time
# # import json
# # import argparse
# # import random
# # import numpy as np
# # import pandas as pd
# # import torch

# # from tqdm import tqdm
# # from transformers import AutoTokenizer, AutoModelForCausalLM
# # from sklearn.metrics import cohen_kappa_score

# # # Optional ICC
# # try:
# #     import pingouin as pg
# #     ICC_AVAILABLE = True
# # except ImportError:
# #     ICC_AVAILABLE = False


# # # =====================================================
# # # Arguments
# # # =====================================================
# # parser = argparse.ArgumentParser(
# #     description="Canada 2021 vote prediction using LLM likelihoods"
# # )
# # parser.add_argument("--model_name", type=str, required=True)
# # parser.add_argument("--data_path", type=str, required=True)
# # parser.add_argument("--out_dir", type=str, default="./output")
# # parser.add_argument("--election_year", type=str, default="2021")
# # parser.add_argument("--save_every", type=int, default=100)
# # parser.add_argument("--sleep", type=float, default=0.1)
# # parser.add_argument("--seed", type=int, default=42)

# # args = parser.parse_args()
# # os.makedirs(args.out_dir, exist_ok=True)

# # random.seed(args.seed)
# # np.random.seed(args.seed)
# # torch.manual_seed(args.seed)


# # # =====================================================
# # # Candidates (Canada 2021 leaders)
# # # =====================================================
# # # CANDIDATES = [
# # #     "Justin Trudeau",
# # #     "Erin O'Toole",
# # #     "Jagmeet Singh",
# # #     "Yves-François Blanchet",
# # #     "Annamie Paul",
# # #     "Maxime Bernier",
# # # ]

# # CANDIDATES = [
# #     "Justin Trudeau",
# #     "Erin O'Toole",
# #     "Others"
# # ]
# # CANDIDATES_LOWER = [c.lower() for c in CANDIDATES]
# # VOTE2ID = {c: i for i, c in enumerate(CANDIDATES)}
# # ID2VOTE = {i: c for c, i in VOTE2ID.items()}


# # # =====================================================
# # # Load dataset
# # # =====================================================
# # with open(args.data_path, "r") as f:
# #     data = json.load(f)

# # print(f"Loaded {len(data)} samples")


# # # =====================================================
# # # Load model
# # # =====================================================
# # print(f"Loading model: {args.model_name}")

# # tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# # model = AutoModelForCausalLM.from_pretrained(
# #     args.model_name,
# #     device_map="auto",
# #     torch_dtype=torch.float16
# # )

# # # Pad token fix
# # tokenizer.pad_token = tokenizer.eos_token
# # tokenizer.pad_token_id = tokenizer.eos_token_id
# # model.config.pad_token_id = tokenizer.eos_token_id
# # model.eval()

# # device = next(model.parameters()).device


# # # =====================================================
# # # Helper functions
# # # =====================================================
# # def normalize_vote(text):
# #     """Extract candidate name from generated text."""
# #     if text is None:
# #         return None
# #     text = text.lower()
# #     for c in CANDIDATES:
# #         if c.lower() in text:
# #             return c
# #     return None


# # def extract_ground_truth(messages):
# #     """Ground truth is stored in assistant turn."""
# #     for m in messages:
# #         if m["role"] == "assistant":
# #             return normalize_vote(m["content"])
# #     return None


# # def get_vote_probs(messages):
# #     """
# #     Deterministic token-level probability of each candidate.
# #     Matches likelihood-based evaluation in the paper.
# #     """
# #     # Remove assistant messages
# #     clean_msgs = [m for m in messages if m["role"] != "assistant"]

# #     prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
# #     prompt += f"\nVote choice ({' or '.join(CANDIDATES)}):"

# #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# #     probs = {}

# #     with torch.no_grad():
# #         for candidate in CANDIDATES:
# #             cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
# #             prob = 1.0
# #             cur_ids = input_ids.clone()

# #             for tok in cand_ids:
# #                 outputs = model(input_ids=cur_ids)
# #                 logits = outputs.logits[:, -1, :]
# #                 token_probs = torch.softmax(logits, dim=-1)
# #                 prob *= token_probs[0, tok].item()
# #                 cur_ids = torch.cat(
# #                     [cur_ids, torch.tensor([[tok]], device=device)], dim=1
# #                 )

# #             probs[candidate] = prob

# #     total = sum(probs.values())
# #     if total > 0:
# #         probs = {c: p / total for c, p in probs.items()}
# #     else:
# #         probs = {c: 1 / len(CANDIDATES) for c in CANDIDATES}

# #     return probs


# # def mutual_information(probs, ground_truth, eps=1e-12):
# #     p = max(probs.get(ground_truth, eps), eps)
# #     return -np.log2(p)


# # def vote_to_numeric(vote):
# #     return VOTE2ID[vote]


# # # =====================================================
# # # Inference loop
# # # =====================================================
# # results = []

# # for idx, entry in tqdm(enumerate(data), total=len(data)):
# #     messages = entry.get("messages", [])
# #     gt = extract_ground_truth(messages)

# #     if gt is None or gt not in CANDIDATES:
# #         continue

# #     probs = get_vote_probs(messages)
# #     pred = max(probs, key=probs.get)

# #     results.append({
# #         "idx": idx,
# #         "ground_truth": gt,
# #         "predicted_vote": pred,
# #         "accuracy": int(pred == gt),
# #         "mutual_information": mutual_information(probs, gt),
# #         "probs": probs,
# #         "messages": messages,
# #     })

# #     if (idx + 1) % args.save_every == 0:
# #         pd.DataFrame(results).to_pickle(
# #             os.path.join(
# #                 args.out_dir,
# #                 f"{args.model_name.replace('/', '_')}_{args.election_year}_partial.pkl"
# #             )
# #         )

# #     time.sleep(args.sleep)

# # df = pd.DataFrame(results)


# # # =====================================================
# # # Metrics
# # # =====================================================
# # anes_votes = df["ground_truth"].map(vote_to_numeric).to_numpy()
# # gpt_votes = df["predicted_vote"].map(vote_to_numeric).to_numpy()

# # metrics = {
# #     "accuracy": df["accuracy"].mean(),
# #     "cohen_kappa": cohen_kappa_score(anes_votes, gpt_votes),
# #     "proportion_agreement": np.mean(anes_votes == gpt_votes),
# #     "mean_mutual_information": df["mutual_information"].mean(),
# # }

# # if ICC_AVAILABLE:
# #     try:
# #         df_long = (
# #             pd.DataFrame({"anes": anes_votes, "gpt": gpt_votes})
# #             .reset_index()
# #             .melt(id_vars="index", var_name="rater", value_name="vote")
# #         )
# #         icc = pg.intraclass_corr(
# #             data=df_long,
# #             targets="index",
# #             raters="rater",
# #             ratings="vote"
# #         )
# #         metrics["ICC"] = icc.loc[icc["Type"] == "ICC2k", "ICC"].values[0]
# #     except Exception:
# #         metrics["ICC"] = None
# # else:
# #     metrics["ICC"] = None

# # for k, v in metrics.items():
# #     df[k] = v


# # # =====================================================
# # # Save final outputs
# # # =====================================================
# # out_base = f"{args.model_name.replace('/', '_')}_{args.election_year}_final"
# # out_pkl = os.path.join(args.out_dir, out_base + ".pkl")
# # out_csv = os.path.join(args.out_dir, out_base + ".csv")

# # df.to_pickle(out_pkl)
# # df.to_csv(out_csv, index=False)

# # print("\n=== Final Metrics ===")
# # for k, v in metrics.items():
# #     print(f"{k}: {v}")

# # print(f"\nSaved results to:\n{out_pkl}\n{out_csv}")
