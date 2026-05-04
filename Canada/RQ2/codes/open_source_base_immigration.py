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
DATA_PATH = "./dataset_test/test_canada_immigration_2021.json"

# =====================================================
# Models
# =====================================================
MODELS = [
    # "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]


CANDIDATES = [
    "More immigrants",
    "Fewer immigrants",
    "About the same number of immigrants as now"
]

os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a classifier trained on Canadian public opinion data. "
    "Given a Canadian respondent's demographic and political attributes, "
    "predict their opinion on immigration in Canada.\n\n"
    "Output exactly one of the following labels:\n"
    "- More immigrants\n"
    "- Fewer immigrants\n"
    "- About the same number of immigrants as now\n\n"
    "Do not explain your answer. Output only the label."    
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
        return None, None

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt, user_text



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



    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception as e:
        print(f"[WARN] FlashAttention failed, falling back to SDPA. Reason: {e}")
    
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    # )

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

        prompt, user_text = build_prompt(tokenizer, messages)
        if prompt is None:
            continue
        
        samples.append((i, prompt, user_text, gt))
        

    # -------------------------------------------------
    # Inference
    # -------------------------------------------------
    results = []

    for start in tqdm(range(0, len(samples), BATCH_SIZE)):
        batch = samples[start : start + BATCH_SIZE]
        # idxs, prompts, gts = zip(*batch)
        idxs, prompts, user_texts, gts = zip(*batch)


        probs_list = score_candidates(model, tokenizer, prompts, device)

        # for idx, gt, probs in zip(idxs, gts, probs_list):
        for idx, user_text, gt, probs in zip(idxs, user_texts, gts, probs_list):

            pred = max(probs, key=probs.get)

            results.append({
                "idx": idx,
                "user_text": user_text,
                "ground_truth": gt,
                "prediction": pred,
                "correct": int(pred == gt),
                "probs": probs,
            })


    df = pd.DataFrame(results)
    print(df.head(5))
    

    metrics = compute_metrics(df)

    # -------------------------------------------------
    # Save
    # -------------------------------------------------
    model_tag = model_name.replace("/", "_")
    out_path = os.path.join(OUT_DIR, f"{model_tag}_results_immigration.csv")
    df.to_csv(out_path, index=False)

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nSaved: {out_path}")

    del model
    torch.cuda.empty_cache()



