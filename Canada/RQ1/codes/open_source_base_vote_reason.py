# =====================================================
# LLM Voting Behavior Evaluation (Reasoning)
# Canada Federal Election 2021
# =====================================================

import os
import json
import random
import numpy as np
import pandas as pd
import torch

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
DATA_PATH = "./dataset_test/test_canada_election_vote_2021_3class_new.json"


# =====================================================
# Models
# =====================================================

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    # "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]

CANDIDATES = [
    "Justin Trudeau",
    "Erin O'Toole",
    "Others",
]

os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================
# System Prompt (Reasoning)
# =====================================================

SYSTEM_PROMPT = (
    "You are a political behavior model that predicts voting choice based on demographic profiles.\n\n"

    "Task:\n"
    "Given a person's demographic and political attributes, predict their MOST LIKELY vote choice "
    "in the 2021 Canadian federal election.\n\n"

    "Reason carefully about demographic characteristics, political alignment, and typical voting "
    "patterns before making your prediction.\n\n"

    "Output format:\n"
    "Return ONLY a valid JSON object with exactly two keys:\n"
    "{\n"
    '  "predict": value of predict,\n'
    '  "reason": brief explanation of the reasoning\n'
    "}\n\n"

    "Rules:\n"
    "- The value of predict must be exactly one of: Justin Trudeau, Erin O'Toole, Others.\n"
    "- The reason should explain the main demographic and political factors considered.\n"
    "- Do not include markdown.\n"
    "- Do not include any text outside the JSON object.\n\n"

    "Definition:\n"
    "'Others' includes Jagmeet Singh, Yves-François Blanchet, Annamie Paul, and Maxime Bernier."
)


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
# Generation
# =====================================================

@torch.no_grad()
def generate_predictions(model, tokenizer, prompts, device):
    outputs = []

    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)

        generated = model.generate(
            **enc,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

        output = tokenizer.decode(
            generated[0][enc.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        outputs.append(output)

    return outputs


def parse_reasoning_output(text):
    try:
        result = json.loads(text)

        prediction = result.get("predict", None)
        reason = result.get("reason", "")

        if prediction not in CANDIDATES:
            prediction = "Others"

        return prediction, reason

    except Exception:
        for c in CANDIDATES:
            if c in text:
                return c, text

        return "Others", text


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

        batch = samples[start:start + BATCH_SIZE]

        idxs, prompts, user_texts, gts = zip(*batch)

        outputs = generate_predictions(
            model,
            tokenizer,
            prompts,
            device
        )

        for idx, user_text, gt, output in zip(
            idxs,
            user_texts,
            gts,
            outputs
        ):

            pred, reason = parse_reasoning_output(output)

            results.append({
                "idx": idx,
                "user_text": user_text,
                "ground_truth": gt,
                "prediction": pred,
                "reason": reason,
                "raw_output": output,
                "correct": int(pred == gt),
            })


    df = pd.DataFrame(results)

    print(df.head(5))

    metrics = compute_metrics(df)


    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    model_tag = model_name.replace("/", "_")

    out_path = os.path.join(
        OUT_DIR,
        f"{model_tag}_results_vote_reasoning.csv"
    )

    df.to_csv(out_path, index=False)

    print("\n=== Metrics ===")

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nSaved: {out_path}")

    del model
    torch.cuda.empty_cache()
