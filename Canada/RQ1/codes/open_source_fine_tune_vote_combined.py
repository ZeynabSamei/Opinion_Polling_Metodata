# =====================================================
# Canada 2021 Election Vote Prediction with LLMs
# Runs multiple models and multiple fine-tune datasets
# =====================================================

import os
import time
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.metrics import cohen_kappa_score
from datasets import Dataset

# Optional ICC
try:
    import pingouin as pg
    ICC_AVAILABLE = True
except ImportError:
    ICC_AVAILABLE = False

# =====================================================
# Configuration
# =====================================================


OUT_DIR = "./result"
ELECTION_YEAR = "2021"
SAVE_EVERY = 10000
SLEEP = 0
SEED = 42
FT_EPOCHS = 1
FT_BATCH_SIZE = 2

# Dataset for evaluation (JSON)
DATA_PATH = "dataset_test/test_canada_election_vote_2021_3class.json"

# Fine-tune datasets
FINE_TUNE_FILES = [
    "dataset_ft/agg_ft_vote.jsonl",
    "dataset_ft/individual_ft_vote_sample2.jsonl",
    "dataset_ft/tweets_ft_vote_sample2.jsonl",
]

# Models to run
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct"
]

# =====================================================
# Candidates
# =====================================================
CANDIDATES = ["Justin Trudeau", "Erin O'Toole", "Others"]
VOTE2ID = {c: i for i, c in enumerate(CANDIDATES)}
ID2VOTE = {i: c for c, i in VOTE2ID.items()}

# =====================================================
# Helper functions
# =====================================================
def normalize_vote(text):
    if text is None:
        return None
    text = text.lower()
    for c in CANDIDATES:
        if c.lower() in text:
            return c
    return None

def extract_ground_truth(messages):
    for m in messages:
        if m["role"] == "assistant":
            return normalize_vote(m["content"])
    return None

def get_vote_probs(messages, model, tokenizer, device):
    clean_msgs = [m for m in messages if m["role"] != "assistant"]
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
    prompt += f"\nVote choice ({' or '.join(CANDIDATES)}):"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    probs = {}

    with torch.no_grad():
        for candidate in CANDIDATES:
            cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
            prob = 1.0
            cur_ids = input_ids.clone()
            for tok in cand_ids:
                outputs = model(input_ids=cur_ids)
                logits = outputs.logits[:, -1, :]
                token_probs = torch.softmax(logits, dim=-1)
                prob *= token_probs[0, tok].item()
                cur_ids = torch.cat([cur_ids, torch.tensor([[tok]], device=device)], dim=1)
            probs[candidate] = prob

    total = sum(probs.values())
    if total > 0:
        probs = {c: p / total for c, p in probs.items()}
    else:
        probs = {c: 1 / len(CANDIDATES) for c in CANDIDATES}

    return probs

def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)

def vote_to_numeric(vote):
    return VOTE2ID[vote]

# =====================================================
# Load evaluation dataset
# =====================================================
with open(DATA_PATH, "r") as f:
    data = json.load(f)
print(f"Loaded {len(data)} evaluation samples")

# =====================================================
# Set seeds
# =====================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================================================
# Main loop: models × fine-tune datasets
# =====================================================
os.makedirs(OUT_DIR, exist_ok=True)

for model_name in MODELS:
    print(f"\n=== Loading model: {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    device = next(model.parameters()).device

    for ft_file in FINE_TUNE_FILES:
        print(f"\n--- Fine-tuning with dataset: {ft_file} ---")
        ft_data = []
        with open(ft_file, "r") as f:
            for line in f:
                ft_data.append(json.loads(line))
        print(f"Loaded {len(ft_data)} fine-tune samples")

        if len(ft_data) > 0:
            # Prepare dataset
            ft_texts = []
            for item in ft_data:
                prompt = "\n".join(f"{m['role']}: {m['content']}" for m in item.get("messages", []))
                target = ""
                for m in item.get("messages", []):
                    if m["role"] == "assistant":
                        target = m["content"]
                ft_texts.append({"text": prompt + tokenizer.eos_token + target + tokenizer.eos_token})

            dataset = Dataset.from_list(ft_texts)

            def tokenize_fn(examples):
                return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

            tokenized_ds = dataset.map(tokenize_fn, batched=True)
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            training_args = TrainingArguments(
                output_dir=os.path.join(OUT_DIR, f"{model_name.replace('/', '_')}_ft"),
                per_device_train_batch_size=FT_BATCH_SIZE,
                gradient_accumulation_steps=8,
                num_train_epochs=FT_EPOCHS,
                logging_steps=50,
                save_strategy="no",
                fp16=False,
                bf16=True,
                seed=SEED
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_ds,
                data_collator=data_collator
            )

            trainer.train()
            print("Fine-tuning completed.")

        # =====================================================
        # Inference
        # =====================================================
        results = []
        for idx, entry in tqdm(enumerate(data), total=len(data)):
            messages = entry.get("messages", [])
            gt = extract_ground_truth(messages)
            if gt is None or gt not in CANDIDATES:
                continue

            probs = get_vote_probs(messages, model, tokenizer, device)
            pred = max(probs, key=probs.get)

            results.append({
                "idx": idx,
                "ground_truth": gt,
                "predicted_vote": pred,
                "accuracy": int(pred == gt),
                "mutual_information": mutual_information(probs, gt),
                "probs": probs,
                "messages": messages,
            })

            if (idx + 1) % SAVE_EVERY == 0:
                pd.DataFrame(results).to_pickle(
                    os.path.join(OUT_DIR, f"{model_name.replace('/', '_')}_{os.path.basename(ft_file).replace('.jsonl','')}_{ELECTION_YEAR}_partial.pkl")
                )

            time.sleep(SLEEP)

        df = pd.DataFrame(results)

        # =====================================================
        # Metrics
        # =====================================================
        anes_votes = df["ground_truth"].map(vote_to_numeric).to_numpy()
        gpt_votes = df["predicted_vote"].map(vote_to_numeric).to_numpy()

        metrics = {
            "accuracy": df["accuracy"].mean(),
            "cohen_kappa": cohen_kappa_score(anes_votes, gpt_votes),
            "proportion_agreement": np.mean(anes_votes == gpt_votes),
            "mean_mutual_information": df["mutual_information"].mean(),
        }

        if ICC_AVAILABLE:
            try:
                df_long = (
                    pd.DataFrame({"anes": anes_votes, "gpt": gpt_votes})
                    .reset_index()
                    .melt(id_vars="index", var_name="rater", value_name="vote")
                )
                icc = pg.intraclass_corr(
                    data=df_long,
                    targets="index",
                    raters="rater",
                    ratings="vote"
                )
                metrics["ICC"] = icc.loc[icc["Type"] == "ICC2k", "ICC"].values[0]
            except Exception:
                metrics["ICC"] = None
        else:
            metrics["ICC"] = None

        for k, v in metrics.items():
            df[k] = v

        # =====================================================
        # Save final outputs
        # =====================================================
        ft_name = os.path.basename(ft_file).replace(".jsonl", "")
        out_base = f"{model_name.replace('/', '_')}_{ft_name}_{ELECTION_YEAR}_final"
        out_pkl = os.path.join(OUT_DIR, out_base + ".pkl")
        out_csv = os.path.join(OUT_DIR, out_base + ".csv")

        df.to_pickle(out_pkl)
        df.to_csv(out_csv, index=False)

        print("\n=== Final Metrics ===")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print(f"\nSaved results to:\n{out_pkl}\n{out_csv}")
