import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, f1_score


# =============================
# Config
# =============================
OUT_DIR = "./results"
DATA_PATH = "dataset_test/test_canada_election_party_2021_3class_new.json"
# FINE_TUNE_FILES = ["dataset_ft/individual_ft_party_3class.jsonl"]

FINE_TUNE_FILES = [
    "dataset_ft/agg_ft_party_2021_3class.jsonl",
    "dataset_ft/individual_ft_party_3class.jsonl",
    "dataset_ft/tweets_ft_party_sample.jsonl",
]

MODEL_NAME = "meta-llama/Llama-3.1-70B-Instruct"

CANDIDATES = ["Liberal Party", "Conservative Party", "Minor Parties"]
VOTE2ID = {c: i for i, c in enumerate(CANDIDATES)}

SEED = 42
MAX_LEN = 512

FT_EPOCHS = 1
FT_BATCH_SIZE = 1
FT_GRAD_ACCUM = 16
EVAL_BATCH_SIZE = 8


# =============================
# Setup
# =============================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)


# =============================
# Load Data
# =============================
with open(DATA_PATH, "r") as f:
    data = json.load(f)

print("Loaded eval:", len(data))


# =============================
# Helpers
# =============================
def normalize_vote(x):
    if x is None:
        return None
    x = str(x).lower()
    for c in CANDIDATES:
        if c.lower() in x:
            return c
    return None


def extract_gt(messages):
    for m in messages:
        if m["role"] == "assistant":
            return normalize_vote(m["content"])
    return None


def build_prompt(tokenizer, system_text, user_text):
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# =============================
# Batched scoring (FAST)
# =============================
def get_candidate_token_ids(tokenizer):
    return [
        tokenizer.encode(c, add_special_tokens=False)
        for c in CANDIDATES
    ]


def get_probs_batched(prompts, model, tokenizer, device, candidate_token_ids):
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(device)

    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    batch_size = input_ids.size(0)

    all_probs = []

    for i in range(batch_size):
        prompt_len = attention_mask[i].sum().item()
        prompt_tokens = input_ids[i, :prompt_len]

        scores = []

        for cand_ids in candidate_token_ids:
            seq = torch.cat([
                prompt_tokens,
                torch.tensor(cand_ids, device=device)
            ])

            with torch.no_grad():
                out = model(seq.unsqueeze(0))

            logits = out.logits[:, :-1, :]
            labels = seq[1:].unsqueeze(0)

            log_probs = F.log_softmax(logits, dim=-1)
            token_logp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

            scores.append(token_logp.sum().item())

        probs = torch.softmax(torch.tensor(scores), dim=0).cpu().numpy()
        all_probs.append(dict(zip(CANDIDATES, probs)))

    return all_probs


# =============================
# Load Model (QLoRA)
# =============================
print("\nLoading model with QLoRA...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

print("Trainable params:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))


# =============================
# Fine-tuning
# =============================
for ft_file in FINE_TUNE_FILES:

    print("\nFT dataset:", ft_file)

    ft_data = [json.loads(line) for line in open(ft_file)]

    system_text = (
        "You are an expert political analyst.\n"
        "Predict vote choice.\n"
        "Output ONLY:\n"
        "Liberal Party\nConservative Party\nMinor Parties"
    )

    train_samples = []

    for item in ft_data:
        msgs = item["messages"]
        gt = extract_gt(msgs)
        if gt is None:
            continue

        user_text = msgs[0]["content"]
        prompt = build_prompt(tokenizer, system_text, user_text)

        train_samples.append({
            "text": prompt + " " + gt + tokenizer.eos_token
        })

    dataset = Dataset.from_list(train_samples)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LEN,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp",
            per_device_train_batch_size=FT_BATCH_SIZE,
            gradient_accumulation_steps=FT_GRAD_ACCUM,
            num_train_epochs=FT_EPOCHS,
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    model.train()
    trainer.train()
    model.eval()


# =============================
# Evaluation function
# =============================
def run_evaluation(model, tokenizer, data, ft_name):
    candidate_token_ids = get_candidate_token_ids(tokenizer)

    system_text_eval = (
        "Predict vote choice.\n"
        "Output ONLY:\n"
        "Liberal Party\nConservative Party\nMinor Parties"
    )

    valid = []

    for i, entry in enumerate(data):
        gt = extract_gt(entry["messages"])
        if gt is None:
            continue

        user_text = entry["messages"][0]["content"]
        prompt = build_prompt(tokenizer, system_text_eval, user_text)

        valid.append((i, gt, prompt))

    results = []

    device = next(model.parameters()).device

    for i in tqdm(range(0, len(valid), EVAL_BATCH_SIZE), desc=f"Eval {ft_name}"):
        batch = valid[i:i + EVAL_BATCH_SIZE]
        prompts = [x[2] for x in batch]

        probs = get_probs_batched(
            prompts, model, tokenizer, device, candidate_token_ids
        )

        for (idx, gt, _), p in zip(batch, probs):
            pred = max(p, key=p.get)

            results.append({
                "idx": idx,
                "gt": gt,
                "pred": pred,
                "acc": int(gt == pred),
                "probs": p
            })

    df = pd.DataFrame(results)

    y_true = df["gt"].map(VOTE2ID).values
    y_pred = df["pred"].map(VOTE2ID).values

    metrics = {
        "acc": float(df["acc"].mean()),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="macro")),
    }

    # ===== Print nicely =====
    print(f"\n=== RESULTS ({ft_name}) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # ===== Save outputs =====
    df_path = os.path.join(OUT_DIR, f"llama70b_{ft_name}_results.csv")
    metrics_path = os.path.join(OUT_DIR, f"llama70b_{ft_name}_metrics.json")

    df.to_csv(df_path, index=False)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved:\n{df_path}\n{metrics_path}")


# =============================
# Fine-tuning + evaluation loop
# =============================
for ft_file in FINE_TUNE_FILES:

    ft_name = os.path.basename(ft_file).replace(".jsonl", "")
    print(f"\n\n============================")
    print(f"FT DATASET: {ft_name}")
    print(f"============================")

    # -------- Reload base model each time (IMPORTANT) --------
    print("Reloading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # -------- Load FT data --------
    ft_data = [json.loads(line) for line in open(ft_file)]

    system_text = (
        "You are an expert political analyst.\n"
        "Predict vote choice.\n"
        "Output ONLY:\n"
        "Liberal Party\nConservative Party\nMinor Parties"
    )

    train_samples = []

    for item in ft_data:
        msgs = item["messages"]
        gt = extract_gt(msgs)
        if gt is None:
            continue

        user_text = msgs[0]["content"]
        prompt = build_prompt(tokenizer, system_text, user_text)

        train_samples.append({
            "text": prompt + " " + gt + tokenizer.eos_token
        })

    dataset = Dataset.from_list(train_samples)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LEN,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./tmp",
            per_device_train_batch_size=FT_BATCH_SIZE,
            gradient_accumulation_steps=FT_GRAD_ACCUM,
            num_train_epochs=FT_EPOCHS,
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    # -------- Train --------
    print("Training...")
    model.train()
    trainer.train()

    # -------- Evaluate immediately --------
    model.eval()
    run_evaluation(model, tokenizer, data, ft_name)

    # -------- Cleanup (VERY important for 70B) --------
    del model
    torch.cuda.empty_cache()
