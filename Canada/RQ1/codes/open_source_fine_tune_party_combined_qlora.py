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


def get_candidate_token_ids(tokenizer):
    return [tokenizer.encode(c, add_special_tokens=False) for c in CANDIDATES]


# =============================
# EVALUATION FUNCTION (FIXED)
# =============================
def run_evaluation(model, tokenizer, data, ft_name):

    candidate_token_ids = get_candidate_token_ids(tokenizer)

    system_text_eval = (
        "You are an expert political analyst specializing in Canadian elections and voting behavior. "
        "Task:\n"
        "Given a person's demographic and political attributes, predict their MOST LIKELY party choice "
        "in the 2021 Canadian federal election.\n\n"
    
        "Rules:\n"
        "- You must choose ONLY ONE label.\n"
        "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
        "Liberal Party\n"
        "Conservative Party\n"
        "Minor Parties\n\n"
    
        "Definition:\n"
        "Minor Parties' includes New Democratic Party(NDP), Bloc Québécois, Green Party, and People's Party of Canada.\n\n"
       
        "Important:\n"
        "- Base your decision on typical voting patterns, demographics, and political alignment.\n"
        "- Do NOT explain your reasoning.\n"
        "- Do NOT repeat the input.\n"
        "- Output ONLY the label."
    )

    device = next(model.parameters()).device

    valid = []
    for i, entry in enumerate(data):
        gt = extract_gt(entry["messages"])
        if gt is None:
            continue

        user_text = entry["messages"][0]["content"]
        prompt = build_prompt(tokenizer, system_text_eval, user_text)

        valid.append((i, gt, prompt))

    results = []

    for i in tqdm(range(0, len(valid), EVAL_BATCH_SIZE), desc=f"Eval {ft_name}"):

        batch = valid[i:i + EVAL_BATCH_SIZE]
        prompts = [x[2] for x in batch]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(device)

        batch_probs = []

        for j in range(len(prompts)):

            prompt_len = enc.attention_mask[j].sum().item()
            prompt_tokens = enc.input_ids[j, :prompt_len]

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
            batch_probs.append(dict(zip(CANDIDATES, probs)))

        for (idx, gt, _), p in zip(batch, batch_probs):
            pred = max(p, key=p.get)

            results.append({
                "idx": idx,
                "gt": gt,
                "pred": pred,
                "acc": int(gt == pred),
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

    # ===== PRINT =====
    print(f"\n=== RESULTS: {ft_name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # ===== SAVE =====
    df_path = os.path.join(OUT_DIR, f"{ft_name}_results.csv")
    met_path = os.path.join(OUT_DIR, f"{ft_name}_metrics.json")

    df.to_csv(df_path, index=False)
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved:\n{df_path}\n{met_path}")


# =============================
# TOKENIZER + QLoRA SETUP
# =============================
print("\nLoading tokenizer + QLoRA config...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


# =============================
# MAIN LOOP
# =============================
for ft_file in FINE_TUNE_FILES:

    ft_name = os.path.basename(ft_file).replace(".jsonl", "")

    print(f"\n\n====================")
    print(f"FT DATASET: {ft_name}")
    print(f"====================")

    # -------- Load model --------
    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.config.pad_token_id = tokenizer.pad_token_id

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

        prompt = build_prompt(tokenizer, system_text, msgs[0]["content"])

        train_samples.append({
            "text": prompt + " " + gt + tokenizer.eos_token
        })

    print(f"Training samples: {len(train_samples)}")

    dataset = Dataset.from_list(train_samples)

    def tokenize(x):
        return tokenizer(x["text"], truncation=True, max_length=MAX_LEN)

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
            seed=SEED,
        ),
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # -------- TRAIN --------
    print("Training...")
    trainer.train()

    # -------- EVAL --------
    print("Evaluating...")
    model.eval()

    run_evaluation(model, tokenizer, data, ft_name)

    # -------- SAVE ADAPTER --------
    model.save_pretrained(os.path.join(OUT_DIR, f"{ft_name}_lora"))

    # -------- CLEANUP --------
    del model
    torch.cuda.empty_cache()
