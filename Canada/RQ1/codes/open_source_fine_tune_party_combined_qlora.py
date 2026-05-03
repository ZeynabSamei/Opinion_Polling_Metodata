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
# CONFIG
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
# SETUP
# =============================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)


# =============================
# LOAD DATA
# =============================
with open(DATA_PATH, "r") as f:
    data = json.load(f)

print("Loaded eval:", len(data))


# =============================
# HELPERS
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
# QLoRA CONFIG
# =============================
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
# FASTER EVALUATION (FIXED LOGIC)
# =============================
def get_probs(prompts, model, tokenizer, device):
    results = []

    for prompt in prompts:
        scores = {}

        for c in CANDIDATES:
            text = prompt + " " + c

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN
            ).to(device)

            with torch.no_grad():
                out = model(**enc)

            logits = out.logits[:, :-1, :]
            labels = enc.input_ids[:, 1:]

            log_probs = F.log_softmax(logits, dim=-1)
            token_logp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

            scores[c] = token_logp.sum().item()

        exp_scores = np.exp(list(scores.values()))
        probs = exp_scores / np.sum(exp_scores)

        results.append(dict(zip(CANDIDATES, probs)))

    return results


# =============================
# EVALUATION FUNCTION
# =============================
def run_evaluation(model, tokenizer, data, ft_name):

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

        probs = get_probs(prompts, model, tokenizer, device)

        for (idx, gt, _), p in zip(batch, probs):
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

    print(f"\n=== RESULTS: {ft_name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    df_path = os.path.join(OUT_DIR, f"llama70b_{ft_name}_results.csv")
    met_path = os.path.join(OUT_DIR, f"llama70b_{ft_name}_metrics.json")

    df.to_csv(df_path, index=False)
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved:\n{df_path}\n{met_path}")


# =============================
# MAIN LOOP
# =============================
for ft_file in FINE_TUNE_FILES:

    ft_name = os.path.basename(ft_file).replace(".jsonl", "")

    print(f"\n\n====================")
    print(f"FT DATASET: {ft_name}")
    print("====================")

    # ---------------- MODEL ----------------
    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_state_dict=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    print("Trainable params:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ---------------- FT DATA ----------------
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

    # ---------------- TRAIN ----------------
    print("Training...")
    trainer.train()

    # ---------------- EVAL ----------------
    print("Evaluating...")
    model.eval()
    run_evaluation(model, tokenizer, data, ft_name)

    # ---------------- SAVE ----------------
    model.save_pretrained(os.path.join(OUT_DIR, f"llama3.1_70b_{ft_name}_lora"))

    del model
    torch.cuda.empty_cache()
