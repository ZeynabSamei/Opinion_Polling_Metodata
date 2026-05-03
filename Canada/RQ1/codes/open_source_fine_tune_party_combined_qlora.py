import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel

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
    "dataset_ft/individual_ft_party_3class.jsonl",
    "dataset_ft/agg_ft_party_2021_3class.jsonl",
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
# SPEED SETTINGS
# =============================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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


def get_candidate_ids(tokenizer):
    return [tokenizer.encode(c, add_special_tokens=False) for c in CANDIDATES]


# =============================
# FAST EVALUATION
# =============================
@torch.no_grad()
def run_evaluation(model, tokenizer, data, ft_name):

    device = next(model.parameters()).device
    cand_ids = get_candidate_ids(tokenizer)

    system_text = (
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

    valid = []

    # preprocess once
    for i, entry in enumerate(data):
        gt = extract_gt(entry["messages"])
        if gt is None:
            continue

        prompt = build_prompt(tokenizer, system_text, entry["messages"][0]["content"])
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

        batch_preds = []

        for j in range(len(prompts)):

            prompt_len = enc.attention_mask[j].sum().item()
            prompt_tokens = enc.input_ids[j, :prompt_len]

            scores = []

            for c in cand_ids:
                seq = torch.cat([
                    prompt_tokens,
                    torch.tensor(c, device=device)
                ])

                out = model(seq.unsqueeze(0))

                logits = out.logits[:, :-1, :]
                labels = seq[1:].unsqueeze(0)

                log_probs = F.log_softmax(logits, dim=-1)
                token_logp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

                scores.append(token_logp.sum().item())

            scores = np.array(scores)
            probs = np.exp(scores - scores.max())
            probs = probs / probs.sum()

            batch_preds.append(dict(zip(CANDIDATES, probs)))

        for (idx, gt, _), p in zip(batch, batch_preds):
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

    df.to_csv(os.path.join(OUT_DIR, f"llama70b_{ft_name}_results.csv"), index=False)

    with open(os.path.join(OUT_DIR, f"llama70b_{ft_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


# =============================
# TOKENIZER + QLORA
# =============================
print("Loading tokenizer...")

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# =============================
# LOAD MODEL ONCE (IMPORTANT FIX)
# =============================
print("Loading base model ONCE (70B)...")


base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)

base_model.config.pad_token_id = tokenizer.pad_token_id
base_model = prepare_model_for_kbit_training(base_model)


# =============================
# MAIN LOOP (FAST NOW)
# =============================
for ft_file in FINE_TUNE_FILES:

    ft_name = os.path.basename(ft_file).replace(".jsonl", "")

    print(f"\n====================")
    print(f"FT DATASET: {ft_name}")
    print("====================")

    # ❗ clone fresh LoRA adapter each time (NOT full model reload)
    model = get_peft_model(base_model, lora_config)
    model = prepare_model_for_kbit_training(model)

    print("Trainable params:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ---------------- FT DATA ----------------
    ft_data = [json.loads(line) for line in open(ft_file)]

    system_text = (
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

    print("Training...")
    trainer.train()

    print("Evaluating...")
    model.eval()
    run_evaluation(model, tokenizer, data, ft_name)

    # save adapter
    model.save_pretrained(os.path.join(OUT_DIR, f"llama3.1_70b_{ft_name}_lora"))

    del model
    torch.cuda.empty_cache()
