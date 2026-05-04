import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, f1_score
from datasets import Dataset

# -----------------------------
# Config
# -----------------------------
OUT_DIR = "./results"
DATA_PATH = "dataset_test/test_canada_election_party_2021_3class_new.json"

# Fine-tune datasets
FINE_TUNE_FILES = [
    # "dataset_ft/agg_ft_party_2021_3class.jsonl",
    "dataset_ft/individual_ft_party_3class.jsonl",
    # "dataset_ft/tweets_ft_party_sample.jsonl",
]

# Models to run
MODELS = [
    # "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    # "Qwen/Qwen2.5-7B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
]


CANDIDATES = [
    "Liberal Party",
    "Conservative Party",
    "Minor Parties"
]
VOTE2ID = {c: i for i, c in enumerate(CANDIDATES)}

SEED = 42
MAX_LEN = 512
FT_EPOCHS = 1
FT_BATCH_SIZE = 2
FT_GRAD_ACCUM = 8
EVAL_BATCH_SIZE = 16

# -----------------------------
# Setup
# -----------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load dataset
# -----------------------------
with open(DATA_PATH, "r") as f:
    data = json.load(f)

print("Loaded:", len(data))

# -----------------------------
# Helpers
# -----------------------------
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
        add_generation_prompt=True
    )


def score_candidate(prompt, candidate, model, tokenizer, device):
    text = prompt + " " + candidate

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

    return token_logp.sum().item()


def get_probs(prompts, model, tokenizer, device):
    results = []

    for p in prompts:
        scores = {}
        for c in CANDIDATES:
            scores[c] = score_candidate(p, c, model, tokenizer, device)

        # softmax over scores
        exp_scores = np.exp(list(scores.values()))
        probs = exp_scores / np.sum(exp_scores)

        results.append(dict(zip(CANDIDATES, probs)))

    return results


# -----------------------------
# Main loop
# -----------------------------
for model_name in MODELS:
    for ft_file in FINE_TUNE_FILES:

        print("\n=== Model:", model_name, "FT:", ft_file, "===")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        device = next(model.parameters()).device

        # -------------------------
        # Fine-tuning data
        # -------------------------
        ft_data = []
        with open(ft_file, "r") as f:
            for line in f:
                ft_data.append(json.loads(line))

        train_texts = []

        # system_text = (
        #     "You are a political behavior model that predicts vote choice.\n"
        #     "Output only one label: Justin Trudeau, Erin O'Toole, Others."
        # )



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

        for item in ft_data:
            msgs = item["messages"]
            gt = extract_gt(msgs)
            if gt is None:
                continue

            user_text = msgs[0]["content"]

            prompt = build_prompt(tokenizer, system_text, user_text)

            train_texts.append({
                "text": prompt + " " + gt + tokenizer.eos_token
            })

        dataset = Dataset.from_list(train_texts)

        def tok(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=MAX_LEN
            )

        tokenized = dataset.map(tok, batched=True, remove_columns=["text"])

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir="./tmp",
                per_device_train_batch_size=FT_BATCH_SIZE,
                gradient_accumulation_steps=FT_GRAD_ACCUM,
                num_train_epochs=FT_EPOCHS,
                bf16=True,
                save_strategy="no",
                report_to="none",
            ),
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        trainer.train()

        model.eval()

        # -------------------------
        # Evaluation
        # -------------------------
        results = []

        # system_text = (
        #     "You are a political behavior model that predicts vote choice.\n"
        #     "Output only one label."
        # )

        system_text = (
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
        

        valid = []
        for i, entry in enumerate(data):
            gt = extract_gt(entry["messages"])
            if gt is None:
                continue

            user_text = entry["messages"][0]["content"]
            prompt = build_prompt(tokenizer, system_text, user_text)

            valid.append((i, gt, user_text, prompt))

            # valid.append((i, gt, prompt))

        for i in tqdm(range(0, len(valid), EVAL_BATCH_SIZE)):
            batch = valid[i:i+EVAL_BATCH_SIZE]
            prompts = [x[3] for x in batch]
            

            probs = get_probs(prompts, model, tokenizer, device)

            # for (idx, gt, _), p in zip(batch, probs):
            for (idx, gt, user_text, _prompt), p in zip(batch, probs):

                pred = max(p, key=p.get)

                results.append({
                    "user_text":user_text,
                    "idx": idx,
                    "gt": gt,
                    "pred": pred,
                    "acc": int(gt == pred),
                    "probs": p
                })

        df = pd.DataFrame(results)
        print(df.head(5))
        

        y_true = df["gt"].map(lambda x: VOTE2ID[x]).values
        y_pred = df["pred"].map(lambda x: VOTE2ID[x]).values

        metrics = {
            "acc": df["acc"].mean(),
            "kappa": cohen_kappa_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="macro"),
        }

        print("\nRESULTS:")
        print(metrics)

        out = f"{model_name.replace('/', '_')}_{os.path.basename(ft_file)}_party.csv"
        df.to_csv(os.path.join(OUT_DIR, out), index=False)

        del model
        torch.cuda.empty_cache()



