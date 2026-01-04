import argparse
import json
import time
import torch
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import pickle
import pandas as pd

# Optional PEFT
try:
    from peft import LoraConfig, get_peft_model, TaskType
    peft_available = True
except ImportError:
    peft_available = False

# =====================================================
# Arguments
# =====================================================
parser = argparse.ArgumentParser(
    description="Fine-tune (optional) + evaluate LLM on ANES interview prompts"
)

parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--fine_tune_data", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./output")
parser.add_argument("--election_year", type=int, choices=[2020, 2024], required=True)
parser.add_argument("--ft_epochs", type=int, default=1)
parser.add_argument("--ft_batch_size", type=int, default=4)
parser.add_argument("--max_seq_length", type=int, default=192)
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sleep", type=float, default=0.1)

args = parser.parse_args()

# =====================================================
# Reproducibility
# =====================================================
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# Paths
# =====================================================
DATA_PATH = Path(args.data_path)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PKL = OUT_DIR / f"anes_{args.election_year}_{args.model_name.replace('/', '_')}_interview.pkl"
OUT_FILE = OUT_DIR / f"anes_{args.election_year}_{args.model_name.replace('/', '_')}_interview.jsonl"

# =====================================================
# Load model
# =====================================================
print(f"\nLoading model: {args.model_name}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

# =====================================================
# Optional LoRA
# =====================================================
if args.use_lora:
    if not peft_available:
        raise ImportError("Install peft to use LoRA")
    print("Applying LoRA...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_cfg)

# =====================================================
# Optional fine-tuning
# =====================================================
if args.fine_tune_data:
    print("Starting fine-tuning...")
    with open(args.fine_tune_data) as f:
        ft_data = [json.loads(l) for l in f]



def build_ft_text(item):
    """
    Normalize different FT dataset formats into a single text string
    """

    # Case 1: Chat-style dataset (messages)
    if "messages" in item:
        return "\n".join(
            f"{m['role']}: {m['content']}"
            for m in item["messages"]
            if "role" in m and "content" in m
        )

    # Case 2: Prompt / completion format
    if "prompt" in item and "completion" in item:
        return item["prompt"] + tokenizer.eos_token + item["completion"]

    # Case 3: OpenAI-style chat completion
    if "prompt" in item and "response" in item:
        return item["prompt"] + tokenizer.eos_token + item["response"]

    # Case 4: Already flattened text
    if "text" in item:
        return item["text"]

    # Case 5: Fallback (safe)
    return json.dumps(item)


print("Starting fine-tuning...")

ft_texts = []
for item in ft_data:
    text = build_ft_text(item)
    if text and isinstance(text, str):
        ft_texts.append({"text": text + tokenizer.eos_token})

    ds = Dataset.from_list(ft_texts)

    def tok_fn(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length
        )

    ds = ds.map(tok_fn, batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=OUT_DIR / "ft_model",
        per_device_train_batch_size=args.ft_batch_size,
        num_train_epochs=args.ft_epochs,
        save_strategy="no",
        logging_steps=50,
        bf16=DEVICE == "cuda",
        learning_rate=1e-4,
        seed=args.seed
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=collator
    )

    trainer.train()
    print("Fine-tuning done.")

model.eval()

# =====================================================
# Allowed answers (UNCHANGED)
# =====================================================
ALLOWED_ANSWERS = {
    "gender": ["male", "female"],
    "race": ["white", "black", "hispanic", "asian", "native american", "mixed race"],
    "church_attendance": ["yes", "no"],
    "pol_interest": [
        "very interested",
        "somewhat interested",
        "not very interested",
        "not at all interested"
    ],
    "vote_choice": [
        "kamala harris",
        "donald trump",
        "someone else"
    ],
    "ideology": [
        "extremely liberal",
        "liberal",
        "slightly liberal",
        "moderate",
        "slightly conservative",
        "conservative",
        "extremely conservative"
    ]
}

# =====================================================
# Load interview data
# =====================================================
with open(DATA_PATH) as f:
    interviews = json.load(f)

print(f"Total prompts: {len(interviews)}")

# =====================================================
# Generation params (UNCHANGED)
# =====================================================
MAX_NEW_TOKENS = 8
TEMPERATURE = 0.7

# =====================================================
# Inference loop (UNCHANGED)
# =====================================================
buffer = []

with open(OUT_FILE, "w") as fout:
    for item in tqdm(interviews):

        system_msg = item["messages"][0]["content"]
        user_msg = item["messages"][1]["content"]
        target = item["omitted_feature"]
        raw_value = item["features_raw"][target]

        if target in ALLOWED_ANSWERS:
            try:
                ground_truth = ALLOWED_ANSWERS[target][int(raw_value)-1]
            except Exception:
                ground_truth = str(raw_value)
        else:
            ground_truth = str(raw_value)

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        pred_raw = decoded.strip()
        pred_norm = pred_raw.lower().strip()

        valid = True
        if target in ALLOWED_ANSWERS:
            valid = pred_norm in ALLOWED_ANSWERS[target]

        row = {
            "election_year": args.election_year,
            "model": args.model_name,
            "omitted_feature": target,
            "ground_truth": ground_truth,
            "prompt": prompt,
            "prediction_raw": pred_raw,
            "prediction_norm": pred_norm,
            "valid": valid,
            "features_raw": item["features_raw"]
        }

        fout.write(json.dumps(row) + "\n")
        buffer.append(row)

        time.sleep(args.sleep)

# =====================================================
# Save PKL (UNCHANGED)
# =====================================================
df = pd.DataFrame(buffer)
with open(OUT_PKL, "wb") as f:
    pickle.dump(df, f)

print(f"\nSaved JSONL to: {OUT_FILE}")
print(f"Saved PKL to: {OUT_PKL}")
print(f"Total rows: {len(df)}")
