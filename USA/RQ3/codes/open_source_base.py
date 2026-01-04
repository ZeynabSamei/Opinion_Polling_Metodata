import argparse
import json
import time
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import pandas as pd

# =====================================================
# Arguments
# =====================================================
parser = argparse.ArgumentParser(
    description="Evaluate open-source LLMs on ANES interview-style prompts"
)

parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--out_dir", type=str, default="./output")
parser.add_argument("--election_year", type=int, choices=[2020, 2024], required=True)
parser.add_argument("--sleep", type=float, default=0.1)
# parser.add_argument("--save_every", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

# =====================================================
# Reproducibility
# =====================================================
torch.manual_seed(args.seed)

# =====================================================
# Paths
# =====================================================
DATA_PATH = Path(args.data_path)
OUT_DIR = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PKL = OUT_DIR / f"anes_{args.election_year}_{args.model_name.replace('/', '_')}_interview.pkl"
OUT_FILE = OUT_DIR / f"anes_{args.election_year}_{args.model_name.replace('/', '_')}_interview.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
model.eval()

# =====================================================
# Allowed answers (strict ANES coding)
# =====================================================
ALLOWED_ANSWERS = {
    "gender": ["male", "female"],
    "race": ["white", "black", "asian", "hispanic"],
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
# Load Interview Data
# =====================================================
print(f"Loading interview data from: {DATA_PATH}")
with open(DATA_PATH) as f:
    interviews = json.load(f)

print(f"Total prompts: {len(interviews)}")

# =====================================================
# Generation Parameters
# =====================================================
MAX_NEW_TOKENS = 8
TEMPERATURE = 0.7

# =====================================================
# Inference Loop
# =====================================================
buffer = []
processed = 0

with open(OUT_FILE, "w") as fout:
    for item in tqdm(interviews):

        system_msg = item["messages"][0]["content"]
        user_msg = item["messages"][1]["content"]
        target = item["omitted_feature"]
        # ground_truth = item["features_raw"][target]

        raw_value = item["features_raw"][target]

        # Map numeric index to text if target is in allowed answers
        if target in ALLOWED_ANSWERS:
            try:
                # If raw_value is an integer index
                ground_truth = ALLOWED_ANSWERS[target][int(raw_value)]
            except (ValueError, IndexError, TypeError):
                # fallback: keep raw value as-is if mapping fails
                ground_truth = str(raw_value)
        else:
            # if target is not in ALLOWED_ANSWERS, just use raw
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

        buffer.append({
            "election_year": args.election_year,
            "model": args.model_name,
            "omitted_feature": target,
            "ground_truth": ground_truth,
            "prompt": prompt,
            "prediction_raw": pred_raw,
            "prediction_norm": pred_norm,
            "valid": valid,
            "features_raw": item["features_raw"]
        })


        # processed += 1

        # # Periodic save
        # if processed % args.save_every == 0:
        #     for row in buffer:
        #         fout.write(json.dumps(row) + "\n")
        #     fout.flush()
        #     buffer = []

        # time.sleep(args.sleep)

    # Save remainder
    for row in buffer:
        fout.write(json.dumps(row) + "\n")

print(f"\nSaved results to: {OUT_FILE}")
# =====================================================
# Save final output as PKL
# =====================================================
df = pd.DataFrame(buffer)

with open(OUT_PKL, "wb") as f:
    pickle.dump(df, f)

print(f"\nSaved final results to: {OUT_PKL}")
print(f"Total rows: {len(df)}")

