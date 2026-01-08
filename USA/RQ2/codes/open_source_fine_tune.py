import os
import json
import random
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import argparse
from tqdm import tqdm

# Optional PEFT / LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType
    peft_available = True
except ImportError:
    peft_available = False

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Fast fine-tune + inference with optional LoRA")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--fine_tune_data", type=str, default=None)
parser.add_argument("--out_dir", type=str, default="./results")
parser.add_argument("--election_year", type=int, choices=[2020, 2024], required=True)
parser.add_argument("--n_samples", type=int, default=10)
parser.add_argument("--ft_epochs", type=int, default=1)
parser.add_argument("--ft_batch_size", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
args = parser.parse_args()

# -----------------------------
# Setup
# -----------------------------
os.makedirs(args.out_dir, exist_ok=True)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# -----------------------------
# Election year → candidates
# -----------------------------
if args.election_year == 2020:
    CANDIDATES = ["Donald Trump", "Joe Biden"]

elif args.election_year == 2024:
    CANDIDATES = ["Donald Trump", "Kamala Harris"]


CANDIDATES_NORM = [c.lower() for c in CANDIDATES]

# -----------------------------
# Load datasets
# -----------------------------
with open(args.data_path, "r") as f:
    test_data = json.load(f)
print(f"Loaded {len(test_data)} test samples")

ft_data = None
if args.fine_tune_data:
    with open(args.fine_tune_data, "r") as f:
        ft_data = [json.loads(line) for line in f]  # JSONL
    print(f"Loaded {len(ft_data)} fine-tune samples")

# -----------------------------
# Load model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.gradient_checkpointing_enable()
model.config.use_cache = False
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.eos_token_id
model.eval()
device = model.device if hasattr(model, "device") else next(model.parameters()).device

# -----------------------------
# Apply LoRA if requested
# -----------------------------
if args.use_lora:
    if not peft_available:
        raise ImportError("PEFT is not installed. Install via `pip install peft` to use LoRA.")
    print("Applying LoRA adapters...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    print("LoRA applied!")

# -----------------------------
# Helper functions
# -----------------------------
def normalize_vote(text):
    if text is None:
        return "Other"
    t = text.lower()
    for c in CANDIDATES:
        if c.lower() in t:
            return c
    return "Other"



def extract_ground_truth(messages):
    for m in messages:
        if m.get("role") == "assistant":
            return normalize_vote(m.get("content"))
    return None

def get_vote_probs(messages):
    clean_msgs = [m for m in messages if m["role"] != "assistant"]
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
    prompt += f"\nVote choice ({' or '.join(CANDIDATES)}):"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    probs = {}

    with torch.no_grad():
        for candidate in CANDIDATES:
            token_ids = tokenizer.encode(candidate, add_special_tokens=False)
            p = 1.0
            curr = input_ids.clone()

            for tid in token_ids:
                logits = model(curr).logits[:, -1, :]
                token_probs = torch.softmax(logits, dim=-1)
                p *= token_probs[0, tid].item()
                curr = torch.cat(
                    [curr, torch.tensor([[tid]], device=device)], dim=1
                )

            probs[candidate] = p

    # Binary normalization
    Z = sum(probs.values())
    if Z > 0:
        probs = {k: v / Z for k, v in probs.items()}
    else:
        probs = {c: 0.5 for c in CANDIDATES}

    return probs

def accuracy_from_probs(probs, ground_truth):
    return int(max(probs, key=probs.get) == ground_truth)

def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)


def probs_to_vector(probs):
    return np.array([
        probs["Donald Trump"],
        probs["Kamala Harris"],
        probs["Other"]
    ])


# -----------------------------
# Fine-tuning (optional, robust)
# -----------------------------
if ft_data:
    print("Starting fine-tuning ...")
    ft_texts = []

    for item in ft_data:
        # Chat-completion format
        if "messages" in item:
            messages = item["messages"]
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m.get("role") != "assistant")
            target = next((m["content"] for m in messages if m.get("role")=="assistant"), "")
        # Completion format
        elif "prompt" in item and "completion" in item:
            prompt = item["prompt"]
            target = item["completion"]
        # Unknown format → skip
        else:
            print(f"Skipping FT item (unknown format): {item}")
            continue

        ft_texts.append({"text": prompt + tokenizer.eos_token + target + tokenizer.eos_token})

    dataset = Dataset.from_list(ft_texts)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=192)
    tokenized_ds = dataset.map(tokenize_fn, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "ft_model"),
        per_device_train_batch_size=args.ft_batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=args.ft_epochs,
        logging_steps=50,
        save_strategy="no",
        bf16=True,
        seed=args.seed,
        learning_rate=1e-4,
        warmup_ratio=0.1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    print("Fine-tuning completed.")

# -----------------------------
# Inference
# -----------------------------
results = []
for idx, entry in tqdm(enumerate(test_data), total=len(test_data)):
    messages = entry.get("messages", [])
    gt = extract_ground_truth(messages)
    if gt is None:
        continue

    probs = get_vote_probs(messages)
    pred = max(probs, key=probs.get)
    acc = accuracy_from_probs(probs, gt)
    mi = mutual_information(probs, gt)

    results.append({
        "idx": idx,
        "messages": messages,
        "ground_truth": gt,
        "predicted_vote": pred,
        "probs": probs,
        "accuracy": acc,
        "mutual_inf": mi
    })

# -----------------------------
# Convert results to DataFrame
# -----------------------------
df_final = pd.DataFrame(results)
df_final["llm_binary"] = df_final["predicted_vote"]
df_final["anes_binary"] = df_final["ground_truth"]

df_final["llm_num"] = df_final["llm_binary"].apply(
    lambda x: 0 if x == CANDIDATES[1] else 1
)
df_final["anes_num"] = df_final["anes_binary"].apply(
    lambda x: 0 if x == CANDIDATES[1] else 1
)


-----------------------------
Tetrachoric correlation
-----------------------------
def tetrachoric_corr_safe(vec1, vec2):
    A = np.sum((vec1 == 0) & (vec2 == 0))
    B = np.sum((vec1 == 0) & (vec2 == 1))
    C = np.sum((vec1 == 1) & (vec2 == 0))
    D = np.sum((vec1 == 1) & (vec2 == 1))
    if (A+B)==0 or (C+D)==0 or (A+C)==0 or (B+D)==0:
        return np.nan
    try:
        return np.cos(np.pi / (1 + np.sqrt((A*D)/(B*C))))
    except:
        return np.nan

tetra = tetrachoric_corr_safe(df_final["llm_num"].values, df_final["anes_num"].values)

# -----------------------------
# Bias computation
# -----------------------------
summary_rows = []
row = {
    "Variable": "Wholesample",
    "n_samples": len(df_final),
    # "Tetra": tetra,
    "Prop.Agree": np.mean(df_final["llm_binary"] == df_final["anes_binary"])
}

for c in CANDIDATES:
    real_pct = np.mean(df_final["anes_binary"] == c)
    llm_pct = np.mean([p[c] for p in df_final["probs"]])
    row[f"RealPct_{c}"] = real_pct
    row[f"LLMPct_{c}"] = llm_pct
    row[f"Bias_{c}"] = llm_pct - real_pct


summary_rows.append(row)
df_summary = pd.DataFrame(summary_rows)


# -----------------------------
# Save outputs
# -----------------------------
final_path = os.path.join(args.out_dir, f"{args.model_name.replace('/', '_')}_{args.election_year}_final.pkl")
summary_path = final_path.replace(".pkl", "_summary.csv")

df_final.to_pickle(final_path)


# -----------------------------
# Save detailed results
# -----------------------------
def safe_name(s):
    return str(s).replace("/", "_").replace(" ", "_")

out_file = os.path.join(
    args.out_dir,
    f"{safe_name(args.model_name)}_{args.election_year}_{safe_name(args.fine_tune_data)}_results.pkl"
)
pd.DataFrame(results).to_pickle(out_file)
pd.DataFrame(results).to_csv(out_file.replace(".pkl",".csv"), index=False)
print(f"Saved results to {out_file}")
print("Finished!........")
print('Acc:',df_final["accuracy"].mean() , 'Tetra:', tetra)




# import os
# import json
# import random
# import numpy as np
# import pandas as pd
# import torch
# from transformers import (
#     AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
#     DataCollatorForLanguageModeling
# )
# from datasets import Dataset
# import argparse
# from tqdm import tqdm

# # Optional PEFT / LoRA
# try:
#     from peft import LoraConfig, get_peft_model, TaskType
#     peft_available = True
# except ImportError:
#     peft_available = False

# # -----------------------------
# # Arguments
# # -----------------------------
# parser = argparse.ArgumentParser(description="Fast fine-tune + inference with optional LoRA")
# parser.add_argument("--model_name", type=str, required=True)
# parser.add_argument("--data_path", type=str, required=True)
# parser.add_argument("--fine_tune_data", type=str, default=None)
# parser.add_argument("--out_dir", type=str, default="./results")
# parser.add_argument("--election_year", type=int, choices=[2020, 2024], required=True)
# parser.add_argument("--n_samples", type=int, default=10)
# parser.add_argument("--ft_epochs", type=int, default=1)
# parser.add_argument("--ft_batch_size", type=int, default=4)
# parser.add_argument("--seed", type=int, default=42)
# parser.add_argument("--use_lora", action="store_true", help="Use LoRA for fine-tuning")
# args = parser.parse_args()

# # -----------------------------
# # Setup
# # -----------------------------
# os.makedirs(args.out_dir, exist_ok=True)
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)

# # -----------------------------
# # Election year → candidates
# # -----------------------------
# if args.election_year == 2020:
#     CANDIDATES = ["Donald Trump", "Joe Biden"]
#     ALL_CLASSES = ["Donald Trump", "Joe Biden", "Other"]

# elif args.election_year == 2024:
#     CANDIDATES = ["Donald Trump", "Kamala Harris"]
#     ALL_CLASSES = ["Donald Trump", "Kamala Harris", "Other"]


# CANDIDATES_NORM = [c.lower() for c in CANDIDATES]

# # -----------------------------
# # Load datasets
# # -----------------------------
# with open(args.data_path, "r") as f:
#     test_data = json.load(f)
# print(f"Loaded {len(test_data)} test samples")

# ft_data = None
# if args.fine_tune_data:
#     with open(args.fine_tune_data, "r") as f:
#         ft_data = [json.loads(line) for line in f]  # JSONL
#     print(f"Loaded {len(ft_data)} fine-tune samples")

# # -----------------------------
# # Load model
# # -----------------------------
# tokenizer = AutoTokenizer.from_pretrained(args.model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     args.model_name,
#     device_map="auto",
#     torch_dtype=torch.bfloat16
# )
# model.gradient_checkpointing_enable()
# model.config.use_cache = False
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = tokenizer.eos_token_id
# model.generation_config.pad_token_id = tokenizer.eos_token_id
# model.eval()
# device = model.device if hasattr(model, "device") else next(model.parameters()).device

# # -----------------------------
# # Apply LoRA if requested
# # -----------------------------
# if args.use_lora:
#     if not peft_available:
#         raise ImportError("PEFT is not installed. Install via `pip install peft` to use LoRA.")
#     print("Applying LoRA adapters...")
#     lora_config = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none"
#     )
#     model = get_peft_model(model, lora_config)
#     print("LoRA applied!")

# # -----------------------------
# # Helper functions
# # -----------------------------
# def normalize_vote(text):
#     if text is None:
#         return "Other"
#     t = text.lower()
#     for c in CANDIDATES:
#         if c.lower() in t:
#             return c
#     return "Other"


# def extract_ground_truth(messages):
#     for m in messages:
#         if m.get("role") == "assistant":
#             return normalize_vote(m.get("content"))
#     return None

# def get_vote_probs(messages):
#     clean_msgs = [m for m in messages if m["role"] != "assistant"]
#     prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
#     prompt += f"\nVote choice ({' or '.join(ALL_CLASSES)}):"
#     prompt += "\nVote choice:"

#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

#     probs = {}

#     with torch.no_grad():
#         for candidate in CANDIDATES:
#             token_ids = tokenizer.encode(candidate, add_special_tokens=False)
#             p = 1.0
#             curr = input_ids.clone()

#             for tid in token_ids:
#                 logits = model(curr).logits[:, -1, :]
#                 token_probs = torch.softmax(logits, dim=-1)
#                 p *= token_probs[0, tid].item()
#                 curr = torch.cat(
#                     [curr, torch.tensor([[tid]], device=device)], dim=1
#                 )

#             probs[candidate] = p

#     # Normalize Trump/Harris
#     Z = sum(probs.values())
#     if Z > 0:
#         probs = {k: v / Z for k, v in probs.items()}
#     else:
#         probs = {k: 0.5 for k in probs}

#     # Residual "Other"
#     probs["Other"] = max(0.0, 1.0 - sum(probs.values()))

#     # Final normalization (numerical safety)
#     Z = sum(probs.values())
#     probs = {k: v / Z for k, v in probs.items()}

#     return probs


# def accuracy_from_probs(probs, ground_truth):
#     return int(max(probs, key=probs.get) == ground_truth)

# def mutual_information(probs, ground_truth, eps=1e-12):
#     p = max(probs.get(ground_truth, eps), eps)
#     return -np.log2(p)


# def probs_to_vector(probs):
#     return np.array([
#         probs["Donald Trump"],
#         probs["Kamala Harris"],
#         probs["Other"]
#     ])


# # -----------------------------
# # Fine-tuning (optional, robust)
# # -----------------------------
# if ft_data:
#     print("Starting fine-tuning ...")
#     ft_texts = []

#     for item in ft_data:
#         # Chat-completion format
#         if "messages" in item:
#             messages = item["messages"]
#             prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages if m.get("role") != "assistant")
#             target = next((m["content"] for m in messages if m.get("role")=="assistant"), "")
#         # Completion format
#         elif "prompt" in item and "completion" in item:
#             prompt = item["prompt"]
#             target = item["completion"]
#         # Unknown format → skip
#         else:
#             print(f"Skipping FT item (unknown format): {item}")
#             continue

#         ft_texts.append({"text": prompt + tokenizer.eos_token + target + tokenizer.eos_token})

#     dataset = Dataset.from_list(ft_texts)

#     def tokenize_fn(examples):
#         return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=192)
#     tokenized_ds = dataset.map(tokenize_fn, batched=True)

#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     training_args = TrainingArguments(
#         output_dir=os.path.join(args.out_dir, "ft_model"),
#         per_device_train_batch_size=args.ft_batch_size,
#         gradient_accumulation_steps=2,
#         num_train_epochs=args.ft_epochs,
#         logging_steps=50,
#         save_strategy="no",
#         bf16=True,
#         seed=args.seed,
#         learning_rate=1e-4,
#         warmup_ratio=0.1,
#     )

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_ds,
#         tokenizer=tokenizer,
#         data_collator=data_collator
#     )

#     trainer.train()
#     print("Fine-tuning completed.")

# # -----------------------------
# # Inference
# # -----------------------------
# results = []
# for idx, entry in tqdm(enumerate(test_data), total=len(test_data)):
#     messages = entry.get("messages", [])
#     gt = extract_ground_truth(messages)
#     if gt is None:
#         continue

#     probs = get_vote_probs(messages)
#     pred = max(probs, key=probs.get)
#     acc = accuracy_from_probs(probs, gt)
#     mi = mutual_information(probs, gt)

#     results.append({
#         "idx": idx,
#         "messages": messages,
#         "ground_truth": gt,
#         "predicted_vote": pred,
#         "probs": probs,
#         "accuracy": acc,
#         "mutual_inf": mi
#     })

# # -----------------------------
# # Convert results to DataFrame
# # -----------------------------
# df_final = pd.DataFrame(results)
# df_final["llm_binary"] = df_final["predicted_vote"]
# df_final["anes_binary"] = df_final["ground_truth"]

# df_final["llm_num"] = df_final["llm_binary"].apply(lambda x: 0 if x == CANDIDATES[1] else 1)
# df_final["anes_num"] = df_final["anes_binary"].apply(lambda x: 0 if x == CANDIDATES[1] else 1)

# # -----------------------------
# # Tetrachoric correlation
# # -----------------------------
# # def tetrachoric_corr_safe(vec1, vec2):
# #     A = np.sum((vec1 == 0) & (vec2 == 0))
# #     B = np.sum((vec1 == 0) & (vec2 == 1))
# #     C = np.sum((vec1 == 1) & (vec2 == 0))
# #     D = np.sum((vec1 == 1) & (vec2 == 1))
# #     if (A+B)==0 or (C+D)==0 or (A+C)==0 or (B+D)==0:
# #         return np.nan
# #     try:
# #         return np.cos(np.pi / (1 + np.sqrt((A*D)/(B*C))))
# #     except:
# #         return np.nan

# # tetra = tetrachoric_corr_safe(df_final["llm_num"].values, df_final["anes_num"].values)

# # -----------------------------
# # Bias computation
# # -----------------------------
# summary_rows = []
# row = {
#     "Variable": "Wholesample",
#     "n_samples": len(df_final),
#     # "Tetra": tetra,
#     "Prop.Agree": np.mean(df_final["llm_binary"] == df_final["anes_binary"])
# }

# for c in ALL_CLASSES:
#     real_pct = np.mean(df_final["anes_binary"] == c)
#     llm_pct = np.mean([p[c] for p in df_final["probs"]])
#     row[f"RealPct_{c}"] = real_pct
#     row[f"LLMPct_{c}"] = llm_pct
#     row[f"Bias_{c}"] = llm_pct - real_pct

# summary_rows.append(row)
# df_summary = pd.DataFrame(summary_rows)


# # -----------------------------
# # Save outputs
# # -----------------------------
# final_path = os.path.join(args.out_dir, f"{args.model_name.replace('/', '_')}_{args.election_year}_final.pkl")
# summary_path = final_path.replace(".pkl", "_summary.csv")

# df_final.to_pickle(final_path)


# # -----------------------------
# # Save detailed results
# # -----------------------------
# def safe_name(s):
#     return str(s).replace("/", "_").replace(" ", "_")

# out_file = os.path.join(
#     args.out_dir,
#     f"{safe_name(args.model_name)}_{args.election_year}_{safe_name(args.fine_tune_data)}_results.pkl"
# )
# pd.DataFrame(results).to_pickle(out_file)
# pd.DataFrame(results).to_csv(out_file.replace(".pkl",".csv"), index=False)
# print(f"Saved results to {out_file}")
# print("Finished!")
# print('Acc:',df_final["accuracy"].mean())


