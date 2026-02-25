import os
import time
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
FT_GRAD_ACCUM = 8
EVAL_BATCH_SIZE = 24
MAX_TRAIN_LENGTH = 256
MAX_PROMPT_LENGTH = 512

# Dataset for evaluation (JSON)
DATA_PATH = "dataset_test/test_canada_election_vote_2021_3class.json"

# Fine-tune datasets
FINE_TUNE_FILES = [
    "dataset_ft/agg_ft_vote.jsonl",
    "dataset_ft/individual_ft_vote.jsonl",
    "dataset_ft/tweets_ft_vote_sample.jsonl",
]

# Models to run
MODELS = [
    # "meta-llama/Llama-3.1-70B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]

# =====================================================
# Candidates
# =====================================================
CANDIDATES = ["Justin Trudeau", "Erin O'Toole", "Others"]
VOTE2ID = {c: i for i, c in enumerate(CANDIDATES)}
ID2VOTE = {i: c for c, i in VOTE2ID.items()}


def set_torch_perf_flags():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


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


def build_prompt(messages):
    clean_msgs = [m for m in messages if m["role"] != "assistant"]
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
    prompt += f"\nVote choice ({' or '.join(CANDIDATES)}):"
    return prompt


def mutual_information(probs, ground_truth, eps=1e-12):
    p = max(probs.get(ground_truth, eps), eps)
    return -np.log2(p)


def vote_to_numeric(vote):
    return VOTE2ID[vote]


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
        )

    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def get_vote_probs_batched(prompts, model, tokenizer, device, candidate_token_ids):
    if not prompts:
        return []

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
        add_special_tokens=False,
    )
    prompt_input_ids = enc.input_ids.to(device)
    prompt_attention_mask = enc.attention_mask.to(device)
    prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()

    seqs = []
    seq_prompt_lens = []
    seq_cand_lens = []

    for i, p_len in enumerate(prompt_lengths):
        prompt_tokens = prompt_input_ids[i, :p_len]
        for cand_ids in candidate_token_ids:
            cand_tensor = torch.tensor(cand_ids, device=device, dtype=torch.long)
            seq = torch.cat([prompt_tokens, cand_tensor], dim=0)
            seqs.append(seq)
            seq_prompt_lens.append(int(p_len))
            seq_cand_lens.append(len(cand_ids))

    max_len = max(s.size(0) for s in seqs)
    n_seq = len(seqs)

    input_ids = torch.full(
        (n_seq, max_len),
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros((n_seq, max_len), dtype=torch.long, device=device)

    for i, seq in enumerate(seqs):
        l = seq.size(0)
        input_ids[i, :l] = seq
        attention_mask[i, :l] = 1

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits, dim=-1)

    seq_logps = []
    for i in range(n_seq):
        p_len = seq_prompt_lens[i]
        c_len = seq_cand_lens[i]
        lp = torch.tensor(0.0, device=device)

        # Each continuation token at position t is scored by logits at t-1.
        for k in range(c_len):
            logit_pos = p_len - 1 + k
            token_id = input_ids[i, p_len + k]
            lp = lp + log_probs[i, logit_pos, token_id]
        seq_logps.append(lp)

    seq_logps = torch.stack(seq_logps).view(len(prompts), len(CANDIDATES))
    cand_probs = torch.softmax(seq_logps, dim=1).detach().cpu().numpy()

    out = []
    for row in cand_probs:
        out.append({c: float(row[j]) for j, c in enumerate(CANDIDATES)})
    return out


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
set_torch_perf_flags()

# =====================================================
# Main loop: models x fine-tune datasets
# =====================================================
os.makedirs(OUT_DIR, exist_ok=True)

for model_name in MODELS:
    print(f"\n=== Loading model: {model_name} ===")
    model, tokenizer = load_model_and_tokenizer(model_name)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    device = next(model.parameters()).device

    candidate_token_ids = [
        tokenizer.encode(candidate, add_special_tokens=False) for candidate in CANDIDATES
    ]

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
                prompt = "\n".join(
                    f"{m['role']}: {m['content']}" for m in item.get("messages", [])
                )
                target = ""
                for m in item.get("messages", []):
                    if m["role"] == "assistant":
                        target = m["content"]
                ft_texts.append({"text": prompt + tokenizer.eos_token + target + tokenizer.eos_token})

            dataset = Dataset.from_list(ft_texts)

            def tokenize_fn(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=MAX_TRAIN_LENGTH,
                )

            tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

            training_args = TrainingArguments(
                output_dir=os.path.join(OUT_DIR, f"{model_name.replace('/', '_')}_ft"),
                per_device_train_batch_size=FT_BATCH_SIZE,
                gradient_accumulation_steps=FT_GRAD_ACCUM,
                num_train_epochs=FT_EPOCHS,
                logging_steps=50,
                save_strategy="no",
                fp16=False,
                bf16=True,
                optim="adamw_torch_fused",
                seed=SEED,
                dataloader_pin_memory=True,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_ds,
                data_collator=data_collator,
            )

            trainer.train()
            print("Fine-tuning completed.")

        # Inference setup for speed.
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()

        # =====================================================
        # Inference (batched)
        # =====================================================
        results = []
        valid_entries = []
        for idx, entry in enumerate(data):
            messages = entry.get("messages", [])
            gt = extract_ground_truth(messages)
            if gt is None or gt not in CANDIDATES:
                continue
            valid_entries.append((idx, messages, gt, build_prompt(messages)))

        for start in tqdm(range(0, len(valid_entries), EVAL_BATCH_SIZE), total=(len(valid_entries) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE):
            chunk = valid_entries[start : start + EVAL_BATCH_SIZE]
            prompts = [x[3] for x in chunk]
            prob_list = get_vote_probs_batched(
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
                device=device,
                candidate_token_ids=candidate_token_ids,
            )

            for (idx, messages, gt, _prompt), probs in zip(chunk, prob_list):
                pred = max(probs, key=probs.get)
                results.append(
                    {
                        "idx": idx,
                        "ground_truth": gt,
                        "predicted_vote": pred,
                        "accuracy": int(pred == gt),
                        "mutual_information": mutual_information(probs, gt),
                        "probs": probs,
                        "messages": messages,
                    }
                )

                if (idx + 1) % SAVE_EVERY == 0:
                    pd.DataFrame(results).to_pickle(
                        os.path.join(
                            OUT_DIR,
                            f"{model_name.replace('/', '_')}_{os.path.basename(ft_file).replace('.jsonl','')}_{ELECTION_YEAR}_partial.pkl",
                        )
                    )

                if SLEEP > 0:
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
                    ratings="vote",
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

        # Re-enable training settings for next fine-tune dataset on same base model.
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()



# # =====================================================
# # Canada 2021 Election Vote Prediction with LLMs
# # Runs multiple models and multiple fine-tune datasets
# # =====================================================

# import os
# import time
# import json
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F

# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from sklearn.metrics import cohen_kappa_score
# from datasets import Dataset

# # Optional ICC
# try:
#     import pingouin as pg
#     ICC_AVAILABLE = True
# except ImportError:
#     ICC_AVAILABLE = False

# # =====================================================
# # Configuration
# # =====================================================


# OUT_DIR = "./result"
# ELECTION_YEAR = "2021"
# SAVE_EVERY = 10000
# SLEEP = 0
# SEED = 42
# FT_EPOCHS = 1
# FT_BATCH_SIZE = 2

# # Dataset for evaluation (JSON)
# DATA_PATH = "dataset_test/test_canada_election_vote_2021_3class.json"

# # Fine-tune datasets
# FINE_TUNE_FILES = [
#     "dataset_ft/agg_ft_vote.jsonl",
#     "dataset_ft/individual_ft_vote.jsonl",
#     "dataset_ft/tweets_ft_vote_sample.jsonl",
# ]

# # Models to run
# MODELS = [
#     # "meta-llama/Llama-3.1-70B-Instruct",
#     "meta-llama/Llama-3.1-8B-Instruct",
#     "Qwen/Qwen2.5-7B-Instruct",
#     "Qwen/Qwen2.5-14B-Instruct"
# ]

# # =====================================================
# # Candidates
# # =====================================================
# CANDIDATES = ["Justin Trudeau", "Erin O'Toole", "Others"]
# VOTE2ID = {c: i for i, c in enumerate(CANDIDATES)}
# ID2VOTE = {i: c for c, i in VOTE2ID.items()}

# # =====================================================
# # Helper functions
# # =====================================================
# def normalize_vote(text):
#     if text is None:
#         return None
#     text = text.lower()
#     for c in CANDIDATES:
#         if c.lower() in text:
#             return c
#     return None

# def extract_ground_truth(messages):
#     for m in messages:
#         if m["role"] == "assistant":
#             return normalize_vote(m["content"])
#     return None

# def get_vote_probs(messages, model, tokenizer, device):
#     clean_msgs = [m for m in messages if m["role"] != "assistant"]
#     prompt = "\n".join(f"{m['role']}: {m['content']}" for m in clean_msgs)
#     prompt += f"\nVote choice ({' or '.join(CANDIDATES)}):"
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#     probs = {}

#     with torch.no_grad():
#         for candidate in CANDIDATES:
#             cand_ids = tokenizer.encode(candidate, add_special_tokens=False)
#             prob = 1.0
#             cur_ids = input_ids.clone()
#             for tok in cand_ids:
#                 outputs = model(input_ids=cur_ids)
#                 logits = outputs.logits[:, -1, :]
#                 token_probs = torch.softmax(logits, dim=-1)
#                 prob *= token_probs[0, tok].item()
#                 cur_ids = torch.cat([cur_ids, torch.tensor([[tok]], device=device)], dim=1)
#             probs[candidate] = prob

#     total = sum(probs.values())
#     if total > 0:
#         probs = {c: p / total for c, p in probs.items()}
#     else:
#         probs = {c: 1 / len(CANDIDATES) for c in CANDIDATES}

#     return probs

# def mutual_information(probs, ground_truth, eps=1e-12):
#     p = max(probs.get(ground_truth, eps), eps)
#     return -np.log2(p)

# def vote_to_numeric(vote):
#     return VOTE2ID[vote]

# # =====================================================
# # Load evaluation dataset
# # =====================================================
# with open(DATA_PATH, "r") as f:
#     data = json.load(f)
# print(f"Loaded {len(data)} evaluation samples")

# # =====================================================
# # Set seeds
# # =====================================================
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

# # =====================================================
# # Main loop: models × fine-tune datasets
# # =====================================================
# os.makedirs(OUT_DIR, exist_ok=True)

# for model_name in MODELS:
#     print(f"\n=== Loading model: {model_name} ===")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         device_map="auto",
#         dtype=torch.bfloat16
#     )
#     model.gradient_checkpointing_enable()
#     model.config.use_cache = False

#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     model.config.pad_token_id = tokenizer.eos_token_id
#     model.eval()
#     device = next(model.parameters()).device

#     for ft_file in FINE_TUNE_FILES:
#         print(f"\n--- Fine-tuning with dataset: {ft_file} ---")
#         ft_data = []
#         with open(ft_file, "r") as f:
#             for line in f:
#                 ft_data.append(json.loads(line))
#         print(f"Loaded {len(ft_data)} fine-tune samples")

#         if len(ft_data) > 0:
#             # Prepare dataset
#             ft_texts = []
#             for item in ft_data:
#                 prompt = "\n".join(f"{m['role']}: {m['content']}" for m in item.get("messages", []))
#                 target = ""
#                 for m in item.get("messages", []):
#                     if m["role"] == "assistant":
#                         target = m["content"]
#                 ft_texts.append({"text": prompt + tokenizer.eos_token + target + tokenizer.eos_token})

#             dataset = Dataset.from_list(ft_texts)

#             def tokenize_fn(examples):
#                 return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

#             tokenized_ds = dataset.map(tokenize_fn, batched=True)
#             data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#             training_args = TrainingArguments(
#                 output_dir=os.path.join(OUT_DIR, f"{model_name.replace('/', '_')}_ft"),
#                 per_device_train_batch_size=FT_BATCH_SIZE,
#                 gradient_accumulation_steps=8,
#                 num_train_epochs=FT_EPOCHS,
#                 logging_steps=50,
#                 save_strategy="no",
#                 fp16=False,
#                 bf16=True,
#                 seed=SEED
#             )

#             trainer = Trainer(
#                 model=model,
#                 args=training_args,
#                 train_dataset=tokenized_ds,
#                 data_collator=data_collator
#             )

#             trainer.train()
#             print("Fine-tuning completed.")

#         # =====================================================
#         # Inference
#         # =====================================================
#         results = []
#         for idx, entry in tqdm(enumerate(data), total=len(data)):
#             messages = entry.get("messages", [])
#             gt = extract_ground_truth(messages)
#             if gt is None or gt not in CANDIDATES:
#                 continue

#             probs = get_vote_probs(messages, model, tokenizer, device)
#             pred = max(probs, key=probs.get)

#             results.append({
#                 "idx": idx,
#                 "ground_truth": gt,
#                 "predicted_vote": pred,
#                 "accuracy": int(pred == gt),
#                 "mutual_information": mutual_information(probs, gt),
#                 "probs": probs,
#                 "messages": messages,
#             })

#             if (idx + 1) % SAVE_EVERY == 0:
#                 pd.DataFrame(results).to_pickle(
#                     os.path.join(OUT_DIR, f"{model_name.replace('/', '_')}_{os.path.basename(ft_file).replace('.jsonl','')}_{ELECTION_YEAR}_partial.pkl")
#                 )

#             time.sleep(SLEEP)

#         df = pd.DataFrame(results)

#         # =====================================================
#         # Metrics
#         # =====================================================
#         anes_votes = df["ground_truth"].map(vote_to_numeric).to_numpy()
#         gpt_votes = df["predicted_vote"].map(vote_to_numeric).to_numpy()

#         metrics = {
#             "accuracy": df["accuracy"].mean(),
#             "cohen_kappa": cohen_kappa_score(anes_votes, gpt_votes),
#             "proportion_agreement": np.mean(anes_votes == gpt_votes),
#             "mean_mutual_information": df["mutual_information"].mean(),
#         }

#         if ICC_AVAILABLE:
#             try:
#                 df_long = (
#                     pd.DataFrame({"anes": anes_votes, "gpt": gpt_votes})
#                     .reset_index()
#                     .melt(id_vars="index", var_name="rater", value_name="vote")
#                 )
#                 icc = pg.intraclass_corr(
#                     data=df_long,
#                     targets="index",
#                     raters="rater",
#                     ratings="vote"
#                 )
#                 metrics["ICC"] = icc.loc[icc["Type"] == "ICC2k", "ICC"].values[0]
#             except Exception:
#                 metrics["ICC"] = None
#         else:
#             metrics["ICC"] = None

#         for k, v in metrics.items():
#             df[k] = v

#         # =====================================================
#         # Save final outputs
#         # =====================================================
#         ft_name = os.path.basename(ft_file).replace(".jsonl", "")
#         out_base = f"{model_name.replace('/', '_')}_{ft_name}_{ELECTION_YEAR}_final"
#         out_pkl = os.path.join(OUT_DIR, out_base + ".pkl")
#         out_csv = os.path.join(OUT_DIR, out_base + ".csv")

#         df.to_pickle(out_pkl)
#         df.to_csv(out_csv, index=False)

#         print("\n=== Final Metrics ===")
#         for k, v in metrics.items():
#             print(f"{k}: {v}")
#         print(f"\nSaved results to:\n{out_pkl}\n{out_csv}")
