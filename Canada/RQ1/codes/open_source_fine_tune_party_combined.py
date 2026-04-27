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
    "dataset_ft/agg_ft_party_2021_3class.jsonl",
    "dataset_ft/individual_ft_party_3class.jsonl",
    "dataset_ft/tweets_ft_party_sample.jsonl",
]

# Models to run
MODELS = [
    "meta-llama/Llama-3.1-70B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
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

            valid.append((i, gt, prompt))

        for i in tqdm(range(0, len(valid), EVAL_BATCH_SIZE)):
            batch = valid[i:i+EVAL_BATCH_SIZE]
            prompts = [x[2] for x in batch]

            probs = get_probs(prompts, model, tokenizer, device)

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









# import os
# import time
# import json
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F

# from tqdm import tqdm
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForLanguageModeling,
# )
# from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, f1_score
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
# FT_GRAD_ACCUM = 8
# EVAL_BATCH_SIZE = 24
# MAX_TRAIN_LENGTH = 256
# MAX_PROMPT_LENGTH = 512


# # Dataset for evaluation (JSON)
# DATA_PATH = "dataset_test/test_canada_election_party_2021_3class_new.json"

# # Fine-tune datasets
# FINE_TUNE_FILES = [
#     "dataset_ft/agg_ft_party_2021_3class.jsonl",
#     "dataset_ft/individual_ft_party_3class.jsonl",
#     "dataset_ft/tweets_ft_party_sample.jsonl",
# ]

# # Models to run
# MODELS = [
#     # "meta-llama/Llama-3.1-70B-Instruct",
#     "meta-llama/Llama-3.1-8B-Instruct",
#     "Qwen/Qwen2.5-7B-Instruct",
#     "Qwen/Qwen2.5-14B-Instruct",
# ]

# # =====================================================
# # Candidates
# # =====================================================
# CANDIDATES = [
#     "Liberal Party",
#     "Conservative Party",
#     "Minor Parties"
# ]

# VOTE2ID = {c: i for i, c in enumerate(CANDIDATES)}
# ID2VOTE = {i: c for c, i in VOTE2ID.items()}


# def set_torch_perf_flags():
#     if torch.cuda.is_available():
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
#         torch.set_float32_matmul_precision("high")


# # =====================================================
# # Helper functions
# # =====================================================
# def normalize_vote(text):
#     if text is None:
#         return None
#     text = str(text).strip().lower()
#     for c in CANDIDATES:
#         if c.lower() == text:
#             return c
#     for c in CANDIDATES:
#         if c.lower() in text:
#             return c
#     return None


# def extract_ground_truth(messages):
#     for m in messages:
#         if m["role"] == "assistant":
#             return normalize_vote(m["content"])
#     return None




# def build_prompt(messages):
#     user_text = None
#     for m in messages:
#         if m["role"] == "user":
#             user_text = m["content"]
#             break

#     if user_text is None:
#         return None

#     system_text = (
#         "You are an expert political analyst specializing in Canadian elections and voting behavior. "
#         "Predict the party choice in the 2021 Canadian party election.\n\n"
#         "Choose exactly one of the following labels:\n"
#         "Liberal Party\n"
#         "Conservative Party\n"
#         "Minor Parties\n\n"
#         "Minor Parties' includes New Democratic Party(NDP), Bloc Québécois, Green Party, and People's Party of Canada.\n\n"
#         "Answer with only one label exactly as written above."
#     )

#     prompt = f"system: {system_text}\nuser: {user_text}"
#     return prompt




# def mutual_information(probs, ground_truth, eps=1e-12):
#     p = max(probs.get(ground_truth, eps), eps)
#     return -np.log2(p)


# def vote_to_numeric(vote):
#     return VOTE2ID[vote]


# def load_model_and_tokenizer(model_name):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id

#     try:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#             attn_implementation="flash_attention_2",
#         )
#     except Exception:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             torch_dtype=torch.bfloat16,
#         )

#     model.config.pad_token_id = tokenizer.pad_token_id
#     return model, tokenizer


# def get_vote_probs_batched(prompts, model, tokenizer, device, candidate_token_ids):
#     if not prompts:
#         return []

#     enc = tokenizer(
#         prompts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=MAX_PROMPT_LENGTH,
#         add_special_tokens=False,
#     )
#     prompt_input_ids = enc.input_ids.to(device)
#     prompt_attention_mask = enc.attention_mask.to(device)
#     prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()

#     seqs = []
#     seq_prompt_lens = []
#     seq_cand_lens = []

#     for i, p_len in enumerate(prompt_lengths):
#         prompt_tokens = prompt_input_ids[i, :p_len]
#         for cand_ids in candidate_token_ids:
#             cand_tensor = torch.tensor(cand_ids, device=device, dtype=torch.long)
#             seq = torch.cat([prompt_tokens, cand_tensor], dim=0)
#             seqs.append(seq)
#             seq_prompt_lens.append(int(p_len))
#             seq_cand_lens.append(len(cand_ids))

#     max_len = max(s.size(0) for s in seqs)
#     n_seq = len(seqs)

#     input_ids = torch.full(
#         (n_seq, max_len),
#         fill_value=tokenizer.pad_token_id,
#         dtype=torch.long,
#         device=device,
#     )
#     attention_mask = torch.zeros((n_seq, max_len), dtype=torch.long, device=device)

#     for i, seq in enumerate(seqs):
#         l = seq.size(0)
#         input_ids[i, :l] = seq
#         attention_mask[i, :l] = 1

#     with torch.inference_mode():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#         log_probs = F.log_softmax(outputs.logits, dim=-1)

#     seq_logps = []
#     for i in range(n_seq):
#         p_len = seq_prompt_lens[i]
#         c_len = seq_cand_lens[i]
#         lp = torch.tensor(0.0, device=device)

#         for k in range(c_len):
#             logit_pos = p_len - 1 + k
#             token_id = input_ids[i, p_len + k]
#             lp = lp + log_probs[i, logit_pos, token_id]

#         seq_logps.append(lp)

#     seq_logps = torch.stack(seq_logps).view(len(prompts), len(CANDIDATES))
#     cand_probs = torch.softmax(seq_logps, dim=1).detach().float().cpu().numpy()

#     out = []
#     for row in cand_probs:
#         out.append({c: float(row[j]) for j, c in enumerate(CANDIDATES)})
#     return out


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
# set_torch_perf_flags()


# # =====================================================
# # Main loop: models x fine-tune datasets
# # =====================================================
# os.makedirs(OUT_DIR, exist_ok=True)

# for model_name in MODELS:
#     for ft_file in FINE_TUNE_FILES:
#         print(f"\n=== Model: {model_name} | FT dataset: {ft_file} ===")

#         # Fresh base model for every FT dataset
#         model, tokenizer = load_model_and_tokenizer(model_name)

#         model.gradient_checkpointing_enable()
#         model.config.use_cache = False
#         model.train()
#         device = next(model.parameters()).device

#         candidate_token_ids = [
#             tokenizer.encode(candidate, add_special_tokens=False)
#             for candidate in CANDIDATES
#         ]

#         # -------------------------------------------------
#         # Load fine-tune data
#         # -------------------------------------------------
#         ft_data = []
#         with open(ft_file, "r") as f:
#             for line in f:
#                 ft_data.append(json.loads(line))
#         print(f"Loaded {len(ft_data)} fine-tune samples")

#         if len(ft_data) > 0:
#             ft_texts = []

#             for item in ft_data:
#                 messages = item.get("messages", [])
#                 prompt = build_prompt(messages)
#                 target = extract_ground_truth(messages)

#                 if target is None or target not in CANDIDATES:
#                     continue

#                 ft_texts.append({
#                     "text": prompt + " " + target + tokenizer.eos_token
#                 })

#             print(f"Usable fine-tune samples: {len(ft_texts)}")

#             dataset = Dataset.from_list(ft_texts)

#             def tokenize_fn(examples):
#                 return tokenizer(
#                     examples["text"],
#                     truncation=True,
#                     max_length=MAX_TRAIN_LENGTH,
#                 )

#             tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
#             data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#             training_args = TrainingArguments(
#                 output_dir=os.path.join(
#                     OUT_DIR,
#                     f"{model_name.replace('/', '_')}_{os.path.basename(ft_file).replace('.jsonl', '')}_ft"
#                 ),
#                 per_device_train_batch_size=FT_BATCH_SIZE,
#                 gradient_accumulation_steps=FT_GRAD_ACCUM,
#                 num_train_epochs=FT_EPOCHS,
#                 logging_steps=50,
#                 save_strategy="no",
#                 fp16=False,
#                 bf16=True,
#                 optim="adamw_torch_fused",
#                 seed=SEED,
#                 dataloader_pin_memory=True,
#                 report_to="none",
#             )

#             trainer = Trainer(
#                 model=model,
#                 args=training_args,
#                 train_dataset=tokenized_ds,
#                 data_collator=data_collator,
#             )

#             trainer.train()
#             print("Fine-tuning completed.")

#         # -------------------------------------------------
#         # Inference setup
#         # -------------------------------------------------
#         model.gradient_checkpointing_disable()
#         model.config.use_cache = True
#         model.eval()

#         results = []
#         valid_entries = []

#         for idx, entry in enumerate(data):
#             messages = entry.get("messages", [])
#             gt = extract_ground_truth(messages)

#             if gt is None or gt not in CANDIDATES:
#                 continue

#             valid_entries.append((idx, messages, gt, build_prompt(messages)))

#         total_batches = (len(valid_entries) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE

#         for start in tqdm(
#             range(0, len(valid_entries), EVAL_BATCH_SIZE),
#             total=total_batches,
#             desc=f"Inference {model_name} | {os.path.basename(ft_file)}",
#         ):
#             chunk = valid_entries[start : start + EVAL_BATCH_SIZE]
#             prompts = [x[3] for x in chunk]

#             prob_list = get_vote_probs_batched(
#                 prompts=prompts,
#                 model=model,
#                 tokenizer=tokenizer,
#                 device=device,
#                 candidate_token_ids=candidate_token_ids,
#             )

#             for (idx, messages, gt, _prompt), probs in zip(chunk, prob_list):
#                 pred = max(probs, key=probs.get)

#                 results.append(
#                     {
#                         "idx": idx,
#                         "ground_truth": gt,
#                         "predicted_vote": pred,
#                         "accuracy": int(pred == gt),
#                         "mutual_information": mutual_information(probs, gt),
#                         "probs": probs,
#                         "messages": messages,
#                     }
#                 )

#                 if (idx + 1) % SAVE_EVERY == 0:
#                     pd.DataFrame(results).to_pickle(
#                         os.path.join(
#                             OUT_DIR,
#                             f"{model_name.replace('/', '_')}_{os.path.basename(ft_file).replace('.jsonl','')}_{ELECTION_YEAR}_partial.pkl",
#                         )
#                     )

#                 if SLEEP > 0:
#                     time.sleep(SLEEP)

#         df = pd.DataFrame(results)
#         if df.empty:
#             print("No valid evaluation rows; skipping save.")
#             del model
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             continue

#         # -------------------------------------------------
#         # Metrics
#         # -------------------------------------------------
#         y_true = df["ground_truth"].map(vote_to_numeric).to_numpy()
#         y_pred = df["predicted_vote"].map(vote_to_numeric).to_numpy()

#         metrics = {
#             "accuracy": df["accuracy"].mean(),
#             "cohen_kappa": cohen_kappa_score(y_true, y_pred),
#             "mcc": matthews_corrcoef(y_true, y_pred),
#             "macro_f1": f1_score(y_true, y_pred, average="macro"),
#             "proportion_agreement": np.mean(y_true == y_pred),
#             "mean_mutual_information": df["mutual_information"].mean(),
#         }

#         if ICC_AVAILABLE:
#             try:
#                 df_long = (
#                     pd.DataFrame({"anes": y_true, "gpt": y_pred})
#                     .reset_index()
#                     .melt(id_vars="index", var_name="rater", value_name="vote")
#                 )
#                 icc = pg.intraclass_corr(
#                     data=df_long,
#                     targets="index",
#                     raters="rater",
#                     ratings="vote",
#                 )
#                 metrics["ICC"] = icc.loc[icc["Type"] == "ICC2k", "ICC"].values[0]
#             except Exception:
#                 metrics["ICC"] = None
#         else:
#             metrics["ICC"] = None

#         for k, v in metrics.items():
#             df[k] = v

#         # -------------------------------------------------
#         # Save final outputs
#         # -------------------------------------------------
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

#         del model
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()











