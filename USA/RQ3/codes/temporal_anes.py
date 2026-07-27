import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


# ---------- Configuration ----------
YEARS = [2008, 2012, 2016, 2020, 2024]

TASK_CONFIGS = {
    "party_choice": {
        "candidates": ["Democrat", "Republican", "Other"],
        "system_text": (
            "You are an expert political analyst specializing in US political attitudes and voting behavior.\n\n"
            "Task:\n"
            "Based on the respondent's demographic and background information, predict their party choice "
            "in the US presidential election.\n\n"
            "Rules:\n"
            "- You must choose ONLY ONE label.\n"
            "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
            "Democrat\n"
            "Republican\n"
            "Other\n\n"
            "Important:\n"
            "- Base your decision on typical voting patterns and demographics.\n"
            "- Do NOT explain your reasoning.\n"
            "- Do NOT repeat the input.\n"
            "- Output ONLY the label."
        ),
    },
    "ideology": {
        "candidates": ["liberal", "moderate", "conservative"],
        "system_text": (
            "You are an expert political analyst specializing in US political attitudes and voting behavior.\n\n"
            "Task:\n"
            "Based on the respondent's demographic and background information, predict their ideological "
            "self-placement.\n\n"
            "Rules:\n"
            "- You must choose ONLY ONE label.\n"
            "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
            "liberal\n"
            "moderate\n"
            "conservative\n\n"
            "Important:\n"
            "- Base your decision on typical voting patterns and demographics.\n"
            "- Do NOT explain your reasoning.\n"
            "- Do NOT repeat the input.\n"
            "- Output ONLY the label."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal Generalization Experiments with ANES data"
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--task", type=str, default="party_choice", choices=["party_choice", "ideology"])
    parser.add_argument("--data_dir", type=str, default="dataset_test/")
    parser.add_argument("--experiment", type=str, default="all", choices=["sequential", "cumulative", "all"])
    parser.add_argument("--out_dir", type=str, default="./results_temporal")
    parser.add_argument("--tmp_dir", type=str, default="./tmp_qlora_temporal")
    parser.add_argument("--seed", type=int, default=42)

    # OPTIMIZED defaults: bigger batch, shorter sequence, fewer epochs
    parser.add_argument("--max_len", type=int, default=256)          # 512→256 (2x faster, less memory)
    parser.add_argument("--epochs", type=float, default=1.0)         # 3.0→1.0 (3x faster)
    parser.add_argument("--train_batch_size", type=int, default=4)   # 1→4 (4x faster)
    parser.add_argument("--grad_accum", type=int, default=4)         # 16→4 (4x fewer forward passes)
    parser.add_argument("--eval_batch_size", type=int, default=32)   # 8→32 (4x faster eval)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--length_normalize_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_base_predictions", action="store_true")
    parser.add_argument("--dataloader_workers", type=int, default=4)  # NEW: parallel data loading

    # NEW: non-reentrant gradient checkpointing toggle (needed for stable multi-GPU
    # model-parallel backward passes with device_map="auto" sharding across GPUs)
    parser.add_argument("--use_reentrant_checkpointing", action=argparse.BooleanOptionalAction, default=False,
        help="Use reentrant gradient checkpointing. Default False (non-reentrant) because "
             "reentrant checkpointing can crash or silently break gradients when the model "
             "is sharded across multiple GPUs via device_map='auto'.")

    # NEW: smoke test flag to validate the full train->eval pipeline quickly without
    # waiting on a full-size run, useful after a long model load.
    parser.add_argument("--smoke_test", action="store_true",
        help="Truncate train/test rows for every year to --smoke_test_n samples and run "
             "a minimal pass to validate the pipeline end-to-end quickly.")
    parser.add_argument("--smoke_test_n", type=int, default=20,
        help="Number of rows per year to keep when --smoke_test is set.")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_qlora_base_model(model_name: str, tokenizer: AutoTokenizer, attn_implementation: str) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    num_gpus = torch.cuda.device_count()
    max_memory = {i: "75GiB" for i in range(num_gpus)} if num_gpus > 0 else {}
    max_memory["cpu"] = "120GiB"
    print(f"  GPUs: {num_gpus}, Memory: {max_memory}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, max_memory=max_memory,
        dtype=torch.bfloat16, attn_implementation=attn_implementation,
        low_cpu_mem_usage=True, device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=args.target_modules, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM")


def normalize_label(value: Any, candidates: List[str]) -> str | None:
    if value is None: return None
    text = str(value).strip().lower()
    for c in candidates:
        if c.lower() in text: return c
    return None


def extract_label(messages: list[dict[str, str]], candidates: List[str]) -> str | None:
    for msg in messages:
        if msg.get("role") == "assistant":
            return normalize_label(msg.get("content"), candidates)
    return None


def first_user_content(messages: list[dict[str, str]]) -> str:
    for msg in messages:
        if msg.get("role") == "user": return msg.get("content", "")
    return messages[0].get("content", "") if messages else ""


def build_prompt(tokenizer: AutoTokenizer, system_text: str, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True)


def build_sft_dataset(rows, tokenizer, candidates, system_text, max_len) -> Dataset:
    """Pre-tokenize all samples at once for speed."""
    prompts, answers = [], []
    for item in rows:
        messages = item.get("messages", [])
        gt = extract_label(messages, candidates)
        if gt is None: continue
        prompts.append(build_prompt(tokenizer, system_text, first_user_content(messages)))
        answers.append(gt + tokenizer.eos_token)

    samples = []
    for prompt, answer in zip(prompts, answers):
        p_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        a_ids = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        max_p = max_len - len(a_ids)
        if max_p <= 0: continue
        p_ids = p_ids[-max_p:]
        samples.append({"input_ids": p_ids + a_ids, "labels": [-100]*len(p_ids) + a_ids, "attention_mask": [1]*len(p_ids + a_ids)})

    if not samples: raise ValueError("No usable SFT samples.")
    return Dataset.from_list(samples)


@dataclass
class DataCollatorForCompletionOnlyLM:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.tokenizer.pad_token_id]*pad)
            batch["attention_mask"].append(f["attention_mask"] + [0]*pad)
            batch["labels"].append(f["labels"] + [-100]*pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def prepare_eval_entries(rows, tokenizer, candidates, system_text) -> list[Tuple[int, str, str, str]]:
    entries = []
    for idx, entry in enumerate(rows):
        messages = entry.get("messages", [])
        gt = extract_label(messages, candidates)
        if gt is None: continue
        entries.append((idx, gt, first_user_content(messages),
                       build_prompt(tokenizer, system_text, first_user_content(messages))))
    return entries


def get_candidate_token_ids(tokenizer, candidates) -> list[list[int]]:
    return [tokenizer(c, add_special_tokens=False)["input_ids"] for c in candidates]


@torch.inference_mode()
def score_candidates_batched(model, tokenizer, prompts, candidate_token_ids, candidates, max_len, length_normalize):
    if not prompts: return []

    device = next(model.parameters()).device
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len, add_special_tokens=False)

    p_ids = encoded["input_ids"]
    p_mask = encoded["attention_mask"]
    p_lens = p_mask.sum(dim=1).tolist()

    # Build all sequences at once
    all_seqs, all_p_lens, all_c_lens = [], [], []
    for row_idx, plen in enumerate(p_lens):
        ptoks = p_ids[row_idx, :plen].tolist()
        for cids in candidate_token_ids:
            max_p = max_len - len(cids)
            if max_p <= 0: raise ValueError("Candidate too long")
            all_seqs.append(ptoks[-max_p:] + cids)
            all_p_lens.append(len(ptoks[-max_p:]))
            all_c_lens.append(len(cids))

    batch_size = len(all_seqs)
    padded = max(len(s) for s in all_seqs)
    input_ids = torch.full((batch_size, padded), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((batch_size, padded), dtype=torch.long, device=device)

    for i, s in enumerate(all_seqs):
        L = len(s)
        input_ids[i, :L] = torch.tensor(s, dtype=torch.long, device=device)
        attn_mask[i, :L] = 1

    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    log_probs = F.log_softmax(outputs.logits.float(), dim=-1)

    scores = []
    for i in range(batch_size):
        plen, clen = all_p_lens[i], all_c_lens[i]
        tok_scores = [log_probs[i, plen-1+off, input_ids[i, plen+off]] for off in range(clen)]
        sc = torch.stack(tok_scores).sum()
        if length_normalize: sc = sc / clen
        scores.append(sc)

    score_tensor = torch.stack(scores).view(len(prompts), len(candidates))
    prob_tensor = torch.softmax(score_tensor, dim=1).detach().cpu().numpy()

    return [{candidates[i]: float(row[i]) for i in range(len(candidates))} for row in prob_tensor]


def evaluate_and_save_predictions(model, tokenizer, eval_entries, candidates, args,
                                   run_name, train_years_label, test_year, is_base_model=False):
    model.eval()
    model.config.use_cache = True

    cids = get_candidate_token_ids(tokenizer, candidates)
    rows = []

    for start in tqdm(range(0, len(eval_entries), args.eval_batch_size), desc=f"Eval {run_name}"):
        batch = eval_entries[start: start + args.eval_batch_size]
        prompts = [e[3] for e in batch]
        probs = score_candidates_batched(model, tokenizer, prompts, cids, candidates, args.max_len, args.length_normalize_eval)

        for (idx, gt, user_text, _), p in zip(batch, probs):
            pred = max(p, key=p.get)
            sp = sorted(p.items(), key=lambda x: x[1], reverse=True)
            rows.append({
                "idx": idx, "gt": gt, "pred": pred, "correct": int(gt == pred),
                "confidence": sp[0][1], "margin": sp[0][1] - (sp[1][1] if len(sp) > 1 else 0),
                "top2_label": sp[1][0] if len(sp) > 1 else None,
                "top2_prob": sp[1][1] if len(sp) > 1 else 0,
                "user_text": user_text,
                **{f"prob_{c}": p[c] for c in candidates},
            })

    df = pd.DataFrame(rows)
    if df.empty: return {"acc": 0.0, "kappa": 0.0, "macro_f1": 0.0, "n": 0}, df

    v2id = {c: i for i, c in enumerate(candidates)}
    y_true = df["gt"].map(v2id).to_numpy()
    y_pred = df["pred"].map(v2id).to_numpy()

    metrics = {
        "acc": float(df["correct"].mean()), "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")), "n": int(len(df)),
        "mean_confidence": float(df["confidence"].mean()), "mean_margin": float(df["margin"].mean()),
    }
    for c in candidates:
        mask = df["gt"] == c
        if mask.sum() > 0:
            metrics[f"acc_{c}"] = float(df[mask]["correct"].mean())
            metrics[f"n_{c}"] = int(mask.sum())

    df["run_name"] = run_name
    df["train_years"] = train_years_label
    df["test_year"] = test_year
    df["is_base_model"] = is_base_model
    df["model_name"] = args.model_name
    df["task"] = args.task

    model.config.use_cache = False
    return metrics, df


def train_and_evaluate(train_rows, test_rows, base_model, tokenizer, candidates, system_text,
                       args, run_name, train_years_label, test_year):
    train_dataset = build_sft_dataset(train_rows, tokenizer, candidates, system_text, args.max_len)
    test_entries = prepare_eval_entries(test_rows, tokenizer, candidates, system_text)
    print(f"  Train: {len(train_dataset)}, Test: {len(test_entries)}")


    if hasattr(base_model, 'peft_config') and hasattr(base_model, 'unload'):
        base_model = base_model.unload()
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    model = get_peft_model(base_model, build_lora_config(args), adapter_name="default")
    model.train()
    model.config.use_cache = False

    total_steps = (len(train_dataset) * args.epochs) // (args.train_batch_size * args.grad_accum)
    warmup_steps = max(1, int(total_steps * 0.03))

    training_args = TrainingArguments(
        output_dir=os.path.join(args.tmp_dir, run_name),
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        # NEW: non-reentrant checkpointing avoids backward-pass crashes/silent gradient
        # bugs when the model is sharded across multiple GPUs (device_map="auto").
        gradient_checkpointing_kwargs={"use_reentrant": args.use_reentrant_checkpointing},
        max_grad_norm=0.3,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=args.dataloader_workers,  # OPTIMIZED: parallel loading
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                      data_collator=DataCollatorForCompletionOnlyLM(tokenizer))
    trainer.train()

    metrics, pred_df = evaluate_and_save_predictions(
        model, tokenizer, test_entries, candidates, args, run_name, train_years_label, test_year)

    del trainer, model
    torch.cuda.empty_cache()
    return metrics, pred_df


def experiment_1_sequential(args, base_model, tokenizer, candidates, system_text, data_by_year):
    print("\n" + "="*70)
    print("EXPERIMENT 1: Sequential (Train on N, Test on N+1)")
    print("="*70)
    results, all_preds = [], []

    for i in range(len(YEARS) - 1):
        ty, tty = YEARS[i], YEARS[i+1]
        print(f"\n--- Train: {ty} → Test: {tty} ---")
        metrics, pred_df = train_and_evaluate(
            data_by_year[ty], data_by_year[tty], base_model, tokenizer, candidates, system_text,
            args, f"train{ty}_test{tty}", str(ty), tty)
        metrics.update({"train_year": ty, "test_year": tty, "train_n": len(data_by_year[ty]), "test_n": len(data_by_year[tty])})
        results.append(metrics); all_preds.append(pred_df)
        print(f"  Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")

    combined = pd.concat(all_preds, ignore_index=True)
    combined.to_csv(os.path.join(args.out_dir, f"predictions_sequential_{args.task}_{args.model_name.split('/')[-1]}_more.csv"), index=False)
    return results, combined


def experiment_2_cumulative(args, base_model, tokenizer, candidates, system_text, data_by_year):
    print("\n" + "="*70)
    print("EXPERIMENT 2: Cumulative → 2024 (Recent First)")
    print("="*70)
    results, all_preds = [], []
    test_rows = data_by_year[2024]
    prev = [y for y in YEARS if y < 2024]

    for n in range(1, len(prev)+1):
        train_years = prev[-n:]
        train_rows = [r for y in train_years for r in data_by_year[y]]
        label = "+".join(map(str, train_years))
        print(f"\n--- Train: {train_years} ({len(train_rows)}) → Test: 2024 ---")
        metrics, pred_df = train_and_evaluate(
            train_rows, test_rows, base_model, tokenizer, candidates, system_text,
            args, f"train_{'_'.join(map(str,train_years))}_test2024", label, 2024)
        metrics.update({"train_years": train_years, "test_year": 2024, "train_n": len(train_rows), "test_n": len(test_rows)})
        results.append(metrics); all_preds.append(pred_df)
        print(f"  Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")

    combined = pd.concat(all_preds, ignore_index=True)
    combined.to_csv(os.path.join(args.out_dir, f"predictions_cumulative_{args.task}_{args.model_name.split('/')[-1]}_more.csv"), index=False)
    return results, combined


def evaluate_base_model(args, base_model, tokenizer, candidates, system_text, data_by_year):
    print("\n" + "="*70)
    print("BASE MODEL EVALUATION (No Fine-Tuning)")
    print("="*70)
    all_preds = []

    for year in YEARS:
        entries = prepare_eval_entries(data_by_year[year], tokenizer, candidates, system_text)
        print(f"\n--- Base → Test: {year} ({len(entries)}) ---")
        metrics, pred_df = evaluate_and_save_predictions(
            base_model, tokenizer, entries, candidates, args, f"base_{year}", "none", year, True)
        all_preds.append(pred_df)
        print(f"  Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")

    combined = pd.concat(all_preds, ignore_index=True)
    combined.to_csv(os.path.join(args.out_dir, f"predictions_base_{args.task}_{args.model_name.split('/')[-1]}_more.csv"), index=False)
    return combined


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    set_seed(args.seed)

    cfg = TASK_CONFIGS[args.task]
    candidates, system_text = cfg["candidates"], cfg["system_text"]

    print("Loading ANES data...")
    data_by_year = {}
    for year in YEARS:
        path = os.path.join(args.data_dir, f"anes_{year}_without_year.jsonl")
        if os.path.exists(path):
            data_by_year[year] = load_jsonl(path)
            print(f"  {year}: {len(data_by_year[year])}")
        else:
            print(f"  WARNING: {path} not found!")

    if 2024 not in data_by_year:
        print("ERROR: 2024 required!"); return

    # NEW: smoke test truncation — validate the pipeline end-to-end quickly instead
    # of waiting on a full-scale run after a long model load.
    if args.smoke_test:
        print(f"\n*** SMOKE TEST MODE: truncating each year to {args.smoke_test_n} rows ***\n")
        for year in data_by_year:
            data_by_year[year] = data_by_year[year][: args.smoke_test_n]
        args.epochs = min(args.epochs, 1.0)

    print("\nLoading tokenizer & model...")
    tokenizer = load_tokenizer(args.model_name)
    base_model = load_qlora_base_model(args.model_name, tokenizer, args.attn_implementation)

    all_results = {"task": args.task, "model": args.model_name, "candidates": candidates, "experiments": {}}

    if args.save_base_predictions:
        evaluate_base_model(args, base_model, tokenizer, candidates, system_text, data_by_year)

    if args.experiment in ["sequential", "all"]:
        seq_results, _ = experiment_1_sequential(args, base_model, tokenizer, candidates, system_text, data_by_year)
        all_results["experiments"]["sequential"] = seq_results

    if args.experiment in ["cumulative", "all"]:
        cum_results, _ = experiment_2_cumulative(args, base_model, tokenizer, candidates, system_text, data_by_year)
        all_results["experiments"]["cumulative"] = cum_results

    with open(os.path.join(args.out_dir, f"metrics_{args.task}_{args.model_name.split('/')[-1]}.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*70 + "\nCOMPLETE\n" + "="*70)


if __name__ == "__main__":
    main()


# import argparse
# import json
# import os
# import random
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# from datasets import Dataset
# from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef
# from tqdm import tqdm
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
#     Trainer,
#     TrainingArguments,
# )


# # ---------- Configuration ----------
# YEARS = [2008, 2012, 2016, 2020, 2024]

# TASK_CONFIGS = {
#     "party_choice": {
#         "candidates": ["Democrat", "Republican", "Other"],
#         "system_text": (
#             "You are an expert political analyst specializing in US political attitudes and voting behavior.\n\n"
#             "Task:\n"
#             "Based on the respondent's demographic and background information, predict their party choice "
#             "in the US presidential election.\n\n"
#             "Rules:\n"
#             "- You must choose ONLY ONE label.\n"
#             "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
#             "Democrat\n"
#             "Republican\n"
#             "Other\n\n"
#             "Important:\n"
#             "- Base your decision on typical voting patterns and demographics.\n"
#             "- Do NOT explain your reasoning.\n"
#             "- Do NOT repeat the input.\n"
#             "- Output ONLY the label."
#         ),
#     },
#     "ideology": {
#         "candidates": ["liberal", "moderate", "conservative"],
#         "system_text": (
#             "You are an expert political analyst specializing in US political attitudes and voting behavior.\n\n"
#             "Task:\n"
#             "Based on the respondent's demographic and background information, predict their ideological "
#             "self-placement.\n\n"
#             "Rules:\n"
#             "- You must choose ONLY ONE label.\n"
#             "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
#             "liberal\n"
#             "moderate\n"
#             "conservative\n\n"
#             "Important:\n"
#             "- Base your decision on typical voting patterns and demographics.\n"
#             "- Do NOT explain your reasoning.\n"
#             "- Do NOT repeat the input.\n"
#             "- Output ONLY the label."
#         ),
#     },
# }


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Temporal Generalization Experiments with ANES data"
#     )
#     parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
#     parser.add_argument("--task", type=str, default="party_choice", choices=["party_choice", "ideology"])
#     parser.add_argument("--data_dir", type=str, default="dataset_test/")
#     parser.add_argument("--experiment", type=str, default="all", choices=["sequential", "cumulative", "all"])
#     parser.add_argument("--out_dir", type=str, default="./results_temporal")
#     parser.add_argument("--tmp_dir", type=str, default="./tmp_qlora_temporal")
#     parser.add_argument("--seed", type=int, default=42)
    
#     # OPTIMIZED defaults: bigger batch, shorter sequence, fewer epochs
#     parser.add_argument("--max_len", type=int, default=256)          # 512→256 (2x faster, less memory)
#     parser.add_argument("--epochs", type=float, default=1.0)         # 3.0→1.0 (3x faster)
#     parser.add_argument("--train_batch_size", type=int, default=1)   # 1→4 (4x faster)
#     parser.add_argument("--grad_accum", type=int, default=16)         # 16→4 (4x fewer forward passes)
#     parser.add_argument("--eval_batch_size", type=int, default=8)   # 8→32 (4x faster eval)
#     parser.add_argument("--learning_rate", type=float, default=2e-4)
#     parser.add_argument("--lora_r", type=int, default=16)
#     parser.add_argument("--lora_alpha", type=int, default=32)
#     parser.add_argument("--lora_dropout", type=float, default=0.05)
#     parser.add_argument("--target_modules", nargs="+",
#         default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
#     parser.add_argument("--attn_implementation", type=str, default="sdpa",
#         choices=["sdpa", "flash_attention_2", "eager"])
#     parser.add_argument("--length_normalize_eval", action=argparse.BooleanOptionalAction, default=True)
#     parser.add_argument("--save_base_predictions", action="store_true")
#     parser.add_argument("--dataloader_workers", type=int, default=4)  # NEW: parallel data loading
#     return parser.parse_args()


# def set_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True


# def load_jsonl(path: str) -> list[dict[str, Any]]:
#     rows = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 rows.append(json.loads(line))
#     return rows


# def load_tokenizer(model_name: str) -> AutoTokenizer:
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#     tokenizer.padding_side = "right"
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.pad_token_id = tokenizer.eos_token_id
#     return tokenizer


# def load_qlora_base_model(model_name: str, tokenizer: AutoTokenizer, attn_implementation: str) -> AutoModelForCausalLM:
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True, bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
#     )
#     num_gpus = torch.cuda.device_count()
#     max_memory = {i: "75GiB" for i in range(num_gpus)} if num_gpus > 0 else {}
#     max_memory["cpu"] = "120GiB"
#     print(f"  GPUs: {num_gpus}, Memory: {max_memory}")
    
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name, quantization_config=bnb_config, max_memory=max_memory,
#         dtype=torch.bfloat16, attn_implementation=attn_implementation,
#         low_cpu_mem_usage=True, device_map="auto",
#     )
#     model.config.pad_token_id = tokenizer.pad_token_id
#     model.config.use_cache = False
#     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
#     return model


# def build_lora_config(args: argparse.Namespace) -> LoraConfig:
#     return LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
#         target_modules=args.target_modules, lora_dropout=args.lora_dropout,
#         bias="none", task_type="CAUSAL_LM")


# def normalize_label(value: Any, candidates: List[str]) -> str | None:
#     if value is None: return None
#     text = str(value).strip().lower()
#     for c in candidates:
#         if c.lower() in text: return c
#     return None


# def extract_label(messages: list[dict[str, str]], candidates: List[str]) -> str | None:
#     for msg in messages:
#         if msg.get("role") == "assistant":
#             return normalize_label(msg.get("content"), candidates)
#     return None


# def first_user_content(messages: list[dict[str, str]]) -> str:
#     for msg in messages:
#         if msg.get("role") == "user": return msg.get("content", "")
#     return messages[0].get("content", "") if messages else ""


# def build_prompt(tokenizer: AutoTokenizer, system_text: str, user_text: str) -> str:
#     return tokenizer.apply_chat_template(
#         [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
#         tokenize=False, add_generation_prompt=True)


# def build_sft_dataset(rows, tokenizer, candidates, system_text, max_len) -> Dataset:
#     """Pre-tokenize all samples at once for speed."""
#     prompts, answers = [], []
#     for item in rows:
#         messages = item.get("messages", [])
#         gt = extract_label(messages, candidates)
#         if gt is None: continue
#         prompts.append(build_prompt(tokenizer, system_text, first_user_content(messages)))
#         answers.append(gt + tokenizer.eos_token)
    
#     samples = []
#     for prompt, answer in zip(prompts, answers):
#         p_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
#         a_ids = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
#         max_p = max_len - len(a_ids)
#         if max_p <= 0: continue
#         p_ids = p_ids[-max_p:]
#         samples.append({"input_ids": p_ids + a_ids, "labels": [-100]*len(p_ids) + a_ids, "attention_mask": [1]*len(p_ids + a_ids)})
    
#     if not samples: raise ValueError("No usable SFT samples.")
#     return Dataset.from_list(samples)


# @dataclass
# class DataCollatorForCompletionOnlyLM:
#     tokenizer: AutoTokenizer
#     pad_to_multiple_of: int | None = 8
    
#     def __call__(self, features):
#         max_len = max(len(f["input_ids"]) for f in features)
#         if self.pad_to_multiple_of:
#             max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
#         batch = {"input_ids": [], "attention_mask": [], "labels": []}
#         for f in features:
#             pad = max_len - len(f["input_ids"])
#             batch["input_ids"].append(f["input_ids"] + [self.tokenizer.pad_token_id]*pad)
#             batch["attention_mask"].append(f["attention_mask"] + [0]*pad)
#             batch["labels"].append(f["labels"] + [-100]*pad)
#         return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# def prepare_eval_entries(rows, tokenizer, candidates, system_text) -> list[Tuple[int, str, str, str]]:
#     entries = []
#     for idx, entry in enumerate(rows):
#         messages = entry.get("messages", [])
#         gt = extract_label(messages, candidates)
#         if gt is None: continue
#         entries.append((idx, gt, first_user_content(messages),
#                        build_prompt(tokenizer, system_text, first_user_content(messages))))
#     return entries


# def get_candidate_token_ids(tokenizer, candidates) -> list[list[int]]:
#     return [tokenizer(c, add_special_tokens=False)["input_ids"] for c in candidates]


# @torch.inference_mode()
# def score_candidates_batched(model, tokenizer, prompts, candidate_token_ids, candidates, max_len, length_normalize):
#     if not prompts: return []
    
#     device = next(model.parameters()).device
#     encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
#                         max_length=max_len, add_special_tokens=False)
    
#     p_ids = encoded["input_ids"]
#     p_mask = encoded["attention_mask"]
#     p_lens = p_mask.sum(dim=1).tolist()
    
#     # Build all sequences at once
#     all_seqs, all_p_lens, all_c_lens = [], [], []
#     for row_idx, plen in enumerate(p_lens):
#         ptoks = p_ids[row_idx, :plen].tolist()
#         for cids in candidate_token_ids:
#             max_p = max_len - len(cids)
#             if max_p <= 0: raise ValueError("Candidate too long")
#             all_seqs.append(ptoks[-max_p:] + cids)
#             all_p_lens.append(len(ptoks[-max_p:]))
#             all_c_lens.append(len(cids))
    
#     batch_size = len(all_seqs)
#     padded = max(len(s) for s in all_seqs)
#     input_ids = torch.full((batch_size, padded), tokenizer.pad_token_id, dtype=torch.long, device=device)
#     attn_mask = torch.zeros((batch_size, padded), dtype=torch.long, device=device)
    
#     for i, s in enumerate(all_seqs):
#         L = len(s)
#         input_ids[i, :L] = torch.tensor(s, dtype=torch.long, device=device)
#         attn_mask[i, :L] = 1
    
#     outputs = model(input_ids=input_ids, attention_mask=attn_mask)
#     log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
    
#     scores = []
#     for i in range(batch_size):
#         plen, clen = all_p_lens[i], all_c_lens[i]
#         tok_scores = [log_probs[i, plen-1+off, input_ids[i, plen+off]] for off in range(clen)]
#         sc = torch.stack(tok_scores).sum()
#         if length_normalize: sc = sc / clen
#         scores.append(sc)
    
#     score_tensor = torch.stack(scores).view(len(prompts), len(candidates))
#     prob_tensor = torch.softmax(score_tensor, dim=1).detach().cpu().numpy()
    
#     return [{candidates[i]: float(row[i]) for i in range(len(candidates))} for row in prob_tensor]


# def evaluate_and_save_predictions(model, tokenizer, eval_entries, candidates, args,
#                                    run_name, train_years_label, test_year, is_base_model=False):
#     model.eval()
#     model.config.use_cache = True
    
#     cids = get_candidate_token_ids(tokenizer, candidates)
#     rows = []
    
#     for start in tqdm(range(0, len(eval_entries), args.eval_batch_size), desc=f"Eval {run_name}"):
#         batch = eval_entries[start: start + args.eval_batch_size]
#         prompts = [e[3] for e in batch]
#         probs = score_candidates_batched(model, tokenizer, prompts, cids, candidates, args.max_len, args.length_normalize_eval)
        
#         for (idx, gt, user_text, _), p in zip(batch, probs):
#             pred = max(p, key=p.get)
#             sp = sorted(p.items(), key=lambda x: x[1], reverse=True)
#             rows.append({
#                 "idx": idx, "gt": gt, "pred": pred, "correct": int(gt == pred),
#                 "confidence": sp[0][1], "margin": sp[0][1] - (sp[1][1] if len(sp) > 1 else 0),
#                 "top2_label": sp[1][0] if len(sp) > 1 else None,
#                 "top2_prob": sp[1][1] if len(sp) > 1 else 0,
#                 "user_text": user_text,
#                 **{f"prob_{c}": p[c] for c in candidates},
#             })
    
#     df = pd.DataFrame(rows)
#     if df.empty: return {"acc": 0.0, "kappa": 0.0, "macro_f1": 0.0, "n": 0}, df
    
#     v2id = {c: i for i, c in enumerate(candidates)}
#     y_true = df["gt"].map(v2id).to_numpy()
#     y_pred = df["pred"].map(v2id).to_numpy()
    
#     metrics = {
#         "acc": float(df["correct"].mean()), "kappa": float(cohen_kappa_score(y_true, y_pred)),
#         "macro_f1": float(f1_score(y_true, y_pred, average="macro")), "n": int(len(df)),
#         "mean_confidence": float(df["confidence"].mean()), "mean_margin": float(df["margin"].mean()),
#     }
#     for c in candidates:
#         mask = df["gt"] == c
#         if mask.sum() > 0:
#             metrics[f"acc_{c}"] = float(df[mask]["correct"].mean())
#             metrics[f"n_{c}"] = int(mask.sum())
    
#     df["run_name"] = run_name
#     df["train_years"] = train_years_label
#     df["test_year"] = test_year
#     df["is_base_model"] = is_base_model
#     df["model_name"] = args.model_name
#     df["task"] = args.task
    
#     model.config.use_cache = False
#     return metrics, df


# def train_and_evaluate(train_rows, test_rows, base_model, tokenizer, candidates, system_text,
#                        args, run_name, train_years_label, test_year):
#     train_dataset = build_sft_dataset(train_rows, tokenizer, candidates, system_text, args.max_len)
#     test_entries = prepare_eval_entries(test_rows, tokenizer, candidates, system_text)
#     print(f"  Train: {len(train_dataset)}, Test: {len(test_entries)}")


#     if hasattr(base_model, 'peft_config') and hasattr(base_model, 'unload'):
#         base_model = base_model.unload()
#         base_model.config.use_cache = False
#         base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
                           
#     model = get_peft_model(base_model, build_lora_config(args), adapter_name="default")
#     model.train()
#     model.config.use_cache = False
    
#     total_steps = (len(train_dataset) * args.epochs) // (args.train_batch_size * args.grad_accum)
#     warmup_steps = max(1, int(total_steps * 0.03))
    
#     training_args = TrainingArguments(
#         output_dir=os.path.join(args.tmp_dir, run_name),
#         per_device_train_batch_size=args.train_batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         num_train_epochs=args.epochs,
#         learning_rate=args.learning_rate,
#         warmup_steps=warmup_steps,
#         lr_scheduler_type="cosine",
#         optim="paged_adamw_8bit",
#         bf16=True,
#         gradient_checkpointing=True,
#         max_grad_norm=0.3,
#         logging_steps=10,
#         save_strategy="no",
#         report_to="none",
#         seed=args.seed,
#         data_seed=args.seed,
#         remove_unused_columns=False,
#         dataloader_pin_memory=True,
#         dataloader_num_workers=args.dataloader_workers,  # OPTIMIZED: parallel loading
#         gradient_checkpointing_kwargs={"use_reentrant": False}
#     )
    
#     trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
#                       data_collator=DataCollatorForCompletionOnlyLM(tokenizer))
#     trainer.train()
    
#     metrics, pred_df = evaluate_and_save_predictions(
#         model, tokenizer, test_entries, candidates, args, run_name, train_years_label, test_year)
    
#     del trainer, model
#     torch.cuda.empty_cache()
#     return metrics, pred_df


# def experiment_1_sequential(args, base_model, tokenizer, candidates, system_text, data_by_year):
#     print("\n" + "="*70)
#     print("EXPERIMENT 1: Sequential (Train on N, Test on N+1)")
#     print("="*70)
#     results, all_preds = [], []
    
#     for i in range(len(YEARS) - 1):
#         ty, tty = YEARS[i], YEARS[i+1]
#         print(f"\n--- Train: {ty} → Test: {tty} ---")
#         metrics, pred_df = train_and_evaluate(
#             data_by_year[ty], data_by_year[tty], base_model, tokenizer, candidates, system_text,
#             args, f"train{ty}_test{tty}", str(ty), tty)
#         metrics.update({"train_year": ty, "test_year": tty, "train_n": len(data_by_year[ty]), "test_n": len(data_by_year[tty])})
#         results.append(metrics); all_preds.append(pred_df)
#         print(f"  Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")
    
#     combined = pd.concat(all_preds, ignore_index=True)
#     combined.to_csv(os.path.join(args.out_dir, f"predictions_sequential_{args.task}_{args.model_name.split('/')[-1]}_more.csv"), index=False)
#     return results, combined


# def experiment_2_cumulative(args, base_model, tokenizer, candidates, system_text, data_by_year):
#     print("\n" + "="*70)
#     print("EXPERIMENT 2: Cumulative → 2024 (Recent First)")
#     print("="*70)
#     results, all_preds = [], []
#     test_rows = data_by_year[2024]
#     prev = [y for y in YEARS if y < 2024]
    
#     for n in range(1, len(prev)+1):
#         train_years = prev[-n:]
#         train_rows = [r for y in train_years for r in data_by_year[y]]
#         label = "+".join(map(str, train_years))
#         print(f"\n--- Train: {train_years} ({len(train_rows)}) → Test: 2024 ---")
#         metrics, pred_df = train_and_evaluate(
#             train_rows, test_rows, base_model, tokenizer, candidates, system_text,
#             args, f"train_{'_'.join(map(str,train_years))}_test2024", label, 2024)
#         metrics.update({"train_years": train_years, "test_year": 2024, "train_n": len(train_rows), "test_n": len(test_rows)})
#         results.append(metrics); all_preds.append(pred_df)
#         print(f"  Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")
    
#     combined = pd.concat(all_preds, ignore_index=True)
#     combined.to_csv(os.path.join(args.out_dir, f"predictions_cumulative_{args.task}_{args.model_name.split('/')[-1]}_more.csv"), index=False)
#     return results, combined


# def evaluate_base_model(args, base_model, tokenizer, candidates, system_text, data_by_year):
#     print("\n" + "="*70)
#     print("BASE MODEL EVALUATION (No Fine-Tuning)")
#     print("="*70)
#     all_preds = []
    
#     for year in YEARS:
#         entries = prepare_eval_entries(data_by_year[year], tokenizer, candidates, system_text)
#         print(f"\n--- Base → Test: {year} ({len(entries)}) ---")
#         metrics, pred_df = evaluate_and_save_predictions(
#             base_model, tokenizer, entries, candidates, args, f"base_{year}", "none", year, True)
#         all_preds.append(pred_df)
#         print(f"  Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")
    
#     combined = pd.concat(all_preds, ignore_index=True)
#     combined.to_csv(os.path.join(args.out_dir, f"predictions_base_{args.task}_{args.model_name.split('/')[-1]}_more.csv"), index=False)
#     return combined


# def main():
#     args = parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)
#     os.makedirs(args.tmp_dir, exist_ok=True)
#     set_seed(args.seed)
    
#     cfg = TASK_CONFIGS[args.task]
#     candidates, system_text = cfg["candidates"], cfg["system_text"]
    
#     print("Loading ANES data...")
#     data_by_year = {}
#     for year in YEARS:
#         path = os.path.join(args.data_dir, f"anes_{year}_without_year.jsonl")
#         if os.path.exists(path):
#             data_by_year[year] = load_jsonl(path)
#             print(f"  {year}: {len(data_by_year[year])}")
#         else:
#             print(f"  WARNING: {path} not found!")
    
#     if 2024 not in data_by_year:
#         print("ERROR: 2024 required!"); return
    
#     print("\nLoading tokenizer & model...")
#     tokenizer = load_tokenizer(args.model_name)
#     base_model = load_qlora_base_model(args.model_name, tokenizer, args.attn_implementation)
    
#     all_results = {"task": args.task, "model": args.model_name, "candidates": candidates, "experiments": {}}
    
#     if args.save_base_predictions:
#         evaluate_base_model(args, base_model, tokenizer, candidates, system_text, data_by_year)
    
#     if args.experiment in ["sequential", "all"]:
#         seq_results, _ = experiment_1_sequential(args, base_model, tokenizer, candidates, system_text, data_by_year)
#         all_results["experiments"]["sequential"] = seq_results
    
#     if args.experiment in ["cumulative", "all"]:
#         cum_results, _ = experiment_2_cumulative(args, base_model, tokenizer, candidates, system_text, data_by_year)
#         all_results["experiments"]["cumulative"] = cum_results
    
#     with open(os.path.join(args.out_dir, f"metrics_{args.task}_{args.model_name.split('/')[-1]}.json"), "w") as f:
#         json.dump(all_results, f, indent=2)
    
#     print("\n" + "="*70 + "\nCOMPLETE\n" + "="*70)


# if __name__ == "__main__":
#     main()

# # import argparse
# # import json
# # import os
# # import random
# # from dataclasses import dataclass
# # from typing import Any, Dict, List, Optional, Tuple
# # from collections import defaultdict

# # import numpy as np
# # import pandas as pd
# # import torch
# # import torch.nn.functional as F
# # from datasets import Dataset, concatenate_datasets
# # from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# # from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef, accuracy_score
# # from tqdm import tqdm
# # from transformers import (
# #     AutoModelForCausalLM,
# #     AutoTokenizer,
# #     BitsAndBytesConfig,
# #     Trainer,
# #     TrainingArguments,
# # )


# # # ---------- Configuration ----------
# # YEARS = [2008, 2012, 2016, 2020, 2024]

# # # Task definitions
# # TASK_CONFIGS = {
# #     "party_choice": {
# #         "candidates": ["Democrat", "Republican", "Other"],
# #         "system_text": (
# #             "You are an expert political analyst specializing in US political attitudes and voting behavior.\n\n"
# #             "Task:\n"
# #             "Based on the respondent's demographic and background information, predict their party choice "
# #             "in the US presidential election.\n\n"
# #             "Rules:\n"
# #             "- You must choose ONLY ONE label.\n"
# #             "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
# #             "Democrat\n"
# #             "Republican\n"
# #             "Other\n\n"
# #             "Important:\n"
# #             "- Base your decision on typical voting patterns and demographics.\n"
# #             "- Do NOT explain your reasoning.\n"
# #             "- Do NOT repeat the input.\n"
# #             "- Output ONLY the label."
# #         ),
# #     },
# #     "ideology": {
# #         "candidates": ["liberal", "moderate", "conservative"],
# #         "system_text": (
# #             "You are an expert political analyst specializing in US political attitudes and voting behavior.\n\n"
# #             "Task:\n"
# #             "Based on the respondent's demographic and background information, predict their ideological "
# #             "self-placement.\n\n"
# #             "Rules:\n"
# #             "- You must choose ONLY ONE label.\n"
# #             "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
# #             "liberal\n"
# #             "moderate\n"
# #             "conservative\n\n"
# #             "Important:\n"
# #             "- Base your decision on typical voting patterns and demographics.\n"
# #             "- Do NOT explain your reasoning.\n"
# #             "- Do NOT repeat the input.\n"
# #             "- Output ONLY the label."
# #         ),
# #     },
# # }


# # def parse_args() -> argparse.Namespace:
# #     parser = argparse.ArgumentParser(
# #         description="Temporal Generalization Experiments with ANES data"
# #     )
# #     parser.add_argument(
# #         "--model_name",
# #         type=str,
# #         default="meta-llama/Llama-3.1-8B-Instruct",
# #         help="Base model to fine-tune"
# #     )
# #     parser.add_argument(
# #         "--task",
# #         type=str,
# #         default="party_choice",
# #         choices=["party_choice", "ideology"],
# #         help="Prediction task"
# #     )
# #     parser.add_argument(
# #         "--data_dir",
# #         type=str,
# #         default="dataset_test/",
# #         help="Directory containing ANES jsonl files named anes_{year}_more.jsonl"
# #     )
# #     parser.add_argument(
# #         "--experiment",
# #         type=str,
# #         default="all",
# #         choices=["sequential", "cumulative", "all"],
# #         help="Experiment type: sequential (train on year N, test on N+1), "
# #              "cumulative (train on all previous, test on 2024), all (both)"
# #     )
# #     parser.add_argument("--out_dir", type=str, default="./results_temporal")
# #     parser.add_argument("--tmp_dir", type=str, default="./tmp_qlora_temporal")
# #     parser.add_argument("--seed", type=int, default=42)
# #     parser.add_argument("--max_len", type=int, default=512)
# #     parser.add_argument("--epochs", type=float, default=3.0)
# #     parser.add_argument("--train_batch_size", type=int, default=1)
# #     parser.add_argument("--grad_accum", type=int, default=16)
# #     parser.add_argument("--eval_batch_size", type=int, default=8)
# #     parser.add_argument("--learning_rate", type=float, default=2e-4)
# #     parser.add_argument("--lora_r", type=int, default=16)
# #     parser.add_argument("--lora_alpha", type=int, default=32)
# #     parser.add_argument("--lora_dropout", type=float, default=0.05)
# #     parser.add_argument(
# #         "--target_modules",
# #         nargs="+",
# #         default=[
# #             "q_proj", "k_proj", "v_proj", "o_proj",
# #             "gate_proj", "up_proj", "down_proj",
# #         ],
# #     )
# #     parser.add_argument(
# #         "--attn_implementation",
# #         type=str,
# #         default="sdpa",
# #         choices=["sdpa", "flash_attention_2", "eager"],
# #     )
# #     parser.add_argument(
# #         "--length_normalize_eval",
# #         action=argparse.BooleanOptionalAction,
# #         default=True,
# #     )
# #     parser.add_argument(
# #         "--save_base_predictions",
# #         action="store_true",
# #         help="Also evaluate and save base model (no FT) predictions"
# #     )
# #     return parser.parse_args()


# # def set_seed(seed: int) -> None:
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     if torch.cuda.is_available():
# #         torch.cuda.manual_seed_all(seed)
# #         torch.backends.cuda.matmul.allow_tf32 = True
# #         torch.backends.cudnn.allow_tf32 = True


# # def load_jsonl(path: str) -> list[dict[str, Any]]:
# #     rows = []
# #     with open(path, "r", encoding="utf-8") as f:
# #         for line in f:
# #             line = line.strip()
# #             if line:
# #                 rows.append(json.loads(line))
# #     return rows


# # def load_tokenizer(model_name: str) -> AutoTokenizer:
# #     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# #     tokenizer.padding_side = "right"
# #     if tokenizer.pad_token is None:
# #         tokenizer.pad_token = tokenizer.eos_token
# #     tokenizer.pad_token_id = tokenizer.eos_token_id
# #     return tokenizer


# # def load_qlora_base_model(
# #     model_name: str,
# #     tokenizer: AutoTokenizer,
# #     attn_implementation: str,
# # ) -> AutoModelForCausalLM:
# #     bnb_config = BitsAndBytesConfig(
# #         load_in_4bit=True,
# #         bnb_4bit_quant_type="nf4",
# #         bnb_4bit_compute_dtype=torch.bfloat16,
# #         bnb_4bit_use_double_quant=True,
# #     )
    
# #     # Dynamically detect available GPUs
# #     num_gpus = torch.cuda.device_count()
# #     if num_gpus == 0:
# #         max_memory = {"cpu": "120GiB"}
# #     elif num_gpus == 1:
# #         max_memory = {0: "75GiB", "cpu": "120GiB"}
# #     else:
# #         max_memory = {i: "75GiB" for i in range(num_gpus)}
# #         max_memory["cpu"] = "120GiB"
    
# #     print(f"  Available GPUs: {num_gpus}, Memory config: {max_memory}")
    
# #     model = AutoModelForCausalLM.from_pretrained(
# #         model_name,
# #         quantization_config=bnb_config,
# #         max_memory=max_memory,
# #         dtype=torch.bfloat16,
# #         attn_implementation=attn_implementation,
# #         low_cpu_mem_usage=True,
# #         device_map="auto",
# #     )
# #     model.config.pad_token_id = tokenizer.pad_token_id
# #     model.config.use_cache = False
# #     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
# #     return model


# # def build_lora_config(args: argparse.Namespace) -> LoraConfig:
# #     return LoraConfig(
# #         r=args.lora_r,
# #         lora_alpha=args.lora_alpha,
# #         target_modules=args.target_modules,
# #         lora_dropout=args.lora_dropout,
# #         bias="none",
# #         task_type="CAUSAL_LM",
# #     )


# # def normalize_label(value: Any, candidates: List[str]) -> str | None:
# #     if value is None:
# #         return None
# #     text = str(value).strip().lower()
# #     for candidate in candidates:
# #         if candidate.lower() in text:
# #             return candidate
# #     return None


# # def extract_label(messages: list[dict[str, str]], candidates: List[str]) -> str | None:
# #     for message in messages:
# #         if message.get("role") == "assistant":
# #             return normalize_label(message.get("content"), candidates)
# #     return None


# # def first_user_content(messages: list[dict[str, str]]) -> str:
# #     for message in messages:
# #         if message.get("role") == "user":
# #             return message.get("content", "")
# #     return messages[0].get("content", "") if messages else ""


# # def build_prompt(tokenizer: AutoTokenizer, system_text: str, user_text: str) -> str:
# #     messages = [
# #         {"role": "system", "content": system_text},
# #         {"role": "user", "content": user_text},
# #     ]
# #     return tokenizer.apply_chat_template(
# #         messages, tokenize=False, add_generation_prompt=True,
# #     )


# # def build_sft_dataset(
# #     rows: list[dict[str, Any]],
# #     tokenizer: AutoTokenizer,
# #     candidates: List[str],
# #     system_text: str,
# #     max_len: int,
# # ) -> Dataset:
# #     """Build SFT dataset from messages format."""
# #     samples = []
# #     for item in rows:
# #         messages = item.get("messages", [])
# #         gt = extract_label(messages, candidates)
# #         if gt is None:
# #             continue
        
# #         prompt = build_prompt(tokenizer, system_text, first_user_content(messages))
# #         answer = gt + tokenizer.eos_token
        
# #         prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
# #         answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        
# #         max_prompt_tokens = max_len - len(answer_ids)
# #         if max_prompt_tokens <= 0:
# #             continue
# #         prompt_ids = prompt_ids[-max_prompt_tokens:]
        
# #         input_ids = prompt_ids + answer_ids
# #         labels = [-100] * len(prompt_ids) + answer_ids
# #         attention_mask = [1] * len(input_ids)
        
# #         samples.append({"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask})
    
# #     if not samples:
# #         raise ValueError("No usable SFT samples found.")
# #     return Dataset.from_list(samples)


# # @dataclass
# # class DataCollatorForCompletionOnlyLM:
# #     tokenizer: AutoTokenizer
# #     pad_to_multiple_of: int | None = 8
    
# #     def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
# #         max_len = max(len(feature["input_ids"]) for feature in features)
# #         if self.pad_to_multiple_of is not None:
# #             multiple = self.pad_to_multiple_of
# #             max_len = ((max_len + multiple - 1) // multiple) * multiple
        
# #         batch = {"input_ids": [], "attention_mask": [], "labels": []}
# #         for feature in features:
# #             pad_len = max_len - len(feature["input_ids"])
# #             batch["input_ids"].append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
# #             batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
# #             batch["labels"].append(feature["labels"] + [-100] * pad_len)
        
# #         return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


# # def prepare_eval_entries(
# #     rows: list[dict[str, Any]],
# #     tokenizer: AutoTokenizer,
# #     candidates: List[str],
# #     system_text: str,
# # ) -> list[Tuple[int, str, str, str]]:
# #     """Returns (idx, gt, user_text, prompt)"""
# #     entries = []
# #     for idx, entry in enumerate(rows):
# #         messages = entry.get("messages", [])
# #         gt = extract_label(messages, candidates)
# #         if gt is None:
# #             continue
# #         user_text = first_user_content(messages)
# #         prompt = build_prompt(tokenizer, system_text, user_text)
# #         entries.append((idx, gt, user_text, prompt))
# #     return entries


# # def get_candidate_token_ids(tokenizer: AutoTokenizer, candidates: List[str]) -> list[list[int]]:
# #     return [tokenizer(c, add_special_tokens=False)["input_ids"] for c in candidates]


# # @torch.inference_mode()
# # def score_candidates_batched(
# #     model: AutoModelForCausalLM,
# #     tokenizer: AutoTokenizer,
# #     prompts: list[str],
# #     candidate_token_ids: list[list[int]],
# #     candidates: List[str],
# #     max_len: int,
# #     length_normalize: bool,
# # ) -> list[dict[str, float]]:
# #     if not prompts:
# #         return []
    
# #     device = next(model.parameters()).device
# #     encoded_prompts = tokenizer(
# #         prompts, return_tensors="pt", padding=True, truncation=True,
# #         max_length=max_len, add_special_tokens=False,
# #     )
    
# #     prompt_input_ids = encoded_prompts["input_ids"]
# #     prompt_attention_mask = encoded_prompts["attention_mask"]
# #     prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()
    
# #     sequences, prompt_lens, candidate_lens = [], [], []
    
# #     for row_idx, prompt_len in enumerate(prompt_lengths):
# #         prompt_tokens = prompt_input_ids[row_idx, :prompt_len].tolist()
# #         for cand_ids in candidate_token_ids:
# #             max_prompt_tokens = max_len - len(cand_ids)
# #             if max_prompt_tokens <= 0:
# #                 raise ValueError("Candidate label is longer than max_len.")
# #             trimmed_prompt = prompt_tokens[-max_prompt_tokens:]
# #             sequences.append(trimmed_prompt + cand_ids)
# #             prompt_lens.append(len(trimmed_prompt))
# #             candidate_lens.append(len(cand_ids))
    
# #     batch_size = len(sequences)
# #     padded_len = max(len(seq) for seq in sequences)
# #     input_ids = torch.full((batch_size, padded_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
# #     attention_mask = torch.zeros((batch_size, padded_len), dtype=torch.long, device=device)
    
# #     for i, seq in enumerate(sequences):
# #         L = len(seq)
# #         input_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=device)
# #         attention_mask[i, :L] = 1
    
# #     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
# #     log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
    
# #     scores = []
# #     for i in range(batch_size):
# #         p_len = prompt_lens[i]
# #         c_len = candidate_lens[i]
# #         token_scores = []
# #         for offset in range(c_len):
# #             logit_pos = p_len - 1 + offset
# #             token_pos = p_len + offset
# #             tid = input_ids[i, token_pos]
# #             token_scores.append(log_probs[i, logit_pos, tid])
# #         c_score = torch.stack(token_scores).sum()
# #         if length_normalize:
# #             c_score = c_score / c_len
# #         scores.append(c_score)
    
# #     score_tensor = torch.stack(scores).view(len(prompts), len(candidates))
# #     prob_tensor = torch.softmax(score_tensor, dim=1).detach().cpu().numpy()
    
# #     return [
# #         {candidates[i]: float(row[i]) for i in range(len(candidates))}
# #         for row in prob_tensor
# #     ]


# # def evaluate_and_save_predictions(
# #     model: AutoModelForCausalLM,
# #     tokenizer: AutoTokenizer,
# #     eval_entries: list[Tuple[int, str, str, str]],
# #     candidates: List[str],
# #     args: argparse.Namespace,
# #     run_name: str,
# #     train_years_label: str,
# #     test_year: int,
# #     is_base_model: bool = False,
# # ) -> Tuple[dict[str, float], pd.DataFrame]:
# #     """Evaluate model and return metrics + detailed predictions DataFrame."""
# #     model.eval()
# #     model.config.use_cache = True
    
# #     candidate_token_ids = get_candidate_token_ids(tokenizer, candidates)
# #     rows = []
    
# #     for start in tqdm(range(0, len(eval_entries), args.eval_batch_size), desc=f"Eval {run_name}"):
# #         batch = eval_entries[start: start + args.eval_batch_size]
# #         prompts = [e[3] for e in batch]
        
# #         probs = score_candidates_batched(
# #             model=model, tokenizer=tokenizer, prompts=prompts,
# #             candidate_token_ids=candidate_token_ids, candidates=candidates,
# #             max_len=args.max_len, length_normalize=args.length_normalize_eval,
# #         )
        
# #         for (idx, gt, user_text, _prompt), p in zip(batch, probs):
# #             pred = max(p, key=p.get)
            
# #             # Get top-2 for margin analysis
# #             sorted_probs = sorted(p.items(), key=lambda x: x[1], reverse=True)
# #             top1_label, top1_prob = sorted_probs[0]
# #             top2_label, top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0)
            
# #             row = {
# #                 "idx": idx,
# #                 "gt": gt,
# #                 "pred": pred,
# #                 "correct": int(gt == pred),
# #                 "confidence": top1_prob,
# #                 "margin": top1_prob - top2_prob,
# #                 "top2_label": top2_label,
# #                 "top2_prob": top2_prob,
# #                 "user_text": user_text,
# #                 **{f"prob_{c}": p[c] for c in candidates},
# #             }
            
# #             rows.append(row)
    
# #     df = pd.DataFrame(rows)
    
# #     if df.empty:
# #         empty_metrics = {"acc": 0.0, "kappa": 0.0, "macro_f1": 0.0, "n": 0}
# #         return empty_metrics, df
    
# #     vote2id = {c: i for i, c in enumerate(candidates)}
# #     y_true = df["gt"].map(vote2id).to_numpy()
# #     y_pred = df["pred"].map(vote2id).to_numpy()
    
# #     metrics = {
# #         "acc": float(df["correct"].mean()),
# #         "kappa": float(cohen_kappa_score(y_true, y_pred)),
# #         "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
# #         "n": int(len(df)),
# #         "mean_confidence": float(df["confidence"].mean()),
# #         "mean_margin": float(df["margin"].mean()),
# #     }
    
# #     # Per-class metrics
# #     for c in candidates:
# #         c_mask = df["gt"] == c
# #         if c_mask.sum() > 0:
# #             metrics[f"acc_{c}"] = float(df[c_mask]["correct"].mean())
# #             metrics[f"n_{c}"] = int(c_mask.sum())
    
# #     # Add metadata
# #     df["run_name"] = run_name
# #     df["train_years"] = train_years_label
# #     df["test_year"] = test_year
# #     df["is_base_model"] = is_base_model
# #     df["model_name"] = args.model_name
# #     df["task"] = args.task
    
# #     model.config.use_cache = False
# #     return metrics, df


# # def train_and_evaluate(
# #     train_rows: list[dict[str, Any]],
# #     test_rows: list[dict[str, Any]],
# #     base_model: AutoModelForCausalLM,
# #     tokenizer: AutoTokenizer,
# #     candidates: List[str],
# #     system_text: str,
# #     args: argparse.Namespace,
# #     run_name: str,
# #     train_years_label: str,
# #     test_year: int,
# # ) -> Tuple[dict[str, float], pd.DataFrame]:
# #     """Train on train_rows, evaluate on test_rows, return metrics + predictions."""
    
# #     # Build datasets
# #     train_dataset = build_sft_dataset(train_rows, tokenizer, candidates, system_text, args.max_len)
# #     test_entries = prepare_eval_entries(test_rows, tokenizer, candidates, system_text)
    
# #     print(f"  Train samples: {len(train_dataset)}, Test samples: {len(test_entries)}")
    
# #     # Create PEFT model
# #     lora_config = build_lora_config(args)
# #     model = get_peft_model(base_model, lora_config, adapter_name="default")
# #     model.train()
# #     model.config.use_cache = False
    
# #     # Training
# #     total_steps = (len(train_dataset) * args.epochs) // (args.train_batch_size * args.grad_accum)
# #     warmup_steps = max(1, int(total_steps * 0.03))
    
# #     training_args = TrainingArguments(
# #         output_dir=os.path.join(args.tmp_dir, run_name),
# #         per_device_train_batch_size=args.train_batch_size,
# #         gradient_accumulation_steps=args.grad_accum,
# #         num_train_epochs=args.epochs,
# #         learning_rate=args.learning_rate,
# #         warmup_steps=warmup_steps,
# #         lr_scheduler_type="cosine",
# #         optim="paged_adamw_8bit",
# #         bf16=True,
# #         gradient_checkpointing=True,
# #         max_grad_norm=0.3,
# #         logging_steps=10,
# #         save_strategy="no",
# #         report_to="none",
# #         seed=args.seed,
# #         data_seed=args.seed,
# #         remove_unused_columns=False,
# #         dataloader_pin_memory=True,
# #     )
    
# #     trainer = Trainer(
# #         model=model,
# #         args=training_args,
# #         train_dataset=train_dataset,
# #         data_collator=DataCollatorForCompletionOnlyLM(tokenizer),
# #     )
    
# #     trainer.train()
    
# #     # Evaluate and get predictions
# #     metrics, pred_df = evaluate_and_save_predictions(
# #         model, tokenizer, test_entries, candidates, args,
# #         run_name, train_years_label, test_year, is_base_model=False
# #     )
    
# #     del trainer, model
# #     torch.cuda.empty_cache()
    
# #     return metrics, pred_df


# # def experiment_1_sequential(args, base_model, tokenizer, candidates, system_text, data_by_year):
# #     """Experiment 1: Train on year N, test on year N+1"""
# #     print("\n" + "="*70)
# #     print("EXPERIMENT 1: Sequential (Train on N, Test on N+1)")
# #     print("="*70)
    
# #     results = []
# #     all_predictions = []
    
# #     for i in range(len(YEARS) - 1):
# #         train_year = YEARS[i]
# #         test_year = YEARS[i + 1]
# #         run_name = f"train{train_year}_test{test_year}"
# #         train_label = str(train_year)
        
# #         print(f"\n--- Train: {train_year} → Test: {test_year} ---")
        
# #         metrics, pred_df = train_and_evaluate(
# #             data_by_year[train_year], data_by_year[test_year],
# #             base_model, tokenizer, candidates, system_text, args, run_name,
# #             train_label, test_year
# #         )
# #         metrics["train_year"] = train_year
# #         metrics["test_year"] = test_year
# #         metrics["train_n"] = len(data_by_year[train_year])
# #         metrics["test_n"] = len(data_by_year[test_year])
# #         results.append(metrics)
# #         all_predictions.append(pred_df)
        
# #         print(f"  Results: Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")
    
# #     # Save combined predictions
# #     combined_df = pd.concat(all_predictions, ignore_index=True)
# #     pred_path = os.path.join(args.out_dir, f"predictions_sequential_{args.task}_{args.model_name.split('/')[-1]}.csv")
# #     combined_df.to_csv(pred_path, index=False)
# #     print(f"\nPredictions saved to: {pred_path}")
    
# #     return results, combined_df


# # def experiment_2_cumulative(args, base_model, tokenizer, candidates, system_text, data_by_year):
# #     """Experiment 2: Train on previous years (most recent first), test on 2024"""
# #     print("\n" + "="*70)
# #     print("EXPERIMENT 2: Cumulative Training → Test on 2024 (Recent First)")
# #     print("="*70)
    
# #     results = []
# #     all_predictions = []
# #     test_rows = data_by_year[2024]
    
# #     # Get all years before 2024, sorted chronologically
# #     previous_years = [y for y in YEARS if y < 2024]  # [2008, 2012, 2016, 2020]
    
# #     for n_years in range(1, len(previous_years) + 1):
# #         train_years = previous_years[-n_years:]  # Take last n years (most recent)
# #         train_rows = []
# #         for y in train_years:
# #             train_rows.extend(data_by_year[y])
        
# #         run_name = f"train_{'_'.join(map(str, train_years))}_test2024"
# #         train_label = "+".join(map(str, train_years))
        
# #         print(f"\n--- Train: {train_years} ({len(train_rows)} samples) → Test: 2024 ---")
        
# #         metrics, pred_df = train_and_evaluate(
# #             train_rows, test_rows,
# #             base_model, tokenizer, candidates, system_text, args, run_name,
# #             train_label, 2024
# #         )
# #         metrics["train_years"] = train_years
# #         metrics["test_year"] = 2024
# #         metrics["train_n"] = len(train_rows)
# #         metrics["test_n"] = len(test_rows)
# #         results.append(metrics)
# #         all_predictions.append(pred_df)
        
# #         print(f"  Results: Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")
    
# #     # Save combined predictions
# #     combined_df = pd.concat(all_predictions, ignore_index=True)
# #     pred_path = os.path.join(args.out_dir, f"predictions_cumulative_{args.task}_{args.model_name.split('/')[-1]}.csv")
# #     combined_df.to_csv(pred_path, index=False)
# #     print(f"\nPredictions saved to: {pred_path}")
    
# #     return results, combined_df


# # def evaluate_base_model(args, base_model, tokenizer, candidates, system_text, data_by_year):
# #     """Evaluate base model (no fine-tuning) on all test years for comparison."""
# #     print("\n" + "="*70)
# #     print("BASE MODEL EVALUATION (No Fine-Tuning)")
# #     print("="*70)
    
# #     all_predictions = []
    
# #     for test_year in YEARS:
# #         run_name = f"base_model_test{test_year}"
# #         test_entries = prepare_eval_entries(data_by_year[test_year], tokenizer, candidates, system_text)
        
# #         print(f"\n--- Base Model → Test: {test_year} ({len(test_entries)} samples) ---")
        
# #         metrics, pred_df = evaluate_and_save_predictions(
# #             base_model, tokenizer, test_entries, candidates, args,
# #             run_name, "none", test_year, is_base_model=True
# #         )
        
# #         all_predictions.append(pred_df)
# #         print(f"  Results: Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")
    
# #     combined_df = pd.concat(all_predictions, ignore_index=True)
# #     pred_path = os.path.join(args.out_dir, f"predictions_base_model_{args.task}_{args.model_name.split('/')[-1]}.csv")
# #     combined_df.to_csv(pred_path, index=False)
# #     print(f"\nBase model predictions saved to: {pred_path}")
    
# #     return combined_df


# # def main():
# #     args = parse_args()
# #     os.makedirs(args.out_dir, exist_ok=True)
# #     os.makedirs(args.tmp_dir, exist_ok=True)
# #     set_seed(args.seed)
    
# #     task_config = TASK_CONFIGS[args.task]
# #     candidates = task_config["candidates"]
# #     system_text = task_config["system_text"]
    
# #     # Load all years' data
# #     print("Loading ANES data...")
# #     data_by_year = {}
# #     for year in YEARS:
# #         path = os.path.join(args.data_dir, f"anes_{year}.jsonl")
# #         if os.path.exists(path):
# #             rows = load_jsonl(path)
# #             data_by_year[year] = rows
# #             print(f"  {year}: {len(rows)} samples")
# #         else:
# #             print(f"  WARNING: {path} not found!")
    
# #     if 2024 not in data_by_year:
# #         print("ERROR: 2024 test data is required!")
# #         return
    
# #     # Load tokenizer and base model
# #     print("\nLoading tokenizer and base model...")
# #     tokenizer = load_tokenizer(args.model_name)
# #     base_model = load_qlora_base_model(args.model_name, tokenizer, args.attn_implementation)
    
# #     all_results = {
# #         "task": args.task, 
# #         "model": args.model_name, 
# #         "candidates": candidates,
# #         "experiments": {}
# #     }
    
# #     # Evaluate base model if requested
# #     if args.save_base_predictions:
# #         base_pred_df = evaluate_base_model(args, base_model, tokenizer, candidates, system_text, data_by_year)
    
# #     # Run experiments
# #     if args.experiment in ["sequential", "all"]:
# #         seq_results, seq_pred_df = experiment_1_sequential(
# #             args, base_model, tokenizer, candidates, system_text, data_by_year
# #         )
# #         all_results["experiments"]["sequential"] = seq_results
    
# #     if args.experiment in ["cumulative", "all"]:
# #         cum_results, cum_pred_df = experiment_2_cumulative(
# #             args, base_model, tokenizer, candidates, system_text, data_by_year
# #         )
# #         all_results["experiments"]["cumulative"] = cum_results
    
# #     # Save metrics summary
# #     metrics_path = os.path.join(args.out_dir, f"metrics_summary_{args.task}_{args.model_name.split('/')[-1]}.json")
# #     with open(metrics_path, "w") as f:
# #         json.dump(all_results, f, indent=2)
# #     print(f"\nMetrics summary saved to: {metrics_path}")
    
# #     # Print summary
# #     print("\n" + "="*70)
# #     print("FINAL SUMMARY")
# #     print("="*70)
    
# #     if "sequential" in all_results["experiments"]:
# #         print("\nExperiment 1: Sequential Transfer")
# #         print(f"{'Train':>8} → {'Test':>8} | {'Acc':>8} | {'Kappa':>8} | {'F1':>8} | {'Conf':>8}")
# #         print("-" * 60)
# #         for r in all_results["experiments"]["sequential"]:
# #             print(f"{r['train_year']:>8} → {r['test_year']:>8} | {r['acc']:>8.4f} | {r['kappa']:>8.4f} | {r['macro_f1']:>8.4f} | {r.get('mean_confidence', 0):>8.4f}")
    
# #     if "cumulative" in all_results["experiments"]:
# #         print("\nExperiment 2: Cumulative → 2024")
# #         print(f"{'Train Years':>25} | {'Acc':>8} | {'Kappa':>8} | {'F1':>8} | {'Conf':>8}")
# #         print("-" * 65)
# #         for r in all_results["experiments"]["cumulative"]:
# #             years_str = "+".join(map(str, r["train_years"]))
# #             print(f"{years_str:>25} | {r['acc']:>8.4f} | {r['kappa']:>8.4f} | {r['macro_f1']:>8.4f} | {r.get('mean_confidence', 0):>8.4f}")


# # if __name__ == "__main__":
# #     main()


# # # import argparse
# # # import json
# # # import os
# # # import random
# # # from dataclasses import dataclass
# # # from typing import Any, Dict, List, Optional, Tuple
# # # from collections import defaultdict

# # # import numpy as np
# # # import pandas as pd
# # # import torch
# # # import torch.nn.functional as F
# # # from datasets import Dataset, concatenate_datasets
# # # from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# # # from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef, accuracy_score
# # # from tqdm import tqdm
# # # from transformers import (
# # #     AutoModelForCausalLM,
# # #     AutoTokenizer,
# # #     BitsAndBytesConfig,
# # #     Trainer,
# # #     TrainingArguments,
# # # )


# # # # ---------- Configuration ----------
# # # YEARS = [2008, 2012, 2016, 2020, 2024]

# # # # Task definitions
# # # TASK_CONFIGS = {
# # #     "party_choice": {
# # #         "candidates": ["Democrat", "Republican", "Other"],
# # #         "system_text": (
# # #             "You are an expert political analyst specializing in US political attitudes and voting behavior.\n\n"
# # #             "Task:\n"
# # #             "Based on the respondent's demographic and background information, predict their party choice "
# # #             "in the US presidential election.\n\n"
# # #             "Rules:\n"
# # #             "- You must choose ONLY ONE label.\n"
# # #             "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
# # #             "Democrat\n"
# # #             "Republican\n"
# # #             "Other\n\n"
# # #             "Important:\n"
# # #             "- Base your decision on typical voting patterns and demographics.\n"
# # #             "- Do NOT explain your reasoning.\n"
# # #             "- Do NOT repeat the input.\n"
# # #             "- Output ONLY the label."
# # #         ),
# # #     },
# # #     "ideology": {
# # #         "candidates": ["liberal", "moderate", "conservative"],
# # #         "system_text": (
# # #             "You are an expert political analyst specializing in US political attitudes and voting behavior.\n\n"
# # #             "Task:\n"
# # #             "Based on the respondent's demographic and background information, predict their ideological "
# # #             "self-placement.\n\n"
# # #             "Rules:\n"
# # #             "- You must choose ONLY ONE label.\n"
# # #             "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
# # #             "liberal\n"
# # #             "moderate\n"
# # #             "conservative\n\n"
# # #             "Important:\n"
# # #             "- Base your decision on typical voting patterns and demographics.\n"
# # #             "- Do NOT explain your reasoning.\n"
# # #             "- Do NOT repeat the input.\n"
# # #             "- Output ONLY the label."
# # #         ),
# # #     },
# # # }


# # # def parse_args() -> argparse.Namespace:
# # #     parser = argparse.ArgumentParser(
# # #         description="Temporal Generalization Experiments with ANES data"
# # #     )
# # #     parser.add_argument(
# # #         "--model_name",
# # #         type=str,
# # #         default="meta-llama/Llama-3.1-8B-Instruct",
# # #         help="Base model to fine-tune"
# # #     )
# # #     parser.add_argument(
# # #         "--task",
# # #         type=str,
# # #         default="party_choice",
# # #         choices=["party_choice", "ideology"],
# # #         help="Prediction task"
# # #     )
# # #     parser.add_argument(
# # #         "--data_dir",
# # #         type=str,
# # #         default="dataset_test/",
# # #         help="Directory containing ANES jsonl files named anes_{year}.jsonl"
# # #     )
# # #     parser.add_argument(
# # #         "--experiment",
# # #         type=str,
# # #         default="all",
# # #         choices=["sequential", "cumulative", "all"],
# # #         help="Experiment type: sequential (train on year N, test on N+1), "
# # #              "cumulative (train on all previous, test on 2024), all (both)"
# # #     )
# # #     parser.add_argument("--out_dir", type=str, default="./results_temporal")
# # #     parser.add_argument("--tmp_dir", type=str, default="./tmp_qlora_temporal")
# # #     parser.add_argument("--seed", type=int, default=42)
# # #     parser.add_argument("--max_len", type=int, default=512)
# # #     parser.add_argument("--epochs", type=float, default=3.0)
# # #     parser.add_argument("--train_batch_size", type=int, default=1)
# # #     parser.add_argument("--grad_accum", type=int, default=16)
# # #     parser.add_argument("--eval_batch_size", type=int, default=8)
# # #     parser.add_argument("--learning_rate", type=float, default=2e-4)
# # #     parser.add_argument("--lora_r", type=int, default=16)
# # #     parser.add_argument("--lora_alpha", type=int, default=32)
# # #     parser.add_argument("--lora_dropout", type=float, default=0.05)
# # #     parser.add_argument(
# # #         "--target_modules",
# # #         nargs="+",
# # #         default=[
# # #             "q_proj", "k_proj", "v_proj", "o_proj",
# # #             "gate_proj", "up_proj", "down_proj",
# # #         ],
# # #     )
# # #     parser.add_argument(
# # #         "--attn_implementation",
# # #         type=str,
# # #         default="sdpa",
# # #         choices=["sdpa", "flash_attention_2", "eager"],
# # #     )
# # #     parser.add_argument(
# # #         "--length_normalize_eval",
# # #         action=argparse.BooleanOptionalAction,
# # #         default=True,
# # #     )
# # #     parser.add_argument(
# # #         "--save_base_predictions",
# # #         action="store_true",
# # #         help="Also evaluate and save base model (no FT) predictions"
# # #     )
# # #     return parser.parse_args()


# # # def set_seed(seed: int) -> None:
# # #     random.seed(seed)
# # #     np.random.seed(seed)
# # #     torch.manual_seed(seed)
# # #     if torch.cuda.is_available():
# # #         torch.cuda.manual_seed_all(seed)
# # #         torch.backends.cuda.matmul.allow_tf32 = True
# # #         torch.backends.cudnn.allow_tf32 = True


# # # def load_jsonl(path: str) -> list[dict[str, Any]]:
# # #     rows = []
# # #     with open(path, "r", encoding="utf-8") as f:
# # #         for line in f:
# # #             line = line.strip()
# # #             if line:
# # #                 rows.append(json.loads(line))
# # #     return rows


# # # def load_tokenizer(model_name: str) -> AutoTokenizer:
# # #     tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# # #     tokenizer.padding_side = "right"
# # #     if tokenizer.pad_token is None:
# # #         tokenizer.pad_token = tokenizer.eos_token
# # #     tokenizer.pad_token_id = tokenizer.eos_token_id
# # #     return tokenizer


# # # def load_qlora_base_model(
# # #     model_name: str,
# # #     tokenizer: AutoTokenizer,
# # #     attn_implementation: str,
# # # ) -> AutoModelForCausalLM:
# # #     bnb_config = BitsAndBytesConfig(
# # #         load_in_4bit=True,
# # #         bnb_4bit_quant_type="nf4",
# # #         bnb_4bit_compute_dtype=torch.bfloat16,
# # #         bnb_4bit_use_double_quant=True,
# # #     )
    
# # #     # FIXED: Dynamically detect available GPUs
# # #     num_gpus = torch.cuda.device_count()
# # #     if num_gpus == 0:
# # #         max_memory = {"cpu": "120GiB"}
# # #     elif num_gpus == 1:
# # #         max_memory = {0: "75GiB", "cpu": "120GiB"}
# # #     else:
# # #         max_memory = {i: "75GiB" for i in range(num_gpus)}
# # #         max_memory["cpu"] = "120GiB"
    
# # #     print(f"  Available GPUs: {num_gpus}, Memory config: {max_memory}")
    
# # #     model = AutoModelForCausalLM.from_pretrained(
# # #         model_name,
# # #         quantization_config=bnb_config,
# # #         max_memory=max_memory,
# # #         dtype=torch.bfloat16,
# # #         attn_implementation=attn_implementation,
# # #         low_cpu_mem_usage=True,
# # #         device_map="auto",
# # #     )
# # #     model.config.pad_token_id = tokenizer.pad_token_id
# # #     model.config.use_cache = False
# # #     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
# # #     return model

# # # def build_lora_config(args: argparse.Namespace) -> LoraConfig:
# # #     return LoraConfig(
# # #         r=args.lora_r,
# # #         lora_alpha=args.lora_alpha,
# # #         target_modules=args.target_modules,
# # #         lora_dropout=args.lora_dropout,
# # #         bias="none",
# # #         task_type="CAUSAL_LM",
# # #     )


# # # def normalize_label(value: Any, candidates: List[str]) -> str | None:
# # #     if value is None:
# # #         return None
# # #     text = str(value).strip().lower()
# # #     for candidate in candidates:
# # #         if candidate.lower() in text:
# # #             return candidate
# # #     return None


# # # def extract_label(messages: list[dict[str, str]], candidates: List[str]) -> str | None:
# # #     for message in messages:
# # #         if message.get("role") == "assistant":
# # #             return normalize_label(message.get("content"), candidates)
# # #     return None


# # # def first_user_content(messages: list[dict[str, str]]) -> str:
# # #     for message in messages:
# # #         if message.get("role") == "user":
# # #             return message.get("content", "")
# # #     return messages[0].get("content", "") if messages else ""


# # # def extract_demographics(user_text: str) -> Dict[str, str]:
# # #     """Extract demographic features from user text for CSV analysis."""
# # #     demographics = {}
    
# # #     # Common patterns in ANES data
# # #     patterns = {
# # #         "race": r"(?:Racially,?\s*I am|I am)\s*(white|black|hispanic|asian|native american|other)",
# # #         "gender": r"I am a\s*(man|woman)",
# # #         "age": r"I am\s*(\d+)\s*years? old",
# # #         "religion": r"I (have ever|have never|regularly)\s*(?:attended?\s*)?(?:church|religious services)?",
# # #         "political_interest": r"I am\s*(very|somewhat|not\s*very|not at all)\s*interested in politics",
# # #         "education": r"(?:I have|I hold)\s*(?:a\s*)?(no degree|high school|some college|bachelor|graduate|phd|associate)",
# # #         "income": r"(?:My household income is|income\s*:\s*)(\$.+?(?:\d|k))",
# # #         "region": r"(?:I live in|from)\s*(?:the\s*)?(northeast|midwest|south|west)",
# # #         "party_id": r"(?:I am a|I identify as)\s*(democrat|republican|independent)",
# # #     }
    
# # #     import re
# # #     for key, pattern in patterns.items():
# # #         match = re.search(pattern, user_text, re.IGNORECASE)
# # #         if match:
# # #             demographics[key] = match.group(1).strip()
    
# # #     return demographics


# # # def build_prompt(tokenizer: AutoTokenizer, system_text: str, user_text: str) -> str:
# # #     messages = [
# # #         {"role": "system", "content": system_text},
# # #         {"role": "user", "content": user_text},
# # #     ]
# # #     return tokenizer.apply_chat_template(
# # #         messages, tokenize=False, add_generation_prompt=True,
# # #     )


# # # def build_sft_dataset(
# # #     rows: list[dict[str, Any]],
# # #     tokenizer: AutoTokenizer,
# # #     candidates: List[str],
# # #     system_text: str,
# # #     max_len: int,
# # # ) -> Dataset:
# # #     """Build SFT dataset from messages format."""
# # #     samples = []
# # #     for item in rows:
# # #         messages = item.get("messages", [])
# # #         gt = extract_label(messages, candidates)
# # #         if gt is None:
# # #             continue
        
# # #         prompt = build_prompt(tokenizer, system_text, first_user_content(messages))
# # #         answer = gt + tokenizer.eos_token
        
# # #         prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
# # #         answer_ids = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        
# # #         max_prompt_tokens = max_len - len(answer_ids)
# # #         if max_prompt_tokens <= 0:
# # #             continue
# # #         prompt_ids = prompt_ids[-max_prompt_tokens:]
        
# # #         input_ids = prompt_ids + answer_ids
# # #         labels = [-100] * len(prompt_ids) + answer_ids
# # #         attention_mask = [1] * len(input_ids)
        
# # #         samples.append({"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask})
    
# # #     if not samples:
# # #         raise ValueError("No usable SFT samples found.")
# # #     return Dataset.from_list(samples)


# # # @dataclass
# # # class DataCollatorForCompletionOnlyLM:
# # #     tokenizer: AutoTokenizer
# # #     pad_to_multiple_of: int | None = 8
    
# # #     def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
# # #         max_len = max(len(feature["input_ids"]) for feature in features)
# # #         if self.pad_to_multiple_of is not None:
# # #             multiple = self.pad_to_multiple_of
# # #             max_len = ((max_len + multiple - 1) // multiple) * multiple
        
# # #         batch = {"input_ids": [], "attention_mask": [], "labels": []}
# # #         for feature in features:
# # #             pad_len = max_len - len(feature["input_ids"])
# # #             batch["input_ids"].append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
# # #             batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
# # #             batch["labels"].append(feature["labels"] + [-100] * pad_len)
        
# # #         return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


# # # def prepare_eval_entries(
# # #     rows: list[dict[str, Any]],
# # #     tokenizer: AutoTokenizer,
# # #     candidates: List[str],
# # #     system_text: str,
# # # ) -> list[Tuple[int, str, str, str, Dict[str, str]]]:
# # #     """Returns (idx, gt, user_text, prompt, demographics_dict)"""
# # #     entries = []
# # #     for idx, entry in enumerate(rows):
# # #         messages = entry.get("messages", [])
# # #         gt = extract_label(messages, candidates)
# # #         if gt is None:
# # #             continue
# # #         user_text = first_user_content(messages)
# # #         prompt = build_prompt(tokenizer, system_text, user_text)
# # #         demographics = extract_demographics(user_text)
# # #         entries.append((idx, gt, user_text, prompt, demographics))
# # #     return entries


# # # def get_candidate_token_ids(tokenizer: AutoTokenizer, candidates: List[str]) -> list[list[int]]:
# # #     return [tokenizer(c, add_special_tokens=False)["input_ids"] for c in candidates]


# # # @torch.inference_mode()
# # # def score_candidates_batched(
# # #     model: AutoModelForCausalLM,
# # #     tokenizer: AutoTokenizer,
# # #     prompts: list[str],
# # #     candidate_token_ids: list[list[int]],
# # #     candidates: List[str],
# # #     max_len: int,
# # #     length_normalize: bool,
# # # ) -> list[dict[str, float]]:
# # #     if not prompts:
# # #         return []
    
# # #     device = next(model.parameters()).device
# # #     encoded_prompts = tokenizer(
# # #         prompts, return_tensors="pt", padding=True, truncation=True,
# # #         max_length=max_len, add_special_tokens=False,
# # #     )
    
# # #     prompt_input_ids = encoded_prompts["input_ids"]
# # #     prompt_attention_mask = encoded_prompts["attention_mask"]
# # #     prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()
    
# # #     sequences, prompt_lens, candidate_lens = [], [], []
    
# # #     for row_idx, prompt_len in enumerate(prompt_lengths):
# # #         prompt_tokens = prompt_input_ids[row_idx, :prompt_len].tolist()
# # #         for cand_ids in candidate_token_ids:
# # #             max_prompt_tokens = max_len - len(cand_ids)
# # #             if max_prompt_tokens <= 0:
# # #                 raise ValueError("Candidate label is longer than max_len.")
# # #             trimmed_prompt = prompt_tokens[-max_prompt_tokens:]
# # #             sequences.append(trimmed_prompt + cand_ids)
# # #             prompt_lens.append(len(trimmed_prompt))
# # #             candidate_lens.append(len(cand_ids))
    
# # #     batch_size = len(sequences)
# # #     padded_len = max(len(seq) for seq in sequences)
# # #     input_ids = torch.full((batch_size, padded_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
# # #     attention_mask = torch.zeros((batch_size, padded_len), dtype=torch.long, device=device)
    
# # #     for i, seq in enumerate(sequences):
# # #         L = len(seq)
# # #         input_ids[i, :L] = torch.tensor(seq, dtype=torch.long, device=device)
# # #         attention_mask[i, :L] = 1
    
# # #     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
# # #     log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
    
# # #     scores = []
# # #     for i in range(batch_size):
# # #         p_len = prompt_lens[i]
# # #         c_len = candidate_lens[i]
# # #         token_scores = []
# # #         for offset in range(c_len):
# # #             logit_pos = p_len - 1 + offset
# # #             token_pos = p_len + offset
# # #             tid = input_ids[i, token_pos]
# # #             token_scores.append(log_probs[i, logit_pos, tid])
# # #         c_score = torch.stack(token_scores).sum()
# # #         if length_normalize:
# # #             c_score = c_score / c_len
# # #         scores.append(c_score)
    
# # #     score_tensor = torch.stack(scores).view(len(prompts), len(candidates))
# # #     prob_tensor = torch.softmax(score_tensor, dim=1).detach().cpu().numpy()
    
# # #     return [
# # #         {candidates[i]: float(row[i]) for i in range(len(candidates))}
# # #         for row in prob_tensor
# # #     ]


# # # def evaluate_and_save_predictions(
# # #     model: AutoModelForCausalLM,
# # #     tokenizer: AutoTokenizer,
# # #     eval_entries: list[Tuple[int, str, str, str, Dict[str, str]]],
# # #     candidates: List[str],
# # #     args: argparse.Namespace,
# # #     run_name: str,
# # #     train_years_label: str,
# # #     test_year: int,
# # #     is_base_model: bool = False,
# # # ) -> Tuple[dict[str, float], pd.DataFrame]:
# # #     """Evaluate model and return metrics + detailed predictions DataFrame."""
# # #     model.eval()
# # #     model.config.use_cache = True
    
# # #     candidate_token_ids = get_candidate_token_ids(tokenizer, candidates)
# # #     rows = []
    
# # #     for start in tqdm(range(0, len(eval_entries), args.eval_batch_size), desc=f"Eval {run_name}"):
# # #         batch = eval_entries[start: start + args.eval_batch_size]
# # #         prompts = [e[3] for e in batch]
        
# # #         probs = score_candidates_batched(
# # #             model=model, tokenizer=tokenizer, prompts=prompts,
# # #             candidate_token_ids=candidate_token_ids, candidates=candidates,
# # #             max_len=args.max_len, length_normalize=args.length_normalize_eval,
# # #         )
        
# # #         for (idx, gt, user_text, _, demographics), p in zip(batch, probs):
# # #             pred = max(p, key=p.get)
            
# # #             # Get top-2 for margin analysis
# # #             sorted_probs = sorted(p.items(), key=lambda x: x[1], reverse=True)
# # #             top1_label, top1_prob = sorted_probs[0]
# # #             top2_label, top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else (None, 0)
            
# # #             row = {
# # #                 "idx": idx,
# # #                 "gt": gt,
# # #                 "pred": pred,
# # #                 "correct": int(gt == pred),
# # #                 "confidence": top1_prob,
# # #                 "margin": top1_prob - top2_prob,
# # #                 "top2_label": top2_label,
# # #                 "top2_prob": top2_prob,
# # #                 "user_text": user_text,
# # #                 **{f"prob_{c}": p[c] for c in candidates},
# # #             }
            
# # #             # Add demographics
# # #             for demo_key, demo_val in demographics.items():
# # #                 row[f"demo_{demo_key}"] = demo_val
            
# # #             rows.append(row)
    
# # #     df = pd.DataFrame(rows)
    
# # #     if df.empty:
# # #         empty_metrics = {"acc": 0.0, "kappa": 0.0, "macro_f1": 0.0, "n": 0}
# # #         return empty_metrics, df
    
# # #     vote2id = {c: i for i, c in enumerate(candidates)}
# # #     y_true = df["gt"].map(vote2id).to_numpy()
# # #     y_pred = df["pred"].map(vote2id).to_numpy()
    
# # #     metrics = {
# # #         "acc": float(df["correct"].mean()),
# # #         "kappa": float(cohen_kappa_score(y_true, y_pred)),
# # #         "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
# # #         "n": int(len(df)),
# # #         "mean_confidence": float(df["confidence"].mean()),
# # #         "mean_margin": float(df["margin"].mean()),
# # #     }
    
# # #     # Per-class metrics
# # #     for c in candidates:
# # #         c_mask = df["gt"] == c
# # #         if c_mask.sum() > 0:
# # #             metrics[f"acc_{c}"] = float(df[c_mask]["correct"].mean())
# # #             metrics[f"n_{c}"] = int(c_mask.sum())
    
# # #     # Add metadata
# # #     df["run_name"] = run_name
# # #     df["train_years"] = train_years_label
# # #     df["test_year"] = test_year
# # #     df["is_base_model"] = is_base_model
# # #     df["model_name"] = args.model_name
# # #     df["task"] = args.task
    
# # #     model.config.use_cache = False
# # #     return metrics, df


# # # def train_and_evaluate(
# # #     train_rows: list[dict[str, Any]],
# # #     test_rows: list[dict[str, Any]],
# # #     base_model: AutoModelForCausalLM,
# # #     tokenizer: AutoTokenizer,
# # #     candidates: List[str],
# # #     system_text: str,
# # #     args: argparse.Namespace,
# # #     run_name: str,
# # #     train_years_label: str,
# # #     test_year: int,
# # # ) -> Tuple[dict[str, float], pd.DataFrame]:
# # #     """Train on train_rows, evaluate on test_rows, return metrics + predictions."""
    
# # #     # Build datasets
# # #     train_dataset = build_sft_dataset(train_rows, tokenizer, candidates, system_text, args.max_len)
# # #     test_entries = prepare_eval_entries(test_rows, tokenizer, candidates, system_text)
    
# # #     print(f"  Train samples: {len(train_dataset)}, Test samples: {len(test_entries)}")
    
# # #     # Create PEFT model
# # #     lora_config = build_lora_config(args)
# # #     model = get_peft_model(base_model, lora_config, adapter_name="default")
# # #     model.train()
# # #     model.config.use_cache = False
    
# # #     # Training
# # #     total_steps = (len(train_dataset) * args.epochs) // (args.train_batch_size * args.grad_accum)
# # #     warmup_steps = max(1, int(total_steps * 0.03))
    
# # #     training_args = TrainingArguments(
# # #         output_dir=os.path.join(args.tmp_dir, run_name),
# # #         per_device_train_batch_size=args.train_batch_size,
# # #         gradient_accumulation_steps=args.grad_accum,
# # #         num_train_epochs=args.epochs,
# # #         learning_rate=args.learning_rate,
# # #         warmup_steps=warmup_steps,
# # #         lr_scheduler_type="cosine",
# # #         optim="paged_adamw_8bit",
# # #         bf16=True,
# # #         gradient_checkpointing=True,
# # #         max_grad_norm=0.3,
# # #         logging_steps=10,
# # #         save_strategy="no",
# # #         report_to="none",
# # #         seed=args.seed,
# # #         data_seed=args.seed,
# # #         remove_unused_columns=False,
# # #         dataloader_pin_memory=True,
# # #     )
    
# # #     trainer = Trainer(
# # #         model=model,
# # #         args=training_args,
# # #         train_dataset=train_dataset,
# # #         data_collator=DataCollatorForCompletionOnlyLM(tokenizer),
# # #     )
    
# # #     trainer.train()
    
# # #     # Evaluate and get predictions
# # #     metrics, pred_df = evaluate_and_save_predictions(
# # #         model, tokenizer, test_entries, candidates, args,
# # #         run_name, train_years_label, test_year, is_base_model=False
# # #     )
    
# # #     del trainer, model
# # #     torch.cuda.empty_cache()
    
# # #     return metrics, pred_df


# # # def experiment_1_sequential(args, base_model, tokenizer, candidates, system_text, data_by_year):
# # #     """Experiment 1: Train on year N, test on year N+1"""
# # #     print("\n" + "="*70)
# # #     print("EXPERIMENT 1: Sequential (Train on N, Test on N+1)")
# # #     print("="*70)
    
# # #     results = []
# # #     all_predictions = []
    
# # #     for i in range(len(YEARS) - 1):
# # #         train_year = YEARS[i]
# # #         test_year = YEARS[i + 1]
# # #         run_name = f"train{train_year}_test{test_year}"
# # #         train_label = str(train_year)
        
# # #         print(f"\n--- Train: {train_year} → Test: {test_year} ---")
        
# # #         metrics, pred_df = train_and_evaluate(
# # #             data_by_year[train_year], data_by_year[test_year],
# # #             base_model, tokenizer, candidates, system_text, args, run_name,
# # #             train_label, test_year
# # #         )
# # #         metrics["train_year"] = train_year
# # #         metrics["test_year"] = test_year
# # #         metrics["train_n"] = len(data_by_year[train_year])
# # #         metrics["test_n"] = len(data_by_year[test_year])
# # #         results.append(metrics)
# # #         all_predictions.append(pred_df)
        
# # #         print(f"  Results: Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")
    
# # #     # Save combined predictions
# # #     combined_df = pd.concat(all_predictions, ignore_index=True)
# # #     pred_path = os.path.join(args.out_dir, f"predictions_sequential_{args.task}_{args.model_name.split('/')[-1]}.csv")
# # #     combined_df.to_csv(pred_path, index=False)
# # #     print(f"\nPredictions saved to: {pred_path}")
    
# # #     return results, combined_df


# # # def experiment_2_cumulative(args, base_model, tokenizer, candidates, system_text, data_by_year):
# # #     """Experiment 2: Train on previous years (most recent first), test on 2024"""
# # #     print("\n" + "="*70)
# # #     print("EXPERIMENT 2: Cumulative Training → Test on 2024 (Recent First)")
# # #     print("="*70)
    
# # #     results = []
# # #     all_predictions = []
# # #     test_rows = data_by_year[2024]
    
# # #     # Get all years before 2024, sorted chronologically
# # #     previous_years = [y for y in YEARS if y < 2024]  # [2008, 2012, 2016, 2020]
    
# # #     for n_years in range(1, len(previous_years) + 1):
# # #         train_years = previous_years[-n_years:]  # Take last n years (most recent)
# # #         train_rows = []
# # #         for y in train_years:
# # #             train_rows.extend(data_by_year[y])
        
# # #         run_name = f"train_{'_'.join(map(str, train_years))}_test2024"
# # #         train_label = "+".join(map(str, train_years))
        
# # #         print(f"\n--- Train: {train_years} ({len(train_rows)} samples) → Test: 2024 ---")
        
# # #         metrics, pred_df = train_and_evaluate(
# # #             train_rows, test_rows,
# # #             base_model, tokenizer, candidates, system_text, args, run_name,
# # #             train_label, 2024
# # #         )
# # #         metrics["train_years"] = train_years
# # #         metrics["test_year"] = 2024
# # #         metrics["train_n"] = len(train_rows)
# # #         metrics["test_n"] = len(test_rows)
# # #         results.append(metrics)
# # #         all_predictions.append(pred_df)
        
# # #         print(f"  Results: Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")
    
# # #     # Save combined predictions
# # #     combined_df = pd.concat(all_predictions, ignore_index=True)
# # #     pred_path = os.path.join(args.out_dir, f"predictions_cumulative_{args.task}_{args.model_name.split('/')[-1]}.csv")
# # #     combined_df.to_csv(pred_path, index=False)
# # #     print(f"\nPredictions saved to: {pred_path}")
    
# # #     return results, combined_df

# # # def evaluate_base_model(args, base_model, tokenizer, candidates, system_text, data_by_year):
# # #     """Evaluate base model (no fine-tuning) on all test years for comparison."""
# # #     print("\n" + "="*70)
# # #     print("BASE MODEL EVALUATION (No Fine-Tuning)")
# # #     print("="*70)
    
# # #     all_predictions = []
    
# # #     for test_year in YEARS:
# # #         run_name = f"base_model_test{test_year}"
# # #         test_entries = prepare_eval_entries(data_by_year[test_year], tokenizer, candidates, system_text)
        
# # #         print(f"\n--- Base Model → Test: {test_year} ({len(test_entries)} samples) ---")
        
# # #         metrics, pred_df = evaluate_and_save_predictions(
# # #             base_model, tokenizer, test_entries, candidates, args,
# # #             run_name, "none", test_year, is_base_model=True
# # #         )
        
# # #         all_predictions.append(pred_df)
# # #         print(f"  Results: Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")
    
# # #     combined_df = pd.concat(all_predictions, ignore_index=True)
# # #     pred_path = os.path.join(args.out_dir, f"predictions_base_model_{args.task}_{args.model_name.split('/')[-1]}.csv")
# # #     combined_df.to_csv(pred_path, index=False)
# # #     print(f"\nBase model predictions saved to: {pred_path}")
    
# # #     return combined_df


# # # def main():
# # #     args = parse_args()
# # #     os.makedirs(args.out_dir, exist_ok=True)
# # #     os.makedirs(args.tmp_dir, exist_ok=True)
# # #     set_seed(args.seed)
    
# # #     task_config = TASK_CONFIGS[args.task]
# # #     candidates = task_config["candidates"]
# # #     system_text = task_config["system_text"]
    
# # #     # Load all years' data
# # #     print("Loading ANES data...")
# # #     data_by_year = {}
# # #     for year in YEARS:
# # #         path = os.path.join(args.data_dir, f"anes_{year}.jsonl")
# # #         if os.path.exists(path):
# # #             rows = load_jsonl(path)
# # #             data_by_year[year] = rows
# # #             print(f"  {year}: {len(rows)} samples")
# # #         else:
# # #             print(f"  WARNING: {path} not found!")
    
# # #     if 2024 not in data_by_year:
# # #         print("ERROR: 2024 test data is required!")
# # #         return
    
# # #     # Load tokenizer and base model
# # #     print("\nLoading tokenizer and base model...")
# # #     tokenizer = load_tokenizer(args.model_name)
# # #     base_model = load_qlora_base_model(args.model_name, tokenizer, args.attn_implementation)
    
# # #     all_results = {
# # #         "task": args.task, 
# # #         "model": args.model_name, 
# # #         "candidates": candidates,
# # #         "experiments": {}
# # #     }
    
# # #     # Evaluate base model if requested
# # #     if args.save_base_predictions:
# # #         base_pred_df = evaluate_base_model(args, base_model, tokenizer, candidates, system_text, data_by_year)
    
# # #     # Run experiments
# # #     if args.experiment in ["sequential", "all"]:
# # #         seq_results, seq_pred_df = experiment_1_sequential(
# # #             args, base_model, tokenizer, candidates, system_text, data_by_year
# # #         )
# # #         all_results["experiments"]["sequential"] = seq_results
    
# # #     if args.experiment in ["cumulative", "all"]:
# # #         cum_results, cum_pred_df = experiment_2_cumulative(
# # #             args, base_model, tokenizer, candidates, system_text, data_by_year
# # #         )
# # #         all_results["experiments"]["cumulative"] = cum_results
    
# # #     # Save metrics summary
# # #     metrics_path = os.path.join(args.out_dir, f"metrics_summary_{args.task}_{args.model_name.split('/')[-1]}.json")
# # #     with open(metrics_path, "w") as f:
# # #         json.dump(all_results, f, indent=2)
# # #     print(f"\nMetrics summary saved to: {metrics_path}")
    
# # #     # Print summary
# # #     print("\n" + "="*70)
# # #     print("FINAL SUMMARY")
# # #     print("="*70)
    
# # #     if "sequential" in all_results["experiments"]:
# # #         print("\nExperiment 1: Sequential Transfer")
# # #         print(f"{'Train':>8} → {'Test':>8} | {'Acc':>8} | {'Kappa':>8} | {'F1':>8} | {'Conf':>8}")
# # #         print("-" * 60)
# # #         for r in all_results["experiments"]["sequential"]:
# # #             print(f"{r['train_year']:>8} → {r['test_year']:>8} | {r['acc']:>8.4f} | {r['kappa']:>8.4f} | {r['macro_f1']:>8.4f} | {r.get('mean_confidence', 0):>8.4f}")
    
# # #     if "cumulative" in all_results["experiments"]:
# # #         print("\nExperiment 2: Cumulative → 2024")
# # #         print(f"{'Train Years':>25} | {'Acc':>8} | {'Kappa':>8} | {'F1':>8} | {'Conf':>8}")
# # #         print("-" * 65)
# # #         for r in all_results["experiments"]["cumulative"]:
# # #             years_str = "+".join(map(str, r["train_years"]))
# # #             print(f"{years_str:>25} | {r['acc']:>8.4f} | {r['kappa']:>8.4f} | {r['macro_f1']:>8.4f} | {r.get('mean_confidence', 0):>8.4f}")


# # # if __name__ == "__main__":
# # #     main()
