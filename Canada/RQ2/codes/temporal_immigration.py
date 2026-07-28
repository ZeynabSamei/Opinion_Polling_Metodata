import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import cohen_kappa_score, f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


# ---------- Configuration ----------
# Annual data, every year from 2019 through 2025.
YEARS = list(range(2019, 2026))

TASK_CONFIGS = {
    "immigration_level": {
        "candidates": ["More immigrants", "Fewer immigrants", "Same amount"],
        "system_text": (
            "You are a survey response prediction model.\n\n"
            "Your task is to predict the survey response associated with this respondent "
            "profile based on the provided characteristics.\n"
            "Survey question: In your opinion, should Canada admit more immigrants, fewer "
            "immigrants, or about the same number of immigrants as now?\n\n"
            "Rules:\n"
            "- You must choose ONLY ONE label.\n"
            "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
            "More immigrants\n"
            "Fewer immigrants\n"
            "Same amount\n\n"
            "Important:\n"
            "- Base your decision on typical attitudes and demographics.\n"
            "- Do NOT explain your reasoning.\n"
            "- Do NOT repeat the input.\n"
            "- Output ONLY the label."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal Generalization Experiments with Canada immigration-attitude survey data"
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--task", type=str, default="immigration_level", choices=["immigration_level"])
    parser.add_argument("--data_dir", type=str, default="dataset_test/")
    parser.add_argument("--experiment", type=str, default="all", choices=["sequential", "cumulative", "all"])
    parser.add_argument("--out_dir", type=str, default="./results_temporal_canada")
    parser.add_argument("--tmp_dir", type=str, default="./tmp_qlora_temporal_canada")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=32)
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
    parser.add_argument("--dataloader_workers", type=int, default=4)

    parser.add_argument("--use_reentrant_checkpointing", action=argparse.BooleanOptionalAction, default=False,
        help="Use reentrant gradient checkpointing. Default False (non-reentrant) because "
             "reentrant checkpointing can crash or silently break gradients when the model "
             "is sharded across multiple GPUs via device_map='auto'.")

    parser.add_argument("--smoke_test", action="store_true",
        help="Truncate train/test rows for every year to --smoke_test_n samples and run "
             "a minimal pass to validate the pipeline end-to-end quickly.")
    parser.add_argument("--smoke_test_n", type=int, default=20,
        help="Number of rows per year to keep when --smoke_test is set.")

    # NEW: last cumulative/sequential "anchor" test year. For ANES this was hardcoded
    # to 2024 (the most recent wave); here it defaults to the most recent year in YEARS.
    parser.add_argument("--final_test_year", type=int, default=None,
        help="Year to use as the fixed cumulative-experiment test set. Defaults to max(YEARS).")

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
    if value is None:
        return None
    text = str(value).strip().lower()
    for c in candidates:
        if c.lower() in text:
            return c
    return None


def extract_label(messages: list[dict[str, str]], candidates: List[str]) -> str | None:
    for msg in messages:
        if msg.get("role") == "assistant":
            return normalize_label(msg.get("content"), candidates)
    return None


def first_user_content(messages: list[dict[str, str]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
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
        if gt is None:
            continue
        prompts.append(build_prompt(tokenizer, system_text, first_user_content(messages)))
        answers.append(gt + tokenizer.eos_token)

    samples = []
    for prompt, answer in zip(prompts, answers):
        p_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        a_ids = tokenizer(answer, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        max_p = max_len - len(a_ids)
        if max_p <= 0:
            continue
        p_ids = p_ids[-max_p:]
        samples.append({"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids,
                         "attention_mask": [1] * len(p_ids + a_ids)})

    if not samples:
        raise ValueError("No usable SFT samples.")
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
            batch["input_ids"].append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def prepare_eval_entries(rows, tokenizer, candidates, system_text) -> list[Tuple[int, str, str, str]]:
    entries = []
    for idx, entry in enumerate(rows):
        messages = entry.get("messages", [])
        gt = extract_label(messages, candidates)
        if gt is None:
            continue
        entries.append((idx, gt, first_user_content(messages),
                         build_prompt(tokenizer, system_text, first_user_content(messages))))
    return entries


def get_candidate_token_ids(tokenizer, candidates) -> list[list[int]]:
    return [tokenizer(c, add_special_tokens=False)["input_ids"] for c in candidates]


@torch.inference_mode()
def score_candidates_batched(model, tokenizer, prompts, candidate_token_ids, candidates, max_len, length_normalize):
    if not prompts:
        return []

    device = next(model.parameters()).device
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_len, add_special_tokens=False)

    p_ids = encoded["input_ids"]
    p_mask = encoded["attention_mask"]
    p_lens = p_mask.sum(dim=1).tolist()

    all_seqs, all_p_lens, all_c_lens = [], [], []
    for row_idx, plen in enumerate(p_lens):
        ptoks = p_ids[row_idx, :plen].tolist()
        for cids in candidate_token_ids:
            max_p = max_len - len(cids)
            if max_p <= 0:
                raise ValueError("Candidate too long")
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
        tok_scores = [log_probs[i, plen - 1 + off, input_ids[i, plen + off]] for off in range(clen)]
        sc = torch.stack(tok_scores).sum()
        if length_normalize:
            sc = sc / clen
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
        probs = score_candidates_batched(model, tokenizer, prompts, cids, candidates, args.max_len,
                                          args.length_normalize_eval)

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
    if df.empty:
        return {"acc": 0.0, "kappa": 0.0, "macro_f1": 0.0, "n": 0}, df

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

    if hasattr(base_model, "peft_config") and hasattr(base_model, "unload"):
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
        gradient_checkpointing_kwargs={"use_reentrant": args.use_reentrant_checkpointing},
        max_grad_norm=0.3,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=args.dataloader_workers,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                       data_collator=DataCollatorForCompletionOnlyLM(tokenizer))
    trainer.train()

    metrics, pred_df = evaluate_and_save_predictions(
        model, tokenizer, test_entries, candidates, args, run_name, train_years_label, test_year)

    del trainer, model
    torch.cuda.empty_cache()
    return metrics, pred_df


def experiment_1_sequential(args, base_model, tokenizer, candidates, system_text, data_by_year, years):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Sequential (Train on N, Test on next available wave)")
    print("=" * 70)
    results, all_preds = [], []

    for i in range(len(years) - 1):
        ty, tty = years[i], years[i + 1]
        print(f"\n--- Train: {ty} -> Test: {tty} ---")
        metrics, pred_df = train_and_evaluate(
            data_by_year[ty], data_by_year[tty], base_model, tokenizer, candidates, system_text,
            args, f"train{ty}_test{tty}", str(ty), tty)
        metrics.update({"train_year": ty, "test_year": tty,
                         "train_n": len(data_by_year[ty]), "test_n": len(data_by_year[tty])})
        results.append(metrics)
        all_preds.append(pred_df)
        print(f"  Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")

    combined = pd.concat(all_preds, ignore_index=True)
    combined.to_csv(os.path.join(
        args.out_dir, f"predictions_sequential_{args.task}_{args.model_name.split('/')[-1]}.csv"), index=False)
    return results, combined


def experiment_2_cumulative(args, base_model, tokenizer, candidates, system_text, data_by_year, years, final_year):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Cumulative -> {final_year} (Recent First)")
    print("=" * 70)
    results, all_preds = [], []
    test_rows = data_by_year[final_year]
    prev = [y for y in years if y < final_year]

    for n in range(1, len(prev) + 1):
        train_years = prev[-n:]
        train_rows = [r for y in train_years for r in data_by_year[y]]
        label = "+".join(map(str, train_years))
        print(f"\n--- Train: {train_years} ({len(train_rows)}) -> Test: {final_year} ---")
        metrics, pred_df = train_and_evaluate(
            train_rows, test_rows, base_model, tokenizer, candidates, system_text,
            args, f"train_{'_'.join(map(str, train_years))}_test{final_year}", label, final_year)
        metrics.update({"train_years": train_years, "test_year": final_year,
                         "train_n": len(train_rows), "test_n": len(test_rows)})
        results.append(metrics)
        all_preds.append(pred_df)
        print(f"  Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")

    combined = pd.concat(all_preds, ignore_index=True)
    combined.to_csv(os.path.join(
        args.out_dir, f"predictions_cumulative_{args.task}_{args.model_name.split('/')[-1]}.csv"), index=False)
    return results, combined


def evaluate_base_model(args, base_model, tokenizer, candidates, system_text, data_by_year, years):
    print("\n" + "=" * 70)
    print("BASE MODEL EVALUATION (No Fine-Tuning)")
    print("=" * 70)
    all_preds = []

    for year in years:
        entries = prepare_eval_entries(data_by_year[year], tokenizer, candidates, system_text)
        print(f"\n--- Base -> Test: {year} ({len(entries)}) ---")
        metrics, pred_df = evaluate_and_save_predictions(
            base_model, tokenizer, entries, candidates, args, f"base_{year}", "none", year, True)
        all_preds.append(pred_df)
        print(f"  Acc={metrics['acc']:.4f}, Kappa={metrics['kappa']:.4f}, F1={metrics['macro_f1']:.4f}")

    combined = pd.concat(all_preds, ignore_index=True)
    combined.to_csv(os.path.join(
        args.out_dir, f"predictions_base_{args.task}_{args.model_name.split('/')[-1]}.csv"), index=False)
    return combined


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    set_seed(args.seed)

    cfg = TASK_CONFIGS[args.task]
    candidates, system_text = cfg["candidates"], cfg["system_text"]

    print("Loading Canada immigration-attitude survey data...")
    data_by_year = {}
    for year in YEARS:
        # Expected file naming: canada_immigration_{year}.jsonl
        path = os.path.join(args.data_dir, f"canada_immigration_{year}.jsonl")
        if os.path.exists(path):
            data_by_year[year] = load_jsonl(path)
            print(f"  {year}: {len(data_by_year[year])}")
        else:
            print(f"  WARNING: {path} not found!")

    years = sorted(data_by_year.keys())
    if not years:
        print("ERROR: no year files found in --data_dir!")
        return

    final_year = args.final_test_year if args.final_test_year is not None else max(years)
    if final_year not in data_by_year:
        print(f"ERROR: final_test_year={final_year} not found in loaded data!")
        return

    if args.smoke_test:
        print(f"\n*** SMOKE TEST MODE: truncating each year to {args.smoke_test_n} rows ***\n")
        for year in data_by_year:
            data_by_year[year] = data_by_year[year][: args.smoke_test_n]
        args.epochs = min(args.epochs, 1.0)

    print("\nLoading tokenizer & model...")
    tokenizer = load_tokenizer(args.model_name)
    base_model = load_qlora_base_model(args.model_name, tokenizer, args.attn_implementation)

    all_results = {"task": args.task, "model": args.model_name, "candidates": candidates,
                   "years": years, "final_test_year": final_year, "experiments": {}}

    if args.save_base_predictions:
        evaluate_base_model(args, base_model, tokenizer, candidates, system_text, data_by_year, years)

    if args.experiment in ["sequential", "all"]:
        seq_results, _ = experiment_1_sequential(args, base_model, tokenizer, candidates, system_text,
                                                  data_by_year, years)
        all_results["experiments"]["sequential"] = seq_results

    if args.experiment in ["cumulative", "all"]:
        cum_results, _ = experiment_2_cumulative(args, base_model, tokenizer, candidates, system_text,
                                                  data_by_year, years, final_year)
        all_results["experiments"]["cumulative"] = cum_results

    with open(os.path.join(args.out_dir, f"metrics_{args.task}_{args.model_name.split('/')[-1]}.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70 + "\nCOMPLETE\n" + "=" * 70)


if __name__ == "__main__":
    main()
