import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig


CANDIDATES = ["Liberal", "Conservative", "Other"]

VOTE2ID = {candidate: i for i, candidate in enumerate(CANDIDATES)}

SYSTEM_TEXT = (
    "You are an expert political analyst specializing in Canadian elections and voting behavior.\n\n"
    "Task:\n"
    "Given a person's demographic and political attributes, predict their MOST LIKELY party choice "
    "in the 2021 Canadian federal election.\n\n"
    "Rules:\n"
    "- You must choose ONLY ONE label.\n"
    "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
    "Liberal\n"
    "Conservative\n"
    "Other\n\n"
    "Definition:\n"
    "Other includes New Democratic Party (NDP), Bloc Quebecois, Green Party, "
    "and People's Party of Canada.\n\n"
    "Important:\n"
    "- Base your decision on typical voting patterns, demographics, and political alignment.\n"
    "- Do NOT explain your reasoning.\n"
    "- Do NOT repeat the input.\n"
    "- Output ONLY the label."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA DPO training + deterministic party-label evaluation."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset_test/test_canada_election_party_2021_3class_new.json",
    )
    parser.add_argument(
        "--ft_files",
        nargs="+",
        default=[
            "dataset_ft/agg_ft_party_2021_3class_dpo.jsonl",
        ],
    )
    parser.add_argument("--out_dir", type=str, default="./results_dpo")
    parser.add_argument("--tmp_dir", type=str, default="./tmp_qlora_dpo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--max_prompt_len", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument(
        "--length_normalize_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average candidate log-probability by token count during evaluation.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter controlling KL divergence penalty.",
    )
    parser.add_argument(
        "--ref_model_name",
        type=str,
        default=None,
        help="Reference model for DPO. If None, uses model_name.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def normalize_vote(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    for candidate in CANDIDATES:
        if candidate.lower() in text:
            return candidate
    return None


def extract_gt(messages: list[dict[str, str]]) -> str | None:
    for message in messages:
        if message.get("role") == "assistant":
            return normalize_vote(message.get("content"))
    return None


def first_user_content(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "")
    return messages[0].get("content", "") if messages else ""


def build_prompt(tokenizer: AutoTokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def load_qlora_base_model(
    model_name: str,
    tokenizer: AutoTokenizer,
    attn_implementation: str,
) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    max_memory = {
        0: "75GiB",
        1: "75GiB",
        "cpu": "120GiB",
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        max_memory=max_memory,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        device_map="auto",
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    return model


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_dpo_dataset(
    ft_rows: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_len: int,
) -> Dataset:
    """Convert DPO-style data (prompt, chosen, rejected) to the format expected by DPOTrainer."""
    samples = []

    for item in ft_rows:
        # Handle DPO format: {"prompt": "...", "chosen": "...", "rejected": "..."}
        if "prompt" in item and "chosen" in item and "rejected" in item:
            prompt = build_prompt(tokenizer, item["prompt"])
            chosen = item["chosen"] + tokenizer.eos_token
            rejected = item["rejected"] + tokenizer.eos_token
            
            samples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
        # Also support the original messages format for compatibility
        elif "messages" in item:
            messages = item.get("messages", [])
            gt = extract_gt(messages)
            if gt is None:
                continue
            
            prompt = build_prompt(tokenizer, first_user_content(messages))
            chosen = gt + tokenizer.eos_token
            
            # Create a simple rejected example (other party)
            rejected_candidates = [c for c in CANDIDATES if c != gt]
            rejected = rejected_candidates[0] + tokenizer.eos_token if rejected_candidates else chosen
            
            samples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

    if not samples:
        raise ValueError("No usable DPO samples found.")

    return Dataset.from_list(samples)


def prepare_eval_entries(
    eval_rows: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
) -> list[tuple[int, str, str, str]]:
    entries = []
    for idx, entry in enumerate(eval_rows):
        messages = entry.get("messages", [])
        gt = extract_gt(messages)
        if gt is None:
            continue

        user_text = first_user_content(messages)
        prompt = build_prompt(tokenizer, user_text)

        entries.append((idx, gt, user_text, prompt))

    return entries


def get_candidate_token_ids(tokenizer: AutoTokenizer) -> list[list[int]]:
    return [
        tokenizer(candidate, add_special_tokens=False)["input_ids"]
        for candidate in CANDIDATES
    ]


@torch.inference_mode()
def score_candidates_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    candidate_token_ids: list[list[int]],
    max_len: int,
    length_normalize: bool,
) -> list[dict[str, float]]:
    if not prompts:
        return []

    device = next(model.parameters()).device
    encoded_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
        add_special_tokens=False,
    )

    prompt_input_ids = encoded_prompts["input_ids"]
    prompt_attention_mask = encoded_prompts["attention_mask"]
    prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()

    sequences = []
    prompt_lens = []
    candidate_lens = []

    for row_idx, prompt_len in enumerate(prompt_lengths):
        prompt_tokens = prompt_input_ids[row_idx, :prompt_len].tolist()
        for candidate_ids in candidate_token_ids:
            max_prompt_tokens = max_len - len(candidate_ids)
            if max_prompt_tokens <= 0:
                raise ValueError("Candidate label is longer than max_len.")

            trimmed_prompt = prompt_tokens[-max_prompt_tokens:]
            sequences.append(trimmed_prompt + candidate_ids)
            prompt_lens.append(len(trimmed_prompt))
            candidate_lens.append(len(candidate_ids))

    batch_size = len(sequences)
    padded_len = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (batch_size, padded_len),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, padded_len),
        dtype=torch.long,
        device=device,
    )

    for i, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[i, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
        attention_mask[i, :length] = 1

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(outputs.logits.float(), dim=-1)

    scores = []
    for i in range(batch_size):
        prompt_len = prompt_lens[i]
        candidate_len = candidate_lens[i]
        token_scores = []

        for offset in range(candidate_len):
            logit_pos = prompt_len - 1 + offset
            token_pos = prompt_len + offset
            token_id = input_ids[i, token_pos]
            token_scores.append(log_probs[i, logit_pos, token_id])

        candidate_score = torch.stack(token_scores).sum()
        if length_normalize:
            candidate_score = candidate_score / candidate_len
        scores.append(candidate_score)

    score_tensor = torch.stack(scores).view(len(prompts), len(CANDIDATES))
    prob_tensor = torch.softmax(score_tensor, dim=1).detach().cpu().numpy()

    return [
        {candidate: float(row[i]) for i, candidate in enumerate(CANDIDATES)}
        for row in prob_tensor
    ]


def safe_model_name(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_name)


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_entries: list[tuple[int, str, str, str]],
    ft_name: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    model.config.use_cache = True

    candidate_token_ids = get_candidate_token_ids(tokenizer)
    rows = []

    for start in tqdm(
        range(0, len(eval_entries), args.eval_batch_size),
        desc=f"Eval {ft_name}",
    ):
        batch = eval_entries[start : start + args.eval_batch_size]
        prompts = [entry[3] for entry in batch]
        
        probabilities = score_candidates_batched(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            candidate_token_ids=candidate_token_ids,
            max_len=args.max_len,
            length_normalize=args.length_normalize_eval,
        )

        for (idx, gt, user_text, _prompt), probs in zip(batch, probabilities):
            pred = max(probs, key=probs.get)
            rows.append(
                {
                    "user_text": user_text,
                    "idx": idx,
                    "gt": gt,
                    "pred": pred,
                    "acc": int(gt == pred),
                    **{f"prob_{candidate}": probs[candidate] for candidate in CANDIDATES},
                }
            )

    df = pd.DataFrame(rows)
    print(df.head(5))

    if df.empty:
        raise ValueError("No valid evaluation samples found.")

    y_true = df["gt"].map(VOTE2ID).to_numpy()
    y_pred = df["pred"].map(VOTE2ID).to_numpy()
    metrics = {
        "acc": float(df["acc"].mean()),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "n_eval": int(len(df)),
    }

    safe_name = safe_model_name(args.model_name)
    result_base = os.path.join(args.out_dir, f"{safe_name}_{ft_name}_dpo_lora")
    df.to_csv(result_base + "_results_dpo.csv", index=False)
    with open(result_base + "_metrics_dpo.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== RESULTS: {ft_name} ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    model.config.use_cache = False
    return metrics


def safe_adapter_name(ft_file: str) -> str:
    name = os.path.basename(ft_file).replace(".jsonl", "")
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    set_seed(args.seed)

    print("Loading evaluation data...")
    eval_rows = load_json(args.data_path)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)
    eval_entries = prepare_eval_entries(eval_rows, tokenizer)
    print(f"Loaded eval rows: {len(eval_rows)}; usable rows: {len(eval_entries)}")

    print("Loading 4-bit base model...")
    base_model = load_qlora_base_model(
        model_name=args.model_name,
        tokenizer=tokenizer,
        attn_implementation=args.attn_implementation,
    )

    # Load reference model for DPO if specified
    ref_model = None
    ref_model_name = args.ref_model_name or args.model_name
    if ref_model_name != args.model_name:
        print(f"Loading reference model: {ref_model_name}")
        ref_model = load_qlora_base_model(
            model_name=ref_model_name,
            tokenizer=tokenizer,
            attn_implementation=args.attn_implementation,
        )

    lora_config = build_lora_config(args)
    model = None
    summary = {}

    for ft_file in args.ft_files:
        ft_name = safe_adapter_name(ft_file)
        print("\n====================")
        print(f"DPO DATASET: {ft_name}")
        print("====================")

        ft_rows = load_jsonl(ft_file)
        train_dataset = build_dpo_dataset(ft_rows, tokenizer, args.max_len)
        print(f"Loaded DPO rows: {len(ft_rows)}; usable rows: {len(train_dataset)}")

        if model is None:
            model = get_peft_model(base_model, lora_config, adapter_name="default")
        else:
            # For multiple datasets, we still need to handle adapters
            # But DPOTrainer requires "default" adapter
            model.add_adapter(ft_name, lora_config)
            model.set_adapter(ft_name)

        # Set up reference model adapter if using separate reference model
        ref_adapter_model = None
        if ref_model is not None:
            ref_adapter_model = get_peft_model(ref_model, lora_config, adapter_name="default")
            # Freeze reference model
            for param in ref_adapter_model.parameters():
                param.requires_grad = False

        model.train()
        model.config.use_cache = False
        model.print_trainable_parameters()

        # DPO Training Arguments
        dpo_config = DPOConfig(
            output_dir=os.path.join(args.tmp_dir, ft_name),
            per_device_train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=10,  # Changed from warmup_ratio
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
            bf16=True,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            beta=args.beta,
            max_length=args.max_len,
        )

        # Check TRL version to use correct API
        import trl
        trl_version = tuple(map(int, trl.__version__.split('.')[:2]))
        
        if trl_version >= (0, 11):
            # TRL >= 0.11: no tokenizer/processing_class parameter
            trainer = DPOTrainer(
                model=model,
                ref_model=ref_adapter_model,
                args=dpo_config,
                train_dataset=train_dataset,
            )
        elif trl_version >= (0, 9):
            # TRL 0.9-0.10: uses processing_class
            trainer = DPOTrainer(
                model=model,
                ref_model=ref_adapter_model,
                args=dpo_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )
        else:
            # TRL < 0.9: uses tokenizer
            trainer = DPOTrainer(
                model=model,
                ref_model=ref_adapter_model,
                args=dpo_config,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
            )

        print("Training with DPO...")
        trainer.train()

        print("Evaluating...")
        # For evaluation, we need to merge and use the model properly
        model.eval()
        metrics = evaluate(model, tokenizer, eval_entries, ft_name, args)
        summary[ft_name] = metrics

        del trainer
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
