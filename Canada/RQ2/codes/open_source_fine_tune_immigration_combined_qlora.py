import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any

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


CANDIDATES = ["Fewer immigrants", "More immigrants", "Same amount"]
VOTE2ID = {candidate: i for i, candidate in enumerate(CANDIDATES)}

SYSTEM_TEXT = (
    "You are a classifier.\n\n"
    "A survey respondent was asked the following question in 2024:\n"
    "In your opinion, should Canada admit more immigrants, fewer immigrants, or about the same number of immigrants as now?"
    
    "Based on the respondent's attributes, predict their answer.\n\n"
    "Output exactly one of the following labels:\n"
    "More immigrants\n"
    "Fewer immigrants\n"
    "Same amount\n\n"
    "Do not explain your answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast, clean QLoRA SFT + deterministic party-label evaluation."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-14B-Instruct"
                # "meta-llama/Llama-3.1-70B-Instruct",
                # "meta-llama/Llama-3.1-8B-Instruct",
                # "Qwen/Qwen2.5-7B-Instruct",
                # "Qwen/Qwen2.5-14B-Instruct"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset_test/test_canada_immigration_2024_new.json",
    )
    parser.add_argument(
        "--ft_files",
        nargs="+",
        default=[
            # "dataset_ft/agg_ft_immigration.jsonl",
            "dataset_ft/individual_ft_immigration.jsonl",
            # "dataset_ft/tweets_ft_immigration.jsonl",
        ],
    )
    parser.add_argument("--out_dir", type=str, default="./results")
    parser.add_argument("--tmp_dir", type=str, default="./tmp_qlora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
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

    # max_memory = {
    #     0: "75GiB",
    #     1: "75GiB",
    #     "cpu": "120GiB",
    # }

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     quantization_config=bnb_config,
    #     device_map="balanced",
    #     max_memory=max_memory,
    #     dtype=torch.bfloat16,
    #     attn_implementation=attn_implementation,
    #     low_cpu_mem_usage=True,
    # )

    max_memory = {
    0: "75GiB",
    "cpu": "120GiB",
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
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


def build_train_dataset(
    ft_rows: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_len: int,
) -> Dataset:
    samples = []

    for item in ft_rows:
        messages = item.get("messages", [])
        gt = extract_gt(messages)
        if gt is None:
            continue

        prompt = build_prompt(tokenizer, first_user_content(messages))
        # answer = " " + gt + tokenizer.eos_token
        answer = gt + tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )["input_ids"]
        answer_ids = tokenizer(
            answer,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )["input_ids"]

        # Keep the supervised answer whenever truncation is needed.
        max_prompt_tokens = max_len - len(answer_ids)
        if max_prompt_tokens <= 0:
            continue
        prompt_ids = prompt_ids[-max_prompt_tokens:]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)

        samples.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        )

    if not samples:
        raise ValueError("No usable fine-tuning samples found after label extraction.")

    return Dataset.from_list(samples)


@dataclass
class DataCollatorForCompletionOnlyLM:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: int | None = 8

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            max_len = ((max_len + multiple - 1) // multiple) * multiple

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad_len)
            batch["labels"].append(feature["labels"] + [-100] * pad_len)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


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
    eval_entries: list[tuple[int, str, str]],
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

        # batch = eval_entries[start : start + args.eval_batch_size]
        # prompts = [entry[2] for entry in batch]
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
    result_base = os.path.join(args.out_dir, f"{safe_name}_{ft_name}_lora")
    # result_base = os.path.join(args.out_dir, f"{args.model_name}_{ft_name}_lora")
    df.to_csv(result_base + "_results.csv", index=False)
    with open(result_base + "_metrics.json", "w", encoding="utf-8") as f:
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

    print("Loading 4-bit base model once...")
    base_model = load_qlora_base_model(
        model_name=args.model_name,
        tokenizer=tokenizer,
        attn_implementation=args.attn_implementation,
    )
    lora_config = build_lora_config(args)

    model = None
    summary = {}

    for ft_file in args.ft_files:
        ft_name = safe_adapter_name(ft_file)
        print("\n====================")
        print(f"FT DATASET: {ft_name}")
        print("====================")

        ft_rows = load_jsonl(ft_file)
        train_dataset = build_train_dataset(ft_rows, tokenizer, args.max_len)
        print(f"Loaded FT rows: {len(ft_rows)}; usable rows: {len(train_dataset)}")

        if model is None:
            model = get_peft_model(base_model, lora_config, adapter_name=ft_name)
        else:
            model.add_adapter(ft_name, lora_config)
            model.set_adapter(ft_name)

        model.train()
        model.config.use_cache = False
        model.print_trainable_parameters()

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=os.path.join(args.tmp_dir, ft_name),
                per_device_train_batch_size=args.train_batch_size,
                gradient_accumulation_steps=args.grad_accum,
                num_train_epochs=args.epochs,
                learning_rate=args.learning_rate,
                warmup_ratio=0.03,
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
            ),
            train_dataset=train_dataset,
            data_collator=DataCollatorForCompletionOnlyLM(tokenizer),
        )

        print("Training...")
        trainer.train()

        print("Evaluating...")
        metrics = evaluate(model, tokenizer, eval_entries, ft_name, args)
        summary[ft_name] = metrics

        del trainer
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
