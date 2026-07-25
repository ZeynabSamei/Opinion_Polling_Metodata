import argparse
import inspect
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import trl
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import DPOTrainer, DPOConfig


CANDIDATES = ["Fewer immigrants", "More immigrants", "Same amount"]
VOTE2ID = {candidate: i for i, candidate in enumerate(CANDIDATES)}

SYSTEM_TEXT = (
    "You are a classifier.\n\n"
    "A survey respondent was asked the following question in 2024:\n"
    "In your opinion, should Canada admit more immigrants, fewer immigrants, or about the same number of immigrants as now?\n\n"
    "Based on the respondent's attributes, predict their answer.\n\n"
    "Output exactly one of the following labels:\n"
    "More immigrants\n"
    "Fewer immigrants\n"
    "Same amount\n\n"
    "Do not explain your answer."
)


# --------------------------------------------------------------------------- #
# Args / setup
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA DPO training + deterministic immigration-preference evaluation."
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset_test/test_canada_immigration_2024_new.json",
    )
    parser.add_argument(
        "--ft_files",
        nargs="+",
        default=["dataset_ft/agg_ft_immigration_2024_dpo.jsonl"],
    )
    parser.add_argument("--out_dir", type=str, default="./results_dpo_immigration")
    parser.add_argument("--tmp_dir", type=str, default="./tmp_qlora_dpo_immigration")

    # >>> MODIFIED (Fix #1 — seed variance): was a single --seed int. Now
    # accepts one or more seeds; every (ft_file, seed) combination is trained
    # and evaluated separately, then aggregated (mean/std), so you can see
    # how much of DPO's result is training noise vs. a real effect.
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])

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
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
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

    # >>> MODIFIED (Fix #2 — pairwise/multiclass mismatch): rpo_alpha adds an
    # SFT/NLL loss term on the chosen response alongside the DPO loss, which
    # counteracts "likelihood displacement" (DPO widening the margin by
    # suppressing both candidates instead of raising P(chosen)). Only applied
    # if the installed TRL's DPOConfig supports it (see build_dpo_config) —
    # otherwise dropped with a printed warning.
    parser.add_argument(
        "--rpo_alpha",
        type=float,
        default=1.0,
        help="Weight of the auxiliary NLL loss on the chosen response (RPO-style). "
             "Set to 0 to disable and use vanilla DPO.",
    )

    parser.add_argument(
        "--ref_model_name",
        type=str,
        default=None,
        help="Reference model for DPO. If None, uses model_name with adapters disabled "
             "(or the SFT adapter, if --sft_adapter_path/--use_sft_as_ref are set).",
    )

    # >>> MODIFIED (Fix #3 — no SFT warm start): path to an already-trained
    # SFT LoRA adapter. If given, the DPO policy is initialized from these
    # weights instead of a fresh zero-init LoRA adapter, matching standard
    # DPO practice (refine preferences of an already-competent policy).
    parser.add_argument(
        "--sft_adapter_path",
        type=str,
        default=None,
        help="Path to a trained SFT LoRA adapter to warm-start the DPO policy from. "
             "If None, DPO starts from a fresh zero-init adapter (old behavior).",
    )
    parser.add_argument(
        "--use_sft_as_ref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If --sft_adapter_path is set, also use those (frozen) SFT weights as the "
             "DPO reference model, instead of the raw base model. Recommended: True.",
    )

    parser.add_argument(
        "--max_memory_gpu_gib",
        type=int,
        default=None,
        help="Optional per-GPU memory cap in GiB. If unset, use the old fixed 75GiB/GPU default.",
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


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
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
    max_memory_gpu_gib: Optional[int] = None,  # >>> MODIFIED: made configurable (was hardcoded 75GiB)
) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # >>> MODIFIED: previously this was hardcoded to {0: "75GiB", 1: "75GiB", "cpu": "120GiB"}
    # regardless of how many GPUs were actually visible. Now derives it from
    # torch.cuda.device_count() (or an explicit --max_memory_gpu_gib override),
    # so the script doesn't silently misconfigure on single-GPU or >2-GPU boxes.
    if max_memory_gpu_gib is not None and torch.cuda.is_available():
        max_memory = {i: f"{max_memory_gpu_gib}GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "120GiB"
    elif torch.cuda.is_available():
        max_memory = {i: "75GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "120GiB"
    else:
        max_memory = None

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

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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
    rng: random.Random,  # >>> MODIFIED: was `max_len` (unused in the body) — now takes an rng for random rejected sampling
) -> Tuple[Dataset, Counter]:
    """Convert DPO-style data (prompt, chosen, rejected) to the format expected by DPOTrainer.

    >>> MODIFIED (Fix #4 — biased rejected sampling): also returns a Counter
    of how often each candidate was used as 'rejected', so you can verify the
    fix isn't secretly still biased on your real data.
    """
    samples = []
    rejected_counter: Counter = Counter()

    for item in ft_rows:
        # Handle DPO format: {"prompt": "...", "chosen": "...", "rejected": "..."}
        if "prompt" in item and "chosen" in item and "rejected" in item:
            prompt = build_prompt(tokenizer, item["prompt"])
            chosen = item["chosen"] + tokenizer.eos_token
            rejected = item["rejected"] + tokenizer.eos_token

            samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            rejected_counter[item["rejected"]] += 1

        # Also support the original messages format for compatibility
        elif "messages" in item:
            messages = item.get("messages", [])
            gt = extract_gt(messages)
            if gt is None:
                continue

            prompt = build_prompt(tokenizer, first_user_content(messages))
            chosen = gt + tokenizer.eos_token

            # >>> MODIFIED (Fix #4): was `rejected_candidates[0]`, which
            # deterministically always picked the same label based on list
            # order — e.g. gt="Fewer immigrants" always got rejected="More
            # immigrants", and "Same amount" was almost never used as a
            # negative. That systematically biased what DPO learned to
            # suppress vs. promote, independent of the real label
            # distribution. Now sampled uniformly at random from the
            # non-gt candidates.
            rejected_candidates = [c for c in CANDIDATES if c != gt]
            rejected_label = rng.choice(rejected_candidates) if rejected_candidates else gt
            rejected = rejected_label + tokenizer.eos_token

            samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            rejected_counter[rejected_label] += 1

    if not samples:
        raise ValueError("No usable DPO samples found.")

    return Dataset.from_list(samples), rejected_counter


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


# >>> MODIFIED: added (was missing in this file, present in the party-choice
# version). Byte-level BPE tokenizers can merge tokens across the
# prompt/candidate boundary, in which case splicing isolated candidate token
# ids after the prompt is misaligned. Cheap check, run once at startup.
def sanity_check_candidate_tokenization(
    tokenizer: AutoTokenizer,
    sample_prompt: str,
    candidate_token_ids: list[list[int]],
) -> None:
    for candidate, cand_ids in zip(CANDIDATES, candidate_token_ids):
        full_ids = tokenizer(sample_prompt + candidate, add_special_tokens=False)["input_ids"]
        tail = full_ids[-len(cand_ids):]
        if tail != cand_ids:
            print(
                f"[WARNING] Tokenization boundary mismatch for candidate {candidate!r}: "
                f"isolated={cand_ids} vs tail-of-full={tail}. "
                "Candidate scoring may be misaligned for this model/tokenizer."
            )


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
        (batch_size, padded_len), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((batch_size, padded_len), dtype=torch.long, device=device)

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


def safe_adapter_name(ft_file: str) -> str:
    name = os.path.basename(ft_file).replace(".jsonl", "")
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_entries: list[tuple[int, str, str, str]],
    ft_name: str,
    seed: int,  # >>> MODIFIED: added, so per-seed runs don't overwrite each other's output files
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    model.config.use_cache = True

    candidate_token_ids = get_candidate_token_ids(tokenizer)
    rows = []

    for start in tqdm(
        range(0, len(eval_entries), args.eval_batch_size),
        desc=f"Eval {ft_name} (seed={seed})",
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

    if df.empty:
        raise ValueError("No valid evaluation samples found.")

    y_true = df["gt"].map(VOTE2ID).to_numpy()
    y_pred = df["pred"].map(VOTE2ID).to_numpy()
    metrics = {
        "acc": float(df["acc"].mean()),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "n_eval": int(len(df)),  # >>> kept, and now also surfaced in aggregate summary below
    }

    safe_name = safe_model_name(args.model_name)
    # >>> MODIFIED: seed now included in output filename so multi-seed runs don't collide
    result_base = os.path.join(args.out_dir, f"{safe_name}_{ft_name}_seed{seed}_dpo_lora")
    df.to_csv(result_base + "_results_dpo_immigration.csv", index=False)
    with open(result_base + "_metrics_dpo_immigration.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== RESULTS: {ft_name} (seed={seed}) ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    model.config.use_cache = False
    return metrics


# >>> MODIFIED: new helper, replaces the inline DPOConfig(...) block in main().
# Adds rpo_alpha support (auto-dropped with a warning if the installed TRL
# version's DPOConfig doesn't accept it) and takes `seed` explicitly for the
# multi-seed loop.
def build_dpo_config(args: argparse.Namespace, ft_name: str, seed: int) -> DPOConfig:
    desired = dict(
        output_dir=os.path.join(args.tmp_dir, f"{ft_name}_seed{seed}"),
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        beta=args.beta,
        max_length=args.max_len,
        max_prompt_length=args.max_prompt_len,  # >>> MODIFIED: was missing from this file's DPOConfig call entirely
    )

    if args.rpo_alpha is not None and args.rpo_alpha > 0:
        desired["rpo_alpha"] = args.rpo_alpha

    accepted_params = set(inspect.signature(DPOConfig.__init__).parameters.keys())
    accepted = {k: v for k, v in desired.items() if k in accepted_params}
    dropped = {k: v for k, v in desired.items() if k not in accepted_params}

    if dropped:
        print(f"[INFO] Installed TRL's DPOConfig does not accept these kwargs; dropping: {sorted(dropped.keys())}")
        if "rpo_alpha" in dropped:
            print(
                "[WARNING] rpo_alpha not supported by this TRL version's DPOConfig — "
                "running vanilla DPO with no auxiliary NLL loss."
            )

    return DPOConfig(**accepted)


# >>> MODIFIED: extracted from the inline trl_version check in main(), unchanged in logic.
def make_dpo_trainer(model, ref_model, dpo_config: DPOConfig, train_dataset: Dataset, tokenizer: AutoTokenizer) -> DPOTrainer:
    trl_version = tuple(map(int, trl.__version__.split(".")[:2]))

    if trl_version >= (0, 11):
        return DPOTrainer(model=model, ref_model=ref_model, args=dpo_config, train_dataset=train_dataset)
    elif trl_version >= (0, 9):
        return DPOTrainer(
            model=model, ref_model=ref_model, args=dpo_config,
            train_dataset=train_dataset, processing_class=tokenizer,
        )
    else:
        return DPOTrainer(
            model=model, ref_model=ref_model, args=dpo_config,
            train_dataset=train_dataset, tokenizer=tokenizer,
        )


# >>> MODIFIED (Fix #3 — SFT warm start): new helper. If --sft_adapter_path
# is given, loads that adapter as the DPO policy's starting point instead of
# a fresh zero-init LoRA adapter.
def build_policy_model(base_model: AutoModelForCausalLM, lora_config: LoraConfig, args: argparse.Namespace) -> PeftModel:
    if args.sft_adapter_path:
        print(f"[INFO] Warm-starting DPO policy from SFT adapter: {args.sft_adapter_path}")
        return PeftModel.from_pretrained(
            base_model, args.sft_adapter_path, adapter_name="default", is_trainable=True
        )
    print("[INFO] No --sft_adapter_path given — starting DPO from a fresh zero-init adapter "
          "(old behavior; more prone to instability, see build_reference_model).")
    return get_peft_model(base_model, lora_config, adapter_name="default")


# >>> MODIFIED (Fix #3 — SFT warm start): new helper, replaces the old
# "ref_model_name or model_name" logic. If use_sft_as_ref is True and an SFT
# adapter path is given, the reference model is the frozen SFT-tuned policy
# (standard DPO setup), instead of always defaulting to the raw base model.
def build_reference_model(tokenizer: AutoTokenizer, lora_config: LoraConfig, args: argparse.Namespace) -> Optional[PeftModel]:
    if args.sft_adapter_path and args.use_sft_as_ref:
        print(f"[INFO] Using SFT adapter as frozen DPO reference model: {args.sft_adapter_path}")
        ref_base = load_qlora_base_model(
            args.model_name, tokenizer, args.attn_implementation, args.max_memory_gpu_gib
        )
        ref_model = PeftModel.from_pretrained(
            ref_base, args.sft_adapter_path, adapter_name="default", is_trainable=False
        )
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()
        return ref_model

    if args.ref_model_name and args.ref_model_name != args.model_name:
        print(f"[INFO] Loading separate reference model: {args.ref_model_name}")
        ref_base = load_qlora_base_model(
            args.ref_model_name, tokenizer, args.attn_implementation, args.max_memory_gpu_gib
        )
        ref_model = get_peft_model(ref_base, lora_config, adapter_name="default")
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()
        return ref_model

    # Old default: ref_model=None -> DPOTrainer uses the policy with adapters
    # disabled (== raw base weights) as reference.
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    print("Loading evaluation data...")
    eval_rows = load_json(args.data_path)

    print("Loading tokenizer...")
    tokenizer = load_tokenizer(args.model_name)
    eval_entries = prepare_eval_entries(eval_rows, tokenizer)
    print(f"Loaded eval rows: {len(eval_rows)}; usable rows: {len(eval_entries)}")

    print("Loading 4-bit base model...")
    base_model = load_qlora_base_model(
        args.model_name, tokenizer, args.attn_implementation, args.max_memory_gpu_gib
    )

    # >>> MODIFIED: added — was missing from this file entirely.
    candidate_token_ids = get_candidate_token_ids(tokenizer)
    if eval_entries:
        sanity_check_candidate_tokenization(tokenizer, eval_entries[0][3], candidate_token_ids)

    lora_config = build_lora_config(args)

    # >>> MODIFIED: reference model is now built once via build_reference_model
    # (Fix #3), replacing the old `ref_model_name or model_name` + per-ft_file
    # `get_peft_model(ref_model, ...)` re-wrap that used to happen inside the loop.
    ref_model = build_reference_model(tokenizer, lora_config, args)

    # >>> MODIFIED (Fix #1 — seed variance): summary is now nested
    # {ft_name: {"per_seed": [...], "aggregate": {...}}} instead of one flat
    # metrics dict per ft_name.
    summary: Dict[str, Dict[str, Any]] = {}

    for ft_file in args.ft_files:
        ft_name = safe_adapter_name(ft_file)
        print("\n====================")
        print(f"DPO DATASET: {ft_name}")
        print("====================")

        ft_rows = load_jsonl(ft_file)
        per_seed_metrics: List[Dict[str, float]] = []

        for seed in args.seeds:
            print(f"\n--- {ft_name} | seed={seed} ---")
            set_seed(seed)
            rng = random.Random(seed)

            train_dataset, rejected_counter = build_dpo_dataset(ft_rows, tokenizer, rng)
            print(f"Loaded DPO rows: {len(ft_rows)}; usable rows: {len(train_dataset)}")
            # >>> MODIFIED: logs the rejected-label distribution so you can
            # verify the sampling fix (Fix #4) isn't secretly still biased.
            print(f"Rejected-label distribution: {dict(rejected_counter)}")

            # >>> MODIFIED: previously the script kept ONE `model` across all
            # ft_files, calling `model.add_adapter(ft_name, ...)` /
            # `model.set_adapter(ft_name)` — adapters accumulated and were
            # never cleaned up between files. Now a fresh policy is built
            # per (ft_file, seed) via build_policy_model, and any previous
            # "default" adapter on base_model is deleted first so runs don't
            # leak into each other.
            if hasattr(base_model, "peft_config") and "default" in getattr(base_model, "peft_config", {}):
                base_model.delete_adapter("default")
            model = build_policy_model(base_model, lora_config, args)
            model.set_adapter("default")
            model.train()
            model.config.use_cache = False
            model.print_trainable_parameters()

            dpo_config = build_dpo_config(args, ft_name, seed)
            trainer = make_dpo_trainer(model, ref_model, dpo_config, train_dataset, tokenizer)

            print("Training with DPO...")
            trainer.train()

            print("Evaluating...")
            model.eval()
            metrics = evaluate(model, tokenizer, eval_entries, ft_name, seed, args)
            metrics["seed"] = seed
            metrics["n_train"] = len(train_dataset)
            per_seed_metrics.append(metrics)

            del trainer
            torch.cuda.empty_cache()

        # >>> MODIFIED (Fix #1 — aggregate across seeds): mean/std per metric.
        agg = {}
        for key in ["acc", "kappa", "mcc", "macro_f1"]:
            values = [m[key] for m in per_seed_metrics]
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        agg["n_eval"] = per_seed_metrics[0]["n_eval"]
        agg["n_train"] = per_seed_metrics[0]["n_train"]
        agg["seeds"] = args.seeds

        summary[ft_name] = {"per_seed": per_seed_metrics, "aggregate": agg}

        print(f"\n=== AGGREGATE ({len(args.seeds)} seeds): {ft_name} ===")
        for key, value in agg.items():
            print(f"{key}: {value}")

    if ref_model is not None:
        del ref_model
        torch.cuda.empty_cache()

    with open(os.path.join(args.out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nAll runs complete. Summary written to summary_metrics.json")


if __name__ == "__main__":
    main()

# import argparse
# import json
# import os
# import random
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional

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
#     TrainingArguments,
# )
# from trl import DPOTrainer, DPOConfig


# CANDIDATES = ["Fewer immigrants", "More immigrants", "Same amount"]
# VOTE2ID = {candidate: i for i, candidate in enumerate(CANDIDATES)}

# SYSTEM_TEXT = (
#     "You are a classifier.\n\n"
#     "A survey respondent was asked the following question in 2024:\n"
#     "In your opinion, should Canada admit more immigrants, fewer immigrants, or about the same number of immigrants as now?\n\n"
    
#     "Based on the respondent's attributes, predict their answer.\n\n"
#     "Output exactly one of the following labels:\n"
#     "More immigrants\n"
#     "Fewer immigrants\n"
#     "Same amount\n\n"
#     "Do not explain your answer."
# )


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="QLoRA DPO training + deterministic immigration-preference evaluation."
#     )
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         default="meta-llama/Llama-3.1-8B-Instruct",
#     )
#     parser.add_argument(
#         "--data_path",
#         type=str,
#         default="dataset_test/test_canada_immigration_2024_new.json",
#     )
#     parser.add_argument(
#         "--ft_files",
#         nargs="+",
#         default=[
#             "dataset_ft/agg_ft_immigration_2024_dpo.jsonl",
#         ],
#     )
#     parser.add_argument("--out_dir", type=str, default="./results_dpo_immigration")
#     parser.add_argument("--tmp_dir", type=str, default="./tmp_qlora_dpo_immigration")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--max_len", type=int, default=512)
#     parser.add_argument("--max_prompt_len", type=int, default=256)
#     parser.add_argument("--epochs", type=float, default=1.0)
#     parser.add_argument("--train_batch_size", type=int, default=1)
#     parser.add_argument("--grad_accum", type=int, default=16)
#     parser.add_argument("--eval_batch_size", type=int, default=8)
#     parser.add_argument("--learning_rate", type=float, default=5e-5)
#     parser.add_argument("--lora_r", type=int, default=16)
#     parser.add_argument("--lora_alpha", type=int, default=32)
#     parser.add_argument("--lora_dropout", type=float, default=0.05)
#     parser.add_argument(
#         "--target_modules",
#         nargs="+",
#         default=[
#             "q_proj",
#             "k_proj",
#             "v_proj",
#             "o_proj",
#             "gate_proj",
#             "up_proj",
#             "down_proj",
#         ],
#     )
#     parser.add_argument(
#         "--attn_implementation",
#         type=str,
#         default="sdpa",
#         choices=["sdpa", "flash_attention_2", "eager"],
#     )
#     parser.add_argument(
#         "--length_normalize_eval",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#         help="Average candidate log-probability by token count during evaluation.",
#     )
#     parser.add_argument(
#         "--beta",
#         type=float,
#         default=0.1,
#         help="DPO beta parameter controlling KL divergence penalty.",
#     )
#     parser.add_argument(
#         "--ref_model_name",
#         type=str,
#         default=None,
#         help="Reference model for DPO. If None, uses model_name.",
#     )
#     return parser.parse_args()


# def set_seed(seed: int) -> None:
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
#         torch.set_float32_matmul_precision("high")


# def normalize_vote(value: Any) -> str | None:
#     if value is None:
#         return None

#     text = str(value).strip().lower()
#     for candidate in CANDIDATES:
#         if candidate.lower() in text:
#             return candidate
#     return None


# def extract_gt(messages: list[dict[str, str]]) -> str | None:
#     for message in messages:
#         if message.get("role") == "assistant":
#             return normalize_vote(message.get("content"))
#     return None


# def first_user_content(messages: list[dict[str, str]]) -> str:
#     for message in messages:
#         if message.get("role") == "user":
#             return message.get("content", "")
#     return messages[0].get("content", "") if messages else ""


# def build_prompt(tokenizer: AutoTokenizer, user_text: str) -> str:
#     messages = [
#         {"role": "system", "content": SYSTEM_TEXT},
#         {"role": "user", "content": user_text},
#     ]
#     return tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )


# def load_json(path: str) -> list[dict[str, Any]]:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


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


# def load_qlora_base_model(
#     model_name: str,
#     tokenizer: AutoTokenizer,
#     attn_implementation: str,
# ) -> AutoModelForCausalLM:
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )

#     max_memory = {
#         0: "75GiB",
#         1: "75GiB",
#         "cpu": "120GiB",
#     }

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         max_memory=max_memory,
#         dtype=torch.bfloat16,
#         attn_implementation=attn_implementation,
#         low_cpu_mem_usage=True,
#         device_map="auto",
#     )

#     model.config.pad_token_id = tokenizer.pad_token_id
#     model.config.use_cache = False

#     model = prepare_model_for_kbit_training(
#         model,
#         use_gradient_checkpointing=True,
#     )
#     return model


# def build_lora_config(args: argparse.Namespace) -> LoraConfig:
#     return LoraConfig(
#         r=args.lora_r,
#         lora_alpha=args.lora_alpha,
#         target_modules=args.target_modules,
#         lora_dropout=args.lora_dropout,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )


# def build_dpo_dataset(
#     ft_rows: list[dict[str, Any]],
#     tokenizer: AutoTokenizer,
#     max_len: int,
# ) -> Dataset:
#     """Convert DPO-style data (prompt, chosen, rejected) to the format expected by DPOTrainer."""
#     samples = []

#     for item in ft_rows:
#         # Handle DPO format: {"prompt": "...", "chosen": "...", "rejected": "..."}
#         if "prompt" in item and "chosen" in item and "rejected" in item:
#             prompt = build_prompt(tokenizer, item["prompt"])
#             chosen = item["chosen"] + tokenizer.eos_token
#             rejected = item["rejected"] + tokenizer.eos_token
            
#             samples.append({
#                 "prompt": prompt,
#                 "chosen": chosen,
#                 "rejected": rejected,
#             })
#         # Also support the original messages format for compatibility
#         elif "messages" in item:
#             messages = item.get("messages", [])
#             gt = extract_gt(messages)
#             if gt is None:
#                 continue
            
#             prompt = build_prompt(tokenizer, first_user_content(messages))
#             chosen = gt + tokenizer.eos_token
            
#             # Create a simple rejected example (other immigration preference)
#             rejected_candidates = [c for c in CANDIDATES if c != gt]
#             rejected = rejected_candidates[0] + tokenizer.eos_token if rejected_candidates else chosen
            
#             samples.append({
#                 "prompt": prompt,
#                 "chosen": chosen,
#                 "rejected": rejected,
#             })

#     if not samples:
#         raise ValueError("No usable DPO samples found.")

#     return Dataset.from_list(samples)


# def prepare_eval_entries(
#     eval_rows: list[dict[str, Any]],
#     tokenizer: AutoTokenizer,
# ) -> list[tuple[int, str, str, str]]:
#     entries = []
#     for idx, entry in enumerate(eval_rows):
#         messages = entry.get("messages", [])
#         gt = extract_gt(messages)
#         if gt is None:
#             continue

#         user_text = first_user_content(messages)
#         prompt = build_prompt(tokenizer, user_text)

#         entries.append((idx, gt, user_text, prompt))

#     return entries


# def get_candidate_token_ids(tokenizer: AutoTokenizer) -> list[list[int]]:
#     return [
#         tokenizer(candidate, add_special_tokens=False)["input_ids"]
#         for candidate in CANDIDATES
#     ]


# @torch.inference_mode()
# def score_candidates_batched(
#     model: AutoModelForCausalLM,
#     tokenizer: AutoTokenizer,
#     prompts: list[str],
#     candidate_token_ids: list[list[int]],
#     max_len: int,
#     length_normalize: bool,
# ) -> list[dict[str, float]]:
#     if not prompts:
#         return []

#     device = next(model.parameters()).device
#     encoded_prompts = tokenizer(
#         prompts,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=max_len,
#         add_special_tokens=False,
#     )

#     prompt_input_ids = encoded_prompts["input_ids"]
#     prompt_attention_mask = encoded_prompts["attention_mask"]
#     prompt_lengths = prompt_attention_mask.sum(dim=1).tolist()

#     sequences = []
#     prompt_lens = []
#     candidate_lens = []

#     for row_idx, prompt_len in enumerate(prompt_lengths):
#         prompt_tokens = prompt_input_ids[row_idx, :prompt_len].tolist()
#         for candidate_ids in candidate_token_ids:
#             max_prompt_tokens = max_len - len(candidate_ids)
#             if max_prompt_tokens <= 0:
#                 raise ValueError("Candidate label is longer than max_len.")

#             trimmed_prompt = prompt_tokens[-max_prompt_tokens:]
#             sequences.append(trimmed_prompt + candidate_ids)
#             prompt_lens.append(len(trimmed_prompt))
#             candidate_lens.append(len(candidate_ids))

#     batch_size = len(sequences)
#     padded_len = max(len(sequence) for sequence in sequences)
#     input_ids = torch.full(
#         (batch_size, padded_len),
#         tokenizer.pad_token_id,
#         dtype=torch.long,
#         device=device,
#     )
#     attention_mask = torch.zeros(
#         (batch_size, padded_len),
#         dtype=torch.long,
#         device=device,
#     )

#     for i, sequence in enumerate(sequences):
#         length = len(sequence)
#         input_ids[i, :length] = torch.tensor(sequence, dtype=torch.long, device=device)
#         attention_mask[i, :length] = 1

#     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#     log_probs = F.log_softmax(outputs.logits.float(), dim=-1)

#     scores = []
#     for i in range(batch_size):
#         prompt_len = prompt_lens[i]
#         candidate_len = candidate_lens[i]
#         token_scores = []

#         for offset in range(candidate_len):
#             logit_pos = prompt_len - 1 + offset
#             token_pos = prompt_len + offset
#             token_id = input_ids[i, token_pos]
#             token_scores.append(log_probs[i, logit_pos, token_id])

#         candidate_score = torch.stack(token_scores).sum()
#         if length_normalize:
#             candidate_score = candidate_score / candidate_len
#         scores.append(candidate_score)

#     score_tensor = torch.stack(scores).view(len(prompts), len(CANDIDATES))
#     prob_tensor = torch.softmax(score_tensor, dim=1).detach().cpu().numpy()

#     return [
#         {candidate: float(row[i]) for i, candidate in enumerate(CANDIDATES)}
#         for row in prob_tensor
#     ]


# def safe_model_name(model_name: str) -> str:
#     return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in model_name)


# def evaluate(
#     model: AutoModelForCausalLM,
#     tokenizer: AutoTokenizer,
#     eval_entries: list[tuple[int, str, str, str]],
#     ft_name: str,
#     args: argparse.Namespace,
# ) -> dict[str, float]:
#     model.eval()
#     model.config.use_cache = True

#     candidate_token_ids = get_candidate_token_ids(tokenizer)
#     rows = []

#     for start in tqdm(
#         range(0, len(eval_entries), args.eval_batch_size),
#         desc=f"Eval {ft_name}",
#     ):
#         batch = eval_entries[start : start + args.eval_batch_size]
#         prompts = [entry[3] for entry in batch]
        
#         probabilities = score_candidates_batched(
#             model=model,
#             tokenizer=tokenizer,
#             prompts=prompts,
#             candidate_token_ids=candidate_token_ids,
#             max_len=args.max_len,
#             length_normalize=args.length_normalize_eval,
#         )

#         for (idx, gt, user_text, _prompt), probs in zip(batch, probabilities):
#             pred = max(probs, key=probs.get)
#             rows.append(
#                 {
#                     "user_text": user_text,
#                     "idx": idx,
#                     "gt": gt,
#                     "pred": pred,
#                     "acc": int(gt == pred),
#                     **{f"prob_{candidate}": probs[candidate] for candidate in CANDIDATES},
#                 }
#             )

#     df = pd.DataFrame(rows)
#     print(df.head(5))

#     if df.empty:
#         raise ValueError("No valid evaluation samples found.")

#     y_true = df["gt"].map(VOTE2ID).to_numpy()
#     y_pred = df["pred"].map(VOTE2ID).to_numpy()
#     metrics = {
#         "acc": float(df["acc"].mean()),
#         "kappa": float(cohen_kappa_score(y_true, y_pred)),
#         "mcc": float(matthews_corrcoef(y_true, y_pred)),
#         "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
#         "n_eval": int(len(df)),
#     }

#     safe_name = safe_model_name(args.model_name)
#     result_base = os.path.join(args.out_dir, f"{safe_name}_{ft_name}_dpo_lora")
#     df.to_csv(result_base + "_results_dpo_immigration.csv", index=False)
#     with open(result_base + "_metrics_dpo_immigration.json", "w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2)

#     print(f"\n=== RESULTS: {ft_name} ===")
#     for key, value in metrics.items():
#         if isinstance(value, float):
#             print(f"{key}: {value:.4f}")
#         else:
#             print(f"{key}: {value}")

#     model.config.use_cache = False
#     return metrics


# def safe_adapter_name(ft_file: str) -> str:
#     name = os.path.basename(ft_file).replace(".jsonl", "")
#     return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


# def main() -> None:
#     args = parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)
#     os.makedirs(args.tmp_dir, exist_ok=True)
#     set_seed(args.seed)

#     print("Loading evaluation data...")
#     eval_rows = load_json(args.data_path)

#     print("Loading tokenizer...")
#     tokenizer = load_tokenizer(args.model_name)
#     eval_entries = prepare_eval_entries(eval_rows, tokenizer)
#     print(f"Loaded eval rows: {len(eval_rows)}; usable rows: {len(eval_entries)}")

#     print("Loading 4-bit base model...")
#     base_model = load_qlora_base_model(
#         model_name=args.model_name,
#         tokenizer=tokenizer,
#         attn_implementation=args.attn_implementation,
#     )

#     # Load reference model for DPO if specified
#     ref_model = None
#     ref_model_name = args.ref_model_name or args.model_name
#     if ref_model_name != args.model_name:
#         print(f"Loading reference model: {ref_model_name}")
#         ref_model = load_qlora_base_model(
#             model_name=ref_model_name,
#             tokenizer=tokenizer,
#             attn_implementation=args.attn_implementation,
#         )

#     lora_config = build_lora_config(args)
#     model = None
#     summary = {}

#     for ft_file in args.ft_files:
#         ft_name = safe_adapter_name(ft_file)
#         print("\n====================")
#         print(f"DPO DATASET: {ft_name}")
#         print("====================")

#         ft_rows = load_jsonl(ft_file)
#         train_dataset = build_dpo_dataset(ft_rows, tokenizer, args.max_len)
#         print(f"Loaded DPO rows: {len(ft_rows)}; usable rows: {len(train_dataset)}")

#         if model is None:
#             model = get_peft_model(base_model, lora_config, adapter_name="default")
#         else:
#             model.add_adapter(ft_name, lora_config)
#             model.set_adapter(ft_name)

#         # Set up reference model adapter if using separate reference model
#         ref_adapter_model = None
#         if ref_model is not None:
#             ref_adapter_model = get_peft_model(ref_model, lora_config, adapter_name="default")
#             # Freeze reference model
#             for param in ref_adapter_model.parameters():
#                 param.requires_grad = False

#         model.train()
#         model.config.use_cache = False
#         model.print_trainable_parameters()

#         # DPO Training Arguments
#         dpo_config = DPOConfig(
#             output_dir=os.path.join(args.tmp_dir, ft_name),
#             per_device_train_batch_size=args.train_batch_size,
#             gradient_accumulation_steps=args.grad_accum,
#             num_train_epochs=args.epochs,
#             learning_rate=args.learning_rate,
#             warmup_steps=10,
#             lr_scheduler_type="cosine",
#             optim="paged_adamw_8bit",
#             bf16=True,
#             gradient_checkpointing=True,
#             max_grad_norm=0.3,
#             logging_steps=10,
#             save_strategy="no",
#             report_to="none",
#             seed=args.seed,
#             data_seed=args.seed,
#             remove_unused_columns=False,
#             dataloader_pin_memory=True,
#             beta=args.beta,
#             max_length=args.max_len,
#         )

#         # Check TRL version to use correct API
#         import trl
#         trl_version = tuple(map(int, trl.__version__.split('.')[:2]))
        
#         if trl_version >= (0, 11):
#             trainer = DPOTrainer(
#                 model=model,
#                 ref_model=ref_adapter_model,
#                 args=dpo_config,
#                 train_dataset=train_dataset,
#             )
#         elif trl_version >= (0, 9):
#             trainer = DPOTrainer(
#                 model=model,
#                 ref_model=ref_adapter_model,
#                 args=dpo_config,
#                 train_dataset=train_dataset,
#                 processing_class=tokenizer,
#             )
#         else:
#             trainer = DPOTrainer(
#                 model=model,
#                 ref_model=ref_adapter_model,
#                 args=dpo_config,
#                 train_dataset=train_dataset,
#                 tokenizer=tokenizer,
#             )

#         print("Training with DPO...")
#         trainer.train()

#         print("Evaluating...")
#         model.eval()
#         metrics = evaluate(model, tokenizer, eval_entries, ft_name, args)
#         summary[ft_name] = metrics

#         del trainer
#         torch.cuda.empty_cache()

#     with open(os.path.join(args.out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2)


# if __name__ == "__main__":
#     main()
