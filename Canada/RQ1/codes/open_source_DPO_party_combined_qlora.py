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


# --------------------------------------------------------------------------- #
# Args / setup
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA DPO training + deterministic party-label evaluation."
    )
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset_test/test_canada_election_party_2021_3class_new.json",
    )
    parser.add_argument(
        "--ft_files",
        nargs="+",
        default=["dataset_ft/agg_ft_party_2021_3class_dpo.jsonl"],
    )
    parser.add_argument("--out_dir", type=str, default="./results_dpo")
    parser.add_argument("--tmp_dir", type=str, default="./tmp_qlora_dpo")

    # >>> MODIFIED (Fix #2 — seed variance): was a single --seed int. Now accepts
    # one or more seeds; every (ft_file, seed) combination is trained and
    # evaluated separately, then aggregated (mean/std) so you can see how much
    # of the DPO-vs-SFT gap is just training noise vs. a real effect.
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
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty).")

    # >>> MODIFIED (Fix #3 — pairwise/multiclass mismatch): rpo_alpha adds an
    # SFT/NLL loss term on the chosen response alongside the DPO loss, which
    # directly counteracts "likelihood displacement" (DPO widening the margin
    # by suppressing both candidates instead of raising P(chosen)). Only
    # applied if the installed TRL's DPOConfig supports it (see
    # build_dpo_config below) — otherwise it's dropped with a printed warning.
    parser.add_argument(
        "--rpo_alpha",
        type=float,
        default=1.0,
        help="Weight of the auxiliary NLL loss on the chosen response (RPO-style). "
             "Set to 0 or None to disable and use vanilla DPO.",
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
    # DPO practice (refine preferences of an already-competent policy,
    # rather than learning the task and the preference signal at once).
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
        help="Optional per-GPU memory cap in GiB. If unset, let accelerate decide.",
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
def normalize_vote(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    for candidate in CANDIDATES:
        if candidate.lower() in text:
            return candidate
    return None


def extract_gt(messages: List[Dict[str, str]]) -> Optional[str]:
    for message in messages:
        if message.get("role") == "assistant":
            return normalize_vote(message.get("content"))
    return None


def first_user_content(messages: List[Dict[str, str]]) -> str:
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


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
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
    max_memory_gpu_gib: Optional[int],
) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    max_memory = None
    if max_memory_gpu_gib is not None and torch.cuda.is_available():
        max_memory = {i: f"{max_memory_gpu_gib}GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "120GiB"

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
    ft_rows: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    rng: random.Random,
) -> Tuple[Dataset, Counter]:
    """Convert DPO-style data (prompt, chosen, rejected) to the format DPOTrainer expects.

    Returns the dataset AND a Counter of how often each candidate was used as
    the 'rejected' label, so you can log/verify the sampling isn't biased.
    """
    samples = []
    rejected_counter: Counter = Counter()

    for item in ft_rows:
        if "prompt" in item and "chosen" in item and "rejected" in item:
            prompt = build_prompt(tokenizer, item["prompt"])
            chosen = item["chosen"] + tokenizer.eos_token
            rejected = item["rejected"] + tokenizer.eos_token

            samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            rejected_counter[item["rejected"]] += 1

        elif "messages" in item:
            # Fallback path: build a DPO pair from plain SFT-style messages.
            messages = item.get("messages", [])
            gt = extract_gt(messages)
            if gt is None:
                continue

            prompt = build_prompt(tokenizer, first_user_content(messages))
            chosen = gt + tokenizer.eos_token

            # (Already fixed in the previous version) random non-gt label,
            # not a fixed index — kept as-is here.
            rejected_candidates = [c for c in CANDIDATES if c != gt]
            rejected_label = rng.choice(rejected_candidates) if rejected_candidates else gt
            rejected = rejected_label + tokenizer.eos_token

            samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            rejected_counter[rejected_label] += 1

    if not samples:
        raise ValueError("No usable DPO samples found.")

    return Dataset.from_list(samples), rejected_counter


def prepare_eval_entries(
    eval_rows: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
) -> List[Tuple[int, str, str, str]]:
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


def get_candidate_token_ids(tokenizer: AutoTokenizer) -> List[List[int]]:
    return [
        tokenizer(candidate, add_special_tokens=False)["input_ids"]
        for candidate in CANDIDATES
    ]


def sanity_check_candidate_tokenization(
    tokenizer: AutoTokenizer,
    sample_prompt: str,
    candidate_token_ids: List[List[int]],
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


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def score_candidates_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    candidate_token_ids: List[List[int]],
    max_len: int,
    length_normalize: bool,
) -> List[Dict[str, float]]:
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
    eval_entries: List[Tuple[int, str, str, str]],
    ft_name: str,
    seed: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    model.config.use_cache = True

    candidate_token_ids = get_candidate_token_ids(tokenizer)
    rows = []

    for start in tqdm(
        range(0, len(eval_entries), args.eval_batch_size), desc=f"Eval {ft_name} (seed={seed})"
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
        # >>> MODIFIED (Fix #5 — visibility into eval set size): n_eval was
        # already computed before, but is now also printed explicitly below
        # and included in the per-seed AND aggregated summary, so panel-width
        # differences in your forest plot can be checked against sample size.
        "n_eval": int(len(df)),
    }

    safe_name = safe_model_name(args.model_name)
    # >>> MODIFIED: seed is now part of the output filename so per-seed runs
    # don't overwrite each other.
    result_base = os.path.join(args.out_dir, f"{safe_name}_{ft_name}_seed{seed}_dpo_lora")
    df.to_csv(result_base + "_results_dpo.csv", index=False)
    with open(result_base + "_metrics_dpo.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== RESULTS: {ft_name} (seed={seed}) ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    model.config.use_cache = False
    return metrics


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
        max_prompt_length=args.max_prompt_len,
    )

    # >>> MODIFIED (Fix #3): add rpo_alpha to the desired kwargs. It will be
    # silently dropped by the filter below (with a printed warning) if the
    # installed TRL version's DPOConfig doesn't support it.
    if args.rpo_alpha is not None and args.rpo_alpha > 0:
        desired["rpo_alpha"] = args.rpo_alpha

    accepted_params = set(inspect.signature(DPOConfig.__init__).parameters.keys())
    accepted = {k: v for k, v in desired.items() if k in accepted_params}
    dropped = {k: v for k, v in desired.items() if k not in accepted_params}

    if dropped:
        print(
            f"[INFO] Installed TRL's DPOConfig does not accept these kwargs; "
            f"dropping them: {sorted(dropped.keys())}"
        )
        if "max_prompt_length" in dropped:
            print(
                "[INFO] max_prompt_length not supported by this DPOConfig — "
                "TRL will infer the prompt/completion split internally."
            )
        if "rpo_alpha" in dropped:
            print(
                "[WARNING] rpo_alpha not supported by this TRL version's DPOConfig — "
                "running vanilla DPO with no auxiliary NLL loss. Consider upgrading TRL "
                "if you want the likelihood-displacement fix."
            )

    return DPOConfig(**accepted)


def make_dpo_trainer(
    model,
    ref_model,
    dpo_config: DPOConfig,
    train_dataset: Dataset,
    tokenizer: AutoTokenizer,
) -> DPOTrainer:
    trl_version = tuple(map(int, trl.__version__.split(".")[:2]))

    if trl_version >= (0, 11):
        return DPOTrainer(model=model, ref_model=ref_model, args=dpo_config, train_dataset=train_dataset)
    elif trl_version >= (0, 9):
        return DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    else:
        return DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
        )


# >>> MODIFIED (Fix #3 — SFT warm start): new helper. Builds the DPO policy
# model either from a fresh zero-init LoRA adapter (old behavior, if
# args.sft_adapter_path is None) or by loading an already-trained SFT LoRA
# adapter as the starting point (new, recommended behavior).
def build_policy_model(
    base_model: AutoModelForCausalLM,
    lora_config: LoraConfig,
    args: argparse.Namespace,
) -> PeftModel:
    if args.sft_adapter_path:
        print(f"[INFO] Warm-starting DPO policy from SFT adapter: {args.sft_adapter_path}")
        model = PeftModel.from_pretrained(
            base_model, args.sft_adapter_path, adapter_name="default", is_trainable=True
        )
    else:
        print("[INFO] No --sft_adapter_path given — starting DPO from a fresh zero-init adapter "
              "(this reproduces the old 'DPO from scratch' behavior and is more prone to instability).")
        model = get_peft_model(base_model, lora_config, adapter_name="default")
    return model


# >>> MODIFIED (Fix #3 — SFT warm start): new helper. Builds the frozen
# reference model used for the DPO KL penalty. If use_sft_as_ref is True and
# an SFT adapter path is given, the reference is the SFT-tuned policy
# (standard DPO setup: refine an already-competent model's preferences).
# Otherwise falls back to the old behavior (raw base model / separate
# ref_model_name).
def build_reference_model(
    tokenizer: AutoTokenizer,
    lora_config: LoraConfig,
    args: argparse.Namespace,
) -> Optional[PeftModel]:
    if args.sft_adapter_path and args.use_sft_as_ref:
        print(f"[INFO] Using SFT adapter as frozen DPO reference model: {args.sft_adapter_path}")
        ref_base = load_qlora_base_model(
            model_name=args.model_name,
            tokenizer=tokenizer,
            attn_implementation=args.attn_implementation,
            max_memory_gpu_gib=args.max_memory_gpu_gib,
        )
        ref_model = PeftModel.from_pretrained(
            ref_base, args.sft_adapter_path, adapter_name="default", is_trainable=False
        )
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()
        return ref_model

    if args.ref_model_name and args.ref_model_name != args.model_name:
        print(f"[INFO] Loading separate reference model: {args.ref_model_name}")
        ref_base = load_qlora_base_model(
            model_name=args.ref_model_name,
            tokenizer=tokenizer,
            attn_implementation=args.attn_implementation,
            max_memory_gpu_gib=args.max_memory_gpu_gib,
        )
        ref_model = get_peft_model(ref_base, lora_config, adapter_name="default")
        for param in ref_model.parameters():
            param.requires_grad = False
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
        model_name=args.model_name,
        tokenizer=tokenizer,
        attn_implementation=args.attn_implementation,
        max_memory_gpu_gib=args.max_memory_gpu_gib,
    )

    candidate_token_ids = get_candidate_token_ids(tokenizer)
    if eval_entries:
        sanity_check_candidate_tokenization(tokenizer, eval_entries[0][3], candidate_token_ids)

    lora_config = build_lora_config(args)

    # >>> MODIFIED: reference model is now built once via build_reference_model,
    # which knows about the SFT-as-reference option (Fix #3).
    ref_model = build_reference_model(tokenizer, lora_config, args)

    # >>> MODIFIED (Fix #2 — seed variance): summary is now nested
    # {ft_name: {seed: metrics}} plus an aggregated {ft_name: {"mean": ..., "std": ...}}
    # entry, instead of one metrics dict per ft_name.
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
            # >>> MODIFIED (Fix #2/#5 — visibility): explicitly print dataset
            # size and rejected-label distribution per run, so you can verify
            # the sampling fix is behaving as expected and check whether small
            # dataset size correlates with unstable results.
            print(f"Loaded DPO rows: {len(ft_rows)}; usable rows: {len(train_dataset)}")
            print(f"Rejected-label distribution: {dict(rejected_counter)}")

            # >>> MODIFIED (Fix #3): build a fresh policy each seed, either
            # warm-started from the SFT adapter or from scratch, per
            # build_policy_model above. Previous adapters are discarded
            # (fresh AutoModelForCausalLM copy pattern would be expensive, so
            # instead we delete/re-add the adapter on the same base_model).
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

        # >>> MODIFIED (Fix #2 — aggregate across seeds): mean/std per metric,
        # so you can plot error bars from training-seed variance, not just
        # from the eval-side bootstrap you were already doing.
        agg = {}
        for key in ["acc", "kappa", "mcc", "macro_f1"]:
            values = [m[key] for m in per_seed_metrics]
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        agg["n_eval"] = per_seed_metrics[0]["n_eval"]
        agg["n_train"] = per_seed_metrics[0]["n_train"]
        agg["seeds"] = args.seeds

        summary[ft_name] = {
            "per_seed": per_seed_metrics,
            "aggregate": agg,
        }

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
# import inspect
# import json
# import os
# import random
# from typing import Any, Dict, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# import trl
# from datasets import Dataset
# from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# from sklearn.metrics import cohen_kappa_score, f1_score, matthews_corrcoef
# from tqdm import tqdm
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     BitsAndBytesConfig,
# )
# from trl import DPOTrainer, DPOConfig


# CANDIDATES = ["Liberal", "Conservative", "Other"]

# VOTE2ID = {candidate: i for i, candidate in enumerate(CANDIDATES)}

# SYSTEM_TEXT = (
#     "You are an expert political analyst specializing in Canadian elections and voting behavior.\n\n"
#     "Task:\n"
#     "Given a person's demographic and political attributes, predict their MOST LIKELY party choice "
#     "in the 2021 Canadian federal election.\n\n"
#     "Rules:\n"
#     "- You must choose ONLY ONE label.\n"
#     "- Output must be EXACTLY one of the following (no explanation, no extra text):\n"
#     "Liberal\n"
#     "Conservative\n"
#     "Other\n\n"
#     "Definition:\n"
#     "Other includes New Democratic Party (NDP), Bloc Quebecois, Green Party, "
#     "and People's Party of Canada.\n\n"
#     "Important:\n"
#     "- Base your decision on typical voting patterns, demographics, and political alignment.\n"
#     "- Do NOT explain your reasoning.\n"
#     "- Do NOT repeat the input.\n"
#     "- Output ONLY the label."
# )


# # --------------------------------------------------------------------------- #
# # Args / setup
# # --------------------------------------------------------------------------- #
# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="QLoRA DPO training + deterministic party-label evaluation."
#     )
#     parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
#     parser.add_argument(
#         "--data_path",
#         type=str,
#         default="dataset_test/test_canada_election_party_2021_3class_new.json",
#     )
#     parser.add_argument(
#         "--ft_files",
#         nargs="+",
#         default=["dataset_ft/agg_ft_party_2021_3class_dpo.jsonl"],
#     )
#     parser.add_argument("--out_dir", type=str, default="./results_dpo")
#     parser.add_argument("--tmp_dir", type=str, default="./tmp_qlora_dpo")
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
#             "q_proj", "k_proj", "v_proj", "o_proj",
#             "gate_proj", "up_proj", "down_proj",
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
#     parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty).")
#     parser.add_argument(
#         "--ref_model_name",
#         type=str,
#         default=None,
#         help="Reference model for DPO. If None, uses model_name with adapters disabled.",
#     )
#     parser.add_argument(
#         "--max_memory_gpu_gib",
#         type=int,
#         default=None,
#         help="Optional per-GPU memory cap in GiB. If unset, let accelerate decide.",
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


# # --------------------------------------------------------------------------- #
# # Data helpers
# # --------------------------------------------------------------------------- #
# def normalize_vote(value: Any) -> Optional[str]:
#     if value is None:
#         return None
#     text = str(value).strip().lower()
#     for candidate in CANDIDATES:
#         if candidate.lower() in text:
#             return candidate
#     return None


# def extract_gt(messages: List[Dict[str, str]]) -> Optional[str]:
#     for message in messages:
#         if message.get("role") == "assistant":
#             return normalize_vote(message.get("content"))
#     return None


# def first_user_content(messages: List[Dict[str, str]]) -> str:
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


# def load_json(path: str) -> List[Dict[str, Any]]:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def load_jsonl(path: str) -> List[Dict[str, Any]]:
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
#     max_memory_gpu_gib: Optional[int],
# ) -> AutoModelForCausalLM:
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )

#     max_memory = None
#     if max_memory_gpu_gib is not None and torch.cuda.is_available():
#         max_memory = {i: f"{max_memory_gpu_gib}GiB" for i in range(torch.cuda.device_count())}
#         max_memory["cpu"] = "120GiB"

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

#     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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
#     ft_rows: List[Dict[str, Any]],
#     tokenizer: AutoTokenizer,
#     rng: random.Random,
# ) -> Dataset:
#     """Convert DPO-style data (prompt, chosen, rejected) to the format DPOTrainer expects."""
#     samples = []

#     for item in ft_rows:
#         if "prompt" in item and "chosen" in item and "rejected" in item:
#             prompt = build_prompt(tokenizer, item["prompt"])
#             chosen = item["chosen"] + tokenizer.eos_token
#             rejected = item["rejected"] + tokenizer.eos_token

#             samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

#         elif "messages" in item:
#             # Fallback path: build a DPO pair from plain SFT-style messages.
#             messages = item.get("messages", [])
#             gt = extract_gt(messages)
#             if gt is None:
#                 continue

#             prompt = build_prompt(tokenizer, first_user_content(messages))
#             chosen = gt + tokenizer.eos_token

#             # Pick a random non-gt label so "rejected" isn't always the same
#             # class for every example (avoids biasing the contrast signal).
#             rejected_candidates = [c for c in CANDIDATES if c != gt]
#             rejected_label = rng.choice(rejected_candidates) if rejected_candidates else gt
#             rejected = rejected_label + tokenizer.eos_token

#             samples.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

#     if not samples:
#         raise ValueError("No usable DPO samples found.")

#     return Dataset.from_list(samples)


# def prepare_eval_entries(
#     eval_rows: List[Dict[str, Any]],
#     tokenizer: AutoTokenizer,
# ) -> List[Tuple[int, str, str, str]]:
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


# def get_candidate_token_ids(tokenizer: AutoTokenizer) -> List[List[int]]:
#     return [
#         tokenizer(candidate, add_special_tokens=False)["input_ids"]
#         for candidate in CANDIDATES
#     ]


# def sanity_check_candidate_tokenization(
#     tokenizer: AutoTokenizer,
#     sample_prompt: str,
#     candidate_token_ids: List[List[int]],
# ) -> None:
#     """
#     Verify that tokenizing each candidate label in isolation matches the tail
#     tokens you'd get from tokenizing (prompt + candidate) together. Byte-level
#     BPE tokenizers can merge across the prompt/candidate boundary, in which
#     case splicing isolated candidate token ids after the prompt is wrong.
#     """
#     for candidate, cand_ids in zip(CANDIDATES, candidate_token_ids):
#         full_ids = tokenizer(sample_prompt + candidate, add_special_tokens=False)["input_ids"]
#         tail = full_ids[-len(cand_ids):]
#         if tail != cand_ids:
#             print(
#                 f"[WARNING] Tokenization boundary mismatch for candidate {candidate!r}: "
#                 f"isolated={cand_ids} vs tail-of-full={tail}. "
#                 "Candidate scoring may be misaligned for this model/tokenizer."
#             )


# # --------------------------------------------------------------------------- #
# # Evaluation
# # --------------------------------------------------------------------------- #
# @torch.inference_mode()
# def score_candidates_batched(
#     model: AutoModelForCausalLM,
#     tokenizer: AutoTokenizer,
#     prompts: List[str],
#     candidate_token_ids: List[List[int]],
#     max_len: int,
#     length_normalize: bool,
# ) -> List[Dict[str, float]]:
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
#         (batch_size, padded_len), tokenizer.pad_token_id, dtype=torch.long, device=device
#     )
#     attention_mask = torch.zeros((batch_size, padded_len), dtype=torch.long, device=device)

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


# def safe_adapter_name(ft_file: str) -> str:
#     name = os.path.basename(ft_file).replace(".jsonl", "")
#     return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)


# def evaluate(
#     model: AutoModelForCausalLM,
#     tokenizer: AutoTokenizer,
#     eval_entries: List[Tuple[int, str, str, str]],
#     ft_name: str,
#     args: argparse.Namespace,
# ) -> Dict[str, float]:
#     model.eval()
#     model.config.use_cache = True

#     candidate_token_ids = get_candidate_token_ids(tokenizer)
#     rows = []

#     for start in tqdm(range(0, len(eval_entries), args.eval_batch_size), desc=f"Eval {ft_name}"):
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
#     if df.empty:
#         raise ValueError("No valid evaluation samples found.")
#     print(df.head(5))

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
#     df.to_csv(result_base + "_results_dpo.csv", index=False)
#     with open(result_base + "_metrics_dpo.json", "w", encoding="utf-8") as f:
#         json.dump(metrics, f, indent=2)

#     print(f"\n=== RESULTS: {ft_name} ===")
#     for key, value in metrics.items():
#         print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

#     model.config.use_cache = False
#     return metrics


# def build_dpo_config(args: argparse.Namespace, ft_name: str) -> DPOConfig:
#     """
#     Build a DPOConfig using only the kwargs the installed TRL version's
#     DPOConfig actually accepts. TRL has renamed/removed/relocated fields
#     like `max_prompt_length` across versions (some infer it automatically
#     from `max_length`, some moved it to the trainer, some dropped it), so
#     hardcoding kwargs breaks on version drift. Instead, propose the full
#     desired set and drop whatever the installed signature doesn't support,
#     warning about anything dropped so it doesn't fail silently.
#     """
#     desired = dict(
#         output_dir=os.path.join(args.tmp_dir, ft_name),
#         per_device_train_batch_size=args.train_batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         num_train_epochs=args.epochs,
#         learning_rate=args.learning_rate,
#         warmup_steps=10,
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
#         beta=args.beta,
#         max_length=args.max_len,
#         max_prompt_length=args.max_prompt_len,
#     )

#     accepted_params = set(inspect.signature(DPOConfig.__init__).parameters.keys())
#     accepted = {k: v for k, v in desired.items() if k in accepted_params}
#     dropped = {k: v for k, v in desired.items() if k not in accepted_params}

#     if dropped:
#         print(
#             f"[INFO] Installed TRL's DPOConfig does not accept these kwargs; "
#             f"dropping them: {sorted(dropped.keys())}"
#         )
#         if "max_prompt_length" in dropped:
#             print(
#                 "[INFO] max_prompt_length not supported by this DPOConfig — "
#                 "TRL will infer the prompt/completion split internally. If you "
#                 "need to enforce a specific prompt budget, truncate `item['prompt']` "
#                 "yourself before calling build_prompt(), or upgrade/pin TRL."
#             )

#     return DPOConfig(**accepted)


# def make_dpo_trainer(
#     model,
#     ref_model,
#     dpo_config: DPOConfig,
#     train_dataset: Dataset,
#     tokenizer: AutoTokenizer,
# ) -> DPOTrainer:
#     """Build a DPOTrainer compatible with the installed TRL version."""
#     trl_version = tuple(map(int, trl.__version__.split(".")[:2]))

#     if trl_version >= (0, 11):
#         return DPOTrainer(model=model, ref_model=ref_model, args=dpo_config, train_dataset=train_dataset)
#     elif trl_version >= (0, 9):
#         return DPOTrainer(
#             model=model,
#             ref_model=ref_model,
#             args=dpo_config,
#             train_dataset=train_dataset,
#             processing_class=tokenizer,
#         )
#     else:
#         return DPOTrainer(
#             model=model,
#             ref_model=ref_model,
#             args=dpo_config,
#             train_dataset=train_dataset,
#             tokenizer=tokenizer,
#         )


# # --------------------------------------------------------------------------- #
# # Main
# # --------------------------------------------------------------------------- #
# def main() -> None:
#     args = parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)
#     os.makedirs(args.tmp_dir, exist_ok=True)
#     set_seed(args.seed)
#     rng = random.Random(args.seed)

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
#         max_memory_gpu_gib=args.max_memory_gpu_gib,
#     )

#     candidate_token_ids = get_candidate_token_ids(tokenizer)
#     if eval_entries:
#         sanity_check_candidate_tokenization(tokenizer, eval_entries[0][3], candidate_token_ids)

#     # Reference model: built once (frozen for the whole run). If a distinct
#     # ref_model_name is given, wrap it in its own frozen LoRA-shaped PEFT
#     # model with zero-initialized adapters (equivalent to the base weights,
#     # but keeps dtypes/module structure consistent with the policy model).
#     # If ref_model_name is None, pass ref_model=None to DPOTrainer, which
#     # will use the policy model with adapters disabled as the reference —
#     # this only works correctly if the policy is a PeftModel, which it is.
#     ref_model = None
#     if args.ref_model_name and args.ref_model_name != args.model_name:
#         print(f"Loading separate reference model: {args.ref_model_name}")
#         ref_base = load_qlora_base_model(
#             model_name=args.ref_model_name,
#             tokenizer=tokenizer,
#             attn_implementation=args.attn_implementation,
#             max_memory_gpu_gib=args.max_memory_gpu_gib,
#         )
#         ref_lora_config = build_lora_config(args)
#         ref_model = get_peft_model(ref_base, ref_lora_config, adapter_name="default")
#         for param in ref_model.parameters():
#             param.requires_grad = False
#         ref_model.eval()

#     lora_config = build_lora_config(args)
#     model: Optional[PeftModel] = None
#     summary: Dict[str, Dict[str, float]] = {}

#     for i, ft_file in enumerate(args.ft_files):
#         ft_name = safe_adapter_name(ft_file)
#         print("\n====================")
#         print(f"DPO DATASET: {ft_name}")
#         print("====================")

#         ft_rows = load_jsonl(ft_file)
#         train_dataset = build_dpo_dataset(ft_rows, tokenizer, rng)
#         print(f"Loaded DPO rows: {len(ft_rows)}; usable rows: {len(train_dataset)}")

#         if model is None:
#             # First dataset: wrap the base model with a fresh adapter.
#             # NOTE: the installed TRL version's DPOTrainer reads
#             # model.peft_config["default"] internally, so the active
#             # adapter name MUST be "default" regardless of which dataset
#             # we're on. ft_name is used only for logging/output filenames.
#             model = get_peft_model(base_model, lora_config, adapter_name="default")
#         else:
#             # Subsequent datasets: drop the previous adapter's weights
#             # before adding a fresh "default" adapter, so adapters never
#             # stack or leak into each other's "reference" (adapters-
#             # disabled) forward pass, while keeping the name TRL expects.
#             model.delete_adapter("default")
#             model.add_adapter("default", lora_config)

#         model.set_adapter("default")
#         model.train()
#         model.config.use_cache = False
#         model.print_trainable_parameters()

#         dpo_config = build_dpo_config(args, ft_name)

#         trainer = make_dpo_trainer(model, ref_model, dpo_config, train_dataset, tokenizer)

#         print("Training with DPO...")
#         trainer.train()

#         print("Evaluating...")
#         model.eval()
#         metrics = evaluate(model, tokenizer, eval_entries, ft_name, args)
#         summary[ft_name] = metrics

#         del trainer
#         torch.cuda.empty_cache()

#         is_last = i == len(args.ft_files) - 1
#         if is_last:
#             # Clean up the trained adapter after the final eval too, so the
#             # process doesn't hold onto trained weights unnecessarily.
#             model.delete_adapter("default")

#     if ref_model is not None:
#         del ref_model
#         torch.cuda.empty_cache()

#     with open(os.path.join(args.out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
#         json.dump(summary, f, indent=2)

#     print("\nAll runs complete. Summary written to summary_metrics.json")


# if __name__ == "__main__":
#     main()
