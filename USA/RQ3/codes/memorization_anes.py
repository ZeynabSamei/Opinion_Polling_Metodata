import argparse
import json
import os
import re
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------- Configuration ----------
YEARS = [2008, 2012, 2016, 2020, 2024]

# Bidirectional shuffle pairs: dem-win vs rep-win contrast, as specified.
# NOTE: the 2020->2024 direction (tagging real 2020 data as "2024") tests
# extrapolation/prior, not recall, since 2024 is post-cutoff for most base models.
# The 2024->2020 direction (tagging real 2024 data as "2020") IS a clean
# memorization test, since 2020 facts are within pretraining knowledge.
SHUFFLE_PAIRS = {
    2012: 2016,
    2016: 2012,
    2020: 2024,
    2024: 2020,
}

TASK_CONFIGS = {
    "party_choice": {
        "candidates": ["Democrat", "Republican", "Other"],
    },
    "ideology": {
        "candidates": ["liberal", "moderate", "conservative"],
    },
}

SYSTEM_TEXT = (
    "You are an expert political analyst specializing in US political attitudes and voting behavior. "
    "Your task is to predict the respondent's ideological self-placement based on the demographic "
    "and background information provided. Output strictly one label: 'liberal', 'moderate', or 'conservative'."
)

QUESTION_OLD = "Based on these characteristics, which political group would I most likely identify with?"
QUESTION_NEW = "Based on these characteristics, which political group would this respondent most likely identify with?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Base-model Memorization Probe (No-Year / True-Year / Shuffled-Year)")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--task", type=str, default="ideology", choices=["party_choice", "ideology"])
    parser.add_argument("--data_dir", type=str, default="dataset_test/")
    parser.add_argument("--out_dir", type=str, default="./results_memorization")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--length_normalize_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn_implementation", type=str, default="sdpa",
                         choices=["sdpa", "flash_attention_2", "eager"])
    return parser.parse_args()


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
    model.config.use_cache = True
    return model


def normalize_label(value: Any, candidates: List[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    for c in candidates:
        if c.lower() in text:
            return c
    return None


def extract_label(messages: list[dict[str, str]], candidates: List[str]) -> Optional[str]:
    for msg in messages:
        if msg.get("role") == "assistant":
            return normalize_label(msg.get("content"), candidates)
    return None


def first_user_content(messages: list[dict[str, str]]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return messages[0].get("content", "") if messages else ""


def reframe_third_person(user_text: str) -> str:
    """Switch the closing question from first-person ('I') to third-person ('this respondent')."""
    text = user_text.strip()
    if QUESTION_OLD in text:
        text = text.replace(QUESTION_OLD, QUESTION_NEW)
    return text


def build_memorization_user_text(user_text: str, year_tag: Optional[int]) -> str:
    """
    year_tag=None      -> no-year condition
    year_tag=<int>      -> true-year or shuffled-year condition (same mechanic either way;
                            which one it is depends on whether year_tag == the row's real test_year)
    """
    demo_text = reframe_third_person(user_text)
    if year_tag is None:
        prefix = "The following characteristics describe a respondent: "
    else:
        prefix = f"The following characteristics describe a respondent surveyed in {year_tag}: "
    return prefix + demo_text


def build_prompt(tokenizer: AutoTokenizer, system_text: str, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True,
    )


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


def build_condition_entries(rows, tokenizer, candidates, system_text, year_tag):
    """Returns list of (idx, gt, user_text_used, prompt) for a given condition."""
    entries = []
    for idx, entry in enumerate(rows):
        messages = entry.get("messages", [])
        gt = extract_label(messages, candidates)
        if gt is None:
            continue
        raw_user_text = first_user_content(messages)
        modified_text = build_memorization_user_text(raw_user_text, year_tag)
        prompt = build_prompt(tokenizer, system_text, modified_text)
        entries.append((idx, gt, modified_text, prompt))
    return entries


def evaluate_condition(model, tokenizer, entries, candidates, args,
                        condition_name, test_year, prompt_year_tag):
    cids = get_candidate_token_ids(tokenizer, candidates)
    rows = []

    for start in tqdm(range(0, len(entries), args.eval_batch_size), desc=f"Eval {condition_name} (test_year={test_year})"):
        batch = entries[start: start + args.eval_batch_size]
        prompts = [e[3] for e in batch]
        probs = score_candidates_batched(model, tokenizer, prompts, cids, candidates,
                                          args.max_len, args.length_normalize_eval)

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
    if not df.empty:
        df["condition"] = condition_name          # "no_year" / "true_year" / "shuffled_year"
        df["test_year"] = test_year                # the real source year of the demographic data
        df["prompt_year_tag"] = prompt_year_tag     # year mentioned in prompt (None for no_year)
        df["is_memorization_clean"] = (
            condition_name != "shuffled_year"
            or not (test_year == 2020 and prompt_year_tag == 2024)  # flag the non-clean direction
        )
    return df


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = TASK_CONFIGS[args.task]
    candidates = cfg["candidates"]

    print("Loading ANES data...")
    data_by_year = {}
    for year in YEARS:
        path = os.path.join(args.data_dir, f"anes_{year}_more.jsonl")
        if os.path.exists(path):
            data_by_year[year] = load_jsonl(path)
            print(f"  {year}: {len(data_by_year[year])}")
        else:
            print(f"  WARNING: {path} not found!")

    print("\nLoading tokenizer & base model...")
    tokenizer = load_tokenizer(args.model_name)
    base_model = load_qlora_base_model(args.model_name, tokenizer, args.attn_implementation)
    base_model.eval()

    all_dfs = []

    for year in YEARS:
        if year not in data_by_year:
            continue
        rows = data_by_year[year]

        # --- Condition A: no year mentioned ---
        entries_a = build_condition_entries(rows, tokenizer, candidates, SYSTEM_TEXT, year_tag=None)
        df_a = evaluate_condition(base_model, tokenizer, entries_a, candidates, args,
                                   "no_year", year, None)
        all_dfs.append(df_a)

        # --- Condition B: true year mentioned ---
        entries_b = build_condition_entries(rows, tokenizer, candidates, SYSTEM_TEXT, year_tag=year)
        df_b = evaluate_condition(base_model, tokenizer, entries_b, candidates, args,
                                   "true_year", year, year)
        all_dfs.append(df_b)

        # --- Condition C: shuffled (wrong) year mentioned ---
        if year in SHUFFLE_PAIRS:
            fake_year = SHUFFLE_PAIRS[year]
            entries_c = build_condition_entries(rows, tokenizer, candidates, SYSTEM_TEXT, year_tag=fake_year)
            df_c = evaluate_condition(base_model, tokenizer, entries_c, candidates, args,
                                       "shuffled_year", year, fake_year)
            all_dfs.append(df_c)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["model_name"] = args.model_name
    combined["task"] = args.task

    out_path = os.path.join(
        args.out_dir,
        f"predictions_memorization_{args.task}_{args.model_name.split('/')[-1]}.csv"
    )
    combined.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()
