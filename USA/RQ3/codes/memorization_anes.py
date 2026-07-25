import argparse
import json
import os
from typing import Any, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------- Configuration ----------
YEARS = [2008, 2012, 2016, 2020, 2024]

SHUFFLE_FILES = {
    # (source_year, fake_year_tag) -> filename stem
    (2012, 2016): "anes_2012_shuffled_2016",
    (2016, 2012): "anes_2016_shuffled_2012",
    (2020, 2024): "anes_2020_shuffled_2024",
    (2024, 2020): "anes_2024_shuffled_2020",
}

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
            "Democrat\nRepublican\nOther\n\n"
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
            "liberal\nmoderate\nconservative\n\n"
            "Important:\n"
            "- Base your decision on typical voting patterns and demographics.\n"
            "- Do NOT explain your reasoning.\n"
            "- Do NOT repeat the input.\n"
            "- Output ONLY the label."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Base-Model Memorization Probe (No-Year / True-Year / Shuffled-Year)")
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


def build_prompt(tokenizer: AutoTokenizer, system_text: str, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True)


def prepare_eval_entries(rows, tokenizer, candidates, system_text) -> list[Tuple[int, Optional[str], str, str]]:
    """
    Returns (idx, gt, user_text, prompt).
    gt may be None for shuffled-year rows that have no assistant label
    (test-time-only prompts); those rows are still evaluated, just without
    an accuracy/kappa contribution.
    """
    entries = []
    for idx, entry in enumerate(rows):
        messages = entry.get("messages", [])
        gt = extract_label(messages, candidates)  # None if no assistant turn present
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
                                   run_name, condition, test_year, prompt_year_tag):
    """Same output-row format/fields as your existing evaluate_and_save_predictions,
    plus condition/test_year/prompt_year_tag metadata columns."""
    model.eval()
    model.config.use_cache = True

    cids = get_candidate_token_ids(tokenizer, candidates)
    rows = []

    for start in tqdm(range(0, len(eval_entries), args.eval_batch_size), desc=f"Eval {run_name}"):
        batch = eval_entries[start: start + args.eval_batch_size]
        prompts = [e[3] for e in batch]
        probs = score_candidates_batched(model, tokenizer, prompts, cids, candidates,
                                          args.max_len, args.length_normalize_eval)

        for (idx, gt, user_text, _), p in zip(batch, probs):
            pred = max(p, key=p.get)
            sp = sorted(p.items(), key=lambda x: x[1], reverse=True)
            rows.append({
                "idx": idx,
                "gt": gt,  # may be None for shuffled-year test-only rows
                "pred": pred,
                "correct": int(gt == pred) if gt is not None else None,
                "confidence": sp[0][1],
                "margin": sp[0][1] - (sp[1][1] if len(sp) > 1 else 0),
                "top2_label": sp[1][0] if len(sp) > 1 else None,
                "top2_prob": sp[1][1] if len(sp) > 1 else 0,
                "user_text": user_text,
                **{f"prob_{c}": p[c] for c in candidates},
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["run_name"] = run_name
        df["condition"] = condition               # "no_year" / "true_year" / "shuffled_year"
        df["test_year"] = test_year                # real source year of the demographic data
        df["prompt_year_tag"] = prompt_year_tag     # year mentioned in the prompt (None for no_year)
        df["is_base_model"] = True
        df["model_name"] = args.model_name
        df["task"] = args.task

    model.config.use_cache = False
    return df


def compute_metrics(df: pd.DataFrame, candidates: List[str], condition_name: str) -> dict:
    """Compute comprehensive metrics for a dataframe"""
    metrics = {}
    
    # Filter rows with ground truth
    eval_df = df[df['gt'].notna()].copy()
    
    if len(eval_df) == 0:
        metrics['n_samples'] = 0
        metrics['message'] = 'No ground truth labels available'
        return metrics
    
    y_true = eval_df['gt'].tolist()
    y_pred = eval_df['pred'].tolist()
    
    # Basic metrics
    metrics['n_samples'] = len(eval_df)
    metrics['n_total'] = len(df)
    metrics['n_no_gt'] = len(df) - len(eval_df)
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 (macro and weighted)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    metrics['precision_weighted'] = precision_weighted
    metrics['recall_weighted'] = recall_weighted
    metrics['f1_weighted'] = f1_weighted
    
    # Per-class metrics
    per_class = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0, labels=candidates)
    for i, label in enumerate(candidates):
        metrics[f'precision_{label}'] = per_class[0][i]
        metrics[f'recall_{label}'] = per_class[1][i]
        metrics[f'f1_{label}'] = per_class[2][i]
        metrics[f'support_{label}'] = per_class[3][i]
    
    # Cohen's Kappa
    try:
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
    except:
        metrics['kappa'] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=candidates)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Confidence statistics
    metrics['mean_confidence'] = eval_df['confidence'].mean()
    metrics['std_confidence'] = eval_df['confidence'].std()
    metrics['mean_margin'] = eval_df['margin'].mean()
    metrics['std_margin'] = eval_df['margin'].std()
    
    # Accuracy by confidence threshold
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for thresh in thresholds:
        high_conf = eval_df[eval_df['confidence'] >= thresh]
        if len(high_conf) > 0:
            metrics[f'accuracy_conf_{thresh}'] = accuracy_score(
                high_conf['gt'].tolist(), 
                high_conf['pred'].tolist()
            )
            metrics[f'n_conf_{thresh}'] = len(high_conf)
        else:
            metrics[f'accuracy_conf_{thresh}'] = None
            metrics[f'n_conf_{thresh}'] = 0
    
    return metrics


def print_metrics(metrics: dict, condition_name: str, year: Optional[int] = None, 
                  shuffle_info: Optional[str] = None):
    """Pretty print metrics"""
    if metrics.get('n_samples', 0) == 0:
        print(f"  No metrics available (no ground truth labels)")
        return
    
    title = f"{condition_name}"
    if year is not None:
        title += f" (Year: {year})"
    if shuffle_info is not None:
        title += f" {shuffle_info}"
    
    print(f"\n  {'='*50}")
    print(f"  {title}")
    print(f"  {'='*50}")
    print(f"  Samples: {metrics['n_samples']} (total: {metrics['n_total']}, no GT: {metrics['n_no_gt']})")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Kappa: {metrics['kappa']:.4f}" if metrics['kappa'] is not None else "  Kappa: N/A")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")
    print(f"\n  Per-class performance:")
    for label in [c for c in ['Democrat', 'Republican', 'Other', 'liberal', 'moderate', 'conservative'] 
                  if label in metrics]:
        if f'precision_{label}' in metrics:
            print(f"    {label:12s} | P: {metrics[f'precision_{label}']:.4f} | "
                  f"R: {metrics[f'recall_{label}']:.4f} | "
                  f"F1: {metrics[f'f1_{label}']:.4f} | "
                  f"N: {metrics[f'support_{label}']}")
    print(f"\n  Confidence: mean={metrics['mean_confidence']:.4f}, "
          f"std={metrics['std_confidence']:.4f}")
    print(f"  Margin: mean={metrics['mean_margin']:.4f}, "
          f"std={metrics['std_margin']:.4f}")


def save_dataframe(df, args, suffix):
    """Helper function to save a single dataframe"""
    if df.empty:
        print(f"Warning: No data to save for {suffix}")
        return
    
    out_path = os.path.join(
        args.out_dir,
        f"predictions_memorization_{args.task}_{args.model_name.split('/')[-1]}_{suffix}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df)} rows)")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = TASK_CONFIGS[args.task]
    candidates, system_text = cfg["candidates"], cfg["system_text"]

    print("\nLoading tokenizer & base model...")
    tokenizer = load_tokenizer(args.model_name)
    base_model = load_qlora_base_model(args.model_name, tokenizer, args.attn_implementation)
    base_model.eval()

    # Separate dataframes and metrics for each condition
    no_year_dfs = []
    true_year_dfs = []
    shuffled_year_dfs = []
    
    no_year_metrics = []
    true_year_metrics = []
    shuffled_year_metrics = []

    # ---------- Condition A: no year ----------
    print("\n" + "=" * 70)
    print("CONDITION: NO YEAR")
    print("=" * 70)
    for year in YEARS:
        path = os.path.join(args.data_dir, f"anes_{year}_without_year.jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        rows = load_jsonl(path)
        entries = prepare_eval_entries(rows, tokenizer, candidates, system_text)
        print(f"\n--- No-year → Test: {year} ({len(entries)}) ---")
        df = evaluate_and_save_predictions(
            base_model, tokenizer, entries, candidates, args,
            f"no_year_{year}", "no_year", year, None)
        if not df.empty:
            no_year_dfs.append(df)
            metrics = compute_metrics(df, candidates, f"no_year_{year}")
            no_year_metrics.append((year, metrics))
            print_metrics(metrics, "No Year", year=year)

    # ---------- Condition B: true year ----------
    print("\n" + "=" * 70)
    print("CONDITION: TRUE YEAR")
    print("=" * 70)
    for year in YEARS:
        path = os.path.join(args.data_dir, f"anes_{year}_with_year.jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        rows = load_jsonl(path)
        entries = prepare_eval_entries(rows, tokenizer, candidates, system_text)
        print(f"\n--- True-year → Test: {year} ({len(entries)}) ---")
        df = evaluate_and_save_predictions(
            base_model, tokenizer, entries, candidates, args,
            f"true_year_{year}", "true_year", year, year)
        if not df.empty:
            true_year_dfs.append(df)
            metrics = compute_metrics(df, candidates, f"true_year_{year}")
            true_year_metrics.append((year, metrics))
            print_metrics(metrics, "True Year", year=year)

    # ---------- Condition C: shuffled year ----------
    print("\n" + "=" * 70)
    print("CONDITION: SHUFFLED YEAR")
    print("=" * 70)
    for (source_year, fake_year), file_stem in SHUFFLE_FILES.items():
        path = os.path.join(args.data_dir, f"{file_stem}.jsonl")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping.")
            continue
        rows = load_jsonl(path)
        entries = prepare_eval_entries(rows, tokenizer, candidates, system_text)
        shuffle_info = f"{source_year} tagged as {fake_year}"
        print(f"\n--- Shuffled: {shuffle_info} ({len(entries)}) ---")
        df = evaluate_and_save_predictions(
            base_model, tokenizer, entries, candidates, args,
            f"shuffled_{source_year}_as_{fake_year}", "shuffled_year", source_year, fake_year)
        if not df.empty:
            shuffled_year_dfs.append(df)
            metrics = compute_metrics(df, candidates, f"shuffled_{source_year}_as_{fake_year}")
            shuffled_year_metrics.append(((source_year, fake_year), metrics))
            print_metrics(metrics, "Shuffled Year", shuffle_info=shuffle_info)

    # ---------- Save Summary Metrics ----------
    print("\n" + "=" * 70)
    print("SUMMARY METRICS")
    print("=" * 70)
    
    # Create summary DataFrames
    summary_rows = []
    
    # No Year summary
    if no_year_metrics:
        for year, metrics in no_year_metrics:
            summary_rows.append({
                'condition': 'no_year',
                'year': year,
                'prompt_year_tag': None,
                'accuracy': metrics.get('accuracy'),
                'kappa': metrics.get('kappa'),
                'f1_macro': metrics.get('f1_macro'),
                'f1_weighted': metrics.get('f1_weighted'),
                'n_samples': metrics.get('n_samples'),
                'n_total': metrics.get('n_total'),
                'n_no_gt': metrics.get('n_no_gt'),
                'mean_confidence': metrics.get('mean_confidence'),
                'mean_margin': metrics.get('mean_margin'),
            })
    
    # True Year summary
    if true_year_metrics:
        for year, metrics in true_year_metrics:
            summary_rows.append({
                'condition': 'true_year',
                'year': year,
                'prompt_year_tag': year,
                'accuracy': metrics.get('accuracy'),
                'kappa': metrics.get('kappa'),
                'f1_macro': metrics.get('f1_macro'),
                'f1_weighted': metrics.get('f1_weighted'),
                'n_samples': metrics.get('n_samples'),
                'n_total': metrics.get('n_total'),
                'n_no_gt': metrics.get('n_no_gt'),
                'mean_confidence': metrics.get('mean_confidence'),
                'mean_margin': metrics.get('mean_margin'),
            })
    
    # Shuffled Year summary
    if shuffled_year_metrics:
        for (source_year, fake_year), metrics in shuffled_year_metrics:
            summary_rows.append({
                'condition': 'shuffled_year',
                'year': source_year,
                'prompt_year_tag': fake_year,
                'accuracy': metrics.get('accuracy'),
                'kappa': metrics.get('kappa'),
                'f1_macro': metrics.get('f1_macro'),
                'f1_weighted': metrics.get('f1_weighted'),
                'n_samples': metrics.get('n_samples'),
                'n_total': metrics.get('n_total'),
                'n_no_gt': metrics.get('n_no_gt'),
                'mean_confidence': metrics.get('mean_confidence'),
                'mean_margin': metrics.get('mean_margin'),
            })
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(
            args.out_dir,
            f"summary_metrics_{args.task}_{args.model_name.split('/')[-1]}.csv"
        )
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary metrics: {summary_path}")
        
        # Print overall summary
        print("\nOverall Summary:")
        print(summary_df.to_string(index=False))
    else:
        print("\nNo metrics to summarize")

    # ---------- Save Three Separate CSV Files ----------
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save No Year data
    if no_year_dfs:
        no_year_combined = pd.concat(no_year_dfs, ignore_index=True)
        save_dataframe(no_year_combined, args, "no_year")
        print(f"  No-year total rows: {len(no_year_combined)}")
    else:
        print("  No-year: No data to save")
    
    # Save True Year data
    if true_year_dfs:
        true_year_combined = pd.concat(true_year_dfs, ignore_index=True)
        save_dataframe(true_year_combined, args, "true_year")
        print(f"  True-year total rows: {len(true_year_combined)}")
    else:
        print("  True-year: No data to save")
    
    # Save Shuffled Year data
    if shuffled_year_dfs:
        shuffled_year_combined = pd.concat(shuffled_year_dfs, ignore_index=True)
        save_dataframe(shuffled_year_combined, args, "shuffled_year")
        print(f"  Shuffled-year total rows: {len(shuffled_year_combined)}")
    else:
        print("  Shuffled-year: No data to save")
    
    # Optional: Save combined file as well
    all_dfs = []
    if no_year_dfs:
        all_dfs.append(pd.concat(no_year_dfs, ignore_index=True))
    if true_year_dfs:
        all_dfs.append(pd.concat(true_year_dfs, ignore_index=True))
    if shuffled_year_dfs:
        all_dfs.append(pd.concat(shuffled_year_dfs, ignore_index=True))
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_path = os.path.join(
            args.out_dir,
            f"predictions_memorization_{args.task}_{args.model_name.split('/')[-1]}_combined.csv"
        )
        combined.to_csv(out_path, index=False)
        print(f"\nSaved combined file: {out_path}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()


# import argparse
# import json
# import os
# from typing import Any, List, Optional, Tuple

# import pandas as pd
# import torch
# import torch.nn.functional as F
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# # ---------- Configuration ----------
# YEARS = [2008, 2012, 2016, 2020, 2024]

# SHUFFLE_FILES = {
#     # (source_year, fake_year_tag) -> filename stem
#     (2012, 2016): "anes_2012_shuffled_2016",
#     (2016, 2012): "anes_2016_shuffled_2012",
#     (2020, 2024): "anes_2020_shuffled_2024",
#     (2024, 2020): "anes_2024_shuffled_2020",
# }

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
#             "Democrat\nRepublican\nOther\n\n"
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
#             "liberal\nmoderate\nconservative\n\n"
#             "Important:\n"
#             "- Base your decision on typical voting patterns and demographics.\n"
#             "- Do NOT explain your reasoning.\n"
#             "- Do NOT repeat the input.\n"
#             "- Output ONLY the label."
#         ),
#     },
# }


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Base-Model Memorization Probe (No-Year / True-Year / Shuffled-Year)")
#     parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
#     parser.add_argument("--task", type=str, default="ideology", choices=["party_choice", "ideology"])
#     parser.add_argument("--data_dir", type=str, default="dataset_test/")
#     parser.add_argument("--out_dir", type=str, default="./results_memorization")
#     parser.add_argument("--max_len", type=int, default=256)
#     parser.add_argument("--eval_batch_size", type=int, default=32)
#     parser.add_argument("--length_normalize_eval", action=argparse.BooleanOptionalAction, default=True)
#     parser.add_argument("--attn_implementation", type=str, default="sdpa",
#                          choices=["sdpa", "flash_attention_2", "eager"])
#     return parser.parse_args()


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
#     model.config.use_cache = True
#     return model


# def normalize_label(value: Any, candidates: List[str]) -> Optional[str]:
#     if value is None:
#         return None
#     text = str(value).strip().lower()
#     for c in candidates:
#         if c.lower() in text:
#             return c
#     return None


# def extract_label(messages: list[dict[str, str]], candidates: List[str]) -> Optional[str]:
#     for msg in messages:
#         if msg.get("role") == "assistant":
#             return normalize_label(msg.get("content"), candidates)
#     return None


# def first_user_content(messages: list[dict[str, str]]) -> str:
#     for msg in messages:
#         if msg.get("role") == "user":
#             return msg.get("content", "")
#     return messages[0].get("content", "") if messages else ""


# def build_prompt(tokenizer: AutoTokenizer, system_text: str, user_text: str) -> str:
#     return tokenizer.apply_chat_template(
#         [{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
#         tokenize=False, add_generation_prompt=True)


# def prepare_eval_entries(rows, tokenizer, candidates, system_text) -> list[Tuple[int, Optional[str], str, str]]:
#     """
#     Returns (idx, gt, user_text, prompt).
#     gt may be None for shuffled-year rows that have no assistant label
#     (test-time-only prompts); those rows are still evaluated, just without
#     an accuracy/kappa contribution.
#     """
#     entries = []
#     for idx, entry in enumerate(rows):
#         messages = entry.get("messages", [])
#         gt = extract_label(messages, candidates)  # None if no assistant turn present
#         entries.append((idx, gt, first_user_content(messages),
#                        build_prompt(tokenizer, system_text, first_user_content(messages))))
#     return entries


# def get_candidate_token_ids(tokenizer, candidates) -> list[list[int]]:
#     return [tokenizer(c, add_special_tokens=False)["input_ids"] for c in candidates]


# @torch.inference_mode()
# def score_candidates_batched(model, tokenizer, prompts, candidate_token_ids, candidates, max_len, length_normalize):
#     if not prompts:
#         return []
#     device = next(model.parameters()).device
#     encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
#                         max_length=max_len, add_special_tokens=False)
#     p_ids = encoded["input_ids"]
#     p_mask = encoded["attention_mask"]
#     p_lens = p_mask.sum(dim=1).tolist()

#     all_seqs, all_p_lens, all_c_lens = [], [], []
#     for row_idx, plen in enumerate(p_lens):
#         ptoks = p_ids[row_idx, :plen].tolist()
#         for cids in candidate_token_ids:
#             max_p = max_len - len(cids)
#             if max_p <= 0:
#                 raise ValueError("Candidate too long")
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
#         tok_scores = [log_probs[i, plen - 1 + off, input_ids[i, plen + off]] for off in range(clen)]
#         sc = torch.stack(tok_scores).sum()
#         if length_normalize:
#             sc = sc / clen
#         scores.append(sc)

#     score_tensor = torch.stack(scores).view(len(prompts), len(candidates))
#     prob_tensor = torch.softmax(score_tensor, dim=1).detach().cpu().numpy()
#     return [{candidates[i]: float(row[i]) for i in range(len(candidates))} for row in prob_tensor]


# def evaluate_and_save_predictions(model, tokenizer, eval_entries, candidates, args,
#                                    run_name, condition, test_year, prompt_year_tag):
#     """Same output-row format/fields as your existing evaluate_and_save_predictions,
#     plus condition/test_year/prompt_year_tag metadata columns."""
#     model.eval()
#     model.config.use_cache = True

#     cids = get_candidate_token_ids(tokenizer, candidates)
#     rows = []

#     for start in tqdm(range(0, len(eval_entries), args.eval_batch_size), desc=f"Eval {run_name}"):
#         batch = eval_entries[start: start + args.eval_batch_size]
#         prompts = [e[3] for e in batch]
#         probs = score_candidates_batched(model, tokenizer, prompts, cids, candidates,
#                                           args.max_len, args.length_normalize_eval)

#         for (idx, gt, user_text, _), p in zip(batch, probs):
#             pred = max(p, key=p.get)
#             sp = sorted(p.items(), key=lambda x: x[1], reverse=True)
#             rows.append({
#                 "idx": idx,
#                 "gt": gt,  # may be None for shuffled-year test-only rows
#                 "pred": pred,
#                 "correct": int(gt == pred) if gt is not None else None,
#                 "confidence": sp[0][1],
#                 "margin": sp[0][1] - (sp[1][1] if len(sp) > 1 else 0),
#                 "top2_label": sp[1][0] if len(sp) > 1 else None,
#                 "top2_prob": sp[1][1] if len(sp) > 1 else 0,
#                 "user_text": user_text,
#                 **{f"prob_{c}": p[c] for c in candidates},
#             })

#     df = pd.DataFrame(rows)
#     if not df.empty:
#         df["run_name"] = run_name
#         df["condition"] = condition               # "no_year" / "true_year" / "shuffled_year"
#         df["test_year"] = test_year                # real source year of the demographic data
#         df["prompt_year_tag"] = prompt_year_tag     # year mentioned in the prompt (None for no_year)
#         df["is_base_model"] = True
#         df["model_name"] = args.model_name
#         df["task"] = args.task

#     model.config.use_cache = False
#     return df


# def main():
#     args = parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)

#     cfg = TASK_CONFIGS[args.task]
#     candidates, system_text = cfg["candidates"], cfg["system_text"]

#     print("\nLoading tokenizer & base model...")
#     tokenizer = load_tokenizer(args.model_name)
#     base_model = load_qlora_base_model(args.model_name, tokenizer, args.attn_implementation)
#     base_model.eval()

#     all_dfs = []

#     # ---------- Condition A: no year ----------
#     print("\n" + "=" * 70)
#     print("CONDITION: NO YEAR")
#     print("=" * 70)
#     for year in YEARS:
#         path = os.path.join(args.data_dir, f"anes_{year}_without_year.jsonl")
#         if not os.path.exists(path):
#             print(f"  WARNING: {path} not found, skipping.")
#             continue
#         rows = load_jsonl(path)
#         entries = prepare_eval_entries(rows, tokenizer, candidates, system_text)
#         print(f"\n--- No-year → Test: {year} ({len(entries)}) ---")
#         df = evaluate_and_save_predictions(
#             base_model, tokenizer, entries, candidates, args,
#             f"no_year_{year}", "no_year", year, None)
#         all_dfs.append(df)

#     # ---------- Condition B: true year ----------
#     print("\n" + "=" * 70)
#     print("CONDITION: TRUE YEAR")
#     print("=" * 70)
#     for year in YEARS:
#         path = os.path.join(args.data_dir, f"anes_{year}_with_year.jsonl")
#         if not os.path.exists(path):
#             print(f"  WARNING: {path} not found, skipping.")
#             continue
#         rows = load_jsonl(path)
#         entries = prepare_eval_entries(rows, tokenizer, candidates, system_text)
#         print(f"\n--- True-year → Test: {year} ({len(entries)}) ---")
#         df = evaluate_and_save_predictions(
#             base_model, tokenizer, entries, candidates, args,
#             f"true_year_{year}", "true_year", year, year)
#         all_dfs.append(df)

#     # ---------- Condition C: shuffled year ----------
#     print("\n" + "=" * 70)
#     print("CONDITION: SHUFFLED YEAR")
#     print("=" * 70)
#     for (source_year, fake_year), file_stem in SHUFFLE_FILES.items():
#         path = os.path.join(args.data_dir, f"{file_stem}.jsonl")
#         if not os.path.exists(path):
#             print(f"  WARNING: {path} not found, skipping.")
#             continue
#         rows = load_jsonl(path)
#         entries = prepare_eval_entries(rows, tokenizer, candidates, system_text)
#         print(f"\n--- Shuffled: {source_year} tagged as {fake_year} ({len(entries)}) ---")
#         df = evaluate_and_save_predictions(
#             base_model, tokenizer, entries, candidates, args,
#             f"shuffled_{source_year}_as_{fake_year}", "shuffled_year", source_year, fake_year)
#         all_dfs.append(df)

#     combined = pd.concat(all_dfs, ignore_index=True)
#     out_path = os.path.join(
#         args.out_dir,
#         f"predictions_memorization_{args.task}_{args.model_name.split('/')[-1]}.csv"
#     )
#     combined.to_csv(out_path, index=False)
#     print(f"\nSaved: {out_path}  ({len(combined)} rows)")


# if __name__ == "__main__":
#     main()
