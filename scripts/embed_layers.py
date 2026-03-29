#!/usr/bin/env python3
import argparse
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    r2_score,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import spearmanr, pearsonr


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x)


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_layers(layer_str: str, num_layers: Optional[int] = None) -> List[int]:
    s = layer_str.strip().lower()
    if s == "all":
        if num_layers is None:
            raise ValueError("num_layers required for layer_str='all'")
        return list(range(num_layers + 1))  # hidden_states includes embedding layer at 0
    if s.startswith("last"):
        if num_layers is None:
            raise ValueError("num_layers required for layer_str='lastK'")
        k = int(s.replace("last", ""))
        start = max(0, num_layers + 1 - k)
        return list(range(start, num_layers + 1))
    return [int(x) for x in layer_str.split(",") if x.strip()]


def build_full_texts(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    full_text = QUESTION + RESPONSE
    prefix_text = QUESTION prefix only, used to identify response span
    """
    if "question" not in df.columns:
        raise ValueError("Input CSV must contain a 'question' column.")
    if "response" not in df.columns:
        raise ValueError("Input CSV must contain a 'response' column for this experiment.")

    full_texts = []
    question_prefixes = []

    for q, r in zip(df["question"].tolist(), df["response"].tolist()):
        q = safe_str(q)
        r = safe_str(r)

        prefix = f"QUESTION:\n{q}\n\nRESPONSE:\n"
        full = prefix + r

        question_prefixes.append(prefix)
        full_texts.append(full)

    return full_texts, question_prefixes


@dataclass
class ExampleIndices:
    q_last_idx: int
    resp_token_indices: List[int]
    resp_early_idx: int
    resp_mid_idx: int
    resp_last_idx: int
    full_len: int
    prefix_len: int


def find_segment_indices(
    tokenizer,
    prefix_text: str,
    full_text: str,
    max_length: int,
) -> Tuple[Dict[str, torch.Tensor], ExampleIndices]:
    full_enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )
    prefix_enc = tokenizer(
        prefix_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    full_attn = full_enc["attention_mask"][0]
    prefix_attn = prefix_enc["attention_mask"][0]

    full_len = int(full_attn.sum().item())
    prefix_len = int(prefix_attn.sum().item())

    q_last_idx = max(0, min(prefix_len - 1, full_len - 1))

    resp_start = min(prefix_len, full_len - 1)
    resp_token_indices = list(range(resp_start, full_len))
    if len(resp_token_indices) == 0:
        resp_token_indices = [full_len - 1]

    def frac_idx(frac: float) -> int:
        pos = int(round(frac * (len(resp_token_indices) - 1)))
        pos = max(0, min(pos, len(resp_token_indices) - 1))
        return resp_token_indices[pos]

    idxs = ExampleIndices(
        q_last_idx=q_last_idx,
        resp_token_indices=resp_token_indices,
        resp_early_idx=frac_idx(0.1),
        resp_mid_idx=frac_idx(0.5),
        resp_last_idx=resp_token_indices[-1],
        full_len=full_len,
        prefix_len=prefix_len,
    )
    return full_enc, idxs


def decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")
    except Exception:
        return "<decode_error>"


def print_sanity_checks(
    df: pd.DataFrame,
    tokenizer,
    full_texts: List[str],
    question_prefixes: List[str],
    max_length: int,
    n_examples: int = 5,
) -> None:
    print("\n" + "=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    print("\n[1] Target distributions")
    print("\ncorrect value counts:")
    print(df["correct"].value_counts(dropna=False).sort_index())

    print("\nconsistency describe:")
    print(df["consistency"].describe())

    print("\nTop rounded consistency values:")
    print(df["consistency"].round(4).value_counts().head(20))

    print("\n[2] Token span checks")
    n_examples = min(n_examples, len(full_texts))
    for i in range(n_examples):
        full_enc, idxs = find_segment_indices(
            tokenizer=tokenizer,
            prefix_text=question_prefixes[i],
            full_text=full_texts[i],
            max_length=max_length,
        )
        input_ids = full_enc["input_ids"][0].tolist()

        q_tok = decode_token(tokenizer, input_ids[idxs.q_last_idx])
        re_tok = decode_token(tokenizer, input_ids[idxs.resp_early_idx])
        rm_tok = decode_token(tokenizer, input_ids[idxs.resp_mid_idx])
        rl_tok = decode_token(tokenizer, input_ids[idxs.resp_last_idx])

        print("\n" + "-" * 80)
        print(f"example idx: {i}")
        print(f"id: {df.iloc[i]['id']}")
        print(f"question[:120]: {safe_str(df.iloc[i]['question'])[:120]!r}")
        print(f"response[:200]: {safe_str(df.iloc[i]['response'])[:200]!r}")
        print(f"full_len={idxs.full_len}, prefix_len={idxs.prefix_len}, n_resp={len(idxs.resp_token_indices)}")
        print(f"q_last_idx={idxs.q_last_idx}, token={q_tok!r}")
        print(f"resp_early_idx={idxs.resp_early_idx}, token={re_tok!r}")
        print(f"resp_mid_idx={idxs.resp_mid_idx}, token={rm_tok!r}")
        print(f"resp_last_idx={idxs.resp_last_idx}, token={rl_tok!r}")
        if idxs.full_len >= max_length:
            print("WARNING: sequence hit max_length truncation")

    print("\n" + "=" * 80)
    print("END SANITY CHECKS")
    print("=" * 80 + "\n")


@torch.no_grad()
def extract_layerwise_reps(
    model,
    tokenizer,
    full_texts: List[str],
    question_prefixes: List[str],
    layers: List[int],
    max_length: int,
    device: str,
) -> Dict[str, np.ndarray]:
    """
    Returns reps[rep_name] = array of shape [N, L, D]
    where L = len(layers).

    Included representations:
      - full_last: last token of the whole input (baseline matching vLLM-style last-token embedding)
      - full_mean: mean over all valid tokens in the whole input
      - q_last: last token of question/prefix segment
      - resp_early: early response token
      - resp_mid: middle response token
      - resp_last: final response token
      - resp_mean: mean over response tokens only
    """
    n = len(full_texts)
    hidden_size = model.config.hidden_size
    num_sel_layers = len(layers)

    out = {
        "full_last": np.zeros((n, num_sel_layers, hidden_size), dtype=np.float32),
        "full_mean": np.zeros((n, num_sel_layers, hidden_size), dtype=np.float32),
        "q_last": np.zeros((n, num_sel_layers, hidden_size), dtype=np.float32),
        "resp_early": np.zeros((n, num_sel_layers, hidden_size), dtype=np.float32),
        "resp_mid": np.zeros((n, num_sel_layers, hidden_size), dtype=np.float32),
        "resp_last": np.zeros((n, num_sel_layers, hidden_size), dtype=np.float32),
        "resp_mean": np.zeros((n, num_sel_layers, hidden_size), dtype=np.float32),
    }

    iterator = zip(full_texts, question_prefixes)
    for i, (full_text, prefix_text) in enumerate(
        tqdm(list(iterator), total=n, desc="Extracting hidden states")
    ):
        enc, idxs = find_segment_indices(
            tokenizer=tokenizer,
            prefix_text=prefix_text,
            full_text=full_text,
            max_length=max_length,
        )

        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(
            **enc,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states  # tuple length = num_layers + 1

        valid_indices = list(range(idxs.full_len))

        for li, layer_idx in enumerate(layers):
            hs = hidden_states[layer_idx][0]  # [T, D]

            out["full_last"][i, li] = hs[idxs.full_len - 1].float().cpu().numpy()
            out["full_mean"][i, li] = hs[valid_indices].mean(dim=0).float().cpu().numpy()

            out["q_last"][i, li] = hs[idxs.q_last_idx].float().cpu().numpy()
            out["resp_early"][i, li] = hs[idxs.resp_early_idx].float().cpu().numpy()
            out["resp_mid"][i, li] = hs[idxs.resp_mid_idx].float().cpu().numpy()
            out["resp_last"][i, li] = hs[idxs.resp_last_idx].float().cpu().numpy()
            out["resp_mean"][i, li] = hs[idxs.resp_token_indices].mean(dim=0).float().cpu().numpy()

    return out


def evaluate_binary_probe(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
) -> Dict[str, float]:
    unique = np.unique(y)
    if len(unique) < 2:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        C=1.0,
        max_iter=5000,
        random_state=seed,
    )
    clf.fit(X_train_s, y_train)
    p = clf.predict_proba(X_test_s)[:, 1]

    return {
        "auroc": roc_auc_score(y_test, p),
        "auprc": average_precision_score(y_test, p),
        "brier": brier_score_loss(y_test, p),
    }


def evaluate_regression_probe(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
) -> Dict[str, float]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    reg = Ridge(alpha=1.0, random_state=seed)
    reg.fit(X_train_s, y_train)
    pred = reg.predict(X_test_s)
    pred_clip = np.clip(pred, 0.0, 1.0)

    baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=np.float64)

    pearson = np.nan
    if len(y_test) > 1 and np.std(pred) > 0 and np.std(y_test) > 0:
        pearson = pearsonr(y_test, pred)[0]

    try:
        spearman = spearmanr(y_test, pred).statistic
    except Exception:
        spearman = np.nan

    return {
        "r2": r2_score(y_test, pred),
        "rmse": math.sqrt(mean_squared_error(y_test, pred)),
        "rmse_clipped": math.sqrt(mean_squared_error(y_test, pred_clip)),
        "baseline_rmse": math.sqrt(mean_squared_error(y_test, baseline_pred)),
        "pearson": pearson,
        "spearman": spearman,
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "y_mean": float(np.mean(y_test)),
        "y_std": float(np.std(y_test)),
    }


def run_probe_sweep(
    reps: Dict[str, np.ndarray],
    df: pd.DataFrame,
    layers: List[int],
    consistency_threshold: float,
    seed: int = 0,
) -> pd.DataFrame:
    if "correct" not in df.columns:
        raise ValueError("CSV must contain 'correct' column.")
    if "consistency" not in df.columns:
        raise ValueError("CSV must contain 'consistency' column.")

    y_correct = df["correct"].astype(int).to_numpy()
    y_cons = df["consistency"].astype(float).to_numpy()
    y_cons_hi = (y_cons >= consistency_threshold).astype(int)

    rows = []

    for rep_name, arr in reps.items():
        for li, layer_idx in enumerate(layers):
            X = arr[:, li, :]

            correct_metrics = evaluate_binary_probe(X, y_correct, seed=seed)
            cons_reg_metrics = evaluate_regression_probe(X, y_cons, seed=seed)
            cons_hi_metrics = evaluate_binary_probe(X, y_cons_hi, seed=seed)

            rows.append({
                "rep": rep_name,
                "layer": layer_idx,
                **{f"correct_{k}": v for k, v in correct_metrics.items()},
                **{f"consistency_reg_{k}": v for k, v in cons_reg_metrics.items()},
                **{f"consistency_hi_{k}": v for k, v in cons_hi_metrics.items()},
            })

    return pd.DataFrame(rows)


def save_reps_npz(
    out_path: str,
    ids: np.ndarray,
    layers: List[int],
    reps: Dict[str, np.ndarray],
) -> None:
    payload = {
        "ids": ids.astype(str),
        "layers": np.array(layers, dtype=np.int32),
    }
    payload.update(reps)
    np.savez_compressed(out_path, **payload)


def print_global_baselines(df: pd.DataFrame, consistency_threshold: float) -> None:
    y = df["consistency"].astype(float).to_numpy()
    mean_pred = np.full_like(y, y.mean(), dtype=np.float64)
    rmse_mean = math.sqrt(mean_squared_error(y, mean_pred))

    print("\nGlobal baseline for raw consistency:")
    print(f"  mean(consistency) = {y.mean():.6f}")
    print(f"  std(consistency)  = {y.std():.6f}")
    print(f"  RMSE(mean baseline on full data) = {rmse_mean:.6f}")

    y_hi = (y >= consistency_threshold).astype(int)
    print(f"\nHigh-consistency threshold = {consistency_threshold}")
    print("High-consistency label counts:")
    vals, cnts = np.unique(y_hi, return_counts=True)
    for v, c in zip(vals, cnts):
        print(f"  {v}: {c}")


def main(args):
    mkdir(args.out_dir)

    print("Loading CSV...")
    df = pd.read_csv(args.in_csv)

    required_cols = ["id", "question", "response", "correct", "consistency"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if args.max_rows is not None and args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    ids = df["id"].astype(str).to_numpy()

    print("Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype_map[args.dtype],
    }
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **model_kwargs,
    )

    if args.device != "auto":
        model = model.to(args.device)
        device = args.device
    else:
        device = next(model.parameters()).device.type
        if device == "cuda":
            device = "cuda"

    model.eval()

    num_layers = model.config.num_hidden_layers
    layers = parse_layers(args.layers, num_layers=num_layers)
    print("Selected layers:", layers)

    full_texts, question_prefixes = build_full_texts(df)

    if args.print_sanity_check:
        print_global_baselines(df, args.consistency_threshold)
        print_sanity_checks(
            df=df,
            tokenizer=tokenizer,
            full_texts=full_texts,
            question_prefixes=question_prefixes,
            max_length=args.max_length,
            n_examples=args.sanity_n_examples,
        )

    reps = extract_layerwise_reps(
        model=model,
        tokenizer=tokenizer,
        full_texts=full_texts,
        question_prefixes=question_prefixes,
        layers=layers,
        max_length=args.max_length,
        device=device,
    )

    emb_path = os.path.join(args.out_dir, f"{args.out_name}_layer_reps.npz")
    print("Saving reps:", emb_path)
    save_reps_npz(emb_path, ids=ids, layers=layers, reps=reps)

    print("Running probe sweep...")
    results_df = run_probe_sweep(
        reps=reps,
        df=df,
        layers=layers,
        consistency_threshold=args.consistency_threshold,
        seed=args.seed,
    )

    results_path = os.path.join(args.out_dir, f"{args.out_name}_probe_results.csv")
    results_df.to_csv(results_path, index=False)
    print("Saved probe results:", results_path)

    print("\nTop rows by correctness AUROC:")
    cols1 = [
        "rep", "layer",
        "correct_auroc",
        "consistency_hi_auroc",
        "consistency_reg_spearman",
        "consistency_reg_r2",
    ]
    print(results_df.sort_values("correct_auroc", ascending=False)[cols1].head(20))

    print("\nTop rows by high-consistency AUROC:")
    cols2 = [
        "rep", "layer",
        "consistency_hi_auroc",
        "consistency_reg_spearman",
        "consistency_reg_r2",
        "correct_auroc",
    ]
    print(results_df.sort_values("consistency_hi_auroc", ascending=False)[cols2].head(20))

    print("\nBaseline layer sweeps:")
    baseline_reps = ["full_last", "full_mean"]
    print(
        results_df[results_df["rep"].isin(baseline_reps)][[
            "rep", "layer",
            "correct_auroc",
            "consistency_hi_auroc",
            "consistency_reg_spearman",
            "consistency_reg_r2",
        ]].sort_values(["rep", "layer"])
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="embed_layers")

    ap.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,28")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--max_rows", type=int, default=None)

    ap.add_argument("--device", type=str, default="auto")  # auto, cuda, cpu
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--consistency_threshold", type=float, default=0.9)

    ap.add_argument("--print_sanity_check", action="store_true")
    ap.add_argument("--sanity_n_examples", type=int, default=5)

    args = ap.parse_args()
    main(args)