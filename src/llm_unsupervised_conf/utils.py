from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from typing import Optional, Tuple
from llm_unsupervised_conf.metrics import get_ece1, get_ece2, get_nll, get_mce
from sklearn.metrics import brier_score_loss, roc_auc_score

import pandas as pd

import numpy as np
import torch


ROOT_DIR = ".."


# Generic boxed extraction: grabs last \boxed{...} of any content
BOXED_ANY_RE = re.compile(r"\\boxed\s*\{\s*(.*?)\s*\}", flags=re.DOTALL)


def qa_correct(ans: str, gts) -> int:
    """QA correctness: substring match against aliases."""
    if ans is None:
        return 0
    a = str(ans).split("\\text{")[-1].lower().replace("\\","")
    # gts should be list[str]
    for gt in gts:
        t = str(gt).lower()
        if (a in t) or (t in a):
            return 1
    return 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clean_answer(ans, lower=True):
    a = str(ans).split("\\text{")[-1].replace("\\","")
    if lower:
        a = a.lower()
    return a


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_str(x) -> str:
    return "" if x is None else str(x)


def extract_last_boxed(text: str) -> str:
    """
    Return the last \\boxed{...} content, stripped.
    If none found, return empty string.
    """
    if not text:
        return ""
    hits = BOXED_ANY_RE.findall(text)
    if not hits:
        return ""
    return str(hits[-1]).strip()


def normalize_run_id(run_id: str) -> str:
    return run_id.strip().lstrip("/")


def run_paths(
    base_dir: str,
    run_id: str,
    *,
    in_ext: str = ".csv",
    out_suffix: str = "",
    out_ext: str = ".csv",
) -> Tuple[str, str]:
    """
    Build input/output paths under base_dir using run_id.
    Example:
      in:  {base_dir}/{run_id}.csv
      out: {base_dir}/{run_id}{out_suffix}.csv
    """
    run_id = normalize_run_id(run_id)
    in_path = os.path.join(base_dir, run_id + in_ext)
    out_path = os.path.join(base_dir, run_id + out_suffix + out_ext)
    return in_path, out_path


@dataclass(frozen=True)
class OutputLayout:
    """
    For the produce_data PKL layout: ../outputs/{model}/{dataset}/...
    root is relative to the script location by default, matching your current usage.
    """
    root: str = "../outputs"

    def dir_for(self, model_name: str, dataset: str) -> str:
        return os.path.join(self.root, model_name, dataset)

    def pkl_path(self, model_name: str, dataset: str, *, n: int, temperature: float, k: int) -> str:
        out_dir = self.dir_for(model_name, dataset)
        fname = f"n_{n}_temp_{temperature}_k_{k}.pkl"
        return os.path.join(out_dir, fname)


def load_verbal_conf_scores(df_save_path_csv: str, ids: pd.Series, stem_idx=None) -> np.ndarray | None:
    """Load <df_save_path_csv[:-4]> + '_verbal_conf.csv' and align to ids by string key."""
    if not df_save_path_csv.endswith(".csv"):
        raise ValueError(f"Expected .csv path, got: {df_save_path_csv}")

    if stem_idx is None:
        path_suffix = "_verbal_conf.csv"
    else:
        path_suffix = f"_verbal_conf_stem_{stem_idx}.csv"

    vc_path = df_save_path_csv[:-4] + path_suffix
    if not os.path.exists(vc_path):
        print(f"[verbal_conf] missing: {vc_path}")
        return None

    vc_df = pd.read_csv(vc_path)
    if not {"id", "verbal_confidence"}.issubset(vc_df.columns):
        print(f"[verbal_conf] bad schema in {vc_path} (need id, verbal_confidence)")
        return None

    left = pd.DataFrame({"id": ids.astype(str).to_numpy()})
    right = vc_df[["id", "verbal_confidence"]].copy()
    right["id"] = right["id"].astype(str)

    merged = left.merge(right, on="id", how="left")
    return pd.to_numeric(merged["verbal_confidence"], errors="coerce").to_numpy(dtype=float)


def impute_mean_clip01(x: np.ndarray, fallback: float = 0.5) -> tuple[np.ndarray, float, int]:
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(x)
    mean_val = float(x[mask].mean()) if mask.any() else float(fallback)
    n_missing = int((~mask).sum())
    x = np.where(mask, x, mean_val)
    x = np.clip(x, 0.0, 1.0)
    return x, mean_val, n_missing


def maybe_add_verbal_conf_row(rows, df_save_path_csv, test_df, correct, n_bins=12, stem_idx=None):
    vc = load_verbal_conf_scores(df_save_path_csv, test_df["id"], stem_idx=stem_idx)
    if vc is None:
        return rows, None

    vc, mean_val, n_missing = impute_mean_clip01(vc, fallback=0.5)
    # if n_missing:
        # print(f"[verbal_conf] imputed {n_missing}/{len(vc)} with mean={mean_val:.4f}")

    rows.append([
        "verbal_conf",
        get_ece1(vc, correct, n_bins=n_bins),
        get_ece2(vc, correct, n_bins=n_bins),
        get_mce(vc, correct, n_bins=n_bins),
        get_nll(vc, correct),
        brier_score_loss(correct, vc),
        roc_auc_score(correct, vc),
    ])
    # print("[verbal_conf] added row")
    return rows, vc


def load_out_df(dataset, model_name, n, temp_train, k_train, temp_test, embedding_text, drop_bad_rows):
    
    exp_save_dir = f"{ROOT_DIR}/outputs/{model_name}/{dataset}"
    df_save_path = f"{exp_save_dir}/n_{n}_temp_{temp_train}_k_{k_train}_out_df_temp_{temp_test}_k_1.csv"
    # print("loading data stored at", df_save_path)
    out_df = pd.read_csv(df_save_path)

    emb_save_path = f"{exp_save_dir}/n_{n}_temp_{temp_train}_k_{k_train}_out_df_temp_{temp_test}_k_1_{embedding_text}_embeddings.npz"
    
    if not os.path.isfile(emb_save_path):

        raise ValueError("Missing embeds")
        
    else:
        # print("loading embeddings...")
        embeddings = np.load(emb_save_path)["embeddings"]

    out_df["embeddings"] = list(embeddings)

    if drop_bad_rows:
        # --------------------
        # Drop invalid answers (keep embeddings aligned)
        # --------------------
        n_before = len(out_df)
    
        # "valid answer" = not NaN AND not empty after stripping
        ans = out_df["answer"]
        valid_mask = ans.notna() & (ans.astype(str).str.strip() != "")
    
        n_dropped = int((~valid_mask).sum())
        # if n_dropped > 0:
        #     print(f"Dropping {n_dropped}/{n_before} rows with missing/empty answers")
    
        out_df = out_df.loc[valid_mask].reset_index(drop=True)
    
        # Optional sanity check
        if len(out_df) == 0:
            raise ValueError("All rows were dropped (no valid answers). Check your parsing/pipeline.")
    
    return out_df, df_save_path