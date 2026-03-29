#!/usr/bin/env python3
import argparse
import os
import pickle as pkl
from typing import List, Tuple

import numpy as np
import pandas as pd

from llm_unsupervised_conf.llms import embed_texts, load_vllm_embed
from llm_unsupervised_conf.utils import run_paths, safe_str


def build_texts(df: pd.DataFrame, mode: str, max_chars: int | None) -> List[str]:
    if mode not in ("question", "question_response"):
        raise ValueError(f"mode must be 'question' or 'question_response', got {mode!r}")

    if "question" not in df.columns:
        raise ValueError("Input CSV must contain a 'question' column.")

    if mode == "question":
        texts = [safe_str(q) for q in df["question"].tolist()]
    else:
        resp_col = df["response"].tolist() if "response" in df.columns else [""] * len(df)
        texts = []
        for q, r in zip(df["question"].tolist(), resp_col):
            q = safe_str(q)
            r = safe_str(r)
            if r.strip():
                t = f"QUESTION:\n{q}\n\nRESPONSE:\n{r}"
            else:
                t = f"QUESTION:\n{q}"
            texts.append(t)

    if max_chars is not None and max_chars > 0:
        texts = [t if len(t) <= max_chars else (t[:max_chars] + " ...[truncated]") for t in texts]

    return texts


def save_embeddings(out_path: str, save_format: str, ids: np.ndarray, embeddings: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if save_format == "npz":
        np.savez_compressed(out_path, ids=ids, embeddings=embeddings)

    elif save_format == "pkl":
        payload = {"id": ids, "embeddings": embeddings}
        with open(out_path, "wb") as f:
            pkl.dump(payload, f)

    elif save_format == "csv":
        df_out = pd.DataFrame({"id": ids.astype(str)})
        df_out["embedding"] = [emb.tolist() for emb in embeddings]
        df_out.to_csv(out_path, index=False)

    else:
        raise ValueError(f"Unknown save_format: {save_format}")


def main(args):
    # preserve your naming: append mode into suffix
    out_suffix = f"_{args.mode}{args.out_suffix}"

    in_csv, out_path = run_paths(
        args.base_dir,
        args.run_id,
        in_ext=".csv",
        out_suffix=out_suffix,
        out_ext={"npz": ".npz", "pkl": ".pkl", "csv": ".csv"}[args.save_format],
    )

    if "gemma" in args.model:
        out_path = out_path.replace("_embeddings", "_gemma_embeddings")

    print("in_csv :", in_csv)
    print("out    :", out_path)

    if not os.path.exists(in_csv):
        # raise FileNotFoundError(f"Input CSV not found: {in_csv}")
        print(f"Input CSV not found: {in_csv}")
        return

    print("Loading df...")
    df = pd.read_csv(in_csv)

    if "id" not in df.columns:
        raise ValueError("Input CSV must contain an 'id' column.")

    texts = build_texts(df, mode=args.mode, max_chars=args.max_chars)
    ids = df["id"].astype(str).to_numpy()

    print("Loading vLLM embedding model...")
    llm = load_vllm_embed(args.model, args.n_device, enforce_eager=args.enforce_eager)

    print(f"Embedding N={len(texts)} texts (mode={args.mode}, batch_size={args.batch_size}) ...")
    embeddings = embed_texts(llm, texts, batch_size=args.batch_size)
    print("Embeddings shape:", embeddings.shape)

    if embeddings.shape[0] != len(ids):
        raise RuntimeError(f"Embedding count mismatch: got {embeddings.shape[0]} embeddings for {len(ids)} ids")

    print("Saving...")
    save_embeddings(out_path, args.save_format, ids=ids, embeddings=embeddings)
    print("Saved:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)

    ap.add_argument(
        "--base_dir",
        type=str,
        default="../outputs",
    )
    ap.add_argument("--out_suffix", type=str, default="_embeddings")

    ap.add_argument("--mode", type=str, default="question", choices=["question", "question_response"])
    ap.add_argument("--max_chars", type=int, default=0)

    ap.add_argument("--n_device", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=512)

    ap.add_argument("--save_format", type=str, default="npz", choices=["npz", "pkl", "csv"])
    ap.add_argument("--enforce_eager", action="store_true")

    args = ap.parse_args()
    if args.max_chars <= 0:
        args.max_chars = None

    main(args)
