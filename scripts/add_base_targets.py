#!/usr/bin/env python3
import argparse
import math
import os
import re
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def infer_base_model(reasoning_model: str) -> str:
    mapping = {
        "Qwen/Qwen3-0.6B": "Qwen/Qwen3-0.6B-Base",
        "Qwen/Qwen3-1.7B": "Qwen/Qwen3-1.7B-Base",
        "Qwen/Qwen3-4B-Thinking-2507": "Qwen/Qwen3-4B-Base",
        "Qwen/Qwen3-8B": "Qwen/Qwen3-8B-Base",
        "Qwen/Qwen3-14B": "Qwen/Qwen3-14B-Base",
    }
    if reasoning_model not in mapping:
        raise ValueError(f"No base-model mapping found for {reasoning_model}")
    return mapping[reasoning_model]


def build_qa_prompt(tokenizer, model_name: str, question: str) -> str:
    """
    Match the base-prompt logic from llms.py for Qwen models.
    """
    if "Qwen3" in model_name:
        stem = "Please reason step by step, and put your final answer within \\boxed{}."
        messages = [
            {
                "role": "user",
                "content": f"{stem}\nQuestion: {question}",
            }
        ]
    else:
        raise ValueError(f"Unsupported base model for this script: {model_name}")

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    return prompt


def build_qa_prompt_base(tokenizer, model_name: str, question: str) -> str:
    """
    Match the base-prompt logic from llms.py for Qwen models.
    """
    if "Qwen3" in model_name:

        return f"Question: {question}\nAnswer: "
    #     messages = [
    #         {
    #             "role": "user",
    #             "content": f"Question: {question}",
    #         }
    #     ]
    # else:
    #     raise ValueError(f"Unsupported base model for this script: {model_name}")

    # prompt = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=True
    # )
    # return prompt +  "\nAnswer: "




BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer_span(response: str, answer_val: Optional[str] = None) -> Optional[str]:
    if answer_val is not None:
        ans = safe_str(answer_val).strip()
        if ans:
            return ans

    text = safe_str(response)

    m = BOXED_RE.search(text)
    if m:
        return m.group(0)

    m = ANSWER_TAG_RE.search(text)
    if m:
        return m.group(1).strip()

    markers = [
        "Final answer:",
        "final answer:",
        "Answer:",
        "The answer is",
        "Therefore, the answer is",
        "So the answer is",
    ]
    for marker in markers:
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx:].strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def encode_no_special(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def load_model_and_tokenizer(model_name: str, dtype: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype_map[dtype],
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    if device != "auto":
        model = model.to(device)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def score_continuations_mean_logprob(
    model,
    tokenizer,
    prompts: List[str],
    continuations: List[str],
    batch_size: int,
) -> Tuple[List[float], List[int]]:
    device = next(model.parameters()).device
    mean_lps: List[float] = []
    ns: List[int] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Scoring batches"):
        batch_prompts = prompts[start:start + batch_size]
        batch_conts = continuations[start:start + batch_size]

        prompt_ids_list = [encode_no_special(tokenizer, p) for p in batch_prompts]
        cont_ids_list = [encode_no_special(tokenizer, c) for c in batch_conts]

        seqs = [p + c for p, c in zip(prompt_ids_list, cont_ids_list)]
        max_len = max(len(s) for s in seqs)

        input_ids = torch.full(
            (len(seqs), max_len),
            fill_value=tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(seqs), max_len),
            dtype=torch.long,
            device=device,
        )

        for i, seq in enumerate(seqs):
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attention_mask[i, :len(seq)] = 1

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits, dim=-1)

        for i, (prompt_ids, cont_ids) in enumerate(zip(prompt_ids_list, cont_ids_list)):
            if len(cont_ids) == 0:
                mean_lps.append(float("nan"))
                ns.append(0)
                continue

            total_lp = 0.0
            p_len = len(prompt_ids)

            for j, tok_id in enumerate(cont_ids):
                abs_pos = p_len + j
                prev_pos = abs_pos - 1
                total_lp += log_probs[i, prev_pos, tok_id].item()

            mean_lps.append(total_lp / len(cont_ids))
            ns.append(len(cont_ids))

    return mean_lps, ns


def main(args):
    mkdir(args.out_dir)

    print("Loading CSV...")
    df = pd.read_csv(args.in_csv)

    required = ["question", "response"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if args.max_rows is not None and args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    base_model_name = args.base_model_name or infer_base_model(args.model_name)

    print(f"Reasoning model: {args.model_name}")
    print(f"Base model:      {base_model_name}")
    print("Loading tokenizer/model...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=base_model_name,
        dtype=args.dtype,
        device=args.device,
    )

    print("Building prompts for full responses...")
    prompts = [
        build_qa_prompt(tokenizer, base_model_name, safe_str(q))
        for q in df["question"].tolist()
    ]
    print(prompts[0])
    responses = [safe_str(x) for x in df["response"].tolist()]

    print("Scoring full responses under base model...")
    resp_lps, resp_ns = score_continuations_mean_logprob(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        continuations=responses,
        batch_size=args.batch_size,
    )

    print("Building prompts for answer-only responses...")
    prompts = [
        build_qa_prompt_base(tokenizer, base_model_name, safe_str(q))
        for q in df["question"].tolist()
    ]
    print(prompts[0])

    # if "answer" in df.columns:
    #     answer_texts = [
    #         extract_answer_span(r, a)
    #         for r, a in zip(df["response"].tolist(), df["answer"].tolist())
    #     ]
    # else:
    #     answer_texts = [
    #         extract_answer_span(r, None)
    #         for r in df["response"].tolist()
    #     ]
    # answer_texts = [a if a is not None else "" for a in answer_texts]
    answer_texts = [str(a) for a in df["answer"].tolist()]

    print("Scoring extracted answers under base model...")
    ans_lps, ans_ns = score_continuations_mean_logprob(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        continuations=answer_texts,
        batch_size=args.batch_size,
    )

    df["base_response_avg_logprob"] = resp_lps
    df["base_response_avg_prob"] = [
        math.exp(x) if not math.isnan(x) else float("nan") for x in resp_lps
    ]
    df["base_response_num_tokens"] = resp_ns

    df["base_answer_avg_logprob"] = ans_lps
    df["base_answer_avg_prob"] = [
        math.exp(x) if not math.isnan(x) else float("nan") for x in ans_lps
    ]
    df["base_answer_num_tokens"] = ans_ns

    out_csv = os.path.join(args.out_dir, f"{args.out_name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--model_name", type=str, required=True)
    ap.add_argument("--base_model_name", type=str, default=None)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--out_name", type=str, default="with_base_targets")

    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--max_rows", type=int, default=None)

    ap.add_argument("--device", type=str, default="auto")  # auto, cuda, cpu
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])

    args = ap.parse_args()
    main(args)