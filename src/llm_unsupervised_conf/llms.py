from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .utils import extract_last_boxed


@dataclass(frozen=True)
class GenerateConfig:
    model: str
    n_device: int
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int
    logprobs: int = 1
    min_p: float = 0.0
    top_k: int = 20


def load_vllm_generate(cfg: GenerateConfig) -> tuple[Any, LLM, SamplingParams]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    llm = LLM(model=cfg.model, tensor_parallel_size=cfg.n_device)
    sampling = SamplingParams(
        n=1,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_new_tokens,
        logprobs=cfg.logprobs,
        seed=cfg.seed,
        min_p=cfg.min_p,
        top_k=cfg.top_k,
    )
    return tokenizer, llm, sampling


def load_vllm_embed(model: str, n_device: int, *, enforce_eager: bool = False) -> LLM:
    return LLM(
        model=model,
        task="embed",
        tensor_parallel_size=n_device,
        enforce_eager=enforce_eager,
    )

def build_qa_prompt(tokenizer, model_name: str, question: str) -> str:
    """
    Verbatim prompt logic matching your original produce_data.py.

    - Llama 3.1 instruct:
        system: "Please reason step by step, and put your final answer within \\boxed{}"
        user:   question
        enable_thinking=True

    - Qwen3 (any model name containing "Qwen3"):
        user: "Please reason step by step, and put your final answer within \\boxed{}.\nQuestion: {question}"
        enable_thinking=True
    """
    if "deepseek" in model_name:
        stem = "Please reason step by step, and put your final answer within \\boxed{}."
        messages = [
            {
                "role": "user",
                "content": f"{stem}\nQuestion: {question}",
            }
        ]
        prompts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

    elif "Llama" in model_name:
        stem = "Please reason step by step, and put your final answer within \\boxed{}.\nBE SURE TO PUT THE ANSWER IN \\boxed{} OR IT WILL BE CONSIDERED WRONG!!"
        messages = [
            {
                "role": "system",
                "content": stem
            },
            {
                "role": "user",
                "content": f"{question}",
            },
        ]
        prompts =  tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    elif "Qwen3" in model_name:
        stem = "Please reason step by step, and put your final answer within \\boxed{}."
        messages = [
            {
                "role": "user",
                "content": f"{stem}\nQuestion: {question}",
            }
        ]
        prompts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

    elif "Nemotron" in model_name:
        stem = "Please reason step by step, and put your final answer within \\boxed{}."
        messages = [
            {
                "role": "user",
                "content": f"{stem}\nQuestion: {question}",
            }
        ]
        prompts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

    elif "Smol" in model_name:
        stem = "Please reason step by step, and put your final answer within \\boxed{}."
        messages = [
            {
                "role": "user",
                "content": f"{stem}\nQuestion: {question}",
            }
        ]
        prompts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

    elif "gemma" in model_name:
        stem = "Please reason step by step, and put your final answer within \\boxed{}"
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": stem}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{question}"}]
            },
        ]
        prompts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    else:
        raise ValueError

    return prompts


def build_qa_prompt_base(tokenizer, model_name: str, question: str) -> str:
    """
    Verbatim prompt logic matching your original produce_data.py.

    - Llama 3.1 instruct:
        system: "Please reason step by step, and put your final answer within \\boxed{}"
        user:   question
        enable_thinking=True

    - Qwen3 (any model name containing "Qwen3"):
        user: "Please reason step by step, and put your final answer within \\boxed{}.\nQuestion: {question}"
        enable_thinking=True
    """
    if "Llama" in model_name:
        stem = "Please answer the question without outputting any intermediate text."
        messages = [
            {
                "role": "system",
                "content": stem
            },
            {
                "role": "user",
                "content": f"{question}",
            },
        ]

    elif "Qwen3" in model_name:
        stem = "Please answer the question without outputting any intermediate text."
        messages = [
            {
                "role": "user",
                "content": f"{stem}\nQuestion: {question}",
            }
        ]

    elif "Nemotron" in model_name:
        stem = "Please answer the question without outputting any intermediate text."
        messages = [
            {
                "role": "user",
                "content": f"{stem}\nQuestion: {question}",
            }
        ]

    elif "gemma" in model_name:
        stem = "Please answer the question without outputting any intermediate text."
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": stem}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"{question}"}]
            },
        ]

    else:
        raise ValueError
    
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return prompts


def gather_prompts(tokenizer, items: list[dict], model_name: str, *, n_print: int = 5) -> list[str]:
    if "base" in model_name.lower():
        prompts = [build_qa_prompt_base(tokenizer, model_name, ex["question"]) for ex in items]
    else:
        prompts = [build_qa_prompt(tokenizer, model_name, ex["question"]) for ex in items]
    if n_print > 0:
        print("prompt samples:")
        for i in range(n_print):
            check = i * 10
            if check < len(prompts):
                print(prompts[check])
                print("--" * 20)
    return prompts


def generate_k(
    llm: LLM,
    prompts: list[str],
    sampling_base: SamplingParams,
    *,
    k: int,
) -> list:
    """
    One vLLM call: generate k samples per prompt.
    """
    params = SamplingParams(
        n=k,
        temperature=sampling_base.temperature,
        top_p=sampling_base.top_p,
        max_tokens=sampling_base.max_tokens,
        logprobs=sampling_base.logprobs,
        seed=sampling_base.seed,
        min_p=getattr(sampling_base, "min_p", 0.0),
        top_k=getattr(sampling_base, "top_k", 20),
    )
    print("in generate k")
    print(params)
    for i in range(5):
        print(prompts[i])
    return llm.generate(prompts, params)


def rows_from_generation(
    items: list[dict],
    prompts: list[str],
    results: list,
    *,
    answer_parser: Callable[[str], str] = extract_last_boxed,
) -> tuple[list[dict], float]:
    """
    Build your PKL row format:
      id, question, ground_truth, prompt, sample_idx, response, answer, logprobs
    Returns (rows, bad_frac)
    """
    rows: list[dict] = []
    num_bad = 0

    for ex, prompt, res in zip(items, prompts, results):
        for j, out in enumerate(res.outputs):
            text = out.text
            ans = answer_parser(text)
            if not ans:
                num_bad += 1

            rows.append(
                {
                    "id": ex["id"],
                    "question": ex["question"],
                    "ground_truth": ex["ground_truth"],
                    "prompt": prompt,
                    "sample_idx": j,
                    "response": text,
                    "answer": ans,
                    "logprobs": out.logprobs,
                }
            )

    bad_frac = (num_bad / len(rows)) if rows else 0.0
    return rows, bad_frac


def embed_texts(llm: LLM, texts: list[str], batch_size: int) -> np.ndarray:
    all_embeds: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        end = min(len(texts), start + batch_size)
        batch = texts[start:end]

        outputs = llm.embed(batch)
        for out in outputs:
            emb = np.asarray(out.outputs.embedding, dtype=np.float32)
            all_embeds.append(emb)

    if not all_embeds:
        return np.zeros((0, 0), dtype=np.float32)

    return np.vstack(all_embeds).astype(np.float32)
