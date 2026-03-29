#!/usr/bin/env python3
import argparse
import os
import pandas as pd

from llm_unsupervised_conf.calibration import build_verbal_conf_prompt, parse_prob
from llm_unsupervised_conf.llms import GenerateConfig, load_vllm_generate
from llm_unsupervised_conf.utils import run_paths, set_seed, clean_answer


def main(args):
    set_seed(args.seed_base)

    args.out_suffix = f"{args.out_suffix}_stem_{args.stem_idx}"

    print("ARGS")
    print(args)
    print("="*50)

    in_csv, out_csv = run_paths(
        args.base_dir,
        args.run_id,
        in_ext=".csv",
        out_suffix=args.out_suffix,
        out_ext=".csv",
    )
    print("in_csv :", in_csv)
    print("out_csv:", out_csv)

    # if not os.path.exists(in_csv):
    #     raise FileNotFoundError(
    #         f"Input CSV not found: {in_csv}\n"
    #         f"Check --base_dir and --run_id. (Expected file at base_dir/run_id.csv)"
    #     )

    if not os.path.exists(in_csv):
        print(
            f"Input CSV not found: {in_csv}\n"
            f"Check --base_dir and --run_id. (Expected file at base_dir/run_id.csv)"
        )
        return

    print("Loading df...")
    df = pd.read_csv(in_csv)

    needed = {"id", "question", "answer"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}. Found: {list(df.columns)}")

    if "response" not in df.columns:
        df["response"] = ""

    cfg = GenerateConfig(
        model=args.model,
        n_device=args.n_device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed_base,
        logprobs=0,
        min_p=0.0,
        top_k=20,
    )
    tokenizer, llm, sampling = load_vllm_generate(cfg)

    print("Building prompts...")
    prompts = [
        build_verbal_conf_prompt(
            tokenizer=tokenizer,
            model_name=args.model,
            question=str(row["question"]),
            answer=clean_answer(str(row["answer"]), lower=False),
            response=str(row.get("response", "")),
            include_response=args.include_response,
            stem_idx=args.stem_idx
        )
        for _, row in df.iterrows()
    ]

    print("Example prompt:")
    print("*-*-*-"*10)
    print(prompts[0])
    print("*-*-*-"*10)
    print("*-*-*-"*10)

    probs = [None] * len(prompts)
    raw_texts = [""] * len(prompts)

    print(f"Generating verbalized confidence for {len(prompts)} rows ...")
    for start in range(0, len(prompts), args.batch_size):
        end = min(len(prompts), start + args.batch_size)
        batch_prompts = prompts[start:end]

        results = llm.generate(batch_prompts, sampling)
        for i, res in enumerate(results):
            idx = start + i
            out_text = res.outputs[0].text if res.outputs else ""
            raw_texts[idx] = out_text
            probs[idx] = parse_prob(out_text)

    out = pd.DataFrame(
        {
            "id": df["id"].astype(str).tolist(),
            "verbal_confidence": probs,
            "verbal_confidence_raw": raw_texts,
        }
    )

    n_bad = sum(v is None for v in probs)
    print(f"Could not parse {n_bad}/{len(probs)} outputs ({n_bad/len(probs):.2%}).")

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--run_id", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)

    ap.add_argument(
        "--base_dir",
        type=str,
        default="../outputs",
    )
    ap.add_argument("--out_suffix", type=str, default="_verbal_conf")

    ap.add_argument("--n_device", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=2048)

    ap.add_argument("--seed_base", type=int, default=12345)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--include_response", action="store_true")

    ap.add_argument("--stem_idx", type=int, default=4096)

    args = ap.parse_args()
    main(args)
