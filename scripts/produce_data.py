import argparse
import os
import pickle as pkl

from llm_unsupervised_conf.data import load_dataset_items
from llm_unsupervised_conf.llms import (
    GenerateConfig,
    gather_prompts,
    load_vllm_generate,
    generate_k,
    rows_from_generation,
)
from llm_unsupervised_conf.utils import OutputLayout, ensure_dir, set_seed


def main(args):
    print("welcome!")
    print(args)

    model_name = args.model.split("/")[-1]
    layout = OutputLayout(root=args.outputs_root)

    out_dir = layout.dir_for(model_name, args.dataset)
    ensure_dir(out_dir)

    out_path = layout.pkl_path(model_name, args.dataset, n=args.n, temperature=args.temperature, k=args.k)
    print("will save to", out_path)

    e += 7

    set_seed(args.seed_base)

    print("-" * 20)
    print("loading data...")
    items = load_dataset_items(
        args.dataset,
        n=args.n,
        seed=args.seed_base,
        polymath_langs=args.polymath_langs,
        polymath_split=args.polymath_split,
        sciq_format_mcq=args.sciq_format_mcq,
        sciq_include_support=args.sciq_include_support,
        sciq_shuffle_choices=not args.sciq_no_shuffle,
    )

    print("-" * 20)
    print("loading model...")
    cfg = GenerateConfig(
        model=args.model,
        n_device=args.n_device,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed_base,
        logprobs=1,
        min_p=0.0,
        top_k=20,
    )
    tokenizer, llm, sampling = load_vllm_generate(cfg)

    print("-" * 20)
    print("preparing prompts...")
    prompts = gather_prompts(tokenizer, items, args.model, n_print=5)

    print("-" * 20)
    print("producing llm responses...")
    results = generate_k(llm, prompts, sampling, k=args.k)
    rows, bad_frac = rows_from_generation(items, prompts, results)
    print("Bad answers:", bad_frac)

    print("-" * 20)
    print("final data samples:")
    for i in range(min(10, len(items))):
        row0 = rows[i * args.k] if (i * args.k) < len(rows) else None
        if not row0:
            break
        for k, v in row0.items():
            if k == "logprobs":
                continue
            print("KEY", k)
            print(v)
            print("-" * 5)
        print("-" * 20)

    print("saving to", out_path)
    with open(out_path, "wb") as f:
        pkl.dump(rows, f)

    print("donezo, goodbye!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="trivia_qa")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507")

    ap.add_argument("--n_device", type=int, default=4, help="Number of GPUs")
    ap.add_argument("--n", type=int, default=2000, help="Number of data examples")
    ap.add_argument("--k", type=int, default=100, help="Teacher samples per question")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=32768)
    ap.add_argument("--seed-base", dest="seed_base", type=int, default=12345)

    # outputs root (keep your prior ../outputs default behavior)
    ap.add_argument("--outputs_root", type=str, default="../outputs")

    # PolyMath
    ap.add_argument("--polymath_langs", type=str, default="en,es,fr,pt,it,ru,ar,id",
                    help="Comma-separated PolyMath language configs to load.")
    ap.add_argument("--polymath_split", type=str, default="low",
                    help="PolyMath split: top/high/medium/low")

    # SciQ options (default stays free-form, as in your file)
    ap.add_argument("--sciq_format_mcq", action="store_true", help="Format SciQ as MCQ with boxed letter.")
    ap.add_argument("--sciq_include_support", action="store_true", help="Include SciQ support context.")
    ap.add_argument("--sciq_no_shuffle", action="store_true", help="Do not shuffle SciQ choices (if MCQ).")

    args = ap.parse_args()
    main(args)
