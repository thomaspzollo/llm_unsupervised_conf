from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from datasets import load_dataset

DEFAULT_POLYMATH_LANGS = ["en", "es", "fr", "pt", "it", "ru", "ar", "id"]
DEFAULT_POLYMATH_SPLIT = "low"


def load_dataset_items(
    name: str,
    *,
    n: int,
    seed: int = 12345,
    polymath_langs: Optional[str] = None,
    polymath_split: str = DEFAULT_POLYMATH_SPLIT,
    sciq_format_mcq: bool = False,
    sciq_include_support: bool = False,
    sciq_shuffle_choices: bool = True,
) -> List[Dict[str, Any]]:
    """
    Returns list of dicts with canonical keys:
      - id: str
      - question: str
      - ground_truth: str | list[str]
    """
    name = name.strip()

    if name == "trivia_qa":
        return _load_trivia_qa(n=n)
    if name == "gsm8k":
        return _load_gsm8k(n=n)
    if name == "polymath":
        return _load_polymath(n=n, polymath_langs=polymath_langs, polymath_split=polymath_split)
    if name == "sciq":
        return _load_sciq(
            n=n,
            seed=seed,
            format_mcq=sciq_format_mcq,
            include_support=sciq_include_support,
            shuffle_choices=sciq_shuffle_choices,
        )
    if name == "webq":
        return _load_webq(n=n)

    raise ValueError(f"Unknown dataset: {name}")


def _load_webq(*, n: int) -> List[Dict[str, Any]]:
    ds = load_dataset("stanfordnlp/web_questions", split="test")
    items = []
    for i, ex in enumerate(ds):
        q = ex.get("question")
        answers = ex.get("answers")
        if not q or not answers:
            continue
        items.append({"id": str(i), "question": str(q), "ground_truth": answers})
        if len(items) >= n:
            break
    return items[:n]


def _load_sciq(
    *,
    n: int,
    seed: int,
    format_mcq: bool,
    include_support: bool,
    shuffle_choices: bool,
) -> List[Dict[str, Any]]:
    split = "test"
    ds = load_dataset("sciq", split=split)

    items = []
    rng = random.Random(seed)

    for i, ex in enumerate(ds):
        q = ex.get("question")
        gt = ex.get("correct_answer")
        if not q or not gt:
            continue

        distractors = [ex.get("distractor1"), ex.get("distractor2"), ex.get("distractor3")]
        distractors = [d for d in distractors if isinstance(d, str) and d.strip()]
        if len(distractors) != 3:
            continue

        support = ex.get("support") if include_support else None
        support = support.strip() if isinstance(support, str) else ""

        if format_mcq:
            choices = distractors + [gt]
            if shuffle_choices:
                rng.shuffle(choices)
            labels = ["A", "B", "C", "D"]
            choice_lines = "\n".join([f"{lab}) {c}" for lab, c in zip(labels, choices)])

            if include_support and support:
                q_text = (
                    f"Context: {support}\n\n"
                    f"Question: {q}\n\nChoices:\n{choice_lines}\n\n"
                    f"Final answer: \\boxed{{<letter>}}"
                )
            else:
                q_text = (
                    f"Question: {q}\n\nChoices:\n{choice_lines}\n\n"
                    f"Final answer: \\boxed{{<letter>}}"
                )

            correct_letter = labels[choices.index(gt)]
            gt_out = correct_letter
        else:
            if include_support and support:
                q_text = f"Context: {support}\n\nQuestion: {q}"
            else:
                q_text = str(q)
            gt_out = str(gt)

        items.append({"id": str(i), "question": q_text, "ground_truth": gt_out})
        if len(items) >= n:
            break

    return items[:n]


def _load_polymath(*, n: int, polymath_langs: Optional[str], polymath_split: str) -> List[Dict[str, Any]]:
    if polymath_langs:
        langs = [x.strip() for x in polymath_langs.split(",") if x.strip()]
    else:
        langs = DEFAULT_POLYMATH_LANGS

    split = polymath_split or DEFAULT_POLYMATH_SPLIT

    items = []
    for lang in langs:
        ds = load_dataset("Qwen/PolyMath", lang, split=split)
        for ex in ds:
            q = ex.get("question")
            ans = ex.get("answer")
            if not q or not ans:
                continue
            uid = str(len(items))
            items.append({"id": uid, "question": str(q), "ground_truth": str(ans)})
            if len(items) >= n:
                return items[:n]
    return items[:n]


def _load_gsm8k(*, n: int) -> List[Dict[str, Any]]:
    def _extract_answer(ex):
        ans = ex.get("answer", "")
        return ans.split("####")[-1].strip()

    ds = load_dataset("openai/gsm8k", "main")["train"]
    items = []
    for i, ex in enumerate(ds):
        q = ex.get("question")
        if not q:
            continue
        items.append({"id": str(i), "question": str(q), "ground_truth": _extract_answer(ex)})
        if len(items) >= n:
            break
    return items[:n]


def _load_trivia_qa(*, n: int) -> List[Dict[str, Any]]:
    def _extract_answers(ex):
        ans = ex.get("answer") or {}
        aliases = ans.get("aliases") or []
        value = ans.get("value")
        out = []
        if isinstance(aliases, list):
            out.extend([str(a) for a in aliases if a])
        if isinstance(value, str) and value:
            out.append(value)
        if not out and isinstance(ex.get("answers"), list):
            out.extend([str(a) for a in ex["answers"] if a])
        # unique while preserving order
        return list(dict.fromkeys(out))

    ds = load_dataset("trivia_qa", "rc.nocontext")["validation"]
    items = []
    for i, ex in enumerate(ds):
        q = ex.get("question")
        if not q:
            continue
        answers = _extract_answers(ex)
        if not answers:
            continue
        items.append({"id": str(i), "question": str(q), "ground_truth": answers})
        if len(items) >= n:
            break
    return items[:n]

