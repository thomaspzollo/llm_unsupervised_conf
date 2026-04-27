from __future__ import annotations

import re
from typing import Optional

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split


# Prefer parsing from \boxed{...} (also tolerate \box{...})
BOXED_RE = re.compile(r"""\\boxed\s*\{\s*([01](?:\.\d+)?)\s*\}""", flags=re.IGNORECASE | re.DOTALL)
BOX_RE_FALLBACK = re.compile(r"""\\box\s*\{\s*([01](?:\.\d+)?)\s*\}""", flags=re.IGNORECASE | re.DOTALL)
FLOAT_RE = re.compile(r"(?<![\d.])(?:0(?:\.\d+)?|1(?:\.0+)?)(?![\d.])")


def parse_prob(text: str) -> Optional[float]:
    if text is None:
        return None
    t = text.strip()

    m = BOXED_RE.search(t)
    if m:
        try:
            v = float(m.group(1))
            return v if 0.0 <= v <= 1.0 else None
        except Exception:
            pass

    m = BOX_RE_FALLBACK.search(t)
    if m:
        try:
            v = float(m.group(1))
            return v if 0.0 <= v <= 1.0 else None
        except Exception:
            pass

    try:
        v = float(t)
        if 0.0 <= v <= 1.0:
            return v
    except Exception:
        pass

    m = FLOAT_RE.search(t)
    if not m:
        return None
    try:
        v = float(m.group(0))
        return v if 0.0 <= v <= 1.0 else None
    except Exception:
        return None


VERBAL_CONF_STEMS = [
    (
        "You are a calibration assistant.\n"
        "Task: estimate the probability that the given ANSWER is correct for the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Write at most THREE short sentences of reasoning (keep it brief).\n"
        "2) On a NEW LINE, output ONLY the probability in the form \\boxed{p} where p is a decimal in [0,1].\n"
        "   - Examples: \\boxed{0.03}, \\boxed{0.62},  \\boxed{0.81}, \\boxed{0.97}\n"
        "   - Do not output those example scores exactly, try to be precise about the exact probability of this answer being correct\n"
        "3) Do NOT write anything after the \\boxed{...} line.\n"
    ),
    (
        "You are a calibration assistant.\n"
        "Your job is to judge how likely the provided ANSWER is to be correct for the given QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Give no more than a few brief sentences explaining your estimate.\n"
        "2) Then, on a NEW LINE, output ONLY a probability as \\boxed{p}, where p is a decimal in [0,1].\n"
        "   - Examples of format only: \\boxed{0.14}, \\boxed{0.58}, \\boxed{0.89}\n"
        "   - Do not copy those example values; choose the most precise probability you can.\n"
        "3) Do NOT write anything after the \\boxed{...} line.\n"
    ),
    (
        "You are an uncertainty estimation assistant.\n"
        "Task: assess the probability that the ANSWER correctly resolves the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Provide a very brief explanation of your reasoning.\n"
        "2) On a NEW LINE, output ONLY the final probability in the exact form \\boxed{p}, with p in [0,1].\n"
        "   - Format examples: \\boxed{0.07}, \\boxed{0.44}, \\boxed{0.93}\n"
        "   - Do not reuse the example numbers; give your own precise estimate.\n"
        "3) Do NOT include any text after the \\boxed{...} line.\n"
    ),
    (
        "You are a confidence scoring assistant.\n"
        "Task: estimate the chance that the supplied ANSWER is correct given the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Write up to four or five concise sentences of reasoning.\n"
        "2) On a separate NEW LINE, output ONLY \\boxed{p}, where p is a decimal probability between 0 and 1.\n"
        "   - Example formatting: \\boxed{0.11}, \\boxed{0.69}, \\boxed{0.95}\n"
        "   - Do not use those exact values; report the most accurate probability you can.\n"
        "3) Do NOT add anything after the \\boxed{...} line.\n"
    ),
    (
        "You are a probability calibration assistant.\n"
        "Task: determine how likely it is that the ANSWER is correct for the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Use at most THREE short reasoning sentences.\n"
        "2) Then on a NEW LINE, output ONLY the probability as \\boxed{p}, where p is a decimal in [0,1].\n"
        "   - Formatting examples: \\boxed{0.05}, \\boxed{0.51}, \\boxed{0.98}\n"
        "   - Do not repeat those example values; instead provide a precise estimate.\n"
        "3) Do NOT write any additional text after the \\boxed{...} line.\n"
    ),
    (
        "You are a reliability assessment assistant.\n"
        "Task: evaluate the probability that the provided ANSWER is the correct response to the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Write no more than 3-6 brief sentences of reasoning.\n"
        "2) On a NEW LINE, output ONLY the probability in the form \\boxed{p}, where p is a decimal in [0,1].\n"
        "   - Examples: \\boxed{0.09}, \\boxed{0.37}, \\boxed{0.75}, \\boxed{0.91}\n"
        "   - Do not copy these example values; provide your own precise probability estimate.\n"
        "3) Do NOT include any additional text after the \\boxed{...} line.\n"
    ),
    (
        "You are a fact-checking and calibration model.\n"
        "Task: determine the likelihood that the ANSWER accurately solves the given QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Use at most FOUR concise sentences to explain your reasoning.\n"
        "2) On a NEW LINE, output ONLY the probability using the format \\boxed{p} (p is a decimal in [0,1]).\n"
        "   - Format examples: \\boxed{0.18}, \\boxed{0.42}, \\boxed{0.66}, \\boxed{0.99}\n"
        "   - Do not reuse these example scores; instead, be as accurate as possible for this specific case.\n"
        "3) Do NOT write any text after the \\boxed{...} line.\n"
    ),
    (
        "You are an answer verification assistant.\n"
        "Task: calculate the probability that the ANSWER is correct given the context of the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Provide at most THREE short sentences of logic or reasoning.\n"
        "2) Then, on a NEW LINE, output ONLY the final probability as \\boxed{p}, where p is in [0,1].\n"
        "   - Examples: \\boxed{0.02}, \\boxed{0.25}, \\boxed{0.55}, \\boxed{0.88}\n"
        "   - Do not use these exact numbers; choose the most precise decimal for the current answer.\n"
        "3) Do NOT add any comments or text after the \\boxed{...} line.\n"
    ),
    (
        "You are an expert calibration system.\n"
        "Task: assign a probability score representing how likely the ANSWER is to be the correct solution for the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Limit your reasoning to at most 4-6 brief sentences.\n"
        "2) On a separate NEW LINE, output ONLY \\boxed{p}, with p being a decimal between 0 and 1.\n"
        "   - Formatting examples: \\boxed{0.13}, \\boxed{0.49}, \\boxed{0.72}, \\boxed{0.94}\n"
        "   - Do not repeat these example values; provide your own calculated probability.\n"
        "3) Do NOT include any characters or text after the \\boxed{...} line.\n"
    ),
    (
        "You are a truth-estimation assistant.\n"
        "Task: predict the probability that the given ANSWER is a correct and valid response to the QUESTION.\n\n"
        "Output format (STRICT):\n"
        "1) Write up to FIVE short sentences explaining your thought process.\n"
        "2) On a NEW LINE, output ONLY the probability in the exact form \\boxed{p}, where p is a decimal in [0,1].\n"
        "   - Format examples: \\boxed{0.06}, \\boxed{0.31}, \\boxed{0.64}, \\boxed{0.92}\n"
        "   - Do not output the example values; give a precise estimate for the answer provided.\n"
        "3) Do NOT write anything at all after the \\boxed{...} line.\n"
    ),
]



def build_verbal_conf_prompt(
    tokenizer,
    model_name: str,
    question: str,
    answer: str,
    response: str | None,
    include_response: bool,
    stem_idx: int=0
) -> str:
    resp_block = ""
    if include_response and isinstance(response, str) and response.strip():
        resp = response.strip()
        if len(resp) > 1200:
            resp = resp[:1200] + " ...[truncated]"
        resp_block = f"\nMODEL_RESPONSE (may include reasoning):\n{resp}\n"

    stem = VERBAL_CONF_STEMS[stem_idx]

    user = (
        f"{stem}\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n"
        f"{resp_block}\n"
    )

    if "Qwen3" in model_name or "Qwen/Qwen3" in model_name:
        messages = [{"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    elif "Llama" in model_name:
        messages = [{"role": "system", "content": stem}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        messages = [{"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return prompt


# ----------------------------
# Probabilistic regressors on embeddings
# ----------------------------

def fit_predict_prob_models(
    X_train: np.ndarray,
    y_train_prob: np.ndarray,
    X_test: np.ndarray,
    methods: list[str],
    *,
    eps: float = 1e-6,
    random_state: int = 42,
    split_frac: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Fit requested models to predict y in [0,1] and return predictions on test set.

    Supported methods (strings):
      - "ridge_clip":          Ridge -> clip to [0,1]  (your current baseline)
      - "isotonic_on_ridge":   Fit ridge_clip, then isotonic maps ridge_pred -> y_train_prob (1D post-hoc, bounded)

    Notes:
      - All outputs are clipped to [0,1] for safety.
      - If you add methods, keep the contract: return dict[name] = probs in [0,1].
    """
    methods = list(dict.fromkeys(methods))  # de-dupe while preserving order
    y_train_prob = np.asarray(y_train_prob, dtype=float)
    y_train_prob = np.clip(y_train_prob, 0.0, 1.0)

    out = {}

    if "ridge_clip" in methods or "isotonic_on_ridge" in methods:
        ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=random_state)),
        ])
        ridge.fit(X_train, y_train_prob)
        ridge_pred = np.clip(ridge.predict(X_test), 0.0, 1.0)
        out["ridge_clip"] = ridge_pred


    if "split_isotonic_on_ridge" in methods:
        X_a, X_b, y_a, y_b = train_test_split(
            X_train,
            y_train_prob,
            test_size=split_frac,
            random_state=random_state,
        )
    
        ridge_a = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=random_state)),
        ])
        ridge_a.fit(X_a, y_a)
        pred_b = ridge_a.predict(X_b)
    
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(pred_b, y_b)
    
        ridge_all = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=random_state)),
        ])
        ridge_all.fit(X_train, y_train_prob)
        pred_test = ridge_all.predict(X_test)
    
        out["split_isotonic_on_ridge"] = iso.predict(pred_test)

    if "split_isotonic_on_ridge_nrt" in methods:
        X_a, X_b, y_a, y_b = train_test_split(
            X_train,
            y_train_prob,
            test_size=split_frac,
            random_state=random_state,
        )
    
        ridge = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=random_state)),
        ])
        ridge.fit(X_a, y_a)
        pred_b = ridge.predict(X_b)
    
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(pred_b, y_b)
    
        pred_test = ridge.predict(X_test)

        out["split_isotonic_on_ridge_nrt"] = iso.predict(pred_test)

    # Return only requested methods (and in the requested order)
    return {m: out[m] for m in methods if m in out}


