from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

import numpy as np


def stack_embeddings(df: pd.DataFrame, col: str = "embeddings") -> np.ndarray:
    """
    Convert df[col] that contains per-row 1D vectors (np.array/list) into an (N, D) float32 matrix.
    """
    emb_list = df[col].tolist()
    # If they're already numpy arrays or lists, this is fast:
    X = np.vstack([np.asarray(e, dtype=np.float32).reshape(-1) for e in emb_list])
    return X
