"""
Dataset Generator for Data-Centric RL Environment.

Generates corrupted sklearn classification datasets with known ground truth.
Each task has deterministic corruptions via seeded random.Random.

CRITICAL: Always produces TWO copies:
  ground_truth  → frozen, only read by grader
  working_copy  → the only thing the agent can mutate
"""

import random
from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


# ── Column metadata schema ──────────────────────────────────────────────────

def _make_col_meta(expected_dtype: str, valid_range=None,
                   valid_categories=None, is_nullable: bool = False) -> Dict:
    return {
        "expected_dtype": expected_dtype,
        "valid_range": valid_range,
        "valid_categories": valid_categories,
        "is_nullable": is_nullable,
    }


# ── Task configurations ─────────────────────────────────────────────────────

TASK_CONFIGS = {
    "task_0_tutorial": {
        "n_samples": 100,
        "n_features": 4,
        "n_classes": 2,
        "n_informative": 3,
        "budget": 30,
        "target_accuracy": 0.73,
        "baseline_accuracy": 0.62,
        "description": "Single-issue tutorial. Fix missing values in 'age' to win.",
    },
    "task_1_easy": {
        "n_samples": 200,
        "n_features": 5,
        "n_classes": 2,
        "n_informative": 4,
        "budget": 25,
        "target_accuracy": 0.79,
        "baseline_accuracy": 0.63,
        "description": "Missing values + mild class imbalance.",
    },
    "task_2_medium": {
        "n_samples": 500,
        "n_features": 7,
        "n_classes": 3,
        "n_informative": 5,
        "budget": 40,
        "target_accuracy": 0.74,
        "baseline_accuracy": 0.58,
        "description": "Missing values, duplicates, class imbalance, type error.",
    },
    "task_3_hard": {
        "n_samples": 900,
        "n_features": 10,
        "n_classes": 4,
        "n_informative": 7,
        "budget": 60,
        "target_accuracy": 0.71,
        "baseline_accuracy": 0.54,
        "description": "Missing values, duplicates, imbalance, type errors, outliers, cross-column errors.",
    },
}


# ── Generic feature names ───────────────────────────────────────────────────

FEATURE_NAMES = ["age", "income", "score", "tenure", "balance",
                 "transactions", "risk_level", "credit", "spend", "savings"]


def _build_column_meta(feature_cols: list, task: str) -> Dict[str, Dict]:
    meta = {}
    for col in feature_cols:
        meta[col] = _make_col_meta("float64", valid_range=(-10.0, 10.0))
    # age gets tighter range for tutorial plausibility
    if "age" in meta:
        meta["age"] = _make_col_meta("float64", valid_range=(0.0, 100.0))
    meta["target"] = _make_col_meta("int64", valid_categories=None)
    return meta


# ── Core generator ──────────────────────────────────────────────────────────

def generate_dataset(task: str, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Generate a corrupted dataset for the given task.

    Returns:
        ground_truth  – clean DataFrame (frozen)
        working_copy  – corrupted DataFrame (agent mutates this)
        metadata      – task config + column metadata + original_length
    """
    cfg = TASK_CONFIGS[task]
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    n = cfg["n_samples"]
    n_feat = cfg["n_features"]
    n_cls = cfg["n_classes"]

    # ── Generate clean classification data ──────────────────────────────────
    X, y = make_classification(
        n_samples=n,
        n_features=n_feat,
        n_informative=cfg["n_informative"],
        n_redundant=max(0, n_feat - cfg["n_informative"] - 1),
        n_classes=n_cls,
        n_clusters_per_class=1,
        weights=None,
        random_state=seed,
    )

    cols = FEATURE_NAMES[:n_feat]
    df_clean = pd.DataFrame(X, columns=cols)
    df_clean["target"] = y

    # Rescale 'age' column to [18, 80] for plausibility
    if "age" in df_clean.columns:
        mn, mx = df_clean["age"].min(), df_clean["age"].max()
        df_clean["age"] = ((df_clean["age"] - mn) / (mx - mn + 1e-9)) * 62 + 18

    ground_truth = deepcopy(df_clean)
    working_copy = deepcopy(df_clean)

    # ── Inject corruptions into working_copy only ────────────────────────────
    _inject_corruptions(working_copy, task, cfg, rng, np_rng, seed)

    col_meta = _build_column_meta(cols, task)
    metadata = {
        **cfg,
        "task": task,
        "seed": seed,
        "feature_cols": cols,
        "col_meta": col_meta,
        "original_length": len(working_copy),
        "class_names": [str(c) for c in sorted(working_copy["target"].unique())],
    }

    return ground_truth, working_copy, metadata


def _inject_corruptions(df: pd.DataFrame, task: str, cfg: dict,
                        rng: random.Random, np_rng: np.random.RandomState,
                        seed: int):
    """Inject task-specific corruptions into df in-place."""

    if task == "task_0_tutorial":
        # Single issue: 20% missing in age only
        _inject_missing(df, ["age"], frac=0.20, rng=rng)

    elif task == "task_1_easy":
        # Missing values 15% + mild class imbalance
        cols = df.columns[:-1].tolist()
        _inject_missing(df, cols[:2], frac=0.15, rng=rng)
        _inject_class_imbalance(df, ratio=0.60, rng=rng, seed=seed)

    elif task == "task_2_medium":
        cols = df.columns[:-1].tolist()
        _inject_missing(df, cols[:3], frac=0.12, rng=rng)
        _inject_duplicates(df, frac=0.05, rng=rng)
        _inject_class_imbalance(df, ratio=0.55, rng=rng, seed=seed)
        _inject_type_error(df, cols[0], rng=rng, frac=0.04)

    elif task == "task_3_hard":
        cols = df.columns[:-1].tolist()
        _inject_missing(df, cols[:4], frac=0.10, rng=rng)
        _inject_duplicates(df, frac=0.05, rng=rng)
        _inject_class_imbalance(df, ratio=0.50, rng=rng, seed=seed)
        _inject_type_error(df, cols[0], rng=rng, frac=0.03)
        _inject_outliers(df, cols[1], rng=rng, frac=0.03)
        _inject_cross_column_errors(df, cols[2], cols[3], rng=rng, frac=0.02)


def _inject_missing(df: pd.DataFrame, cols: list, frac: float, rng: random.Random):
    for col in cols:
        if col not in df.columns:
            continue
        indices = rng.sample(range(len(df)), int(len(df) * frac))
        df.loc[indices, col] = np.nan


def _inject_duplicates(df: pd.DataFrame, frac: float, rng: random.Random):
    n_dups = max(1, int(len(df) * frac))
    dup_indices = rng.choices(range(len(df)), k=n_dups)
    dups = df.iloc[dup_indices].copy()
    new_df = pd.concat([df, dups], ignore_index=True)
    # Mutate the caller's DataFrame in-place by clearing and re-populating
    df.drop(df.index, inplace=True)
    df.drop(df.columns, axis=1, inplace=True)
    for col in new_df.columns:
        df[col] = new_df[col].values
    df.reset_index(drop=True, inplace=True)


def _inject_class_imbalance(df: pd.DataFrame, ratio: float,
                             rng: random.Random, seed: int):
    """Make class 0 account for `ratio` of rows, drop minority excess."""
    target_col = "target"
    classes = df[target_col].unique()
    if len(classes) < 2:
        return
    major = int(classes[0])
    n_major = int(len(df) * ratio)
    major_idx = df[df[target_col] == major].index.tolist()
    if len(major_idx) > n_major:
        drop_n = len(major_idx) - n_major
        to_drop = rng.sample(major_idx, drop_n)
        df.drop(to_drop, inplace=True)
        df.reset_index(drop=True, inplace=True)


def _inject_type_error(df: pd.DataFrame, col: str, rng: random.Random, frac: float):
    """Replace some float values with string 'ERR' to simulate type errors."""
    if col not in df.columns:
        return
    indices = rng.sample(range(len(df)), max(1, int(len(df) * frac)))
    df[col] = df[col].astype(object)
    for i in indices:
        df.at[i, col] = "ERR"


def _inject_outliers(df: pd.DataFrame, col: str, rng: random.Random, frac: float):
    if col not in df.columns:
        return
    indices = rng.sample(range(len(df)), max(1, int(len(df) * frac)))
    for i in indices:
        df.at[i, col] = rng.choice([999.0, -999.0])


def _inject_cross_column_errors(df: pd.DataFrame, col_a: str, col_b: str,
                                 rng: random.Random, frac: float):
    """Make col_a < col_b for some rows (e.g. min > max violations)."""
    if col_a not in df.columns or col_b not in df.columns:
        return
    indices = rng.sample(range(len(df)), max(1, int(len(df) * frac)))
    for i in indices:
        try:
            a = float(df.at[i, col_a])
            b = float(df.at[i, col_b])
            if a >= b:
                df.at[i, col_a], df.at[i, col_b] = b - 1.0, a + 1.0
        except (ValueError, TypeError):
            pass
