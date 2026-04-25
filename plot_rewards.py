"""
Plot Rewards — Data-Centric AI RL Environment
=============================================
Reads JSONL training logs and produces judge-ready plots with labeled axes.

Log format (one JSON object per line in logs/training.jsonl):
  {
    "ts": 1714000000.0,       # Unix timestamp
    "episode": 42,            # Episode number
    "task": "task_1_easy",    # Task name
    "level": 1,               # Curriculum level (0=tutorial ... 3=hard)
    "reward": 0.34,           # Episode reward
    "accuracy_gain": 0.08,    # Accuracy delta vs baseline
    "steps_used": 18,         # Steps consumed
    "success": true           # Reached target accuracy?
  }

Output (saved to plots/):
  reward_curve.png     — Episode reward with rolling mean
  success_rate.png     — Success rate per curriculum level
  accuracy_gain.png    — Accuracy gain distribution
  curriculum.png       — Curriculum level over episodes

Usage:
  python plot_rewards.py                          # default log path
  python plot_rewards.py --log logs/training.jsonl --out plots/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless/Colab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Style ────────────────────────────────────────────────────────────────────

LEVEL_COLORS = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868", 3: "#C44E52"}
LEVEL_NAMES  = {0: "tutorial", 1: "easy", 2: "medium", 3: "hard"}
FIGSIZE = (10, 4)
DPI = 150

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "grid.alpha": 0.3,
})


# ── Load log ─────────────────────────────────────────────────────────────────

def load_log(log_path: str) -> pd.DataFrame:
    """Load JSONL training log. Returns empty DataFrame if file not found."""
    path = Path(log_path)
    if not path.exists():
        print(f"[plot_rewards] Log not found: {log_path}")
        return pd.DataFrame()

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        print(f"[plot_rewards] Log is empty: {log_path}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Normalise column names — handle both old and new log formats
    col_map = {
        "mean_total_reward": "reward",
        "mean_env_reward": "accuracy_gain",
        "stage": "task",
    }
    df.rename(columns=col_map, inplace=True)
    if "episode" not in df.columns:
        df["episode"] = range(len(df))
    if "level" not in df.columns:
        df["level"] = 0
    if "success" not in df.columns:
        df["success"] = df.get("accuracy_gain", 0) > 0.05
    if "accuracy_gain" not in df.columns:
        df["accuracy_gain"] = 0.0
    if "reward" not in df.columns:
        df["reward"] = 0.0

    df.sort_values("episode", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[plot_rewards] Loaded {len(df)} episodes from {log_path}")
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_reward_curve(df: pd.DataFrame, out_dir: Path, window: int = 20):
    """Plot 1: Episode reward over training with rolling mean."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(df["episode"], df["reward"], alpha=0.25, color="steelblue",
            linewidth=0.8, label="Raw reward")

    if len(df) >= window:
        smooth = df["reward"].rolling(window, min_periods=1).mean()
        ax.plot(df["episode"], smooth, color="steelblue", linewidth=2.2,
                label=f"Rolling mean (window={window})")

    # Mark curriculum level transitions
    level_changes = df[df["level"].diff() != 0]
    for _, row in level_changes.iterrows():
        if row["level"] > 0:
            ax.axvline(row["episode"], color=LEVEL_COLORS.get(int(row["level"]), "gray"),
                       linewidth=1.0, linestyle="--", alpha=0.7)
            ax.text(row["episode"] + 0.5, ax.get_ylim()[1] * 0.95,
                    LEVEL_NAMES.get(int(row["level"]), ""), fontsize=8,
                    color=LEVEL_COLORS.get(int(row["level"]), "gray"), rotation=90, va="top")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Training Reward over Episodes")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()

    out_path = out_dir / "reward_curve.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"[plot_rewards] Saved: {out_path}")


def plot_success_rate(df: pd.DataFrame, out_dir: Path, window: int = 20):
    """Plot 2: Success rate per curriculum level."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    levels = sorted(df["level"].unique())
    for level in levels:
        subset = df[df["level"] == level].copy()
        subset = subset.sort_values("episode").reset_index(drop=True)
        rate = subset["success"].rolling(window, min_periods=1).mean()
        color = LEVEL_COLORS.get(int(level), "gray")
        label = f"Level {int(level)}: {LEVEL_NAMES.get(int(level), 'unknown')}"
        ax.plot(subset["episode"], rate, color=color, linewidth=2, label=label)

    ax.axhline(0.60, color="red", linewidth=1.0, linestyle="--", alpha=0.6,
               label="Advancement threshold (60%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Success rate (rolling mean, window={window})")
    ax.set_title("Success Rate per Curriculum Level")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()

    out_path = out_dir / "success_rate.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"[plot_rewards] Saved: {out_path}")


def plot_accuracy_gain(df: pd.DataFrame, out_dir: Path, window: int = 20):
    """Plot 3: Accuracy gain over training."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(df["episode"], df["accuracy_gain"], alpha=0.25, color="green",
            linewidth=0.8, label="Raw accuracy gain")

    if len(df) >= window:
        smooth = df["accuracy_gain"].rolling(window, min_periods=1).mean()
        ax.plot(df["episode"], smooth, color="green", linewidth=2.2,
                label=f"Rolling mean (window={window})")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Accuracy gain vs baseline")
    ax.set_title("Accuracy Gain per Episode")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()

    out_path = out_dir / "accuracy_gain.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"[plot_rewards] Saved: {out_path}")


def plot_curriculum(df: pd.DataFrame, out_dir: Path):
    """Plot 4: Curriculum level progression over time."""
    fig, ax = plt.subplots(figsize=FIGSIZE)

    colors = [LEVEL_COLORS.get(int(l), "gray") for l in df["level"]]
    ax.scatter(df["episode"], df["level"], c=colors, s=4, alpha=0.5, zorder=2)

    # Smooth line
    ax.plot(df["episode"], df["level"].rolling(10, min_periods=1).mean(),
            color="black", linewidth=1.5, alpha=0.6, label="Rolling mean level")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Curriculum level")
    ax.set_title("Curriculum Progression")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0: tutorial", "1: easy", "2: medium", "3: hard"])
    ax.grid(True, axis="x")

    patches = [mpatches.Patch(color=c, label=f"{l}: {LEVEL_NAMES[l]}")
               for l, c in LEVEL_COLORS.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    fig.tight_layout()

    out_path = out_dir / "curriculum.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"[plot_rewards] Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def plot_all(log_path: str = "logs/training.jsonl", out_dir: str = "plots/",
             window: int = 20):
    df = load_log(log_path)
    if df.empty:
        print("[plot_rewards] No data to plot.")
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_reward_curve(df, out, window)
    plot_success_rate(df, out, window)
    plot_accuracy_gain(df, out, window)
    plot_curriculum(df, out)

    print(f"\n[plot_rewards] All plots saved to {out}/")
    print(f"  Episodes: {len(df)} | "
          f"Avg reward: {df['reward'].mean():.3f} | "
          f"Success rate: {df['success'].mean():.1%} | "
          f"Max level reached: {int(df['level'].max())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training reward curves")
    parser.add_argument("--log", default="logs/training.jsonl",
                        help="Path to JSONL training log")
    parser.add_argument("--out", default="plots/",
                        help="Output directory for plots")
    parser.add_argument("--window", type=int, default=20,
                        help="Rolling mean window size")
    args = parser.parse_args()
    plot_all(args.log, args.out, args.window)
