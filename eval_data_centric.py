"""
eval_data_centric.py — Evaluation script for DataCentricEnv.

Runs two agents on identical seeds for fair comparison:
  - Random Agent:  picks valid commands at random (baseline)
  - Trained Agent: uses the fine-tuned model from ./data-centric-adapter

Produces eval_results.json for use by plot_rewards.py.
"""

import json
import os
import random
import signal
import subprocess
import sys
import time
from typing import Optional

import requests
import torch
from unsloth import FastLanguageModel

# WebSocket client for stateful episode rollouts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import DataCentricEnv
from models import DataCentricAction
from agent_utils import (
    VALID_COMMANDS, SYSTEM_PROMPT, build_user_prompt,
    start_server, stop_server,
)

# ════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════

BASE_URL = os.environ.get("ENV_URL", "http://localhost:8000")
ADAPTER_PATH = "./data-centric-adapter"
MAX_SEQ_LENGTH = 1024
EPISODES_PER_TASK = 10
TASKS = ["task_0_tutorial", "task_1_easy", "task_2_medium", "task_3_hard"]

# ════════════════════════════════════════════════════════
# SERVER MANAGEMENT
# ════════════════════════════════════════════════════════



# ════════════════════════════════════════════════════════
# MODEL LOADING
# ════════════════════════════════════════════════════════

def load_trained_model():
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(
            f"Adapter not found at {ADAPTER_PATH}. "
            "Run train_data_centric.py first."
        )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ════════════════════════════════════════════════════════
# EPISODE METRICS
# ════════════════════════════════════════════════════════

def episode_metrics(
    task: str,
    seed: int,
    final_obs: dict,
    actions: list,
    baseline_accuracy: float,
    max_steps: int,
) -> dict:
    """Compute per-episode metrics for a single completed episode."""
    final_accuracy = final_obs.get("current_accuracy", baseline_accuracy)
    budget_remaining = final_obs.get("budget_remaining", 0)
    target_accuracy = final_obs.get("target_accuracy", 1.0)
    budget_used = max_steps - budget_remaining

    accuracy_improvement = final_accuracy - baseline_accuracy
    target_hit = final_accuracy >= target_accuracy
    budget_efficiency = (
        accuracy_improvement / max(budget_used, 1)
    )

    # Format rate: % actions that are valid commands
    valid_count = sum(
        1 for a in actions
        if any(a.strip().startswith(cmd.split()[0]) for cmd in VALID_COMMANDS)
    )
    format_rate = valid_count / max(len(actions), 1)

    # Strategy rate: % query→apply consecutive pairs
    strategy_hits = 0
    for i in range(1, len(actions)):
        if (actions[i - 1].startswith("query_")
                and actions[i].startswith("apply")):
            strategy_hits += 1
    strategy_rate = strategy_hits / max(len(actions) - 1, 1)

    return {
        "task":                 task,
        "seed":                 seed,
        "final_accuracy":       round(final_accuracy, 4),
        "baseline_accuracy":    round(baseline_accuracy, 4),
        "target_accuracy":      round(target_accuracy, 4),
        "accuracy_improvement": round(accuracy_improvement, 4),
        "target_hit":           target_hit,
        "budget_used":          budget_used,
        "budget_efficiency":    round(budget_efficiency, 6),
        "format_rate":          round(format_rate, 4),
        "strategy_rate":        round(strategy_rate, 4),
        "n_actions":            len(actions),
    }


# ════════════════════════════════════════════════════════
# RANDOM AGENT
# ════════════════════════════════════════════════════════

def run_random_episode(task: str, seed: int) -> Optional[dict]:
    """Run one episode with a random agent using the WebSocket client."""
    try:
        with DataCentricEnv(base_url=BASE_URL).sync() as env:
            r_reset = env.reset(task=task, seed=seed)
            obs = r_reset.observation
            baseline_accuracy = obs.baseline_accuracy
            max_steps = obs.max_steps
            actions = []

            while not obs.done:
                action = random.choice(VALID_COMMANDS)
                actions.append(action)
                try:
                    step_result = env.step(DataCentricAction(message=action))
                    obs = step_result.observation
                except Exception:
                    break

            return episode_metrics(
                task, seed,
                {"current_accuracy": obs.current_accuracy,
                 "budget_remaining": obs.budget_remaining,
                 "target_accuracy":  obs.target_accuracy,
                 "done":             obs.done},
                actions, baseline_accuracy, max_steps
            )
    except Exception as e:
        print(f"  [random] Episode failed: {e}")
        return None


# ════════════════════════════════════════════════════════
# TRAINED AGENT
# ════════════════════════════════════════════════════════

def run_trained_episode(
    model, tokenizer, task: str, seed: int
) -> Optional[dict]:
    """Run one episode with the trained model using the WebSocket client."""
    try:
        with DataCentricEnv(base_url=BASE_URL).sync() as env:
            r_reset = env.reset(task=task, seed=seed)
            obs = r_reset.observation
            baseline_accuracy = obs.baseline_accuracy
            max_steps = obs.max_steps
            actions = []

            while not obs.done:
                obs_dict = {
                    "current_accuracy":        obs.current_accuracy,
                    "target_accuracy":         obs.target_accuracy,
                    "estimated_quality":       obs.estimated_quality,
                    "rows_preserved_pct":      obs.rows_preserved_pct,
                    "budget_remaining":        obs.budget_remaining,
                    "validate_calls_remaining":obs.validate_calls_remaining,
                    "active_session":          obs.active_session,
                    "response":                obs.response,
                }
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(obs_dict)},
                ]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    max_length=MAX_SEQ_LENGTH - 60,
                    truncation=True,
                    add_generation_prompt=True,
                ).to(model.device)

                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=50,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                action = tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip().split("\n")[0].strip()[:200]

                actions.append(action)
                try:
                    step_result = env.step(DataCentricAction(message=action))
                    obs = step_result.observation
                except Exception as e:
                    break

            return episode_metrics(
                task, seed,
                {"current_accuracy": obs.current_accuracy,
                 "budget_remaining": obs.budget_remaining,
                 "target_accuracy":  obs.target_accuracy,
                 "done":             obs.done},
                actions, baseline_accuracy, max_steps
            )
    except Exception as e:
        print(f"  [trained] Episode failed: {e}")
        return None


# ════════════════════════════════════════════════════════
# AGGREGATION
# ════════════════════════════════════════════════════════

def aggregate(episodes: list) -> dict:
    """Compute mean metrics across a list of episode result dicts."""
    if not episodes:
        return {}
    keys = [
        "accuracy_improvement", "target_hit", "budget_used",
        "budget_efficiency", "format_rate", "strategy_rate",
    ]
    return {
        k: round(sum(ep[k] for ep in episodes) / len(episodes), 4)
        for k in keys
    }


def print_comparison_table(random_agg: dict, trained_agg: dict):
    """Print a formatted comparison table to stdout."""
    def pct_change(r, t):
        if r == 0:
            return "—"
        return f"{(t - r) / abs(r) * 100:+.0f}%"

    def pp_change(r, t):
        return f"{(t - r) * 100:+.0f}pp"

    rows = [
        ("Accuracy gain",      f"{random_agg.get('accuracy_improvement',0):.3f}",
                               f"{trained_agg.get('accuracy_improvement',0):.3f}",
                               pct_change(random_agg.get('accuracy_improvement',0),
                                          trained_agg.get('accuracy_improvement',0))),
        ("Target hit rate",    f"{random_agg.get('target_hit',0):.0%}",
                               f"{trained_agg.get('target_hit',0):.0%}",
                               pp_change(random_agg.get('target_hit',0),
                                         trained_agg.get('target_hit',0))),
        ("Budget efficiency",  f"{random_agg.get('budget_efficiency',0):.4f}",
                               f"{trained_agg.get('budget_efficiency',0):.4f}",
                               pct_change(random_agg.get('budget_efficiency',0),
                                          trained_agg.get('budget_efficiency',0))),
        ("Format rate",        "random",
                               f"{trained_agg.get('format_rate',0):.0%}", "—"),
        ("Strategy rate",      "0%",
                               f"{trained_agg.get('strategy_rate',0):.0%}", "—"),
    ]

    header = f"{'Metric':<20} {'Random':>12} {'Trained':>13} {'Improvement':>14}"
    sep    = "─" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for metric, rand, trained, improvement in rows:
        print(f"  {metric:<18} {rand:>12} {trained:>13} {improvement:>14}")
    print(sep + "\n")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    server_proc = start_server()

    try:
        print(f"\nLoading trained model from {ADAPTER_PATH}...")
        model, tokenizer = load_trained_model()

        # Use fixed seeds so both agents see identical tasks
        seeds = list(range(EPISODES_PER_TASK))

        results = {
            "random":  {"all_episodes": [], "by_task": {}},
            "trained": {"all_episodes": [], "by_task": {}},
        }

        for task in TASKS:
            print(f"\n{'='*50}")
            print(f"Task: {task}")
            print("─" * 50)

            random_eps, trained_eps = [], []

            for seed in seeds:
                print(f"  Seed {seed:2d}:", end="  ")

                # Random agent
                sys.stdout.write("[random] ")
                sys.stdout.flush()
                r_ep = run_random_episode(task, seed)
                if r_ep:
                    random_eps.append(r_ep)
                    sys.stdout.write(
                        f"acc={r_ep['final_accuracy']:.3f} "
                        f"hit={'✓' if r_ep['target_hit'] else '✗'}  "
                    )

                # Trained agent (same seed)
                sys.stdout.write("[trained] ")
                sys.stdout.flush()
                t_ep = run_trained_episode(model, tokenizer, task, seed)
                if t_ep:
                    trained_eps.append(t_ep)
                    sys.stdout.write(
                        f"acc={t_ep['final_accuracy']:.3f} "
                        f"hit={'✓' if t_ep['target_hit'] else '✗'}"
                    )

                print()

            results["random"]["by_task"][task]  = aggregate(random_eps)
            results["trained"]["by_task"][task] = aggregate(trained_eps)
            results["random"]["all_episodes"].extend(random_eps)
            results["trained"]["all_episodes"].extend(trained_eps)

        # Overall aggregates
        results["random"]["overall"]  = aggregate(results["random"]["all_episodes"])
        results["trained"]["overall"] = aggregate(results["trained"]["all_episodes"])

        # Print comparison table
        print_comparison_table(
            results["random"]["overall"],
            results["trained"]["overall"],
        )

        # Print per-task breakdown
        print("Per-task summary:")
        for task in TASKS:
            r = results["random"]["by_task"].get(task, {})
            t = results["trained"]["by_task"].get(task, {})
            print(
                f"  {task:<22}  "
                f"random: acc+{r.get('accuracy_improvement',0):.3f} "
                f"hit={r.get('target_hit',0):.0%}  |  "
                f"trained: acc+{t.get('accuracy_improvement',0):.3f} "
                f"hit={t.get('target_hit',0):.0%}"
            )

        # Save full results
        json.dump(results, open("eval_results.json", "w"), indent=2)
        print("\nResults saved to eval_results.json")
        print("Run plot_rewards.py to visualise.")

    finally:
        stop_server(server_proc)
