"""
Heuristic baseline agent for the Data-Centric RL Environment.

Verifies the environment works correctly before any LLM training.
Run on all 4 tasks, 5 seeds each. Prints a results table.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DataCentricAction, DataCentricObservation
from server.data_centric_environment import DataCentricEnvironment
from server.dataset_generator import TASK_CONFIGS


def heuristic_agent(obs: DataCentricObservation, step: int, state: dict) -> str:
    """
    Simple heuristic agent that follows:
    inspect → query_cleaner → apply 1 → apply 2 → validate → query_balancer
    → apply 1 → validate → submit
    """
    if step == 0:
        return "inspect_dataset"
    if not state.get("queried_cleaner"):
        state["queried_cleaner"] = True
        return "query_cleaner"
    if state.get("cleaner_applies", 0) < 2:
        n = state.get("cleaner_applies", 0) + 1
        state["cleaner_applies"] = n
        return f"apply {n}"
    if not state.get("validated"):
        state["validated"] = True
        return "validate"
    if obs.current_accuracy < obs.target_accuracy and not state.get("queried_balancer"):
        state["queried_balancer"] = True
        return "query_balancer"
    if state.get("queried_balancer") and not state.get("balancer_applied"):
        state["balancer_applied"] = True
        return "apply 1"
    if state.get("queried_balancer") and state.get("balancer_applied") and not state.get("validated2"):
        state["validated2"] = True
        return "validate"
    return "submit"


def run_heuristic(task: str, seed: int) -> dict:
    env = DataCentricEnvironment()
    obs = env.reset(task=task, seed=seed)
    state = {}
    total_reward = 0.0

    for step in range(TASK_CONFIGS[task]["budget"]):
        action_msg = heuristic_agent(obs, step, state)
        result_obs = env.step(DataCentricAction(message=action_msg))
        total_reward += result_obs.reward
        obs = result_obs
        if obs.done:
            break

    return {
        "task": task,
        "seed": seed,
        "final_accuracy": obs.current_accuracy,
        "target": obs.target_accuracy,
        "hit": obs.current_accuracy >= obs.target_accuracy,
        "budget_used": obs.step_number,
        "total_reward": round(total_reward, 4),
    }


def main():
    tasks = list(TASK_CONFIGS.keys())
    seeds = [0, 1, 2, 3, 4]

    print(f"\n{'Task':<20} {'Seed':<6} {'Accuracy':<12} {'Target':<10} {'Hit?':<6} {'Budget':<10} {'Reward'}")
    print("-" * 80)

    hits = 0
    total = 0
    for task in tasks:
        for seed in seeds:
            r = run_heuristic(task, seed)
            hit_str = "Y" if r["hit"] else "N"
            if r["hit"]:
                hits += 1
            total += 1
            print(
                f"{r['task']:<20} {r['seed']:<6} {r['final_accuracy']:<12.4f} "
                f"{r['target']:<10.4f} {hit_str:<6} {r['budget_used']:<10} {r['total_reward']}"
            )

    print("-" * 80)
    print(f"Hit rate: {hits}/{total} ({100*hits/total:.0f}%)")
    print()
    if hits / total >= 0.6:
        print("  PASS: Heuristic agent validation passed.")
    else:
        print("  WARN: Hit rate below 60%. Check environment tuning.")


if __name__ == "__main__":
    main()
