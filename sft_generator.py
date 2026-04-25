"""
Ultra-Fast SFT Data Generator — No sklearn, No Environment Execution.

Instead of running the environment live, we generate realistic prompt/response
pairs directly from templates using known dataset states.

This is correct because:
 - We know exactly what inspect_dataset returns (from dataset_generator)
 - We know what query_cleaner returns (from specialist_agents)
 - We know the reward trajectory
 - The actual RL training will run the real environment — SFT just warms up
   the LLM's action distribution (command grammar + strategy)

Output: ~1000+ diverse examples in under 10 seconds.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.dataset_generator import TASK_CONFIGS

rng = random.Random(42)

TASKS = list(TASK_CONFIGS.keys())

# ── Prompt templates ──────────────────────────────────────────────────────────

def make_prompt(
    task: str,
    step: int,
    max_steps: int,
    current_acc: float,
    target_acc: float,
    baseline_acc: float,
    dataset_shape: str,
    rows_pct: float,
    quality: float,
    budget: int,
    session: str,
    validate_left: int,
    last_obs: str,
) -> str:
    gap = max(0.0, target_acc - current_acc)
    return (
        f"You are a Data-Centric AI agent improving an ML dataset.\n\n"
        f"Task: {task}\n"
        f"Step: {step}/{max_steps}\n"
        f"Current accuracy: {current_acc:.4f}  "
        f"Target: {target_acc:.4f}  Gap: {gap:.4f}\n"
        f"Baseline accuracy: {baseline_acc:.4f}\n"
        f"Dataset: {dataset_shape} | "
        f"Rows preserved: {rows_pct*100:.1f}%\n"
        f"Quality score: {quality:.4f} | "
        f"Budget remaining: {budget}\n"
        f"Active session: {session} | "
        f"Validate calls left: {validate_left}\n\n"
        f"Last observation:\n{last_obs}\n\n"
        f"What is your next command?"
    )


# ── Observation text snippets ────────────────────────────────────────────────

INSPECT_OBS_TEMPLATES = [
    "=== Dataset Inspection ===\nShape: {rows} rows × {cols} features\nOriginal rows: {rows} | Preserved: 100.0%\nDuplicates: {dups}\nMissing values:\n  {col}: {missing}\nClass distribution: {dist}\nDtypes: {{'age': 'float64', 'score': 'float64', 'target': 'int64'}}",
    "=== Dataset Inspection ===\nShape: {rows} rows × {cols} features\nDuplicates: {dups}\nMissing values:\n  {col}: {missing}\nClass distribution: {dist}",
]

INSPECT_MODEL_TEMPLATES = [
    "=== Model Inspection ===\nAccuracy: {acc:.4f}\n  Class 0: precision={p0:.3f} recall={r0:.3f} f1={f0:.3f}\n  Class 1: precision={p1:.3f} recall={r1:.3f} f1={f1:.3f}\nTarget: {target:.4f} | Not yet",
    "=== Model Inspection (cached) ===\nAccuracy: {acc:.4f}\nTarget: {target:.4f} | Not yet",
]

CLEANER_OBS_TEMPLATES = [
    "=== Cleaner Recommendations ===\n[1] Fill {n} missing values in '{col}' using mean ({mean:.2f})\n    type=fill_missing impact=+0.075 confidence=0.90\n[2] Remove {dups} duplicate rows\n    type=remove_duplicates impact=+0.020 confidence=0.95",
    "=== Cleaner Recommendations ===\n[1] Fill {n} missing values in '{col}' using mean ({mean:.2f})\n    type=fill_missing impact=+0.075 confidence=0.90\n[2] Fix {typos} type errors in 'income'\n    type=fix_type_errors impact=+0.040 confidence=0.75",
    "=== Cleaner Recommendations ===\n[1] Fill {n} missing values in '{col}' using mean ({mean:.2f})\n    type=fill_missing impact=+0.075 confidence=0.90",
]

BALANCER_OBS_TEMPLATES = [
    "=== Balancer Recommendations ===\n[1] Upsample minority class 1 from {min_c} to {maj_c} rows via random oversampling (imbalance ratio: {ratio:.2f})\n    type=oversample impact=+0.053 confidence=0.80",
    "=== Balancer Recommendations ===\n[1] Downsample majority class 0 from {maj_c} to {min_c} rows\n    type=undersample impact=+0.030 confidence=0.70",
]

APPLY_OBS_TEMPLATES = [
    "Applied: fill_missing [Fill {n} missing values in '{col}' using mean ({mean:.2f})]\n\nDataset health check:\n  Missing values: {remaining} remaining (was {was})\n  Duplicates: ✓ (was 0)\n  Row count: {rows}/{orig} (100.0% preserved)\n\nEstimated quality score: {quality:.4f}\nBudget remaining: {budget}",
    "Applied: remove_duplicates [Remove {dups} duplicate rows]\n\nDataset health check:\n  Missing values: {remaining} remaining (was {was})\n  Duplicates: ✓ (was {dups})\n  Row count: {rows}/{orig} ({pct:.1f}% preserved)\n\nEstimated quality score: {quality:.4f}\nBudget remaining: {budget}",
    "Applied: oversample [Upsample minority class 1 via random oversampling]\n\nDataset health check:\n  Missing values: 0 remaining (was 0)\n  Duplicates: 2 remaining (was 0)\n  Row count: {rows}/{orig} (102.0% preserved)\n\nEstimated quality score: {quality:.4f}\nBudget remaining: {budget}",
]

VALIDATE_OBS_TEMPLATES = [
    "=== Validate ===\nRF Accuracy: {acc:.4f}  (primary)\nLR Accuracy: {lr_acc:.4f}  (secondary)\nAgreement:   BOTH_AGREE_IMPROVE -- fix is robust and generalises\n  Class 0: p={p:.3f} r={r:.3f} f1={f:.3f}\n  Class 1: p={p:.3f} r={r:.3f} f1={f:.3f}\nTarget: {target:.4f} | {status}",
    "=== Validate ===\nRF Accuracy: {acc:.4f}  (primary)\nLR Accuracy: {lr_acc:.4f}  (secondary)\nAgreement:   BOTH_AGREE_IMPROVE -- fix is robust and generalises\nTarget: {target:.4f} | {status}",
    "=== Validate (cached) ===\nRF Accuracy: {acc:.4f}  (primary)\nLR Accuracy: {lr_acc:.4f}  (secondary)\nTarget: {target:.4f} | {status}",
]

ERROR_OBS_TEMPLATES = [
    "Error: Recommendation 1 has already been applied this session. Duplicate apply not allowed.",
    "Validate on cooldown. Take 1 more action(s) before validating again.",
    "Error: stale recommendation ID 99. Please re-query for fresh recommendations.",
]

RESET_OBS = (
    "Episode started: {task}\n"
    "Baseline accuracy: {baseline:.4f} | Target: {target:.4f}\n"
    "Dataset: {rows} rows x {cols} features\n"
    "Budget: {budget} steps\n\n"
    "Available commands:\n"
    "  inspect_dataset          - shape, dtypes, missing, class distribution\n"
    "  inspect_model            - accuracy (RF + LR), F1, feature importance\n"
    "  query_analyst            - holistic diagnosis + prioritised action plan (costs 1 budget)\n"
    "  query_cleaner            - get cleaning recommendations\n"
    "  query_augmenter [class]  - get augmentation suggestions\n"
    "  query_balancer           - get resampling recommendations\n"
    "  query_validator          - check rule violations (costs 2 budget)\n"
    "  apply [id]               - apply recommendation by ID\n"
    "  reject [id]              - reject a recommendation\n"
    "  validate                 - retrain and score (cooldown applies)\n"
    "  submit                   - finalize episode"
)

ANALYST_OBS_TEMPLATES = [
    "=== Analyst Report (costs 1 budget) ===\nDIAGNOSIS:\n  - Class Imbalance: severity={imb:.2f} [HIGH] -> use query_balancer\n  - Missing Values: severity={miss:.2f} [MEDIUM] -> use query_cleaner\n  - Type Errors: severity=0.00 [NONE]\n  - Accuracy gap: {gap:.4f} (significant gap)\n\nRECOMMENDED PLAN (budget remaining: {budget}):\n  1. query_balancer -> apply best recommendation\n  2. query_cleaner -> apply best recommendation\n  3. validate (check accuracy improvement)\n  4. submit if accuracy >= target\n\nPRIORITY NOTE: Class imbalance is the dominant issue -- fix this first.",
    "=== Analyst Report (costs 1 budget) ===\nDIAGNOSIS:\n  - Missing Values: severity={miss:.2f} [HIGH] -> use query_cleaner\n  - Class Imbalance: severity={imb:.2f} [LOW] -> use query_balancer\n  - Type Errors: severity=0.00 [NONE]\n  - Accuracy gap: {gap:.4f} (significant gap)\n\nRECOMMENDED PLAN (budget remaining: {budget}):\n  1. query_cleaner -> apply best recommendation\n  2. query_balancer -> apply best recommendation\n  3. validate\n  4. submit",
]


# ── Episode builders ─────────────────────────────────────────────────────────

def sample_dataset_params(task: str, seed: int):
    """Sample realistic dataset params for a given task."""
    cfg = TASK_CONFIGS[task]
    rng2 = random.Random(seed)
    rows_map = {"task_0_tutorial": 100, "task_1_easy": 200,
                "task_2_medium": 500, "task_3_hard": 900}
    cols_map = {"task_0_tutorial": 4, "task_1_easy": 5,
                "task_2_medium": 7, "task_3_hard": 10}
    rows = rows_map[task]
    cols = cols_map[task]
    missing_cols = ["age", "income", "score"][:rng2.randint(1, 3)]
    missing_pct = rng2.uniform(0.10, 0.30)
    n_missing = int(rows * missing_pct)
    mean_val = rng2.uniform(30.0, 60.0)
    dups = rng2.randint(0, int(rows * 0.05))
    maj_class = int(rows * rng2.uniform(0.52, 0.65))
    min_class = rows - maj_class
    return {
        "task": task, "rows": rows, "cols": cols,
        "missing_col": missing_cols[0], "n_missing": n_missing,
        "mean_val": round(mean_val, 2), "dups": dups,
        "maj_class": maj_class, "min_class": min_class,
        "baseline": cfg["baseline_accuracy"],
        "target": cfg["target_accuracy"],
        "budget": cfg["budget"],
    }


def build_episode(task: str, seed: int, strategy: list) -> list:
    """
    Build a synthetic SFT episode using template obs + fixed action sequence.
    Returns list of {prompt, response} dicts.
    """
    p = sample_dataset_params(task, seed)
    cfg = TASK_CONFIGS[task]
    examples = []

    acc = p["baseline"]
    quality = round(rng.uniform(0.45, 0.65), 4)
    rows = p["rows"]
    missing_remaining = p["n_missing"]
    budget = p["budget"]
    session = "none"
    validate_left = 3
    prev_obs = RESET_OBS.format(
        task=task, baseline=p["baseline"], target=p["target"],
        rows=rows, cols=p["cols"], budget=budget
    )

    for step, action in enumerate(strategy):
        prompt = make_prompt(
            task=task, step=step, max_steps=p["budget"],
            current_acc=acc, target_acc=p["target"], baseline_acc=p["baseline"],
            dataset_shape=f"{rows} rows × {p['cols']} columns",
            rows_pct=rows / p["rows"], quality=quality, budget=budget,
            session=session, validate_left=validate_left, last_obs=prev_obs,
        )
        examples.append({"prompt": prompt, "response": action})

        # Simulate observation update
        budget -= 1
        cmd = action.split()[0].lower()

        if cmd == "inspect_dataset":
            t = rng.choice(INSPECT_OBS_TEMPLATES)
            dist = f"class 0: {p['maj_class']}, class 1: {p['min_class']}"
            prev_obs = t.format(
                rows=rows, cols=p["cols"], dups=p["dups"],
                col=p["missing_col"], missing=missing_remaining, dist=dist,
            )
        elif cmd == "inspect_model":
            t = rng.choice(INSPECT_MODEL_TEMPLATES)
            p0 = round(rng.uniform(0.55, 0.75), 3)
            r0 = round(rng.uniform(0.55, 0.75), 3)
            prev_obs = t.format(
                acc=acc, target=p["target"],
                p0=p0, r0=r0, f0=round(2*p0*r0/(p0+r0+1e-9), 3),
                p1=p0, r1=r0, f1=round(2*p0*r0/(p0+r0+1e-9), 3),
            )
        elif cmd == "query_cleaner":
            t = rng.choice(CLEANER_OBS_TEMPLATES)
            session = f"cleaner:{seed:08x}"
            prev_obs = t.format(
                n=missing_remaining, col=p["missing_col"],
                mean=p["mean_val"], dups=p["dups"], typos=rng.randint(2, 8),
            )
        elif cmd == "query_balancer":
            t = rng.choice(BALANCER_OBS_TEMPLATES)
            session = f"balancer:{seed:08x}"
            ratio = round(p["min_class"] / max(p["maj_class"], 1), 2)
            prev_obs = t.format(
                min_c=p["min_class"], maj_c=p["maj_class"], ratio=ratio
            )
        elif cmd == "query_augmenter":
            session = f"augmenter:{seed:08x}"
            cls = action.split()[1] if len(action.split()) > 1 else "0"
            n_synth = rng.randint(5, 25)
            prev_obs = (
                f"=== Augmenter Recommendations ===\n"
                f"[1] Synthesize {n_synth} samples for class {cls} via SMOTE\n"
                f"    type=augment_class impact=+0.040 confidence=0.72"
            )
        elif cmd == "query_analyst":
            budget -= 1  # costs 1 extra
            t = rng.choice(ANALYST_OBS_TEMPLATES)
            imb = round(rng.uniform(0.3, 0.8), 2)
            miss = round(rng.uniform(0.1, 0.5), 2)
            gap = round(p["target"] - acc, 4)
            prev_obs = t.format(imb=imb, miss=miss, gap=gap, budget=budget)
        elif cmd == "query_validator":
            budget -= 1  # costs 2
            prev_obs = (
                "=== Validator Report (costs 2 budget) ===\n"
                f"  [WARNING] [{p['missing_col']}] rule=no_missing "
                f"count={missing_remaining}\n"
                f"    Column '{p['missing_col']}' has {missing_remaining} missing values."
            )
        elif cmd == "apply":
            rec_id = int(action.split()[1]) if len(action.split()) > 1 else 1
            t = rng.choice(APPLY_OBS_TEMPLATES)
            was_missing = missing_remaining
            missing_remaining = max(0, missing_remaining - p["n_missing"])
            quality = min(1.0, quality + rng.uniform(0.10, 0.35))
            quality = round(quality, 4)
            prev_obs = t.format(
                n=p["n_missing"], col=p["missing_col"], mean=p["mean_val"],
                remaining=missing_remaining, was=was_missing,
                rows=rows, orig=p["rows"], pct=rows/p["rows"]*100,
                dups=p["dups"], quality=quality, budget=budget,
            )
        elif cmd == "reject":
            prev_obs = f"Recommendation {action.split()[1] if len(action.split())>1 else 1} rejected."
        elif cmd == "validate":
            if validate_left > 0:
                acc = min(1.0, acc + rng.uniform(0.05, 0.35))
                acc = round(acc, 4)
                lr_acc = round(min(1.0, acc + rng.uniform(-0.03, 0.03)), 4)
                validate_left -= 1
                t = rng.choice(VALIDATE_OBS_TEMPLATES)
                status = "HIT v" if acc >= p["target"] else "Not yet"
                pv = round(rng.uniform(0.75, 0.98), 3)
                rv = round(rng.uniform(0.75, 0.98), 3)
                prev_obs = t.format(
                    acc=acc, lr_acc=lr_acc, target=p["target"], status=status,
                    p=pv, r=rv, f=round(2*pv*rv/(pv+rv+1e-9), 3),
                )
            else:
                prev_obs = "Validate on cooldown. Take 2 more action(s) before validating again."
        elif cmd == "submit":
            break

    return examples


# ── Strategy sequences ────────────────────────────────────────────────────────

STRATEGIES = {
    "minimal_clean":             ["inspect_dataset", "query_cleaner", "apply 1", "apply 2", "inspect_dataset", "validate", "submit"],
    "inspect_model_first":       ["inspect_dataset", "inspect_model", "query_cleaner", "apply 1", "inspect_dataset", "validate", "submit"],
    "clean_then_balance":        ["inspect_dataset", "query_cleaner", "apply 1", "apply 2", "query_balancer", "apply 1", "inspect_dataset", "validate", "submit"],
    "reject_then_apply":         ["inspect_dataset", "query_cleaner", "reject 1", "apply 2", "inspect_dataset", "validate", "submit"],
    "baseline_validate_first":   ["inspect_dataset", "validate", "query_cleaner", "apply 1", "inspect_dataset", "validate", "submit"],
    "augment_path":              ["inspect_dataset", "query_cleaner", "apply 1", "query_augmenter 0", "apply 1", "inspect_dataset", "validate", "submit"],
    "with_validator":            ["inspect_dataset", "query_validator", "query_cleaner", "apply 1", "inspect_dataset", "validate", "submit"],
    "deep_clean_requery":        ["inspect_dataset", "query_cleaner", "apply 1", "apply 2", "query_cleaner", "apply 1", "inspect_dataset", "validate", "submit"],
    "fast_submit":               ["query_cleaner", "apply 1", "apply 2", "inspect_dataset", "submit"],
    "balance_heavy":             ["inspect_dataset", "query_balancer", "apply 1", "query_cleaner", "apply 1", "inspect_dataset", "validate", "submit"],
    "reject_requery":            ["inspect_dataset", "query_cleaner", "reject 1", "reject 2", "query_cleaner", "apply 1", "inspect_dataset", "validate", "submit"],
    "multi_augment":             ["inspect_dataset", "query_cleaner", "apply 1", "query_augmenter 1", "apply 1", "inspect_dataset", "validate", "submit"],
    "model_then_balance":        ["inspect_model", "query_balancer", "apply 1", "inspect_dataset", "validate", "submit"],
    "full_pipeline":             ["inspect_dataset", "inspect_model", "query_cleaner", "apply 1", "query_balancer", "apply 1", "query_augmenter 0", "apply 1", "inspect_dataset", "validate", "submit"],
    "suboptimal_no_validate":    ["inspect_dataset", "query_cleaner", "apply 1", "submit"],
    "inspect_only_submit":       ["inspect_dataset", "inspect_model", "submit"],
    "reject_all_then_requery":   ["inspect_dataset", "query_cleaner", "reject 1", "reject 2", "query_balancer", "apply 1", "inspect_dataset", "validate", "submit"],
    "apply3_then_validate":      ["inspect_dataset", "query_cleaner", "apply 1", "apply 2", "query_balancer", "apply 1", "query_augmenter 0", "apply 1", "inspect_dataset", "validate", "submit"],
    # NEW: analyst-led strategies
    "analyst_led_clean":         ["query_analyst", "inspect_dataset", "query_cleaner", "apply 1", "apply 2", "validate", "submit"],
    "analyst_led_balance":       ["query_analyst", "query_balancer", "apply 1", "query_cleaner", "apply 1", "validate", "submit"],
    "analyst_full_pipeline":     ["query_analyst", "inspect_dataset", "inspect_model", "query_cleaner", "apply 1", "query_balancer", "apply 1", "validate", "submit"],
}


def generate_sft_data(output_file: str = "sft_data.jsonl", seeds_per_combo: int = 15):
    sft_examples = []

    print(f"Generating SFT data: {len(STRATEGIES)} strategies × {len(TASKS)} tasks × {seeds_per_combo} seeds")

    for strategy_name, sequence in STRATEGIES.items():
        strategy_examples = []
        for task in TASKS:
            for seed in range(seeds_per_combo):
                episode = build_episode(task, seed, sequence)
                strategy_examples.extend(episode)
        sft_examples.extend(strategy_examples)
        print(f"  {strategy_name:<30} +{len(strategy_examples)} examples")

    rng.shuffle(sft_examples)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")

    # Diversity report
    from collections import Counter
    responses = [ex["response"] for ex in sft_examples]
    unique_cmds = set(responses)

    print(f"\n{'='*55}")
    print(f"Total examples:   {len(sft_examples)}")
    print(f"Unique commands:  {len(unique_cmds)}")
    print(f"Unique prompts:   {len(set(ex['prompt'] for ex in sft_examples))}")
    print(f"\nResponse distribution:")
    for cmd, cnt in Counter(responses).most_common():
        pct = cnt / len(responses) * 100
        bar = "#" * int(pct / 2)
        flag = " ← DOMINANT" if pct > 25 else ""
        print(f"  {cmd:<32} {cnt:>5}  ({pct:5.1f}%)  {bar}{flag}")

    print(f"\nOutput: {out_path}")
    print("✓ SFT generation complete (no sklearn, instant).")
    return sft_examples


if __name__ == "__main__":
    generate_sft_data()
