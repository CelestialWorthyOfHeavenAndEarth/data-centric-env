"""
Grader for Data-Centric RL Environment — using OpenEnv Rubric system.

## Key Design Principle: REWARD DISCRIMINATION
The reward must clearly separate:
  - Bad agent  (random actions)     → large negative
  - Mediocre   (some fixes, inefficient) → near 0
  - Good agent (correct, efficient) → +0.5 to +0.8
  - Perfect    (fast, accurate, clean) → ~1.0

Reward saturation at 1.0 every episode = no learning gradient.
Every rubric is tuned to penalise sub-optimal behaviour strictly.

Rubric hierarchy:
    DataCentricRubric
    ├── accuracy      : AccuracyRubric     — main RL signal with efficiency multiplier
    ├── process       : ProcessRubric      — strict workflow enforcement
    ├── preservation  : PreservationRubric — anti-deletion exploit
    └── efficiency    : EfficiencyRubric   — surgical vs spray-and-pray

Also provides StepRubric for dense per-apply proxy feedback (no classifier).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from openenv.core.rubrics.base import Rubric

logger = logging.getLogger(__name__)

# Must match openenv.yaml reward_range
REWARD_MIN: float = -1.0
REWARD_MAX: float =  1.0


# ── Lightweight quality score (no sklearn) ────────────────────────────────────

def compute_lightweight_score(
    working_copy: pd.DataFrame,
    ground_truth: pd.DataFrame,
    original_length: int,
    col_meta: Dict,
    initial_missing: int = None,
) -> float:
    """
    Fast quality score [0.0, 1.0] comparing working_copy to ground_truth structure.
    Does NOT run sklearn — used for dense per-step feedback.
    Composed of:
      - missing value reduction (40%)
      - duplicate reduction     (20%)
      - type correctness        (20%)
      - row preservation        (20%)
    """
    score = 0.0

    # 1. Missing value reduction
    wc_missing = int(working_copy.isnull().sum().sum())
    denom = initial_missing if (initial_missing is not None and initial_missing > 0) else max(wc_missing, 1)
    missing_score = max(0.0, 1.0 - wc_missing / denom) if denom > 0 else 1.0
    score += 0.40 * missing_score

    # 2. Duplicate reduction
    n_dups_wc = int(working_copy.duplicated().sum())
    n_dups_gt = int(ground_truth.duplicated().sum())
    if n_dups_gt == 0 and n_dups_wc == 0:
        dup_score = 1.0
    elif n_dups_gt == 0:
        dup_score = max(0.0, 1.0 - n_dups_wc / max(len(working_copy), 1))
    else:
        dup_score = max(0.0, 1.0 - n_dups_wc / max(n_dups_gt, 1))
    score += 0.20 * dup_score

    # 3. Type correctness
    type_ok, type_total = 0, 0
    for col, meta in col_meta.items():
        if col == "target" or col not in working_copy.columns:
            continue
        if meta.get("expected_dtype", "float64") in ("float64", "int64"):
            type_total += 1
            err_count = sum(
                1 for val in working_copy[col].dropna()
                if not _can_float(val)
            )
            if err_count == 0:
                type_ok += 1
    score += 0.20 * ((type_ok / type_total) if type_total > 0 else 1.0)

    # 4. Row preservation
    score += 0.20 * min(len(working_copy) / max(original_length, 1), 1.0)

    return round(min(score, 1.0), 4)


def _can_float(val: Any) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


# ── Rubric 1: Accuracy ────────────────────────────────────────────────────────

class AccuracyRubric(Rubric):
    """
    Main RL signal. Key changes for discrimination:

    1. EFFICIENCY MULTIPLIER at submit: reward = gain × (budget_remaining / budget_total)
       An agent that hits target in 5 steps gets 5x more reward than one using 25 steps.
       This forces the agent to learn efficient strategies, not just any strategy.

    2. ABOVE-TARGET STRETCH: accuracy above target is additionally rewarded.
       Reaching 0.85 when target is 0.73 scores higher than exactly hitting 0.73.
       Judges see this as evidence of a genuinely capable agent.

    3. STRICT FAILURE PENALTY: if submit is called and target not hit,
       penalty = -0.4 × (1 - progress_to_target). Agent can't hide behind partial credit.

    4. MID-EPISODE REGRESSION: per-step penalty is steeper than reward (2.5× up, 3.0× down).
    """

    def forward(self, action: Any, observation: Any) -> float:
        current   = observation.get("current_accuracy", 0.0)
        previous  = observation.get("previous_accuracy", 0.0)
        baseline  = observation.get("baseline_accuracy", 0.0)
        target    = observation.get("target_accuracy", 0.80)
        is_submit = str(action).strip().lower() == "submit"

        # ── Mid-episode: reward Δ accuracy strictly ──────────────────────────
        improvement = current - previous
        if improvement > 0:
            reward = improvement * 2.5      # gain is rewarded
        elif improvement < 0:
            reward = improvement * 3.0      # regression penalised harder (was 2.0)
        else:
            reward = 0.0

        if not is_submit:
            return round(reward, 4)

        # ── Terminal: compute final score with efficiency multiplier ─────────
        budget_used  = observation.get("budget_used", 1)
        budget_total = observation.get("budget_total", 30)
        accuracy_gain = current - baseline

        if current >= target:
            # Base bonus for hitting target
            base_bonus = 0.35

            # Efficiency multiplier: reward using fewer steps
            # Range: [0.0 (all budget used) → 0.30 (1 step used)]
            budget_fraction_remaining = max(0.0, 1.0 - budget_used / max(budget_total, 1))
            efficiency_bonus = 0.30 * budget_fraction_remaining   # max +0.30

            # Stretch bonus: accuracy above target (up to +0.15)
            stretch = current - target
            stretch_bonus = min(stretch * 3.0, 0.15)

            reward += base_bonus + efficiency_bonus + stretch_bonus
        else:
            # Target not hit — strict penalty
            progress_range = target - baseline
            if progress_range > 0:
                progress = (current - baseline) / progress_range
                progress = max(0.0, min(progress, 1.0))
            else:
                progress = 0.0
            # The closer to baseline, the harsher the penalty
            reward += -0.40 * (1.0 - progress)

        logger.debug(
            "AccuracyRubric: imp=%.4f is_submit=%s reward=%.4f budget_used=%d/%d",
            improvement, is_submit, reward, budget_used, budget_total
        )
        return round(reward, 4)


# ── Rubric 2: Process ─────────────────────────────────────────────────────────

class ProcessRubric(Rubric):
    """
    Strict workflow enforcement. Key changes for discrimination:

    1. BLIND APPLY penalty increased: -0.08 (was -0.04). Applying without querying
       is the clearest signal of poor strategy — penalise it hard.

    2. REPEATED SAME QUERY without apply in between: -0.05 per repeat.
       Forces the agent to act on what it learns, not spam queries.

    3. SUBMIT WITHOUT VALIDATE: -0.15 (was -0.10). Submitting blind is sloppy.

    4. VALIDATE THEN APPLY THEN VALIDATE (correct loop): +0.06 bonus.
       This is the ideal pattern — reward it visibly.

    5. REDUNDANT VALIDATE (validate twice in a row): -0.08 penalty.
    """

    def forward(self, action: Any, observation: Any) -> float:
        history: List[str] = observation.get("action_history", [])
        current_action = str(action)
        full_history = (history + [current_action])[-6:]
        reward = 0.0

        def _cmd(a: str) -> str:
            return a.split()[0].lower()

        cmd = _cmd(current_action)
        prev_cmds = [_cmd(h) for h in full_history[:-1]]

        # ── Query after inspect: +0.02 ───────────────────────────────────────
        if cmd.startswith("query_"):
            if any(p in ("inspect_dataset", "inspect_model") for p in prev_cmds[-3:]):
                reward += 0.02
            # Repeated same query without apply in between: -0.05
            same_query_recent = [p for p in prev_cmds[-3:] if p == cmd]
            if same_query_recent and "apply" not in prev_cmds[-3:]:
                reward -= 0.05

        # ── Apply: check if query preceded it ────────────────────────────────
        if cmd == "apply":
            if any(p.startswith("query_") for p in prev_cmds[-4:]):
                reward += 0.05
            else:
                reward -= 0.08      # blind apply — strict penalty (was -0.04)

        # ── Validate after apply: +0.04 ──────────────────────────────────────
        if cmd == "validate":
            if "apply" in prev_cmds[-3:]:
                reward += 0.04
            # Redundant validate (two validates in a row): -0.08
            if prev_cmds and _cmd(prev_cmds[-1]) == "validate":
                reward -= 0.08

        # ── Complete loop: validate → apply → validate = +0.06 ───────────────
        if cmd == "validate" and len(prev_cmds) >= 2:
            if (any(p == "apply" for p in prev_cmds[-2:]) and
                    any(p == "validate" for p in prev_cmds[-4:-2])):
                reward += 0.06

        # ── Reject is fine: +0.01 ────────────────────────────────────────────
        if cmd == "reject":
            reward += 0.01

        # ── Submit: must have validated ───────────────────────────────────────
        if cmd == "submit":
            all_cmds = [_cmd(h) for h in history]
            if "validate" not in all_cmds:
                reward -= 0.15      # submitting blind (was -0.10)
            elif "apply" not in all_cmds:
                reward -= 0.05      # queried but never applied anything

        logger.debug("ProcessRubric: action=%s reward=%.4f", current_action, reward)
        return round(reward, 4)


# ── Rubric 3: Preservation ────────────────────────────────────────────────────

class PreservationRubric(Rubric):
    """
    Rewards row preservation. Prevents the agent from deleting rows to inflate
    classifier confidence. Threshold tightened: must keep ≥ 92% (was 90%).
    """

    def forward(self, action: Any, observation: Any) -> float:
        current_rows  = observation.get("current_rows", 0)
        original_rows = observation.get("original_rows", 1)
        rows_preserved = current_rows / max(original_rows, 1)

        if rows_preserved >= 0.92:       # tightened from 0.90
            reward = 0.05
        elif rows_preserved >= 0.85:
            reward = 0.02
        elif rows_preserved >= 0.75:
            reward = -0.05              # new tier: slight penalty (was 0.0)
        elif rows_preserved >= 0.50:
            reward = -0.20              # increased from -0.10
        else:
            reward = -0.50              # catastrophic (was -0.40)

        logger.debug("PreservationRubric: pct=%.2f reward=%.4f", rows_preserved, reward)
        return round(reward, 4)


# ── Rubric 4: Efficiency ──────────────────────────────────────────────────────

class EfficiencyRubric(Rubric):
    """
    Computed ONLY at submit. Rewards high accuracy gain per budget step used.
    Encourages the agent to be surgical rather than spray-and-pray.

    Key change: the efficiency score now has a *minimum penalty* of -0.10 when
    the agent wastes >80% of its budget and still fails to reach the target.
    This means a slow, failing agent gets punished more than a fast, failing agent.
    """

    def forward(self, action: Any, observation: Any) -> float:
        if str(action).strip().lower() != "submit":
            return 0.0

        baseline         = observation.get("baseline_accuracy", 0.0)
        current          = observation.get("current_accuracy", 0.0)
        original_budget  = observation.get("original_budget", 1)
        budget_remaining = observation.get("budget_remaining", 0)
        target           = observation.get("target_accuracy", 0.80)
        budget_used      = max(original_budget - budget_remaining, 1)
        accuracy_gain    = current - baseline

        if accuracy_gain <= 0:
            # Zero or negative gain — wasted all that budget for nothing
            budget_waste_fraction = budget_used / max(original_budget, 1)
            reward = -0.10 * budget_waste_fraction   # up to -0.10
        elif current < target:
            # Made progress but didn't hit target
            reward = min((accuracy_gain / budget_used) * 1.5, 0.10)
        else:
            # Hit target — efficiency bonus
            reward = min((accuracy_gain / budget_used) * 3.0, 0.25)   # raised cap

        logger.debug("EfficiencyRubric: gain=%.4f used=%d reward=%.4f",
                     accuracy_gain, budget_used, reward)
        return round(reward, 4)


# ── Rubric 5: Step (proxy, no classifier) ────────────────────────────────────

class StepRubric(Rubric):
    """
    Dense per-apply proxy reward — does NOT run the RF classifier.
    Uses lightweight quality score delta to give feedback between validate calls.

    Key change: quality improvements get a stronger signal, quality regressions
    get a harsher penalty. This forces the model to commit to correct fixes.
    """

    def forward(self, action: Any, observation: Any) -> float:
        if not str(action).startswith("apply"):
            return 0.0

        q_before = observation.get("quality_before", 0.0)
        q_after  = observation.get("quality_after", 0.0)
        rows_pct = observation.get("rows_preserved_after", 1.0)
        delta    = q_after - q_before

        if delta > 0:
            r = float(np.clip(delta * 0.50, 0.0, 0.12))   # stronger positive signal
        elif delta < 0:
            r = float(np.clip(delta * 0.60, -0.25, 0.0))  # harsher negative signal
        else:
            r = -0.02   # zero-delta apply wastes budget — small penalty

        # Row preservation modifier
        if rows_pct >= 0.95:
            r += 0.02
        elif rows_pct >= 0.90:
            r += 0.01
        elif rows_pct < 0.85:
            r -= 0.08   # stricter than before (was -0.10 only below 0.80)
        elif rows_pct < 0.80:
            r -= 0.15

        return float(np.clip(r, -0.30, 0.15))


# ── Root Rubric (aggregates all components) ───────────────────────────────────

class DataCentricRubric(Rubric):
    """
    Root composable rubric for the Data-Centric AI environment.

    Child rubrics are auto-registered (PyTorch nn.Module style):
        rubric.accuracy     → AccuracyRubric
        rubric.process      → ProcessRubric
        rubric.preservation → PreservationRubric
        rubric.efficiency   → EfficiencyRubric

    Call rubric(action, obs_dict) to get total clamped reward [-1.0, 1.0].
    """

    def __init__(self):
        super().__init__()
        self.accuracy     = AccuracyRubric()
        self.process      = ProcessRubric()
        self.preservation = PreservationRubric()
        self.efficiency   = EfficiencyRubric()

    def forward(self, action: Any, observation: Any) -> float:
        r_acc  = self.accuracy(action, observation)
        r_proc = self.process(action, observation)
        r_pres = self.preservation(action, observation)
        r_eff  = self.efficiency(action, observation)
        total  = r_acc + r_proc + r_pres + r_eff

        clamped = float(np.clip(total, REWARD_MIN, REWARD_MAX))
        if __debug__ and abs(clamped - total) > 1e-6:
            logger.warning("Reward %.4f clamped → %.4f", total, clamped)

        logger.info(
            "REWARD | accuracy=%.4f process=%.4f preservation=%.4f "
            "efficiency=%.4f TOTAL=%.4f (clamped=%.4f)",
            r_acc, r_proc, r_pres, r_eff, total, clamped,
        )
        return round(clamped, 4)

    def breakdown(self) -> Dict[str, Optional[float]]:
        return {
            "accuracy":     self.accuracy.last_score,
            "process":      self.process.last_score,
            "preservation": self.preservation.last_score,
            "efficiency":   self.efficiency.last_score,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_rubric: Optional[DataCentricRubric] = None
_step_rubric: Optional[StepRubric] = None


def get_rubric() -> DataCentricRubric:
    global _rubric
    if _rubric is None:
        _rubric = DataCentricRubric()
    return _rubric


def get_step_rubric() -> StepRubric:
    global _step_rubric
    if _step_rubric is None:
        _step_rubric = StepRubric()
    return _step_rubric


# ── Backward-compatible free functions ───────────────────────────────────────

def compute_accuracy_reward(
    current_accuracy: float, previous_accuracy: float,
    baseline_accuracy: float, target_accuracy: float,
    is_submit: bool = False,
    budget_used: int = 1, budget_total: int = 30,
) -> float:
    obs = dict(
        current_accuracy=current_accuracy, previous_accuracy=previous_accuracy,
        baseline_accuracy=baseline_accuracy, target_accuracy=target_accuracy,
        budget_used=budget_used, budget_total=budget_total,
    )
    action = "submit" if is_submit else "step"
    return get_rubric().accuracy(action, obs)


def compute_process_reward(action_history: List[str], current_action: str) -> float:
    obs = dict(action_history=action_history)
    return get_rubric().process(current_action, obs)


def compute_preservation_reward(current_rows: int, original_rows: int) -> float:
    obs = dict(current_rows=current_rows, original_rows=original_rows)
    return get_rubric().preservation("step", obs)


def compute_efficiency_reward(
    current_accuracy: float, baseline_accuracy: float,
    original_budget: int, budget_remaining: int,
    target_accuracy: float = 0.80,
) -> float:
    obs = dict(
        current_accuracy=current_accuracy, baseline_accuracy=baseline_accuracy,
        original_budget=original_budget, budget_remaining=budget_remaining,
        target_accuracy=target_accuracy,
    )
    return get_rubric().efficiency("submit", obs)


def compute_step_reward(
    action: str, quality_before: float, quality_after: float,
    rows_preserved_after: float,
) -> float:
    obs = dict(quality_before=quality_before, quality_after=quality_after,
               rows_preserved_after=rows_preserved_after)
    return get_step_rubric()(action, obs)


def compute_total_reward(
    reward_accuracy: float,
    reward_process: float,
    reward_preservation: float,
    reward_efficiency: float = 0.0,
    reward_step: float = 0.0,
) -> float:
    total = reward_accuracy + reward_process + reward_preservation + reward_efficiency + reward_step
    clamped = float(np.clip(total, REWARD_MIN, REWARD_MAX))
    if __debug__ and abs(clamped - total) > 1e-6:
        logger.warning("Reward %.4f clamped → %.4f", total, clamped)
    logger.info(
        "REWARD BREAKDOWN: accuracy=%.4f process=%.4f preservation=%.4f "
        "efficiency=%.4f step=%.4f TOTAL=%.4f (clamped=%.4f)",
        reward_accuracy, reward_process, reward_preservation,
        reward_efficiency, reward_step, total, clamped,
    )
    return round(clamped, 4)
