"""
Grader for Data-Centric RL Environment — using OpenEnv Rubric system.

Implements 4 composable Rubric subclasses (nn.Module style, auto-registered
as child rubrics) plus a root DataCentricRubric that aggregates them.

Rubric hierarchy:
    DataCentricRubric
    ├── accuracy      : AccuracyRubric
    ├── process       : ProcessRubric
    ├── preservation  : PreservationRubric
    └── efficiency    : EfficiencyRubric

Also provides StepRubric for dense per-apply proxy feedback (no classifier).

Backward-compatible: compute_*() free functions still work for existing callers.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from openenv.core.rubrics.base import Rubric

logger = logging.getLogger(__name__)

# Must match openenv.yaml reward_range — enforced by DataCentricRubric
REWARD_MIN: float = -1.0
REWARD_MAX: float = 1.0


# ── Lightweight quality score (no sklearn) ────────────────────────────────────

def compute_lightweight_score(
    working_copy: pd.DataFrame,
    ground_truth: pd.DataFrame,
    original_length: int,
    col_meta: Dict,
    initial_missing: int = None,
) -> float:
    """
    Fast quality score comparing working_copy to ground_truth structure.
    Does NOT run sklearn — used for dense per-step feedback.

    Score is [0.0, 1.0] composed of:
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
    Main RL signal: rewards accuracy improvement, penalises regression.
    Adds a large terminal bonus when the agent submits and crosses the target.
    """

    def forward(self, action: Any, observation: Any) -> float:
        current   = observation.get("current_accuracy", 0.0)
        previous  = observation.get("previous_accuracy", 0.0)
        baseline  = observation.get("baseline_accuracy", 0.0)
        target    = observation.get("target_accuracy", 0.80)
        is_submit = str(action).strip().lower() == "submit"

        improvement = current - previous
        if improvement > 0:
            reward = improvement * 2.5
        elif improvement < 0:
            reward = improvement * 2.0   # regression penalised harder
        else:
            reward = 0.0

        if is_submit:
            if current >= target:
                reward += 0.50            # big terminal bonus
            else:
                progress_range = target - baseline
                if progress_range > 0:
                    progress = (current - baseline) / progress_range
                    reward += max(0.0, progress) * 0.25

        logger.debug("AccuracyRubric: imp=%.4f reward=%.4f submit=%s", improvement, reward, is_submit)
        return round(reward, 4)


# ── Rubric 2: Process ─────────────────────────────────────────────────────────

class ProcessRubric(Rubric):
    """
    Rewards smart workflow patterns (inspect → query → apply → validate).
    Penalises blind apply-without-query and submit-without-validate.
    """

    def forward(self, action: Any, observation: Any) -> float:
        history: List[str] = observation.get("action_history", [])
        current_action = str(action)
        full_history = (history + [current_action])[-5:]
        reward = 0.0

        def _cmd(a: str) -> str:
            return a.split()[0].lower()

        cmd = _cmd(current_action)
        prev_cmds = [_cmd(h) for h in full_history[:-1][-3:]]

        if cmd.startswith("query_"):
            if "inspect_dataset" in prev_cmds or "inspect_model" in prev_cmds:
                reward += 0.02

        if cmd == "apply":
            if any(p.startswith("query_") for p in prev_cmds):
                reward += 0.05
            else:
                reward -= 0.04

        if cmd == "validate" and "apply" in prev_cmds:
            reward += 0.03

        if cmd == "reject":
            reward += 0.01

        if cmd == "submit":
            all_cmds = [_cmd(h) for h in history]
            if "validate" not in all_cmds:
                reward -= 0.10

        logger.debug("ProcessRubric: action=%s reward=%.4f", current_action, reward)
        return round(reward, 4)


# ── Rubric 3: Preservation ────────────────────────────────────────────────────

class PreservationRubric(Rubric):
    """
    Rewards row preservation. Independent of accuracy — prevents the agent
    from 'cheating' by deleting rows to inflate classifier confidence.
    """

    def forward(self, action: Any, observation: Any) -> float:
        current_rows  = observation.get("current_rows", 0)
        original_rows = observation.get("original_rows", 1)
        rows_preserved = current_rows / max(original_rows, 1)

        if rows_preserved >= 0.90:
            reward = 0.05
        elif rows_preserved >= 0.80:
            reward = 0.02
        elif rows_preserved >= 0.70:
            reward = 0.00
        elif rows_preserved >= 0.50:
            reward = -0.10
        else:
            reward = -0.40

        logger.debug("PreservationRubric: pct=%.2f reward=%.4f", rows_preserved, reward)
        return round(reward, 4)


# ── Rubric 4: Efficiency ──────────────────────────────────────────────────────

class EfficiencyRubric(Rubric):
    """
    Computed ONLY at submit. Rewards high accuracy gain per budget step used.
    Encourages the agent to be surgical rather than spray-and-pray.
    """

    def forward(self, action: Any, observation: Any) -> float:
        if str(action).strip().lower() != "submit":
            return 0.0

        baseline         = observation.get("baseline_accuracy", 0.0)
        current          = observation.get("current_accuracy", 0.0)
        original_budget  = observation.get("original_budget", 1)
        budget_remaining = observation.get("budget_remaining", 0)
        budget_used      = max(original_budget - budget_remaining, 1)
        accuracy_gain    = current - baseline

        if accuracy_gain <= 0:
            reward = -0.05
        else:
            reward = min((accuracy_gain / budget_used) * 3.0, 0.20)

        logger.debug("EfficiencyRubric: gain=%.4f used=%d reward=%.4f",
                     accuracy_gain, budget_used, reward)
        return round(reward, 4)


# ── Rubric 5: Step (proxy, no classifier) ────────────────────────────────────

class StepRubric(Rubric):
    """
    Dense per-apply proxy reward — does NOT run the RF classifier.
    Uses lightweight quality score delta to give feedback between validate calls.
    Registered as a standalone rubric (not a child of DataCentricRubric)
    because it fires on every apply step, not just at episode end.
    """

    def forward(self, action: Any, observation: Any) -> float:
        if not str(action).startswith("apply"):
            return 0.0

        q_before = observation.get("quality_before", 0.0)
        q_after  = observation.get("quality_after", 0.0)
        rows_pct = observation.get("rows_preserved_after", 1.0)

        r = float(np.clip((q_after - q_before) * 0.3, -0.20, 0.10))

        if rows_pct >= 0.95:
            r += 0.02
        elif rows_pct >= 0.90:
            r += 0.01
        elif rows_pct < 0.80:
            r -= 0.10

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
    Inspect rubric.accuracy.last_score etc. for per-component breakdown.
    """

    def __init__(self):
        super().__init__()
        # Assigned as attributes — auto-registered as children by Rubric.__setattr__
        self.accuracy    = AccuracyRubric()
        self.process     = ProcessRubric()
        self.preservation = PreservationRubric()
        self.efficiency  = EfficiencyRubric()

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
        """Return last_score for each child rubric — useful for logging."""
        return {
            "accuracy":     self.accuracy.last_score,
            "process":      self.process.last_score,
            "preservation": self.preservation.last_score,
            "efficiency":   self.efficiency.last_score,
        }


# ── Singleton — reuse across episode steps ────────────────────────────────────

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
# (called by data_centric_environment.py — no changes needed there)

def compute_accuracy_reward(
    current_accuracy: float, previous_accuracy: float,
    baseline_accuracy: float, target_accuracy: float,
    is_submit: bool = False,
) -> float:
    obs = dict(current_accuracy=current_accuracy, previous_accuracy=previous_accuracy,
               baseline_accuracy=baseline_accuracy, target_accuracy=target_accuracy)
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
) -> float:
    obs = dict(current_accuracy=current_accuracy, baseline_accuracy=baseline_accuracy,
               original_budget=original_budget, budget_remaining=budget_remaining)
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
