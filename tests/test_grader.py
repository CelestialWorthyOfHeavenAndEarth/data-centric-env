"""
Unit tests for server/grader.py

Tests every reward component individually plus the range clamp.
Run with: pytest tests/test_grader.py -v
"""
import importlib.util
import sys
import os
import pytest
import pandas as pd

# Import grader directly (avoids server/__init__.py → openenv-core chain)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GRADER = os.path.join(_ROOT, "server", "grader.py")
spec = importlib.util.spec_from_file_location("grader", _GRADER)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

REWARD_MAX = _mod.REWARD_MAX
REWARD_MIN = _mod.REWARD_MIN
compute_accuracy_reward = _mod.compute_accuracy_reward
compute_efficiency_reward = _mod.compute_efficiency_reward
compute_preservation_reward = _mod.compute_preservation_reward
compute_process_reward = _mod.compute_process_reward
compute_step_reward = _mod.compute_step_reward
compute_total_reward = _mod.compute_total_reward
compute_lightweight_score = _mod.compute_lightweight_score


# ── Accuracy reward ───────────────────────────────────────────────────────────

class TestAccuracyReward:

    def test_improvement_positive(self):
        r = compute_accuracy_reward(0.70, 0.62, 0.62, 0.80)
        assert r > 0, f"Improvement should give positive reward, got {r}"

    def test_regression_negative(self):
        r = compute_accuracy_reward(0.60, 0.70, 0.62, 0.80)
        assert r < 0, f"Regression should give negative reward, got {r}"

    def test_no_change_zero(self):
        r = compute_accuracy_reward(0.65, 0.65, 0.62, 0.80)
        assert r == 0.0

    def test_submit_success_bonus(self):
        r = compute_accuracy_reward(0.80, 0.75, 0.62, 0.80, is_submit=True)
        assert r > 0.5, f"Submit success should add bonus, got {r}"

    def test_submit_fail_partial_credit(self):
        """Halfway to target should give some credit."""
        # baseline=0.62, target=0.80, current=0.71 → 50% of the way there
        r = compute_accuracy_reward(0.71, 0.70, 0.62, 0.80, is_submit=True)
        # Current is at 0.71, previous was 0.70 → small improvement, partial credit
        assert r > 0, f"Partial progress at submit should give credit, got {r}"


# ── Preservation reward ───────────────────────────────────────────────────────

class TestPreservationReward:

    def test_above_90_bonus(self):
        r = compute_preservation_reward(97, 100)
        assert r > 0

    def test_below_90_zero_or_neg(self):
        r = compute_preservation_reward(85, 100)
        assert r <= 0.02  # at best neutral at 85%

    def test_below_50_catastrophic(self):
        r = compute_preservation_reward(40, 100)
        assert r <= -0.40, f"Expected catastrophic penalty, got {r}"

    def test_full_preservation(self):
        r = compute_preservation_reward(100, 100)
        assert r == 0.05


# ── Process reward ────────────────────────────────────────────────────────────

class TestProcessReward:

    def test_query_after_inspect_rewarded(self):
        history = ["inspect_dataset"]
        r = compute_process_reward(history, "query_cleaner")
        assert r > 0

    def test_apply_without_query_penalized(self):
        history = ["inspect_dataset", "inspect_model"]
        r = compute_process_reward(history, "apply 1")
        assert r < 0

    def test_apply_after_query_rewarded(self):
        history = ["inspect_dataset", "query_cleaner"]
        r = compute_process_reward(history, "apply 1")
        assert r > 0

    def test_submit_without_validate_penalized(self):
        history = ["inspect_dataset", "query_cleaner", "apply 1"]
        r = compute_process_reward(history, "submit")
        assert r < 0


# ── Step reward ───────────────────────────────────────────────────────────────

class TestStepReward:

    def test_quality_improvement_positive(self):
        r = compute_step_reward("apply 1", quality_before=0.5, quality_after=0.7,
                                rows_preserved_after=0.97)
        assert r > 0

    def test_quality_degradation_negative(self):
        r = compute_step_reward("apply 1", quality_before=0.7, quality_after=0.4,
                                rows_preserved_after=0.97)
        assert r < 0

    def test_non_apply_zero(self):
        r = compute_step_reward("validate", quality_before=0.5, quality_after=0.7,
                                rows_preserved_after=0.97)
        assert r == 0.0

    def test_low_preservation_penalty(self):
        r = compute_step_reward("apply 1", quality_before=0.5, quality_after=0.6,
                                rows_preserved_after=0.75)
        # Row preservation penalty should reduce reward
        r_without = compute_step_reward("apply 1", quality_before=0.5, quality_after=0.6,
                                        rows_preserved_after=0.97)
        assert r < r_without


# ── Total reward range ────────────────────────────────────────────────────────

class TestRewardRange:

    def test_within_declared_range(self):
        """All reward combinations must stay within [-1.0, 1.0]."""
        test_cases = [
            (0.5, 0.1, 0.05, 0.2, 0.1),
            (-0.8, -0.1, -0.4, -0.05, -0.2),
            (1.0, 0.2, 0.05, 0.2, 0.15),   # might need clamping
            (-1.5, -0.2, -0.4, -0.05, -0.3),  # definitely needs clamping
        ]
        for acc, proc, pres, eff, step in test_cases:
            r = compute_total_reward(acc, proc, pres, eff, step)
            assert REWARD_MIN <= r <= REWARD_MAX, (
                f"Reward {r} out of [{REWARD_MIN}, {REWARD_MAX}] "
                f"for inputs acc={acc} proc={proc} pres={pres}"
            )

    def test_clamping_applied(self):
        """Extreme inputs should be clamped, not crash."""
        r = compute_total_reward(10.0, 5.0, 5.0)
        assert r == REWARD_MAX

        r = compute_total_reward(-10.0, -5.0, -5.0)
        assert r == REWARD_MIN


# ── Lightweight quality score ─────────────────────────────────────────────────

class TestLightweightScore:

    def _make_df(self, n_rows=10, n_missing=0, n_dups=0):
        """Create a minimal test dataframe."""
        df = pd.DataFrame({
            "feature_0": [float(i) for i in range(n_rows)],
            "target": [i % 2 for i in range(n_rows)],
        })
        if n_missing:
            df.loc[:n_missing - 1, "feature_0"] = float("nan")
        if n_dups:
            df = pd.concat([df, df.iloc[:n_dups]], ignore_index=True)
        return df

    def test_clean_df_high_score(self):
        df = self._make_df()
        score = compute_lightweight_score(df, df.copy(), len(df),
                                          {"feature_0": {"expected_dtype": "float64"}})
        assert score >= 0.80

    def test_many_missing_low_score(self):
        df = self._make_df(n_missing=8)
        score = compute_lightweight_score(df, df.copy(), 10,
                                          {"feature_0": {"expected_dtype": "float64"}},
                                          initial_missing=8)
        assert score < 0.70

    def test_score_in_range(self):
        df = self._make_df(n_missing=3, n_dups=2)
        score = compute_lightweight_score(df, df.copy(), 10,
                                          {"feature_0": {"expected_dtype": "float64"}},
                                          initial_missing=3)
        assert 0.0 <= score <= 1.0
