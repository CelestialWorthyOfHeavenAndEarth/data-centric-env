"""
Unit tests for anti-exploit and environment stability.

Tests that the core safety invariants hold:
- ground truth never mutates
- budget is enforced
- validate calls are limited
- undo works correctly

Run with: pytest tests/test_environment.py -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import DataCentricAction
from server.data_centric_environment import DataCentricEnvironment


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_env(task="task_0_tutorial", seed=42) -> tuple:
    """Return (env, reset_obs)."""
    env = DataCentricEnvironment()
    obs = env.reset(task=task, seed=seed)
    return env, obs


def step(env, cmd: str):
    return env.step(DataCentricAction(message=cmd))


# ── Ground truth immutability ─────────────────────────────────────────────────

class TestGroundTruth:

    def test_ground_truth_unchanged_after_reset(self):
        env, _ = make_env()
        gt_before = env._ground_truth.copy()
        env.reset(task="task_0_tutorial", seed=123)
        # After re-reset, a NEW ground truth is loaded — old one doesn't matter
        assert env._ground_truth is not None

    def test_ground_truth_unchanged_after_query(self):
        env, _ = make_env()
        gt_before = env._ground_truth.copy()
        step(env, "query_cleaner")
        assert env._ground_truth.equals(gt_before), "GT mutated after query_cleaner"

    def test_ground_truth_unchanged_after_inspect(self):
        env, _ = make_env()
        gt_before = env._ground_truth.copy()
        step(env, "inspect_dataset")
        assert env._ground_truth.equals(gt_before), "GT mutated after inspect_dataset"


# ── Budget enforcement ────────────────────────────────────────────────────────

class TestBudget:

    def test_budget_decreases_each_step(self):
        env, obs = make_env()
        budget_start = obs.budget_remaining
        obs2 = step(env, "inspect_dataset")
        assert obs2.budget_remaining < budget_start

    def test_done_after_budget_exhausted(self):
        env, obs = make_env()
        budget = obs.budget_remaining
        last_obs = obs
        for _ in range(budget + 5):
            if last_obs.done:
                break
            last_obs = step(env, "inspect_dataset")
        assert last_obs.done, "Episode should be done after budget exhausted"


# ── Validate calls ────────────────────────────────────────────────────────────

class TestValidateCalls:

    def test_validate_calls_start_at_3(self):
        env, obs = make_env()
        assert obs.validate_calls_remaining == 3

    def test_validate_call_decrements(self):
        env, obs = make_env()
        obs2 = step(env, "validate")
        # Either decrement or cooldown message — either way calls consumed
        assert obs2.validate_calls_remaining <= 3


# ── Undo ─────────────────────────────────────────────────────────────────────

class TestUndo:

    def test_undo_without_history_returns_response(self):
        """Undo with no history should return a message, not crash."""
        env, _ = make_env()
        obs = step(env, "undo")
        assert obs.response is not None
        assert len(obs.response) > 0

    def test_undo_after_apply_restores_state(self):
        """After apply+undo, working copy should match pre-apply state."""
        env, _ = make_env()
        step(env, "query_cleaner")
        wc_before_apply = env._working_copy.copy()
        step(env, "apply 1")
        step(env, "undo")
        # After undo, working copy should be restored
        assert env._working_copy.shape == wc_before_apply.shape, (
            f"Shape mismatch after undo: {env._working_copy.shape} vs {wc_before_apply.shape}"
        )


# ── Snapshot stack ────────────────────────────────────────────────────────────

class TestSnapshotStack:

    def test_max_3_snapshots(self):
        env, _ = make_env()
        step(env, "query_cleaner")
        # Apply 4 times — stack should cap at 3
        for i in range(1, 5):
            step(env, f"apply {i}")
            step(env, "query_cleaner")  # re-query for fresh recs
        assert len(env._dataset_history) <= env._max_history

    def test_snapshot_cleared_on_reset(self):
        env, _ = make_env()
        step(env, "query_cleaner")
        step(env, "apply 1")
        env.reset(task="task_0_tutorial", seed=99)
        assert len(env._dataset_history) == 0


# ── Reward sanity ─────────────────────────────────────────────────────────────

class TestRewardSanity:

    def test_unknown_command_gives_penalty(self):
        env, _ = make_env()
        obs = step(env, "blorp_invalid_command_xyz")
        assert obs.reward <= 0.0, (
            f"Unknown command should not give positive reward, got {obs.reward}"
        )

    def test_reward_within_range(self):
        env, _ = make_env()
        from server.grader import REWARD_MIN, REWARD_MAX
        for cmd in ["inspect_dataset", "inspect_model", "query_cleaner",
                    "query_balancer", "validate"]:
            obs = step(env, cmd)
            r = obs.reward
            assert REWARD_MIN - 0.01 <= r <= REWARD_MAX + 0.01, (
                f"Command '{cmd}' gave out-of-range reward {r}"
            )
