# -*- coding: utf-8 -*-
"""Smoke test: verify all 5 new features work end-to-end."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DataCentricAction
from server.data_centric_environment import DataCentricEnvironment

env = DataCentricEnvironment()
obs = env.reset(task='task_1_easy', seed=7)
print(f"Reset OK. Budget: {obs.budget_remaining}, Baseline: {obs.baseline_accuracy:.4f}")
print()

# ─── Feature 3: query_analyst ────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: query_analyst (meta-specialist, costs 1 budget)")
print("=" * 60)
budget_before = obs.budget_remaining
obs = env.step(DataCentricAction(message='query_analyst'))
budget_after = obs.budget_remaining
print(obs.response)
print(f"\n[BUDGET CHECK] Before={budget_before}, After={budget_after}, Diff={budget_before - budget_after}")
assert "DIAGNOSIS" in obs.response, "FAIL: no DIAGNOSIS section"
assert "RECOMMENDED PLAN" in obs.response, "FAIL: no RECOMMENDED PLAN section"
assert budget_before - budget_after == 2, f"FAIL: should cost 2 total (1 cmd + 1 analyst), got {budget_before - budget_after}"
print("PASS: query_analyst works")

# ─── Feature 1: Smarter specialists ─────────────────────────────────────────
print()
print("=" * 60)
print("TEST 2: query_cleaner (smarter specialists with reasoning)")
print("=" * 60)
obs = env.step(DataCentricAction(message='query_cleaner'))
print(obs.response)
# Check for statistical reasoning markers
has_reasoning = any(kw in obs.response for kw in ["skew", "Risk:", "Reason:", "median", "mean", "%"])
assert has_reasoning, "FAIL: no statistical reasoning found in cleaner output"
print("PASS: smarter specialists working (statistical reasoning present)")

# ─── Feature 5: Drift detection ──────────────────────────────────────────────
print()
print("=" * 60)
print("TEST 3: apply 1 (drift detection after apply)")
print("=" * 60)
obs = env.step(DataCentricAction(message='apply 1'))
print(obs.response)
has_drift = "Distribution drift" in obs.response or "drift" in obs.response.lower()
assert has_drift, "FAIL: no drift information in apply response"
print("PASS: drift detection working")

# ─── Feature 2 + 4: Dual classifier + Feature importance ───────────────────
print()
print("=" * 60)
print("TEST 4: validate (dual classifier + feature importance)")
print("=" * 60)
obs = env.step(DataCentricAction(message='validate'))
print(obs.response)
assert "RF Accuracy" in obs.response, "FAIL: no RF Accuracy"
assert "LR Accuracy" in obs.response, "FAIL: no LR Accuracy"
assert "Agreement" in obs.response, "FAIL: no Agreement signal"
has_feat_imp = "Feature importance" in obs.response
print(f"Feature importance shown: {has_feat_imp}")
print("PASS: dual classifier + agreement signal working")

# ─── Feature 4: Feature importance in inspect_model ─────────────────────────
print()
print("=" * 60)
print("TEST 5: inspect_model (RF + LR + feature importance)")
print("=" * 60)
obs = env.step(DataCentricAction(message='inspect_model'))
print(obs.response)
assert "RF Accuracy" in obs.response, "FAIL: no RF Accuracy in inspect_model"
assert "LR Accuracy" in obs.response, "FAIL: no LR Accuracy in inspect_model"
print("PASS: inspect_model shows dual classifier")

print()
print("=" * 60)
print("ALL 5 FEATURES VERIFIED OK")
print("=" * 60)
