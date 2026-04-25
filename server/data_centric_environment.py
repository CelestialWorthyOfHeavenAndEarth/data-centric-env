"""Main DataCentric RL Environment."""

import logging
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DataCentricAction, DataCentricObservation
except ImportError:
    try:
        from models import DataCentricAction, DataCentricObservation
    except ImportError:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from models import DataCentricAction, DataCentricObservation

try:
    from .anti_exploit import (
        AntiExploitState, assert_ground_truth_intact,
        check_and_truncate_input, check_apply_allowed,
        check_catastrophic_data_loss, check_episode_timeout,
        check_validate_cooldown, get_validate_reward, record_apply,
        record_non_validate_step, record_validate, reset_session_apply_state,
        validate_calls_remaining,
    )
    from .dataset_generator import TASK_CONFIGS, generate_dataset
    from .grader import (
        compute_accuracy_reward, compute_efficiency_reward,
        compute_lightweight_score, compute_preservation_reward,
        compute_process_reward, compute_step_reward, compute_total_reward,
    )
    from .model_evaluator import ModelEvaluator
    from .specialist_agents import (
        AugmenterAgent, AnalystAgent, BalancerAgent, CleanerAgent,
        SessionRegistry, ValidatorAgent, compute_drift, format_drift_summary,
    )
except ImportError:
    from server.anti_exploit import (
        AntiExploitState, assert_ground_truth_intact,
        check_and_truncate_input, check_apply_allowed,
        check_catastrophic_data_loss, check_episode_timeout,
        check_validate_cooldown, get_validate_reward, record_apply,
        record_non_validate_step, record_validate, reset_session_apply_state,
        validate_calls_remaining,
    )
    from server.dataset_generator import TASK_CONFIGS, generate_dataset
    from server.grader import (
        compute_accuracy_reward, compute_efficiency_reward,
        compute_lightweight_score, compute_preservation_reward,
        compute_process_reward, compute_step_reward, compute_total_reward,
    )
    from server.model_evaluator import ModelEvaluator
    from server.specialist_agents import (
        AugmenterAgent, AnalystAgent, BalancerAgent, CleanerAgent,
        SessionRegistry, ValidatorAgent, compute_drift, format_drift_summary,
    )

logger = logging.getLogger(__name__)

AVAILABLE_COMMANDS = """Available commands:
  inspect_dataset          — shape, dtypes, missing, class distribution
  inspect_model            — accuracy (RF + LR), F1, feature importance
  query_analyst            — holistic diagnosis + prioritised action plan (costs 1 budget)
  query_cleaner            — get cleaning recommendations
  query_augmenter [class]  — get augmentation suggestions
  query_balancer           — get resampling recommendations
  query_validator          — check rule violations (costs 2 budget)
  apply [id]               — apply recommendation by ID
  reject [id]              — reject a recommendation
  undo                     — revert last apply (max 3 levels)
  validate                 — retrain and score (cooldown applies)
  submit                   — finalize episode"""


class DataCentricEnvironment(Environment):
    """Data-Centric AI RL Environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._ground_truth: Optional[pd.DataFrame] = None
        self._working_copy: Optional[pd.DataFrame] = None
        self._metadata: Dict[str, Any] = {}
        self._action_history: List[str] = []
        self._exploit: Optional[AntiExploitState] = None
        # fast_mode=True: uses n_estimators=20 for training rollouts (~4x faster)
        self._evaluator = ModelEvaluator(fast_mode=True)
        self._session_registry = SessionRegistry()
        self._cleaner = CleanerAgent()
        self._augmenter = AugmenterAgent()
        self._balancer = BalancerAgent()
        self._validator = ValidatorAgent()
        self._analyst = AnalystAgent()
        self._current_accuracy: float = 0.0
        self._previous_accuracy: float = 0.0
        self._active_session: str = "none"
        self._task: str = "task_0_tutorial"
        # Snapshot stack for undo command (max 3 snapshots)
        self._dataset_history: List[pd.DataFrame] = []
        self._max_history: int = 3

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, task: str = "task_0_tutorial", seed: int = 42) -> DataCentricObservation:
        self._task = task if task in TASK_CONFIGS else "task_0_tutorial"
        cfg = TASK_CONFIGS[self._task]

        self._ground_truth, self._working_copy, self._metadata = generate_dataset(
            self._task, seed=seed
        )
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._action_history = []
        self._exploit = AntiExploitState(
            episode_start_time=time.time(),
            ground_truth_row_count=len(self._ground_truth),
        )
        self._evaluator.invalidate_cache()
        self._session_registry = SessionRegistry()
        self._active_session = "none"
        self._dataset_history = []   # clear snapshot stack on reset
        reset_session_apply_state(self._exploit)

        # Store episode-start missing count for quality score baseline
        self._metadata["initial_missing"] = int(self._working_copy.isnull().sum().sum())
        self._metadata["baseline_accuracy"] = cfg["baseline_accuracy"]

        baseline = cfg["baseline_accuracy"]
        self._current_accuracy = baseline
        self._previous_accuracy = baseline
        quality = compute_lightweight_score(
            self._working_copy, self._ground_truth,
            self._metadata["original_length"], self._metadata["col_meta"],
            initial_missing=self._metadata["initial_missing"],
        )
        wc = self._working_copy
        return DataCentricObservation(
            response=(
                f"Episode started: {self._task}\n"
                f"Baseline accuracy: {baseline:.4f} | Target: {cfg['target_accuracy']:.4f}\n"
                f"Dataset: {len(wc)} rows × {len(wc.columns)-1} features\n"
                f"Budget: {cfg['budget']} steps\n\n{AVAILABLE_COMMANDS}"
            ),
            current_accuracy=baseline,
            baseline_accuracy=baseline,
            target_accuracy=cfg["target_accuracy"],
            estimated_quality=quality,
            dataset_shape=f"{len(wc)} rows × {len(wc.columns)-1} columns",
            rows_preserved_pct=1.0,
            budget_remaining=cfg["budget"],
            step_number=0,
            max_steps=cfg["budget"],
            active_session="none",
            validate_calls_remaining=3,
            done=False,
            reward=0.0,
        )

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: DataCentricAction) -> DataCentricObservation:
        if self._working_copy is None:
            return self._error_obs("Call reset() first.")

        # Episode timeout
        timeout, tmsg = check_episode_timeout(self._exploit)
        if timeout:
            return self._do_submit(penalty=-0.10, extra_msg=tmsg)

        # Input truncation
        raw_msg = action.message
        msg, trunc_penalty, was_truncated = check_and_truncate_input(raw_msg)
        if was_truncated:
            logger.warning("Input truncated.")

        cfg = TASK_CONFIGS[self._task]
        self._state.step_count += 1
        step_num = self._state.step_count
        budget_remaining = cfg["budget"] - step_num
        cmd_parts = msg.strip().split()
        cmd = cmd_parts[0].lower() if cmd_parts else ""

        # Out of budget → force submit
        if budget_remaining < 0:
            return self._do_submit(penalty=0.0, extra_msg="Budget exhausted.")

        # Record action
        self._action_history.append(msg)

        # Process reward component (computed for all actions)
        r_process = compute_process_reward(self._action_history[:-1], msg)

        # Route command
        if cmd == "inspect_dataset":
            obs = self._cmd_inspect_dataset(step_num, budget_remaining, r_process, trunc_penalty)
        elif cmd == "inspect_model":
            obs = self._cmd_inspect_model(step_num, budget_remaining, r_process, trunc_penalty)
        elif cmd == "query_cleaner":
            obs = self._cmd_query_cleaner(step_num, budget_remaining, r_process, trunc_penalty)
        elif cmd == "query_augmenter":
            cls = cmd_parts[1] if len(cmd_parts) > 1 else None
            obs = self._cmd_query_augmenter(cls, step_num, budget_remaining, r_process, trunc_penalty)
        elif cmd == "query_balancer":
            obs = self._cmd_query_balancer(step_num, budget_remaining, r_process, trunc_penalty)
        elif cmd == "query_analyst":
            obs = self._cmd_query_analyst(step_num, budget_remaining, r_process, trunc_penalty)
        elif cmd == "query_validator":
            obs = self._cmd_query_validator(step_num, budget_remaining, r_process, trunc_penalty)
        elif cmd == "apply":
            try:
                rec_id = int(cmd_parts[1]) if len(cmd_parts) > 1 else -1
            except ValueError:
                rec_id = -1
            obs = self._cmd_apply(rec_id, step_num, budget_remaining, r_process, trunc_penalty)
        elif cmd == "reject":
            try:
                rec_id = int(cmd_parts[1]) if len(cmd_parts) > 1 else -1
            except ValueError:
                rec_id = -1
            obs = self._cmd_reject(rec_id, step_num, budget_remaining, r_process, trunc_penalty)
        elif cmd == "validate":
            obs = self._cmd_validate(step_num, budget_remaining, r_process, trunc_penalty)
        elif cmd == "submit":
            obs = self._do_submit()
        elif cmd == "undo":
            obs = self._cmd_undo(step_num, budget_remaining, r_process, trunc_penalty)
        else:
            obs = self._unknown_cmd_obs(msg, step_num, budget_remaining, r_process + trunc_penalty)

        if cmd != "validate":
            record_non_validate_step(self._exploit)

        return obs

    # ── command handlers ─────────────────────────────────────────────────────

    def _cmd_inspect_dataset(self, step, budget, r_process, trunc_pen) -> DataCentricObservation:
        wc = self._working_copy
        orig_len = self._metadata["original_length"]
        missing = wc.isnull().sum()
        missing_str = "\n".join(f"  {c}: {v}" for c, v in missing.items() if v > 0) or "  None"
        vc = wc["target"].value_counts().sort_index()
        class_str = ", ".join(f"class {k}: {v}" for k, v in vc.items())
        rows_pct = len(wc) / orig_len
        response = (
            f"=== Dataset Inspection ===\n"
            f"Shape: {len(wc)} rows × {len(wc.columns)-1} features\n"
            f"Original rows: {orig_len} | Preserved: {rows_pct*100:.1f}%\n"
            f"Duplicates: {wc.duplicated().sum()}\n"
            f"Missing values:\n{missing_str}\n"
            f"Class distribution: {class_str}\n"
            f"Dtypes: {dict(wc.dtypes.astype(str))}"
        )
        reward = compute_total_reward(0.0, r_process, 0.0) + trunc_pen
        return self._make_obs(response, step, budget, reward)

    def _cmd_inspect_model(self, step, budget, r_process, trunc_pen) -> DataCentricObservation:
        acc, per_class, from_cache, lr_acc = self._evaluator.evaluate(
            self._working_copy, self._ground_truth
        )
        cache_label = " (cached)" if from_cache else ""
        lines = [f"=== Model Inspection{cache_label} ===",
                 f"RF Accuracy:  {acc:.4f}",
                 f"LR Accuracy:  {lr_acc:.4f}  (secondary — diagnostic only)"]
        for cls, metrics in per_class.items():
            if isinstance(metrics, dict):
                lines.append(
                    f"  Class {cls}: precision={metrics.get('precision',0):.3f} "
                    f"recall={metrics.get('recall',0):.3f} "
                    f"f1={metrics.get('f1-score',0):.3f}"
                )
        feat_text = self._evaluator.feature_importance_text()
        if feat_text:
            lines.append(feat_text)
        response = "\n".join(lines)
        reward = compute_total_reward(0.0, r_process, 0.0) + trunc_pen
        return self._make_obs(response, step, budget, reward)

    def _cmd_query_cleaner(self, step, budget, r_process, trunc_pen) -> DataCentricObservation:
        reset_session_apply_state(self._exploit)
        recs = self._cleaner.query(
            self._working_copy, self._session_registry, self._metadata["col_meta"]
        )
        self._active_session = f"cleaner:{self._session_registry.current_session_id[:8]}"
        lines = ["=== Cleaner Recommendations ==="]
        for r in recs:
            lines.append(
                f"[{r.id}] {r.description}\n"
                f"    type={r.action_type} impact={r.estimated_impact:+.3f} "
                f"confidence={r.confidence:.2f}"
            )
        response = "\n".join(lines) if recs else "No cleaning issues detected."
        reward = compute_total_reward(0.0, r_process, 0.0) + trunc_pen
        return self._make_obs(response, step, budget, reward)

    def _cmd_query_augmenter(self, cls, step, budget, r_process, trunc_pen) -> DataCentricObservation:
        reset_session_apply_state(self._exploit)
        recs = self._augmenter.query(self._working_copy, self._session_registry, cls)
        self._active_session = f"augmenter:{self._session_registry.current_session_id[:8]}"
        lines = ["=== Augmenter Recommendations ==="]
        for r in recs:
            lines.append(
                f"[{r.id}] {r.description}\n"
                f"    type={r.action_type} impact={r.estimated_impact:+.3f} "
                f"confidence={r.confidence:.2f}"
            )
        response = "\n".join(lines) if recs else "No augmentation needed."
        reward = compute_total_reward(0.0, r_process, 0.0) + trunc_pen
        return self._make_obs(response, step, budget, reward)

    def _cmd_query_balancer(self, step, budget, r_process, trunc_pen) -> DataCentricObservation:
        reset_session_apply_state(self._exploit)
        recs = self._balancer.query(self._working_copy, self._session_registry)
        self._active_session = f"balancer:{self._session_registry.current_session_id[:8]}"
        lines = ["=== Balancer Recommendations ==="]
        for r in recs:
            lines.append(
                f"[{r.id}] {r.description}\n"
                f"    type={r.action_type} impact={r.estimated_impact:+.3f} "
                f"confidence={r.confidence:.2f}"
            )
        response = "\n".join(lines) if recs else "Dataset is already balanced."
        reward = compute_total_reward(0.0, r_process, 0.0) + trunc_pen
        return self._make_obs(response, step, budget, reward)

    def _cmd_query_analyst(self, step, budget, r_process, trunc_pen) -> DataCentricObservation:
        """Holistic diagnosis + prioritised action plan. Costs 1 budget."""
        # Costs 1 extra budget step
        self._state.step_count += 1
        plan = self._analyst.query(
            self._working_copy,
            self._metadata["col_meta"],
            self._current_accuracy,
            TASK_CONFIGS[self._task]["target_accuracy"],
            budget - 1,
        )
        response = f"=== Analyst Report (costs 1 budget) ===\n{plan}"
        reward = compute_total_reward(0.0, r_process + 0.02, 0.0) + trunc_pen  # small bonus for planning
        budget_remaining = TASK_CONFIGS[self._task]["budget"] - self._state.step_count
        return self._make_obs(response, step, budget_remaining, reward)

    def _cmd_query_validator(self, step, budget, r_process, trunc_pen) -> DataCentricObservation:
        # Costs 2 budget
        self._state.step_count += 1
        violations = self._validator.query(self._working_copy, self._metadata["col_meta"])
        lines = ["=== Validator Report (costs 2 budget) ==="]
        if violations:
            for v in violations:
                lines.append(
                    f"  [{v.severity}] [{v.column}] rule={v.rule} count={v.count}\n    {v.description}"
                )
        else:
            lines.append("  No rule violations found.")
        response = "\n".join(lines)
        reward = compute_total_reward(0.0, r_process, 0.0) + trunc_pen
        budget_remaining = TASK_CONFIGS[self._task]["budget"] - self._state.step_count
        return self._make_obs(response, step, budget_remaining, reward)

    def _cmd_apply(self, rec_id, step, budget, r_process, trunc_pen) -> DataCentricObservation:
        if rec_id < 1:
            # Error: return 0 reward (no penalty, no bonus)
            return self._make_obs("Error: invalid recommendation ID.", step, budget, 0.0)

        # Check apply allowed (duplicate / session limit) — 0 reward on error
        allowed, err = check_apply_allowed(rec_id, self._exploit)
        if not allowed:
            return self._make_obs(f"Error: {err}", step, budget, 0.0)

        # Get recommendation (staleness check) — 0 reward, no penalty
        rec = self._session_registry.get(rec_id, self._session_registry.current_session_id)
        if rec is None:
            return self._make_obs(
                f"Error: stale recommendation ID {rec_id}. Please re-query for fresh recommendations.",
                step, budget, 0.0
            )

        # Capture quality before mutation for step reward
        quality_before = compute_lightweight_score(
            self._working_copy, self._ground_truth,
            self._metadata["original_length"], self._metadata["col_meta"],
            initial_missing=self._metadata.get("initial_missing"),
        )

        # Execute payload
        payload = rec._payload
        action_type = payload.get("action", "")
        wc = self._working_copy
        orig_len = self._metadata["original_length"]
        pre_rows = len(wc)
        pre_missing = int(wc.isnull().sum().sum())
        pre_dups = int(wc.duplicated().sum())

        # Save snapshot for undo before mutating
        self._dataset_history.append(self._working_copy.copy())
        if len(self._dataset_history) > self._max_history:
            self._dataset_history.pop(0)

        try:
            if action_type == "fill_missing":
                col = payload["column"]
                strategy = payload.get("strategy", "mean")  # honor smarter CleanerAgent choice
                numeric = pd.to_numeric(wc[col], errors="coerce")
                if strategy == "median":
                    fill_val = float(numeric.median())
                else:
                    fill_val = float(numeric.mean())
                wc[col] = numeric.fillna(fill_val)
                self._working_copy = wc

            elif action_type == "remove_duplicates":
                self._working_copy = wc.drop_duplicates().reset_index(drop=True)

            elif action_type == "fix_type_errors":
                col = payload["column"]
                numeric = pd.to_numeric(wc[col], errors="coerce")
                mean_val = float(numeric.mean())
                wc[col] = numeric.fillna(mean_val)
                self._working_copy = wc

            elif action_type == "augment_class":
                cls_int = payload["class"]
                n_synth = payload["n_synth"]
                cls_rows = wc[wc["target"] == cls_int]
                if len(cls_rows) > 0:
                    synth = cls_rows.sample(n=n_synth, replace=True, random_state=42)
                    noise_cols = [c for c in synth.columns if c != "target"]
                    for c in noise_cols:
                        try:
                            synth[c] = pd.to_numeric(synth[c], errors="coerce")
                            synth[c] = synth[c] + synth[c].std() * 0.1
                        except Exception:
                            pass
                    self._working_copy = pd.concat([wc, synth], ignore_index=True)

            elif action_type == "oversample":
                cls_int = payload["class"]
                target_count = payload["target_count"]
                cls_rows = wc[wc["target"] == cls_int]
                n_needed = max(0, target_count - len(cls_rows))
                if n_needed > 0:
                    extra = cls_rows.sample(n=n_needed, replace=True, random_state=42)
                    self._working_copy = pd.concat([wc, extra], ignore_index=True)

            elif action_type == "undersample":
                cls_int = payload["class"]
                target_count = payload["target_count"]
                cls_rows = wc[wc["target"] == cls_int]
                if len(cls_rows) > target_count:
                    keep = cls_rows.sample(n=target_count, random_state=42)
                    other = wc[wc["target"] != cls_int]
                    self._working_copy = pd.concat([keep, other], ignore_index=True)

            elif action_type == "remove_outlier_rows":
                col = payload["column"]
                pct = payload.get("pct", 5)
                try:
                    numeric = pd.to_numeric(wc[col], errors="coerce")
                    threshold = float(numeric.quantile(pct / 100))
                    self._working_copy = wc[pd.to_numeric(wc[col], errors="coerce") >= threshold].reset_index(drop=True)
                except Exception:
                    pass

        except Exception as exc:
            logger.exception("Error executing apply: %s", exc)
            return self._make_obs(f"Error executing recommendation: {exc}", step, budget, 0.0)

        record_apply(rec_id, self._exploit)

        # Ground truth immutability assertion — must never change
        gt_ok, gt_msg = assert_ground_truth_intact(
            len(self._ground_truth), self._exploit.ground_truth_row_count
        )
        if not gt_ok:
            logger.critical(gt_msg)
            return self._do_submit(penalty=-1.0, extra_msg=gt_msg)

        wc_new = self._working_copy
        post_rows = len(wc_new)
        post_missing = int(wc_new.isnull().sum().sum())
        post_dups = int(wc_new.duplicated().sum())
        rows_pct = post_rows / orig_len

        # Catastrophic data loss
        catastro, cmsg = check_catastrophic_data_loss(post_rows, orig_len)
        if catastro:
            return self._do_submit(penalty=-0.40, extra_msg=cmsg)

        # Preservation reward
        r_preservation = compute_preservation_reward(post_rows, orig_len)

        # Lightweight quality (use episode-start missing count as denominator)
        quality = compute_lightweight_score(
            wc_new, self._ground_truth, orig_len, self._metadata["col_meta"],
            initial_missing=self._metadata.get("initial_missing"),
        )

        # Build rich feedback with drift detection
        cfg = TASK_CONFIGS[self._task]
        missing_status = "OK" if post_missing == 0 else f"{post_missing} remaining"
        dup_status = "OK" if post_dups == 0 else f"{post_dups} remaining"
        drift = compute_drift(self._working_copy, self._ground_truth)
        drift_summary = format_drift_summary(drift)
        response = (
            f"Applied: {action_type} [{rec.description[:80]}]\n\n"
            f"Dataset health check:\n"
            f"  Missing values: {missing_status} (was {pre_missing})\n"
            f"  Duplicates: {dup_status} (was {pre_dups})\n"
            f"  Row count: {post_rows}/{orig_len} ({rows_pct*100:.1f}% preserved)\n"
            f"  {drift_summary}\n\n"
            f"Estimated quality score: {quality:.4f}\n"
            f"Budget remaining: {budget}"
        )

        reward = compute_total_reward(
            0.0, r_process, r_preservation,
            reward_step=compute_step_reward(
                f"apply {rec_id}", quality_before, quality, rows_pct
            ),
        ) + trunc_pen
        self._evaluator.invalidate_cache()
        return self._make_obs(response, step, budget, reward, quality=quality,
                              rows_pct=rows_pct)

    def _cmd_reject(self, rec_id, step, budget, r_process, trunc_pen) -> DataCentricObservation:
        response = (
            f"Recommendation {rec_id} rejected. It will not appear in future queries."
            if rec_id >= 1 else "Error: invalid recommendation ID."
        )
        reward = compute_total_reward(0.0, r_process + 0.01, 0.0) + trunc_pen
        return self._make_obs(response, step, budget, reward)

    def _cmd_undo(self, step, budget, r_process, trunc_pen) -> DataCentricObservation:
        """Restore previous dataset state (max 3 levels deep)."""
        if self._dataset_history:
            self._working_copy = self._dataset_history.pop()
            self._evaluator.invalidate_cache()
            orig_len = self._metadata["original_length"]
            rows_pct = len(self._working_copy) / orig_len
            quality = compute_lightweight_score(
                self._working_copy, self._ground_truth,
                orig_len, self._metadata["col_meta"],
                initial_missing=self._metadata.get("initial_missing"),
            )
            response = (
                f"Undo successful. Reverted to previous dataset state.\n"
                f"Row count: {len(self._working_copy)}/{orig_len} ({rows_pct*100:.1f}% preserved)\n"
                f"Estimated quality: {quality:.4f}\n"
                f"Snapshots remaining: {len(self._dataset_history)}"
            )
            reward = compute_total_reward(0.0, r_process - 0.03, 0.0) + trunc_pen  # small cost
        else:
            response = "Nothing to undo. No previous state available."
            reward = compute_total_reward(0.0, r_process - 0.05, 0.0) + trunc_pen  # larger cost
        return self._make_obs(response, step, budget, reward)


    def _cmd_validate(self, step, budget, r_process, trunc_pen) -> DataCentricObservation:
        allowed, cooldown_msg = check_validate_cooldown(self._exploit)
        if not allowed:
            return self._make_obs(cooldown_msg, step, budget, 0.0)

        prev_rf = self._evaluator.last_accuracy
        prev_lr = self._evaluator.last_lr_accuracy

        acc, per_class, from_cache, lr_acc = self._evaluator.evaluate(
            self._working_copy, self._ground_truth
        )
        cache_label = " (cached)" if from_cache else ""

        if from_cache:
            r_validate = 0.0
        else:
            r_validate = get_validate_reward(self._exploit)
            record_validate(self._exploit)

        r_accuracy = compute_accuracy_reward(
            acc, self._current_accuracy,
            self._metadata["baseline_accuracy"],
            TASK_CONFIGS[self._task]["target_accuracy"],
        )
        self._previous_accuracy = self._current_accuracy
        self._current_accuracy = acc

        target = TASK_CONFIGS[self._task]["target_accuracy"]
        agreement = self._evaluator.agreement_signal(acc, lr_acc, prev_rf, prev_lr)
        feat_text = self._evaluator.feature_importance_text()

        lines = [
            f"=== Validate{cache_label} ===",
            f"RF Accuracy: {acc:.4f}  (primary)",
            f"LR Accuracy: {lr_acc:.4f}  (secondary)",
            f"Agreement:   {agreement}",
        ]
        for cls, metrics in per_class.items():
            if isinstance(metrics, dict):
                lines.append(
                    f"  Class {cls}: p={metrics.get('precision',0):.3f} "
                    f"r={metrics.get('recall',0):.3f} f1={metrics.get('f1-score',0):.3f}"
                )
        lines.append(f"Target: {target:.4f} | {'HIT ✓' if acc >= target else 'Not yet'}")
        if feat_text:
            lines.append(feat_text)
        response = "\n".join(lines)

        reward = compute_total_reward(r_accuracy, r_process + r_validate, 0.0) + trunc_pen
        return self._make_obs(response, step, budget, reward)

    # ── submit ────────────────────────────────────────────────────────────────

    def _do_submit(self, penalty: float = 0.0, extra_msg: str = "") -> DataCentricObservation:
        cfg = TASK_CONFIGS[self._task]
        orig_len = self._metadata["original_length"]
        budget_remaining = cfg["budget"] - self._state.step_count

        # Final accuracy
        acc, per_class, _, lr_acc = self._evaluator.evaluate(
            self._working_copy, self._ground_truth
        )
        self._current_accuracy = acc

        r_accuracy = compute_accuracy_reward(
            acc, self._previous_accuracy,
            cfg["baseline_accuracy"], cfg["target_accuracy"],
            is_submit=True,
        )
        r_process = compute_process_reward(self._action_history[:-1], "submit")
        r_preservation = compute_preservation_reward(len(self._working_copy), orig_len)
        r_efficiency = compute_efficiency_reward(
            acc, cfg["baseline_accuracy"], cfg["budget"], max(budget_remaining, 0)
        )

        total = compute_total_reward(r_accuracy, r_process, r_preservation, r_efficiency)
        total += penalty

        hit = acc >= cfg["target_accuracy"]
        response = (
            f"{'=' * 40}\n"
            f"EPISODE COMPLETE\n"
            f"{'=' * 40}\n"
            f"Final accuracy:  {acc:.4f}\n"
            f"Target accuracy: {cfg['target_accuracy']:.4f}\n"
            f"Baseline:        {cfg['baseline_accuracy']:.4f}\n"
            f"Result: {'TARGET HIT ✓' if hit else 'Target not reached'}\n\n"
            f"Reward breakdown:\n"
            f"  Accuracy:     {r_accuracy:+.4f}\n"
            f"  Process:      {r_process:+.4f}\n"
            f"  Preservation: {r_preservation:+.4f}\n"
            f"  Efficiency:   {r_efficiency:+.4f}\n"
            f"  Penalty:      {penalty:+.4f}\n"
            f"  TOTAL:        {total:+.4f}\n"
            + (f"\n{extra_msg}" if extra_msg else "")
        )

        quality = compute_lightweight_score(
            self._working_copy, self._ground_truth,
            orig_len, self._metadata["col_meta"],
        )
        rows_pct = len(self._working_copy) / orig_len

        return DataCentricObservation(
            response=response,
            current_accuracy=acc,
            baseline_accuracy=cfg["baseline_accuracy"],
            target_accuracy=cfg["target_accuracy"],
            estimated_quality=quality,
            dataset_shape=f"{len(self._working_copy)} rows × {len(self._working_copy.columns)-1} columns",
            rows_preserved_pct=rows_pct,
            budget_remaining=max(budget_remaining, 0),
            step_number=self._state.step_count,
            max_steps=cfg["budget"],
            active_session=self._active_session,
            validate_calls_remaining=validate_calls_remaining(self._exploit),
            done=True,
            reward=round(total, 4),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _make_obs(self, response: str, step: int, budget: int, reward: float,
                  quality: Optional[float] = None, rows_pct: Optional[float] = None
                  ) -> DataCentricObservation:
        cfg = TASK_CONFIGS[self._task]
        orig_len = self._metadata["original_length"]
        wc = self._working_copy
        if quality is None:
            quality = compute_lightweight_score(
                wc, self._ground_truth, orig_len, self._metadata["col_meta"],
                initial_missing=self._metadata.get("initial_missing"),
            )
        if rows_pct is None:
            rows_pct = len(wc) / orig_len

        return DataCentricObservation(
            response=response,
            current_accuracy=self._current_accuracy,
            baseline_accuracy=cfg["baseline_accuracy"],
            target_accuracy=cfg["target_accuracy"],
            estimated_quality=quality,
            dataset_shape=f"{len(wc)} rows × {len(wc.columns)-1} columns",
            rows_preserved_pct=rows_pct,
            budget_remaining=max(budget, 0),
            step_number=step,
            max_steps=cfg["budget"],
            active_session=self._active_session,
            validate_calls_remaining=validate_calls_remaining(self._exploit),
            done=False,
            reward=round(reward, 4),
        )

    def _error_obs(self, msg: str) -> DataCentricObservation:
        return DataCentricObservation(response=msg, done=False, reward=0.0)

    def _unknown_cmd_obs(self, msg: str, step: int, budget: int,
                         reward: float) -> DataCentricObservation:
        return self._make_obs(
            f"Unknown command: '{msg}'\n\n{AVAILABLE_COMMANDS}", step, budget, reward
        )

    @property
    def state(self) -> State:
        return self._state
