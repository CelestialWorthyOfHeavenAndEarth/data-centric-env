"""
Model Evaluator for Data-Centric RL Environment.

Uses sklearn RandomForestClassifier (primary) + LogisticRegression (secondary)
with hash-based caching. Trains on working_copy, evaluates on held-out
ground_truth test split.

Primary accuracy (RF) drives all rewards and grading.
Secondary accuracy (LR) is diagnostic — shows whether improvements generalise
beyond the RF decision boundary (overfitting detection).
"""

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ModelEvaluator:
    """Caching dual-classifier evaluator.

    Args:
        fast_mode: If True, uses n_estimators=20 (for GRPO rollouts, ~4x faster).
                   If False, uses n_estimators=100 (for final eval, more accurate).
    """

    def __init__(self, fast_mode: bool = False):
        self._cache_hash: Optional[str] = None
        self._cached_accuracy: float = 0.0
        self._cached_per_class: Dict = {}
        self._cached_lr_accuracy: float = 0.0
        self._cached_feature_importance: Dict[str, float] = {}
        self._cached = False
        self._fast_mode = fast_mode
        self._n_estimators = 20 if fast_mode else 100

    def _compute_hash(self, df: pd.DataFrame) -> str:
        try:
            return hashlib.md5(
                pd.util.hash_pandas_object(df, index=True).values.tobytes()
            ).hexdigest()
        except Exception:
            return hashlib.md5(df.to_json().encode()).hexdigest()

    def evaluate(
        self,
        working_copy: pd.DataFrame,
        ground_truth: pd.DataFrame,
        test_size: float = 0.25,
        seed: int = 42,
    ) -> Tuple[float, Dict, bool, float]:
        """
        Train on working_copy; evaluate on held-out ground_truth test split.

        Returns:
            rf_accuracy    – float  (primary, used for rewards)
            per_class      – dict from classification_report
            from_cache     – True if result came from cache (no retrain)
            lr_accuracy    – float  (secondary, diagnostic only)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split

        current_hash = self._compute_hash(working_copy)

        if self._cached and current_hash == self._cache_hash:
            return self._cached_accuracy, self._cached_per_class, True, self._cached_lr_accuracy

        # Prepare train set from working_copy
        wc = working_copy.copy()
        wc = wc.dropna(subset=["target"])
        for col in wc.columns:
            if col != "target":
                wc[col] = pd.to_numeric(wc[col], errors="coerce")
        wc = wc.dropna()

        _empty = (0.0, {}, False, 0.0)
        if len(wc) < 10:
            self._cache_hash = current_hash
            self._cached_accuracy = 0.0
            self._cached_per_class = {}
            self._cached_lr_accuracy = 0.0
            self._cached_feature_importance = {}
            self._cached = True
            return _empty

        X_train = wc.drop("target", axis=1).values
        y_train = wc["target"].astype(int).values

        # Build test set from ground_truth
        gt = ground_truth.copy().dropna()
        for col in gt.columns:
            if col != "target":
                gt[col] = pd.to_numeric(gt[col], errors="coerce")
        gt = gt.dropna()

        if len(gt) < 10:
            self._cache_hash = current_hash
            self._cached_accuracy = 0.0
            self._cached_per_class = {}
            self._cached_lr_accuracy = 0.0
            self._cached_feature_importance = {}
            self._cached = True
            return _empty

        _, X_test_df, _, y_test_df = train_test_split(
            gt.drop("target", axis=1),
            gt["target"],
            test_size=test_size,
            random_state=seed,
            stratify=gt["target"] if gt["target"].nunique() > 1 else None,
        )
        y_test = y_test_df.astype(int).values

        # Align columns
        train_cols = list(wc.drop("target", axis=1).columns)
        test_cols = list(X_test_df.columns)
        shared = [c for c in train_cols if c in test_cols]
        if not shared:
            self._cache_hash = current_hash
            self._cached_accuracy = 0.0
            self._cached_per_class = {}
            self._cached_lr_accuracy = 0.0
            self._cached_feature_importance = {}
            self._cached = True
            return _empty

        X_train_arr = wc[shared].values
        X_test_arr = X_test_df[shared].values

        # ── Primary: Random Forest ──────────────────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=self._n_estimators,
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X_train_arr, y_train)
        y_pred_rf = rf.predict(X_test_arr)

        rf_accuracy = float(rf.score(X_test_arr, y_test))
        try:
            per_class = classification_report(
                y_test, y_pred_rf, output_dict=True, zero_division=0
            )
        except Exception:
            per_class = {}

        # Feature importance (RF gives this for free)
        feature_importance: Dict[str, float] = {}
        if hasattr(rf, "feature_importances_"):
            importances = rf.feature_importances_
            for col, imp in zip(shared, importances):
                feature_importance[col] = round(float(imp), 4)

        # ── Secondary: Logistic Regression (diagnostic) ─────────────────────
        lr_accuracy = 0.0
        if not self._fast_mode:  # skip LR in fast_mode to keep GRPO rollouts quick
            try:
                lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=1)
                lr.fit(X_train_arr, y_train)
                lr_accuracy = float(lr.score(X_test_arr, y_test))
            except Exception:
                lr_accuracy = 0.0
        else:
            # In fast_mode, reuse RF accuracy as placeholder (not shown to agent)
            lr_accuracy = rf_accuracy

        # Update cache
        self._cache_hash = current_hash
        self._cached_accuracy = rf_accuracy
        self._cached_per_class = per_class
        self._cached_lr_accuracy = lr_accuracy
        self._cached_feature_importance = feature_importance
        self._cached = True

        return rf_accuracy, per_class, False, lr_accuracy

    def agreement_signal(self, rf_acc: float, lr_acc: float,
                         prev_rf: float, prev_lr: float) -> str:
        """
        Compare RF vs LR improvement direction.
        Returns a signal string for the agent to reason about.
        """
        rf_improved = rf_acc > prev_rf + 0.005
        lr_improved = lr_acc > prev_lr + 0.005
        rf_declined = rf_acc < prev_rf - 0.005
        lr_declined = lr_acc < prev_lr - 0.005

        if rf_improved and lr_improved:
            return "BOTH_AGREE_IMPROVE — fix is robust and generalises"
        elif rf_improved and lr_declined:
            return "DISAGREE — RF improved but LR declined (possible RF-specific overfitting)"
        elif rf_declined and lr_declined:
            return "BOTH_DECLINED — last change hurt both classifiers, consider undo"
        elif not rf_improved and not rf_declined:
            return "NO_CHANGE — last operation had no measurable effect"
        else:
            return "MIXED — marginal changes, continue and validate again"

    def feature_importance_text(self, top_n: int = 5) -> str:
        """Return formatted feature importance string for agent observation."""
        if not self._cached_feature_importance:
            return ""
        sorted_feats = sorted(
            self._cached_feature_importance.items(),
            key=lambda x: -x[1]
        )[:top_n]
        parts = [f"{col} ({imp:.3f})" for col, imp in sorted_feats]
        return "Feature importance: " + " > ".join(parts)

    def invalidate_cache(self):
        self._cached = False
        self._cache_hash = None

    @property
    def last_accuracy(self) -> float:
        return self._cached_accuracy if self._cached else 0.0

    @property
    def last_lr_accuracy(self) -> float:
        return self._cached_lr_accuracy if self._cached else 0.0

    @property
    def last_feature_importance(self) -> Dict[str, float]:
        return self._cached_feature_importance if self._cached else {}
