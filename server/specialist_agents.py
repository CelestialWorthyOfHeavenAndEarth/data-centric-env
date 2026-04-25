"""
Specialist Agents for Data-Centric RL Environment.

Agents:
  CleanerAgent    — detects missing values, duplicates, type errors
  AugmenterAgent  — suggests synthetic minority class samples
  BalancerAgent   — recommends resampling strategies
  ValidatorAgent  — checks column metadata rule violations (costs 2 budget)
  AnalystAgent    — holistic diagnosis + prioritised action plan (costs 1 budget)

Also exports:
  compute_drift() — per-column distribution drift score (no scipy needed)
"""

import hashlib
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ── Recommendation / Violation dataclasses ──────────────────────────────────

@dataclass
class Recommendation:
    id: int
    description: str
    action_type: str
    estimated_impact: float
    confidence: float
    session_id: str
    _payload: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class Violation:
    column: str
    rule: str
    count: int
    description: str
    severity: str = "WARNING"   # NEW: CRITICAL / WARNING / INFO


# ── Session registry ─────────────────────────────────────────────────────────

class SessionRegistry:
    """Tracks the active recommendation session to detect stale IDs."""

    def __init__(self):
        self.current_session_id: str = ""
        self.recommendations: Dict[int, Recommendation] = {}

    def new_session(self) -> str:
        self.current_session_id = str(uuid.uuid4())
        self.recommendations = {}
        return self.current_session_id

    def register(self, recs: List[Recommendation]) -> None:
        for r in recs:
            self.recommendations[r.id] = r

    def get(self, rec_id: int, session_id: str) -> Optional[Recommendation]:
        if session_id != self.current_session_id:
            return None
        return self.recommendations.get(rec_id)

    def is_valid_session(self, session_id: str) -> bool:
        return session_id == self.current_session_id


# ── Shared helpers ───────────────────────────────────────────────────────────

def _seeded_rng(df: pd.DataFrame, salt: str = "") -> random.Random:
    h = hashlib.md5((df.to_json() + salt).encode()).hexdigest()
    return random.Random(int(h[:8], 16))


def _col_stats(series: pd.Series) -> Dict[str, float]:
    """Return basic stats dict for a numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "skew": 0.0, "n": 0}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
        "skew": float(s.skew()) if len(s) > 2 else 0.0,
        "n": len(s),
    }


def _impute_strategy(stats: Dict[str, float]) -> tuple:
    """Choose mean vs median based on skewness. Returns (strategy, value, reason)."""
    skew = abs(stats["skew"])
    if skew > 1.0:
        return "median", stats["median"], f"right-skewed (skew={stats['skew']:.2f}), median more robust"
    elif skew > 0.5:
        return "median", stats["median"], f"moderately skewed (skew={stats['skew']:.2f}), median preferred"
    else:
        return "mean", stats["mean"], f"near-symmetric (skew={stats['skew']:.2f}), mean appropriate"


# ── Drift Detection ──────────────────────────────────────────────────────────

def compute_drift(working_copy: pd.DataFrame, ground_truth: pd.DataFrame) -> Dict[str, float]:
    """
    Per-column distribution drift score comparing working_copy to ground_truth.
    Uses mean-shift + std-ratio (no scipy dependency).
    Returns dict: column -> drift_score (0.0 = no drift, >1.0 = HIGH drift).
    """
    drift = {}
    for col in working_copy.columns:
        if col == "target":
            continue
        try:
            wc_vals = pd.to_numeric(working_copy[col], errors="coerce").dropna()
            gt_vals = pd.to_numeric(ground_truth[col], errors="coerce").dropna()
            if len(wc_vals) == 0 or len(gt_vals) == 0:
                drift[col] = 0.0
                continue
            mean_shift = abs(wc_vals.mean() - gt_vals.mean()) / (gt_vals.std() + 1e-8)
            std_ratio = wc_vals.std() / (gt_vals.std() + 1e-8)
            drift[col] = round(float(mean_shift + abs(1.0 - std_ratio)), 3)
        except Exception:
            drift[col] = 0.0
    return drift


def _drift_label(score: float) -> str:
    if score < 0.2:
        return "NONE"
    elif score < 0.5:
        return "LOW"
    elif score < 1.0:
        return "MEDIUM"
    else:
        return "HIGH"


def format_drift_summary(drift: Dict[str, float]) -> str:
    """Return one-line drift summary for agent observation."""
    if not drift:
        return ""
    parts = [f"{col} ({_drift_label(v)})" for col, v in sorted(drift.items(), key=lambda x: -x[1])]
    return "Distribution drift: " + " | ".join(parts[:5])  # top 5 most drifted


# ── CleanerAgent ─────────────────────────────────────────────────────────────

class CleanerAgent:
    """
    Analyses working_copy for missing values, duplicates, type mismatches.
    Returns 2-4 recommendations with statistical reasoning.

    Hidden flaw (15% of calls, deterministic): occasionally recommends
    removing rows that are valid. Detectable because estimated_impact < 0.
    """

    def query(self, df: pd.DataFrame, session_registry: SessionRegistry,
              col_meta: Dict) -> List[Recommendation]:
        sid = session_registry.new_session()
        rng = _seeded_rng(df, "cleaner")
        recs = []
        rec_id = 1
        n_rows = len(df)

        # --- Missing value recommendations with statistical reasoning ---
        for col in df.columns:
            if col == "target":
                continue
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue

            pct_missing = n_missing / n_rows * 100
            stats = _col_stats(df[col])
            strategy, value, reason = _impute_strategy(stats)

            # Confidence: lower if >30% missing (imputation less reliable)
            confidence = round(max(0.60, 0.92 - (pct_missing / 100) * 0.5), 2)

            # Risk label
            if pct_missing < 5:
                risk = "LOW"
            elif pct_missing < 20:
                risk = "MEDIUM"
            else:
                risk = "HIGH — imputation may introduce bias"

            mean_median_delta = abs(stats["mean"] - stats["median"])
            description = (
                f"Fill {n_missing}/{n_rows} ({pct_missing:.1f}%) missing values in '{col}' "
                f"using {strategy} ({value:.2f}). "
                f"Reason: {reason}. "
                f"Mean={stats['mean']:.2f}, Median={stats['median']:.2f} "
                f"(delta={mean_median_delta:.2f}). Risk: {risk}."
            )

            recs.append(Recommendation(
                id=rec_id,
                description=description,
                action_type="fill_missing",
                estimated_impact=round(min(0.03 + pct_missing / 100 * 0.3, 0.12), 3),
                confidence=confidence,
                session_id=sid,
                _payload={"action": "fill_missing", "column": col, "strategy": strategy},
            ))
            rec_id += 1

        # --- Duplicate recommendation ---
        n_dups = int(df.duplicated().sum())
        if n_dups > 0:
            pct_dups = n_dups / n_rows * 100
            recs.append(Recommendation(
                id=rec_id,
                description=(
                    f"Remove {n_dups} duplicate rows ({pct_dups:.1f}% of dataset). "
                    f"Duplicates bias the classifier toward overrepresented patterns. Risk: LOW."
                ),
                action_type="remove_duplicates",
                estimated_impact=round(min(0.02 + n_dups / n_rows * 0.15, 0.08), 3),
                confidence=0.92,
                session_id=sid,
                _payload={"action": "remove_duplicates"},
            ))
            rec_id += 1

        # --- Type error recommendations ---
        for col in df.columns:
            if col == "target":
                continue
            meta = col_meta.get(col, {})
            expected = meta.get("expected_dtype", "float64")
            if expected in ("float64", "int64"):
                n_errors = sum(1 for val in df[col].dropna()
                               if not _is_numeric(val))
                if n_errors > 0:
                    pct_err = n_errors / n_rows * 100
                    recs.append(Recommendation(
                        id=rec_id,
                        description=(
                            f"Fix {n_errors} type errors ({pct_err:.1f}%) in '{col}' "
                            f"(non-numeric values coerced to NaN, then filled with mean). "
                            f"Expected dtype: {expected}. Risk: LOW."
                        ),
                        action_type="fix_type_errors",
                        estimated_impact=round(min(0.04 + n_errors / n_rows * 0.2, 0.10), 3),
                        confidence=0.88,
                        session_id=sid,
                        _payload={"action": "fix_type_errors", "column": col},
                    ))
                    rec_id += 1

        # --- Hidden flaw: ~15% chance of recommending valid row removal ---
        flaw_hash = int(hashlib.md5(df.to_json().encode()).hexdigest()[:4], 16)
        if flaw_hash % 100 < 15 and len(recs) < 4:
            col = rng.choice([c for c in df.columns if c != "target"])
            recs.append(Recommendation(
                id=rec_id,
                description=(
                    f"Remove rows where '{col}' is below the 5th percentile "
                    f"(suspected outliers). Confidence LOW — verify with query_validator first."
                ),
                action_type="remove_outlier_rows",
                estimated_impact=round(rng.uniform(-0.05, 0.01), 3),
                confidence=round(rng.uniform(0.55, 0.70), 2),
                session_id=sid,
                _payload={"action": "remove_outlier_rows", "column": col, "pct": 5},
            ))

        # Keep top 4 by estimated_impact, re-number
        recs = sorted(recs, key=lambda r: -r.estimated_impact)[:4]
        for i, r in enumerate(recs, 1):
            r.id = i
        session_registry.register(recs)
        return recs


def _is_numeric(val) -> bool:
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


# ── AugmenterAgent ───────────────────────────────────────────────────────────

class AugmenterAgent:
    """
    Detects underrepresented classes and suggests synthetic samples.
    Returns 1-3 recommendations with class distribution reasoning.

    Hidden flaw: sometimes suggests out-of-distribution samples (flagged).
    """

    def query(self, df: pd.DataFrame, session_registry: SessionRegistry,
              class_name: Optional[str] = None) -> List[Recommendation]:
        sid = session_registry.new_session()
        rng = _seeded_rng(df, "augmenter")
        value_counts = df["target"].value_counts()
        total = len(df)
        recs = []
        rec_id = 1

        targets = [class_name] if class_name else [str(c) for c in value_counts.index]

        # Class distribution context
        dist_str = ", ".join(f"class {k}: {v} ({v/total*100:.1f}%)"
                             for k, v in value_counts.items())

        for cls in targets[:3]:
            try:
                cls_int = int(cls)
            except (ValueError, TypeError):
                continue
            if cls_int not in value_counts.index:
                continue

            count = value_counts[cls_int]
            max_count = value_counts.max()
            gap = max_count - count
            if gap <= 0:
                continue

            n_synth = min(gap, max(5, int(gap * 0.5)))
            ratio_before = count / max_count
            ratio_after = (count + n_synth) / max_count

            flaw_hash = int(hashlib.md5((df.to_json() + cls).encode()).hexdigest()[:4], 16)
            is_ood = flaw_hash % 100 < 20
            impact = round(min(0.04 + n_synth / total * 0.4, 0.10), 3)
            if is_ood:
                impact = round(rng.uniform(-0.02, 0.02), 3)

            ood_note = " [WARNING: high OOD risk — run query_validator before applying]" if is_ood else ""
            risk = "HIGH" if is_ood else ("MEDIUM" if ratio_before < 0.3 else "LOW")

            description = (
                f"Generate {n_synth} synthetic samples for class '{cls}' via Gaussian perturbation. "
                f"Distribution: {dist_str}. "
                f"Imbalance ratio before: {ratio_before:.2f} → after: {ratio_after:.2f}. "
                f"Risk: {risk}.{ood_note}"
            )

            recs.append(Recommendation(
                id=rec_id,
                description=description,
                action_type="augment_class",
                estimated_impact=impact,
                confidence=round(0.60 if is_ood else 0.82, 2),
                session_id=sid,
                _payload={
                    "action": "augment_class",
                    "class": cls_int,
                    "n_synth": n_synth,
                    "ood": is_ood,
                },
            ))
            rec_id += 1

        session_registry.register(recs)
        return recs


# ── BalancerAgent ─────────────────────────────────────────────────────────────

class BalancerAgent:
    """
    Recommends resampling strategies for class imbalance.
    Returns 1-2 recommendations with entropy and ratio reasoning.

    Hidden flaw: occasionally over-balances (minority becomes too large).
    """

    def query(self, df: pd.DataFrame, session_registry: SessionRegistry) -> List[Recommendation]:
        sid = session_registry.new_session()
        rng = _seeded_rng(df, "balancer")
        value_counts = df["target"].value_counts()
        recs = []
        rec_id = 1

        if len(value_counts) < 2:
            session_registry.register([])
            return []

        min_cls = int(value_counts.idxmin())
        max_cls = int(value_counts.idxmax())
        min_count = int(value_counts.min())
        max_count = int(value_counts.max())
        imbalance_ratio = min_count / max_count

        # Class distribution entropy (0=perfectly imbalanced, 1=perfectly balanced)
        probs = value_counts / value_counts.sum()
        entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
        max_entropy = np.log2(len(value_counts))
        entropy_pct = entropy / max_entropy * 100 if max_entropy > 0 else 0

        flaw_hash = int(hashlib.md5(df.to_json().encode()).hexdigest()[:4], 16)
        is_overbalance = flaw_hash % 100 < 20
        target_count = max_count if not is_overbalance else int(max_count * 1.5)
        ratio_after = min(1.0, min_count / target_count) if target_count > 0 else 1.0

        overbalance_note = (
            " [WARNING: target exceeds majority class size — may over-correct and hurt generalisation]"
            if is_overbalance else ""
        )
        risk = "HIGH" if is_overbalance else ("MEDIUM" if imbalance_ratio < 0.3 else "LOW")

        recs.append(Recommendation(
            id=rec_id,
            description=(
                f"Upsample minority class {min_cls} from {min_count} to {target_count} rows "
                f"via random oversampling. "
                f"Imbalance ratio: {imbalance_ratio:.2f} → {ratio_after:.2f}. "
                f"Class entropy: {entropy_pct:.1f}% of maximum. Risk: {risk}.{overbalance_note}"
            ),
            action_type="oversample",
            estimated_impact=round(min(0.05 + (1 - imbalance_ratio) * 0.15, 0.12), 3),
            confidence=round(0.60 if is_overbalance else 0.80, 2),
            session_id=sid,
            _payload={
                "action": "oversample",
                "class": min_cls,
                "target_count": target_count,
                "overbalance": is_overbalance,
            },
        ))
        rec_id += 1

        if imbalance_ratio < 0.5:
            undersample_target = min_count * 2
            recs.append(Recommendation(
                id=rec_id,
                description=(
                    f"Downsample majority class {max_cls} from {max_count} to {undersample_target} rows "
                    f"via random undersampling. "
                    f"Warning: loses {max_count - undersample_target} majority-class examples. "
                    f"Risk: MEDIUM — use only if dataset is large enough."
                ),
                action_type="undersample",
                estimated_impact=round(min(0.03 + (1 - imbalance_ratio) * 0.08, 0.08), 3),
                confidence=0.75,
                session_id=sid,
                _payload={
                    "action": "undersample",
                    "class": max_cls,
                    "target_count": undersample_target,
                },
            ))

        session_registry.register(recs)
        return recs


# ── ValidatorAgent ────────────────────────────────────────────────────────────

class ValidatorAgent:
    """
    Checks working_copy against column metadata for rule violations.
    Returns list of Violation objects (diagnostic only — not recommendations).
    Costs 2 budget per call.
    ~10% false positive rate (flagged with [FALSE POSITIVE WARNING]).
    """

    def query(self, df: pd.DataFrame, col_meta: Dict) -> List[Violation]:
        rng = _seeded_rng(df, "validator")
        violations = []

        for col, meta in col_meta.items():
            if col == "target" or col not in df.columns:
                continue

            expected_dtype = meta.get("expected_dtype", "float64")
            valid_range = meta.get("valid_range")

            # Type violations
            if expected_dtype in ("float64", "int64"):
                n_errors = sum(1 for val in df[col].dropna() if not _is_numeric(val))
                if n_errors > 0:
                    pct = n_errors / len(df) * 100
                    violations.append(Violation(
                        column=col,
                        rule=f"dtype={expected_dtype}",
                        count=n_errors,
                        description=f"{n_errors} non-numeric values in '{col}' ({pct:.1f}%). Recommend fix_type_errors.",
                        severity="CRITICAL" if pct > 10 else "WARNING",
                    ))

            # Range violations
            if valid_range:
                lo, hi = valid_range
                try:
                    numeric_vals = pd.to_numeric(df[col], errors="coerce").dropna()
                    n_out = int(((numeric_vals < lo) | (numeric_vals > hi)).sum())
                    if n_out > 0:
                        max_val = float(numeric_vals.max())
                        min_val = float(numeric_vals.min())
                        std = float(numeric_vals.std()) or 1.0
                        z_max = abs(max_val - numeric_vals.mean()) / std
                        violations.append(Violation(
                            column=col,
                            rule=f"range=[{lo},{hi}]",
                            count=n_out,
                            description=(
                                f"{n_out} values in '{col}' outside [{lo}, {hi}]. "
                                f"Observed range: [{min_val:.1f}, {max_val:.1f}] "
                                f"(max Z-score: {z_max:.1f}). "
                                f"Severity: {'CRITICAL — likely data corruption' if z_max > 5 else 'WARNING — possible outliers'}."
                            ),
                            severity="CRITICAL" if z_max > 5 else "WARNING",
                        ))
                except Exception:
                    pass

        # ~10% false positive
        fp_hash = int(hashlib.md5(df.to_json().encode()).hexdigest()[:4], 16)
        if fp_hash % 100 < 10:
            feature_cols = [c for c in df.columns if c != "target"]
            if feature_cols:
                fp_col = rng.choice(feature_cols)
                violations.append(Violation(
                    column=fp_col,
                    rule="distribution_check",
                    count=rng.randint(1, 5),
                    description=(
                        f"[FALSE POSITIVE WARNING] Unusual value distribution in '{fp_col}' "
                        f"— may not be a real issue. Verify before acting."
                    ),
                    severity="INFO",
                ))

        return violations


# ── AnalystAgent ──────────────────────────────────────────────────────────────

class AnalystAgent:
    """
    Meta-specialist that performs holistic dataset diagnosis.
    Returns a prioritised action plan rather than individual recommendations.
    Costs 1 budget.

    Analyses:
    - Missing value severity
    - Class imbalance severity
    - Type error severity
    - Remaining accuracy gap
    Then ranks problems and recommends an ordered sequence of specialist calls.
    """

    def query(
        self,
        df: pd.DataFrame,
        col_meta: Dict,
        current_accuracy: float,
        target_accuracy: float,
        budget_remaining: int,
    ) -> str:
        """Return a formatted diagnostic + action plan string."""

        n_rows = max(len(df), 1)
        n_cells = n_rows * max(len(df.columns) - 1, 1)

        # ── Score each problem dimension (0.0 – 1.0) ──────────────────────

        # 1. Missing value severity
        total_missing = int(df.isnull().sum().sum())
        missing_severity = min(1.0, total_missing / n_cells * 5)

        # 2. Class imbalance severity
        vc = df["target"].value_counts()
        if len(vc) >= 2:
            imbalance_severity = 1.0 - (vc.min() / vc.max())
        else:
            imbalance_severity = 0.0

        # 3. Type error severity
        n_type_errors = 0
        for col in df.columns:
            if col == "target":
                continue
            meta = col_meta.get(col, {})
            if meta.get("expected_dtype", "float64") in ("float64", "int64"):
                n_type_errors += sum(1 for val in df[col].dropna() if not _is_numeric(val))
        type_severity = min(1.0, n_type_errors / n_cells * 10)

        # 4. Accuracy gap
        accuracy_gap = max(0.0, target_accuracy - current_accuracy)

        # ── Rank problems ─────────────────────────────────────────────────
        problems = [
            ("class imbalance",  imbalance_severity, "query_balancer"),
            ("missing values",   missing_severity,   "query_cleaner"),
            ("type errors",      type_severity,      "query_cleaner"),
        ]
        problems.sort(key=lambda x: -x[1])

        # ── Build diagnosis section ───────────────────────────────────────
        diagnosis_lines = ["DIAGNOSIS:"]
        for name, severity, specialist in problems:
            if severity < 0.05:
                level = "NONE"
            elif severity < 0.3:
                level = "LOW"
            elif severity < 0.6:
                level = "MEDIUM"
            else:
                level = "HIGH"
            diagnosis_lines.append(
                f"  - {name.title()}: severity={severity:.2f} [{level}] -> use {specialist}"
            )
        diagnosis_lines.append(
            f"  - Accuracy gap: {accuracy_gap:.4f} "
            f"({'within reach' if accuracy_gap < 0.05 else 'significant gap'})"
        )

        # ── Build action plan ─────────────────────────────────────────────
        plan_lines = [f"\nRECOMMENDED PLAN (budget remaining: {budget_remaining}):"]
        step = 1

        # Recommend top 2 non-trivial problems
        for name, severity, specialist in problems:
            if severity >= 0.1:
                plan_lines.append(f"  {step}. {specialist} → apply best recommendation")
                step += 1
            if step > 3:
                break

        # Always validate after fixes
        plan_lines.append(f"  {step}. validate (check accuracy improvement)")
        step += 1

        # Budget guidance
        if budget_remaining <= 8:
            plan_lines.append(
                f"  {step}. submit NOW — budget is critically low ({budget_remaining} steps left)"
            )
            plan_lines.append("  NOTE: Skip query_validator (costs 2 budget).")
        elif accuracy_gap < 0.02:
            plan_lines.append(f"  {step}. submit — you are very close to target")
        else:
            plan_lines.append(f"  {step}. Repeat if accuracy gap remains > 0.02, then submit")

        # Feature note
        if imbalance_severity > missing_severity and imbalance_severity > 0.2:
            plan_lines.append("\nPRIORITY NOTE: Class imbalance is the dominant issue — fix this first.")
        elif missing_severity > 0.2:
            plan_lines.append("\nPRIORITY NOTE: High missing-value rate — clean data before augmenting.")

        return "\n".join(diagnosis_lines + plan_lines)
