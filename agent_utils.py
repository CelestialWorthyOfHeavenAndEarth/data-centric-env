"""
Shared utilities for training and evaluation scripts.

Centralises the system prompt, user-prompt builder, and server lifecycle
helpers so that train_data_centric.py and eval_data_centric.py stay in sync
automatically.
"""

import subprocess
import time

import requests


# ── Valid commands (used for format-reward checking) ─────────────────────────

VALID_COMMANDS = [
    "inspect_dataset", "inspect_model", "query_analyst",
    "query_cleaner", "query_augmenter", "query_balancer", "query_validator",
    "apply", "reject", "undo", "validate", "submit",
]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Data-Centric AI agent. Your job is to improve a \
Machine learning dataset so a fixed classifier achieves higher accuracy.

STRATEGY — use this order:
1. query_analyst to get a prioritised action plan (costs 2 budget total, worth it)
2. inspect_dataset to understand the data
3. query the recommended specialist (query_cleaner, query_augmenter, query_balancer)
4. apply the best recommendation by number (e.g. apply 1)
5. validate to check if accuracy improved
6. repeat until you hit the target or run low on budget
7. submit to finalize

IMPORTANT RULES:
- Start with query_analyst — it tells you the biggest problem to fix first.
- Always query a specialist before applying. Never apply without querying first.
- Check the Agreement signal after validate: DISAGREE means possible overfitting.
- Validate after every 2-3 applies to track progress.
- Do not spam validate — it costs budget after 3 uses.
- query_validator costs 2 budget total — use only when suspicious of data quality.
- submit when accuracy >= target or budget < 5.

Reply with exactly ONE command per message. No explanation. Just the command."""


# ── User prompt builder ───────────────────────────────────────────────────────

def build_user_prompt(obs: dict) -> str:
    """Build the per-step user prompt from an observation dict."""
    improvement_needed = obs.get("target_accuracy", 0) - obs.get("current_accuracy", 0)
    return (
        f"Current situation:\n"
        f"Accuracy: {obs.get('current_accuracy', 0):.1%} → "
        f"Target: {obs.get('target_accuracy', 0):.1%}\n"
        f"Still need: {max(0, improvement_needed):.1%} improvement\n"
        f"Quality score: {obs.get('estimated_quality', 0):.2f}/1.00\n"
        f"Rows preserved: {obs.get('rows_preserved_pct', 1.0):.1%}\n"
        f"Budget remaining: {obs.get('budget_remaining', 0)} steps\n"
        f"Free validates left: {obs.get('validate_calls_remaining', 0)}\n"
        f"Active query session: {obs.get('active_session', 'none')}\n\n"
        f"Last result:\n{str(obs.get('response', ''))[:400]}\n\n"
        f"What is your next action? (one command only)"
    )


# ── Server lifecycle helpers ──────────────────────────────────────────────────

def start_server(base_url: str = "http://localhost:8000") -> subprocess.Popen:
    """Start the environment server as a subprocess and wait until healthy."""
    proc = subprocess.Popen(
        ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Poll until ready (max 30 seconds)
    for i in range(30):
        try:
            r = requests.get(f"{base_url}/health", timeout=1)
            if r.status_code == 200:
                print(f"Server ready after {i + 1}s")
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("Environment server failed to start in 30 seconds")


def stop_server(proc: subprocess.Popen) -> None:
    """Terminate the environment server subprocess."""
    proc.terminate()  # cross-platform (SIGTERM on Linux, TerminateProcess on Windows)
    proc.wait()
    print("Server stopped.")
