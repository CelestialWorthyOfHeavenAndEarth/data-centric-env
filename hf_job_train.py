#!/usr/bin/env python3
"""
HF Job training script for Data-Centric AI Agent.

Run this as an HF Job pointing to the deployed HF Space as the environment:
    hf job run --gpu t4-small --env ENV_URL=https://aswini-kumar-data-centric-env.hf.space \
        python hf_job_train.py

Or via Python API:
    from huggingface_hub import HfApi
    api = HfApi()
    api.run_job(...)

The HF Space (server) must be running BEFORE submitting this job.
"""

import os
import sys
import time
import requests

# ── Verify the HF Space is reachable before starting training ─────────────────

ENV_URL = os.environ.get("ENV_URL", "https://aswini-kumar-data-centric-env.hf.space")
print(f"[HF Job] Environment URL: {ENV_URL}")

print("[HF Job] Checking environment server health...")
for attempt in range(12):
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=10)
        if r.status_code == 200:
            print(f"[HF Job] Server healthy: {r.json()}")
            break
    except Exception as e:
        print(f"[HF Job] Attempt {attempt+1}/12: {e}")
        time.sleep(10)
else:
    raise RuntimeError(
        f"HF Space at {ENV_URL} is not responding after 2 minutes.\n"
        "Make sure the Space is Running before submitting this job."
    )

# ── Install dependencies ───────────────────────────────────────────────────────

print("[HF Job] Installing dependencies...")
os.system(
    'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" '
    'trl datasets transformers accelerate scikit-learn pandas numpy matplotlib '
    '"openenv-core[core]>=0.2.1" --quiet'
)

# ── Pull latest environment code ──────────────────────────────────────────────

REPO = "https://github.com/CelestialWorthyOfHeavenAndEarth/data-centric-env.git"
WORK_DIR = "/tmp/data-centric-env"

if not os.path.exists(f"{WORK_DIR}/pyproject.toml"):
    os.system(f"git clone {REPO} {WORK_DIR}")
else:
    os.system(f"git -C {WORK_DIR} pull origin main")

os.chdir(WORK_DIR)
sys.path.insert(0, WORK_DIR)
os.system("pip install -e . --quiet")
print(f"[HF Job] Working dir: {os.getcwd()}")
os.system("git log --oneline -3")

# ── Set ENV_URL so train_data_centric.py uses the HF Space ───────────────────

os.environ["ENV_URL"] = ENV_URL
print(f"[HF Job] ENV_URL = {ENV_URL}")

# ── Generate SFT data ─────────────────────────────────────────────────────────

if not os.path.exists("sft_data.jsonl"):
    print("[HF Job] Generating SFT data...")
    os.system("python sft_generator.py")
else:
    count = sum(1 for _ in open("sft_data.jsonl"))
    print(f"[HF Job] SFT data exists: {count} examples")

# ── Run full training pipeline ────────────────────────────────────────────────

from train_data_centric import load_model, run_sft_warmup, run_grpo_training, save_model
import glob

print("[HF Job] Loading model...")
model, tokenizer = load_model()

print("[HF Job] Phase 1: SFT warmup...")
model = run_sft_warmup(model, tokenizer)
print("[HF Job] SFT complete")

print("[HF Job] Phase 2: GRPO training (connecting to HF Space)...")
resume_from = None
ckpt_dir = "./data-centric-checkpoints"
if os.path.exists(ckpt_dir):
    checkpoints = sorted(glob.glob(f"{ckpt_dir}/checkpoint-*"))
    if checkpoints:
        resume_from = checkpoints[-1]
        print(f"[HF Job] Resuming from: {resume_from}")

model = run_grpo_training(model, tokenizer, resume_from_checkpoint=resume_from)
print("[HF Job] GRPO complete")

print("[HF Job] Saving model...")
save_model(model, tokenizer)

print("[HF Job] Generating reward plots...")
os.system("python plot_rewards.py --log logs/training.jsonl --out plots/")

print("[HF Job] Done! Results in ./logs/ and ./plots/")
