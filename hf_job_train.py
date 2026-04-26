#!/usr/bin/env python3
"""
HF Jobs training script for Data-Centric AI Agent.

Submit via:
    python submit_job.py

Required secrets in HF Job:
    HF_TOKEN  - HuggingFace write token
    ENV_URL   - HF Space URL (set automatically by submit_job.py)
"""
import os, sys, time, glob, subprocess, requests

ENV_URL   = os.environ.get("ENV_URL", "https://aswini-kumar-data-centric-env.hf.space")
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
SPACE_REPO = "Aswinis-Kumar/data-centric-env"

print(f"[Job] ENV_URL   : {ENV_URL}")
print(f"[Job] WorkingDir: {os.getcwd()}")
print(f"[Job] Python    : {sys.version}")

# ── 1. Install dependencies ────────────────────────────────────────────────────
print("\n[Job] Installing dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torchao==0.6.1",
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    "trl>=0.15.0", "datasets>=2.0.0", "transformers>=4.40.0",
    "accelerate>=0.30.0", "openenv-core[core]>=0.2.1",
    "scikit-learn>=1.3.0", "pandas>=2.0.0", "numpy>=1.24.0",
    "matplotlib", "huggingface_hub",
])

# ── 2. Verify HF Space is healthy ─────────────────────────────────────────────
print(f"\n[Job] Waiting for Space at {ENV_URL} ...")
for attempt in range(18):          # 3 min max
    try:
        r = requests.get(f"{ENV_URL}/health", timeout=10)
        if r.status_code == 200:
            print(f"[Job] Space healthy: {r.json()}")
            break
    except Exception as e:
        print(f"[Job] Attempt {attempt+1}/18: {e}")
    time.sleep(10)
else:
    raise RuntimeError("Space not responding after 3 minutes — is it Running?")

os.environ["ENV_URL"] = ENV_URL

# ── 3. Generate SFT data if missing ───────────────────────────────────────────
if not os.path.exists("sft_data.jsonl"):
    print("\n[Job] Generating SFT warmup data...")
    subprocess.check_call([sys.executable, "sft_generator.py"])

count = sum(1 for _ in open("sft_data.jsonl"))
print(f"[Job] SFT data: {count} examples")

# ── 4. Train ──────────────────────────────────────────────────────────────────
from train_data_centric import load_model, run_sft_warmup, run_grpo_training, save_model

print("\n[Job] Loading Qwen2.5-3B-Instruct (4-bit)...")
model, tokenizer = load_model()

print("\n[Job] Phase 1 — SFT warmup...")
model = run_sft_warmup(model, tokenizer)

print("\n[Job] Phase 2 — GRPO training (env rollouts via Space)...")
checkpoints = sorted(glob.glob("./data-centric-checkpoints/checkpoint-*"))
resume = checkpoints[-1] if checkpoints else None
if resume:
    print(f"[Job] Resuming from {resume}")
model = run_grpo_training(model, tokenizer, resume_from_checkpoint=resume)

print("\n[Job] Saving model...")
save_model(model, tokenizer)

# ── 5. Generate plots ─────────────────────────────────────────────────────────
print("\n[Job] Generating reward curves...")
subprocess.check_call([
    sys.executable, "plot_rewards.py",
    "--log", "logs/training.jsonl", "--out", "plots/",
])

# ── 6. Push results back to HF Space repo ─────────────────────────────────────
if not HF_TOKEN:
    print("\n[Job] No HF_TOKEN — skipping upload. Download outputs manually.")
    sys.exit(0)

print("\n[Job] Pushing plots + log to HF Space repo...")
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)

files_to_push = glob.glob("plots/*.png") + ["logs/training.jsonl"]
for fpath in files_to_push:
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fpath,
            repo_id=SPACE_REPO,
            repo_type="space",
            commit_message="training results: plots + log [HF Job]",
        )
        print(f"[Job] Uploaded: {fpath}")

print("\n[Job] Pushing trained adapter to HF Hub...")
api.create_repo("data-centric-ai-agent", repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path="./data-centric-adapter",
    repo_id=f"{api.whoami()['name']}/data-centric-ai-agent",
    repo_type="model",
    commit_message="Data-Centric AI Agent — GRPO trained",
)

print("""
[Job] ✅ DONE

Plots are now in the HF Space repo → Files tab.
Next step (local):
    git pull hf main
    git add plots/ logs/
    git commit -m "Add training results"
    git push origin main
""")
