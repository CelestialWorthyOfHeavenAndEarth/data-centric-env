"""
Run this locally to submit the training job to HF infrastructure.
Cost: T4 small = $0.40/hr × ~5 hrs ≈ $2-3 out of your $30 credits.

Usage:
    pip install huggingface_hub
    python submit_job.py
"""
import os
from huggingface_hub import HfApi

# Get token from env var or prompt
TOKEN = os.environ.get("HF_TOKEN") or input("Enter your HF token (hf_...): ").strip()

ENV_URL   = "https://aswini-kumar-data-centric-env.hf.space"
CODE_REPO = "Aswini-Kumar/data-centric-env"

api = HfApi(token=TOKEN)

print("Submitting HF training job...")
print(f"  Code repo : {CODE_REPO}")
print(f"  Env URL   : {ENV_URL}")
print(f"  Hardware  : gpu-t4-small (~$0.40/hr)")

job = api.run_job(
    repo_id=CODE_REPO,
    repo_type="space",
    command="python hf_job_train.py",
    environment={
        "ENV_URL":  ENV_URL,
        "HF_TOKEN": TOKEN,
    },
    hardware="gpu-t4-small",
)

print(f"\nJob submitted!")
print(f"Job ID  : {job.job_id}")
print(f"Monitor : https://huggingface.co/settings/jobs/{job.job_id}")
print(f"\nWhen DONE (~5 hrs), pull results:")
print(f"  git pull hf main")
print(f"  git add plots/ logs/")
print(f'  git commit -m "Add training results"')
print(f"  git push origin main")
print(f"  git push hf main")
