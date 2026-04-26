"""
Submit the Data-Centric AI training job to HF infrastructure.

Cost: T4 small = $0.40/hr × ~5 hrs ≈ $2-3 from your $30 credits.

Usage:
    set HF_TOKEN=hf_yourtoken   (Windows)
    python submit_job.py
"""
import os, sys
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN") or input("Enter your HF token (hf_...): ").strip()

# The Docker image built by your HF Space
SPACE_IMAGE  = "registry.hf.space/aswini-kumar/data-centric-env:latest"
ENV_URL      = "https://aswini-kumar-data-centric-env.hf.space"

api = HfApi(token=TOKEN)

print("Submitting HF training job...")
print(f"  Docker image: {SPACE_IMAGE}")
print(f"  ENV_URL     : {ENV_URL}")
print(f"  Hardware    : gpu-t4-small (~$0.40/hr)")

job = api.run_job(
    image=SPACE_IMAGE,
    command=[
        "bash", "-c",
        # Install GPU deps first (not in base image), then train
        "pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' "
        "trl>=0.15.0 datasets>=2.0.0 transformers>=4.40.0 accelerate>=0.30.0 matplotlib && "
        "python hf_job_train.py"
    ],
    env={
        "ENV_URL":  ENV_URL,
        "HF_TOKEN": TOKEN,
    },
    flavor="t4-small",
)

print(f"\n✅ Job submitted!")
print(f"   Job ID  : {job.id}")
print(f"   Status  : {job.status}")
print(f"   Monitor : https://huggingface.co/settings/jobs")
print(f"\nWhen DONE (~5 hrs), run:")
print(f"  git pull hf main")
print(f"  git add plots/ logs/")
print(f'  git commit -m "Add training results"')
print(f"  git push origin main && git push hf main")
