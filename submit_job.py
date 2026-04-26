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

# We use a stable PyTorch image (2.5.1) to avoid Unsloth/TorchAO incompatibility in 2.6
SPACE_IMAGE  = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
ENV_URL      = "https://aswini-kumar-data-centric-env.hf.space"
REPO_URL     = "https://huggingface.co/spaces/Aswini-Kumar/data-centric-env"

api = HfApi(token=TOKEN)

print("Submitting HF training job...")
print(f"  Docker image: {SPACE_IMAGE}")
print(f"  ENV_URL     : {ENV_URL}")
print(f"  Hardware    : a10g-large (Fast GPU)")

# Command installs git, clones your repository, and runs training
bash_cmd = f"""
apt-get update && apt-get install -y git && \\
git clone {REPO_URL} /app && cd /app && \\
pip install -q torchao==0.6.1 && \\
pip install -q 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' \\
trl>=0.15.0 datasets>=2.0.0 transformers>=4.40.0 accelerate>=0.30.0 matplotlib && \\
pip install -e . && \\
python hf_job_train.py
"""

job = api.run_job(
    image=SPACE_IMAGE,
    command=["bash", "-c", bash_cmd],
    env={
        "ENV_URL":  ENV_URL,
        "HF_TOKEN": TOKEN,
    },
    flavor="a10g-large",
)

print(f"\nJob submitted successfully!")
print(f"   Job ID  : {job.id}")
print(f"   Status  : {job.status}")
print(f"   Monitor : https://huggingface.co/settings/jobs")
print(f"\nWhen DONE (~5 hrs), run:")
print(f"  git pull hf main")
print(f"  git add plots/ logs/")
print(f'  git commit -m "Add training results"')
print(f"  git push origin main && git push hf main")
