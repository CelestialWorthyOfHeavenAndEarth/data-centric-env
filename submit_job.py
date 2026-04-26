"""
Submit the Data-Centric AI training job to HF infrastructure.

⚠️  RECOMMENDED: Use the Colab notebook instead — it's more reliable.
    https://colab.research.google.com/github/CelestialWorthyOfHeavenAndEarth/data-centric-env/blob/main/train_colab.ipynb

HF Jobs is provided as an alternative for automated / unattended runs.

Usage (Windows):
    set HF_TOKEN=hf_yourtoken
    python submit_job.py

Usage (Linux/Mac):
    HF_TOKEN=hf_yourtoken python submit_job.py
"""
import os, sys
from huggingface_hub import HfApi

TOKEN = os.environ.get("HF_TOKEN") or input("Enter your HF token (hf_...): ").strip()

ENV_URL  = "https://aswini-kumar-data-centric-env.hf.space"
REPO_URL = "https://huggingface.co/spaces/Aswini-Kumar/data-centric-env"

# Use official Unsloth Docker image — has torch 2.4.1 + compatible torchao pre-installed
# See: https://hub.docker.com/r/unsloth/unsloth/tags
DOCKER_IMAGE = "unsloth/unsloth:latest-torch241"

api = HfApi(token=TOKEN)

print("Submitting HF training job...")
print(f"  Docker image: {DOCKER_IMAGE}")
print(f"  ENV_URL     : {ENV_URL}")
print(f"  Hardware    : a10g-large")

# Clone repo + run training (torchao + unsloth are pre-installed in the image)
bash_cmd = f"""
apt-get update -qq && apt-get install -y -qq git && \\
git clone {REPO_URL} /app && cd /app && \\
pip install -q openenv-core[core]>=0.2.1 scikit-learn>=1.3.0 pandas>=2.0.0 numpy matplotlib && \\
pip install -e . && \\
python hf_job_train.py
"""

job = api.run_job(
    image=DOCKER_IMAGE,
    command=["bash", "-c", bash_cmd],
    env={
        "ENV_URL":  ENV_URL,
        "HF_TOKEN": TOKEN,
    },
    flavor="a10g-large",
)

print(f"\nJob submitted!")
print(f"   Job ID  : {job.id}")
print(f"   Status  : {job.status}")
print(f"   Monitor : https://huggingface.co/jobs/Aswini-Kumar/{job.id}")
print(f"\n⚡ Alternatively, use Colab for a more reliable run:")
print(f"   https://colab.research.google.com/github/CelestialWorthyOfHeavenAndEarth/data-centric-env/blob/main/train_colab.ipynb")
