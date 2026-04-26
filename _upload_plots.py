"""
Upload all changed text files + plot images to HF Space via huggingface_hub API.
This bypasses git entirely for the HF remote — no Xet/LFS issues.
"""
import os, sys

try:
    from huggingface_hub import HfApi
except ImportError:
    os.system(f"{sys.executable} -m pip install huggingface_hub -q")
    from huggingface_hub import HfApi

SPACE_REPO = "Aswinis-Kumar/data-centric-env"
TOKEN = os.environ.get("HF_TOKEN", "")

if not TOKEN:
    print("ERROR: set HF_TOKEN first:")
    print("  $env:HF_TOKEN = 'hf_...'")
    sys.exit(1)

api = HfApi(token=TOKEN)

# Files to upload — text/code files
TEXT_FILES = [
    "README.md", "BLOG.md", "openenv.yaml",
    "train_data_centric.py", "train_colab.ipynb",
    "eval_data_centric.py", "hf_job_train.py",
    "models.py", "client.py", "agent_utils.py",
    "plot_rewards.py", "server/grader.py",
    "server/data_centric_environment.py",
    "tests/test_grader.py", ".gitattributes",
]

# Plot images
PLOT_FILES = [f"plots/{f}" for f in os.listdir("plots") if f.endswith((".png", ".jpg"))]

all_files = TEXT_FILES + PLOT_FILES
uploaded = 0

for fpath in all_files:
    if not os.path.exists(fpath):
        print(f"  SKIP (not found): {fpath}")
        continue
    try:
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fpath,
            repo_id=SPACE_REPO,
            repo_type="space",
            commit_message=f"Update {fpath}",
        )
        print(f"  ✓ {fpath}")
        uploaded += 1
    except Exception as e:
        print(f"  ✗ {fpath}: {e}")

print(f"\nUploaded {uploaded}/{len(all_files)} files to {SPACE_REPO}")
