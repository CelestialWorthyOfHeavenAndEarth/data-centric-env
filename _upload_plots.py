"""Upload plots directly to HF Space repo via huggingface_hub API."""
import os, sys

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Installing huggingface_hub...")
    os.system(f"{sys.executable} -m pip install huggingface_hub -q")
    from huggingface_hub import HfApi

SPACE_REPO = "Aswinis-Kumar/data-centric-env"
TOKEN = os.environ.get("HF_TOKEN", "")

if not TOKEN:
    print("ERROR: set HF_TOKEN environment variable first")
    print("  $env:HF_TOKEN='hf_...'")
    sys.exit(1)

api = HfApi(token=TOKEN)
plots_dir = "plots"

for fname in os.listdir(plots_dir):
    if fname.endswith((".png", ".jpg")):
        local_path = os.path.join(plots_dir, fname)
        repo_path  = f"plots/{fname}"
        print(f"Uploading {local_path} → {repo_path}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=SPACE_REPO,
            repo_type="space",
            commit_message=f"Add training plot: {fname}",
        )
        print(f"  ✓ {fname}")

print("\nDone. Also pushing README/BLOG/code changes via git...")
