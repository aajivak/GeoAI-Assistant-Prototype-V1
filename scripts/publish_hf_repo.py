"""Helper script to publish the model card and configuration to Hugging Face Hub.

Fulfills Deliverable #2:
A public repository on Hugging Face Hub with model card and open license.
"""
import os
import sys
from pathlib import Path


def publish_repo(repo_id: str, hf_token: str = None):
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return

    api = HfApi(token=hf_token)
    print(f"Creating/verifying Hugging Face repository: {repo_id}...")
    try:
        url = create_repo(repo_id=repo_id, repo_type="model", token=hf_token, exist_ok=True)
        print(f"Repository URL: {url}")
    except Exception as e:
        print(f"Repository check: {e}")

    model_card_path = Path("docs/hf_model_card.md")
    if model_card_path.exists():
        print("Uploading README.md (Model Card)...")
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=hf_token,
        )
        print("Successfully published Model Card to Hugging Face Hub!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python publish_hf_repo.py <hf_username/repo_name> [hf_token]")
        sys.exit(1)
    repo = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("HF_TOKEN")
    publish_repo(repo, token)
