"""Publish the GeoAI model card, integration metadata, and loader example.

Authenticate first with: hf auth login
Never pass access tokens as command-line arguments.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main(repo_id: str) -> None:
    api = HfApi()
    root = Path(__file__).resolve().parents[1]
    card = root / "docs" / "hf_model_card.md"
    config = root / "docs" / "geoai_config.json"
    loader = root / "scripts" / "load_base_model.py"
    for file in (card, config, loader):
        if not file.is_file():
            raise FileNotFoundError(f"Missing publication file: {file}")

    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    for local_file, remote_name in (
        (card, "README.md"),
        (config, "geoai_config.json"),
        (loader, "load_base_model.py"),
    ):
        print(f"Uploading {remote_name}...")
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Publish {remote_name}",
        )
    print(f"Published: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/publish_hf_repo.py OWNER/REPO")
    main(sys.argv[1])
