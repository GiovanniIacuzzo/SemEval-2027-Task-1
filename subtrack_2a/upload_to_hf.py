#!/usr/bin/env python3
"""Careful Hugging Face Hub uploader for RETECO research artifacts."""
from __future__ import annotations
import argparse, json, os, shutil, tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
    from huggingface_hub import HfApi, whoami
except ImportError as e:
    raise SystemExit("Install dependencies: pip install huggingface_hub python-dotenv pyyaml") from e
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

ROOT = Path(__file__).resolve().parent.parent

def config_load(path):
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Expected a YAML mapping in config")
    return data

def auth_token(dry_run):
    if load_dotenv:
        load_dotenv(ROOT / ".env", override=False)
        load_dotenv(override=False)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token and not dry_run:
        raise SystemExit("Missing HF_TOKEN. Set it in the repository-root .env or use `hf auth login`.")
    return token

def upload(stage, repo_id, token, private, revision, commit_message, dry_run):
    print(f"Repository: {repo_id}")
    print("Staged files:")
    for p in sorted(stage.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(stage)} ({p.stat().st_size:,} bytes)")
    if dry_run:
        print("DRY RUN: validation passed; nothing uploaded.")
        return
    try:
        identity = whoami(token=token)
    except Exception as e:
        raise RuntimeError(f"Could not validate HF token: {e}") from e
    print(f"Authenticated as: {identity.get('name', 'unknown')}")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    result = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(stage),
        revision=revision, commit_message=commit_message, token=token,
        ignore_patterns=[".env", "*.env", "**/.env", "**/*token*"]
    )
    print(f"Upload completed: {result}")
    print(f"URL: https://huggingface.co/{repo_id}")

def main():
    ap = argparse.ArgumentParser(description="Upload Sub-track 2a retrieval artifacts to Hugging Face.")
    ap.add_argument("--config", type=Path, default=ROOT/"subtrack_2a/config/config.yaml")
    ap.add_argument("--checkpoint", type=Path, default=ROOT/"checkpoints/subtrack_2a/bi_encoder/best_model.pt")
    ap.add_argument("--repo-id", default=os.getenv("HF_REPO_ID_2A"))
    ap.add_argument("--private", action="store_true", help="Keep the Hub repository private.")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--commit-message", default="Upload RETECO Sub-track 2a artifacts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra-files", nargs="*", type=Path, default=[])
    args = ap.parse_args()
    token = auth_token(args.dry_run)
    args.repo_id = args.repo_id or os.getenv("HF_REPO_ID_2A")
    if not args.repo_id:
        raise SystemExit("Set HF_REPO_ID_2A in .env or pass --repo-id username/repository.")
    config_path, checkpoint = args.config.resolve(), args.checkpoint.resolve()
    cfg = config_load(config_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Pass the trained checkpoint path.")
    with tempfile.TemporaryDirectory(prefix="reteco_2a_hf_") as td:
        stage = Path(td)
        (stage/"checkpoint").mkdir()
        shutil.copy2(checkpoint, stage/"checkpoint"/checkpoint.name)
        shutil.copy2(config_path, stage/"config.yaml")
        for rel in ["subtrack_2a/models/model.py", "subtrack_2a/dataset/dataset.py",
                    "subtrack_2a/utils/utils.py", "subtrack_2a/inference.py"]:
            src = ROOT/rel
            if src.is_file():
                dst = stage/"source"/src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        for extra in args.extra_files:
            extra = extra.resolve()
            if not extra.is_file(): raise FileNotFoundError(f"Extra file not found: {extra}")
            dst = stage/"evaluation"/extra.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extra, dst)
        manifest = {
            "task": "SemEval-2027 Task 1 / RETECO Sub-track 2a",
            "repo_id": args.repo_id, "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint.name, "run_tag": cfg.get("general", {}).get("run_tag"),
            "artifact_type": "project-specific PyTorch checkpoint bundle",
            "warning": "Not guaranteed to load with transformers.AutoModel.from_pretrained()."
        }
        (stage/"artifact_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        (stage/"README.md").write_text(f"""---
language:
- en
license: other
library_name: pytorch
tags:
- semeval-2027
- ret eco
- information-retrieval
- dense-retrieval
- research
---

# RETECO SemEval-2027 — Sub-track 2a

Project-specific checkpoint bundle for conversational passage retrieval.

- **Run tag:** `{cfg.get('general', {}).get('run_tag', 'not recorded')}`
- **Checkpoint:** `checkpoint/{checkpoint.name}`
- **Configuration:** `config.yaml`

## Important loading note

This is a custom PyTorch checkpoint bundle, not necessarily a standalone
Transformers model. Use the project's `source/` files and inference pipeline,
and ensure the model architecture matches the checkpoint before loading.

## Evaluation

No benchmark score is asserted by this upload. Add metrics only after validating
the exact split and evaluation protocol. Keep local diagnostics separate from
official dev results.

## Intended use and limitations

Research use only. Results depend on domain, corpus, preprocessing, retrieval
settings and evaluation protocol. Review the source model and data licenses
before redistributing or publishing artifacts.
""", encoding="utf-8")
        upload(stage, args.repo_id, token, args.private, args.revision,
               args.commit_message, args.dry_run)

if __name__ == "__main__":
    main()
