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

def copy_checkpoint(src, dst):
    if not src.is_dir():
        raise NotADirectoryError(f"Checkpoint directory not found: {src}")
    copied = []
    excluded = {".env", "training_state.pt", "optimizer.pt", "scheduler.pt", "scaler.pt"}
    for p in src.rglob("*"):
        if not p.is_file(): continue
        rel = p.relative_to(src)
        if any(part.startswith(".") for part in rel.parts): continue
        if any(part.lower() in excluded for part in rel.parts): continue
        if p.suffix.lower() in {".log", ".tmp"}: continue
        if "checkpoint-" in str(p): continue
        target = dst/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        copied.append(str(rel))
    if not copied:
        raise ValueError(f"No eligible files found in {src}")
    names = {p.name for p in src.iterdir() if p.is_file()}
    if "adapter_config.json" in names or "adapter_model.safetensors" in names or "adapter_model.bin" in names:
        kind = "peft_adapter"
    elif "config.json" in names and any(n in names for n in ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"]):
        kind = "full_transformers_model"
    else:
        kind = "project_specific_or_unrecognized"
    return copied, kind

def main():
    ap = argparse.ArgumentParser(description="Upload Sub-track 2b generation artifacts to Hugging Face.")
    ap.add_argument("--config", type=Path, default=ROOT/"subtrack_2b/config/config.yaml")
    ap.add_argument("--checkpoint", type=Path, default=ROOT/"checkpoints/subtrack_2b/best_model")
    ap.add_argument("--repo-id", default=os.getenv("HF_REPO_ID_2B"))
    ap.add_argument("--private", action="store_true", help="Keep the Hub repository private.")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--commit-message", default="Upload RETECO Sub-track 2b artifacts")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra-files", nargs="*", type=Path, default=[])
    args = ap.parse_args()
    token = auth_token(args.dry_run)
    args.repo_id = args.repo_id or os.getenv("HF_REPO_ID_2B")
    if not args.repo_id:
        raise SystemExit("Set HF_REPO_ID_2B in .env or pass --repo-id username/repository.")
    config_path, checkpoint = args.config.resolve(), args.checkpoint.resolve()
    cfg = config_load(config_path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}. Pass the actual trained checkpoint directory.")
    with tempfile.TemporaryDirectory(prefix="reteco_2b_hf_") as td:
        stage = Path(td)
        model_dir = stage/"model"
        model_dir.mkdir()
        copied, kind = copy_checkpoint(checkpoint, model_dir)
        shutil.copy2(config_path, stage/"config.yaml")
        for rel in ["subtrack_2b/models/model.py", "subtrack_2b/dataset/dataset.py",
                    "subtrack_2b/utils/utils.py", "subtrack_2b/inference.py"]:
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
        base_model = cfg.get("model", {}).get("name", "Qwen/Qwen2.5-7B-Instruct")
        manifest = {
            "task": "SemEval-2027 Task 1 / RETECO Sub-track 2b",
            "repo_id": args.repo_id, "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_source": str(checkpoint), "checkpoint_type": kind,
            "files_copied": copied, "base_model": base_model,
            "run_tag": cfg.get("general", {}).get("run_tag"),
            "warning": "No benchmark score is asserted by this upload."
        }
        (stage/"artifact_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        if kind == "peft_adapter":
            load_note = f"This repository contains a PEFT adapter for `{base_model}`. Load the base model first, then attach the adapter with `PeftModel.from_pretrained(base_model, adapter_path)`. The adapter does not contain the base weights."
        elif kind == "full_transformers_model":
            load_note = "This repository appears to contain a full Transformers model. Test loading with the documented package versions before relying on it."
        else:
            load_note = "Checkpoint format was not recognized automatically. Use the project source and inspect files before loading."
        (stage/"README.md").write_text(f"""---
language:
- en
license: other
base_model: {base_model}
library_name: transformers
pipeline_tag: text-generation
tags:
- semeval-2027
- ret eco
- grounded-generation
- conversational-ai
- peft
- lora
- research
---

# RETECO SemEval-2027 — Sub-track 2b

Supervised fine-tuning artifacts for conversational generation grounded in
gold passages.

- **Base model:** `{base_model}`
- **Artifact type:** `{kind}`
- **Run tag:** `{cfg.get('general', {}).get('run_tag', 'not recorded')}`
- **Input token limit:** `{cfg.get('data', {}).get('max_input_tokens', 'see config.yaml')}`
- **Output token limit:** `{cfg.get('data', {}).get('max_output_tokens', 'see config.yaml')}`

## Loading

{load_note}

Use the project's inference implementation to reproduce prompt construction.
See `config.yaml` and `source/` for experiment context.

## Evaluation status

This card does not assert benchmark performance. A smoke test proves only that
the training path executes; add official dev metrics after running the full
evaluation pipeline.

## Limitations and licensing

The generator may hallucinate or omit evidence. Evaluate faithfulness,
correctness, completeness, relevance and conversational coherence. Check the
base model's license and terms before publishing weights.
""", encoding="utf-8")
        upload(stage, args.repo_id, token, args.private, args.revision,
               args.commit_message, args.dry_run)

if __name__ == "__main__":
    main()
