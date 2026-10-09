#!/usr/bin/env python3
"""Export and upload only model/tokenizer assets for RETECO Sub-track 2b.

Create the model repository on Hugging Face first and write README.md manually
there. This script never creates, generates, replaces, or deletes the Hub README.
It uploads only model configuration, weights/adapter files, and tokenizer assets.
Use --dry-run to build and inspect a local bundle without uploading.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "subtrack_2b" / "config" / "config.yaml"
DEFAULT_CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "subtrack_2b"
DEFAULT_OUTPUT = ROOT_DIR / "hf_release" / "subtrack_2b"

HF_ARTIFACT_NAMES = {
    "config.json", "generation_config.json", "model.safetensors", "pytorch_model.bin",
    "model.safetensors.index.json", "pytorch_model.bin.index.json",
    "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.txt", "vocab.json", "merges.txt", "spiece.model", "sentencepiece.bpe.model",
    "tokenizer.model", "chat_template.jinja",
}


def log(message: str) -> None:
    print(f"[RETECO-2b-HF] {message}", flush=True)


def resolve_path(path: str | Path, root: Path = ROOT_DIR) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log(f"Attenzione: JSON non leggibile, ignorato: {path.name}")
        return None


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        log(f"Config YAML non trovato; userò i metadati del checkpoint: {path}")
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Dipendenza mancante: installa PyYAML con `pip install pyyaml`.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"La config deve contenere una mappa YAML: {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def is_local_absolute_model_ref(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    return Path(text).is_absolute() or bool(re.match(r"^[A-Za-z]:[\\/]", text)) or text.startswith("\\\\")


def is_shard(name: str) -> bool:
    return bool(re.fullmatch(r"(?:model|pytorch_model)-\d{5}-of-\d{5}\.(?:safetensors|bin)", name))


def is_hf_artifact(name: str) -> bool:
    return name in HF_ARTIFACT_NAMES or is_shard(name)


def copy_hf_artifacts(source: Path, destination: Path, *, only_missing: bool = False) -> list[str]:
    if not source.is_dir():
        return []
    copied: list[str] = []
    for item in sorted(source.iterdir()):
        if not item.is_file() or not is_hf_artifact(item.name):
            continue
        target = destination / item.name
        if only_missing and target.exists():
            continue
        shutil.copy2(item, target)
        copied.append(item.name)
        log(f"Copiato: {item.name} (da {source.name})")
    return copied


def get_base_model_from_config(cfg: dict[str, Any]) -> str | None:
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    value = model_cfg.get("name") or model_cfg.get("model_name_or_path") or model_cfg.get("base_model_name_or_path")
    return str(value) if value else None


def get_base_model_from_adapter(adapter_cfg: Any) -> str | None:
    if isinstance(adapter_cfg, dict):
        value = adapter_cfg.get("base_model_name_or_path")
        return str(value) if value else None
    return None


def validate_model_bundle(out: Path) -> str:
    adapter = (out / "adapter_config.json").is_file() and any(
        (out / n).is_file() for n in ("adapter_model.safetensors", "adapter_model.bin")
    )
    full = (out / "config.json").is_file() and any(
        (out / n).is_file() or any(out.glob(n))
        for n in ("model.safetensors", "pytorch_model.bin", "model-*.safetensors", "pytorch_model-*.bin")
    )
    if adapter:
        return "lora_adapter"
    if full:
        return "full_model"
    raise RuntimeError(
        "Checkpoint non valido: sono richiesti adapter_config.json + adapter weights, "
        "oppure config.json + model weights. Non carico cartelle incomplete o solo training-state."
    )



UPLOAD_PATTERNS = [
    # Model configuration and weights (including sharded checkpoints).
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "model-*.safetensors",
    "pytorch_model-*.bin",
    # PEFT / LoRA adapter.
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    # Tokenizer assets needed for inference.
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "spiece.model",
    "sentencepiece.bpe.model",
    "tokenizer.model",
    "chat_template.jinja",
]

# Defensive exclusions: this script must never replace hand-written Hub docs.
IGNORE_PATTERNS = [
    "README.md",
    "**/README.md",
    "model_metadata.json",
    "training_recipe.json",
    "artifact_manifest.json",
    ".gitattributes",
]



def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--repo-id", help="Hugging Face repository, e.g. username/reteco-semeval2027-subtrack-2b")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config YAML used by train.py")
    p.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoint root used by train.py; defaults to paths.checkpoint_dir in config")
    p.add_argument("--model-dir", type=Path, default=None, help="Selected checkpoint directory; default: <checkpoint-dir>/best_model")
    p.add_argument("--tokenizer-dir", type=Path, default=None, help="Tokenizer directory; default: <checkpoint-dir>/tokenizer")
    p.add_argument("--base-model-id", default=None, help="Public HF ID for the exact base model, only needed when adapter metadata contains a local filesystem path")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Local release bundle directory")
    p.add_argument("--revision", default="main", help="Hub branch/revision")
    p.add_argument("--token", default=None, help="HF token; otherwise uses HF_TOKEN or cached login")
    p.add_argument("--dry-run", action="store_true", help="Build and validate bundle without uploading")
    p.add_argument("--overwrite-output", action="store_true", help="Replace an existing output directory")
    p.add_argument("--allow-base-model-mismatch", action="store_true", help="Override a mismatch between checkpoint adapter and the current YAML config")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.dry_run and not args.repo_id:
        raise SystemExit("Errore: specifica --repo-id oppure usa --dry-run.")
    repo_id = args.repo_id or "YOUR_USERNAME/reteco-semeval2027-subtrack-2b"
    if args.repo_id and ("/" not in args.repo_id or args.repo_id.startswith("/") or args.repo_id.endswith("/")):
        raise SystemExit("--repo-id deve avere il formato username/nome-repository.")

    config_path = resolve_path(args.config)
    cfg = read_yaml(config_path)
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    configured_ckpt = paths_cfg.get("checkpoint_dir", "checkpoints/subtrack_2b")
    checkpoint_dir = resolve_path(args.checkpoint_dir or configured_ckpt)
    model_dir = resolve_path(args.model_dir) if args.model_dir else checkpoint_dir / "best_model"
    tokenizer_dir = resolve_path(args.tokenizer_dir) if args.tokenizer_dir else checkpoint_dir / "tokenizer"

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Best checkpoint non trovato: {model_dir}. Esegui il training e verifica paths.checkpoint_dir."
        )

    out = resolve_path(args.output_dir)
    if out.exists():
        if not args.overwrite_output:
            raise SystemExit(f"La directory di output esiste già: {out}\nUsa --overwrite-output per sostituirla.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=False)

    copied_model = copy_hf_artifacts(model_dir, out)
    if not copied_model:
        raise RuntimeError(
            f"Nessun artifact HF riconosciuto in {model_dir}. "
            "Il checkpoint deve essere salvato tramite save_pretrained()/PEFT, non solo come training_state.pt."
        )

    # Copy tokenizer/config artifacts saved by train.py, without overwriting files
    # already provided by the selected checkpoint.
    copy_hf_artifacts(tokenizer_dir, out, only_missing=True)
    adapter_config_path = out / "adapter_config.json"
    adapter_cfg = read_json(adapter_config_path)
    selected_base = get_base_model_from_adapter(adapter_cfg)
    if args.base_model_id:
        selected_base = args.base_model_id
        if isinstance(adapter_cfg, dict):
            adapter_cfg["base_model_name_or_path"] = args.base_model_id
            write_json(adapter_config_path, adapter_cfg)
    elif is_local_absolute_model_ref(selected_base):
        raise RuntimeError(
            f"L'adapter fa riferimento a un percorso locale ({selected_base}). "
            "Per evitare di pubblicare un path non portabile, specifica --base-model-id ID_HF_DELLO_STESSO_MODELLO."
        )
    yaml_base = get_base_model_from_config(cfg)
    if selected_base and yaml_base and selected_base.rstrip("/") != yaml_base.rstrip("/"):
        message = (
            f"Il checkpoint indica base model '{selected_base}', mentre la config corrente indica '{yaml_base}'. "
            "È possibile che best_model appartenga a una run precedente (ad esempio uno smoke test)."
        )
        if not args.allow_base_model_mismatch:
            raise RuntimeError(message + " Verifica i file o passa --allow-base-model-mismatch solo dopo averli controllati.")
        log("ATTENZIONE: mismatch forzato. Le impostazioni di training della config corrente non saranno presentate come certamente associate al checkpoint.")

    full_cfg_path = out / "config.json"
    full_cfg = read_json(full_cfg_path)
    base_model = selected_base or yaml_base
    if not base_model and isinstance(full_cfg, dict):
        full_cfg_base = full_cfg.get("_name_or_path")
        if not is_local_absolute_model_ref(full_cfg_base):
            base_model = full_cfg_base
    if not (out / "adapter_config.json").is_file() and isinstance(full_cfg, dict):
        stored_name = full_cfg.get("_name_or_path")
        if args.base_model_id:
            base_model = args.base_model_id
            full_cfg["_name_or_path"] = args.base_model_id
            write_json(full_cfg_path, full_cfg)
        elif is_local_absolute_model_ref(stored_name):
            # Complete model weights are included; do not expose the original
            # machine-local model path as a public config field.
            full_cfg["_name_or_path"] = repo_id
            write_json(full_cfg_path, full_cfg)
            if is_local_absolute_model_ref(base_model):
                base_model = None
    if is_local_absolute_model_ref(base_model):
        base_model = None

    has_tokenizer = any((out / name).is_file() for name in (
        "tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json", "spiece.model", "tokenizer.model"
    ))
    if not has_tokenizer:
        if not base_model:
            raise RuntimeError("Tokenizer non trovato e base model non determinabile dai metadati/config.")
        try:
            from transformers import AutoTokenizer
            model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            )
            tokenizer.save_pretrained(str(out))
            log(f"Tokenizer recuperato dal modello base {base_model}.")
        except Exception as exc:
            raise RuntimeError(f"Impossibile recuperare il tokenizer da {base_model}: {exc}") from exc

    export_type = validate_model_bundle(out)

    log(f"Bundle validato: {out}")
    log(f"Tipo export: {export_type}")
    log("File di modello/tokenizer candidati all'upload (README escluso):")
    for p in sorted(out.iterdir()):
        if p.is_file():
            log(f"  {p.name} ({p.stat().st_size:,} bytes)")
    if args.dry_run:
        log("Dry-run: upload non eseguito. Il README su Hugging Face non viene generato né modificato.")
        return 0

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Dipendenza mancante: `pip install -U huggingface_hub`.") from exc
    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=repo_id, repo_type="model", revision=args.revision)
    except Exception as exc:
        raise RuntimeError(
            f"Repository HF non trovato o non accessibile: {repo_id}. "
            "Crealo prima manualmente su huggingface.co e verifica i permessi del token."
        ) from exc
    api.upload_folder(
        folder_path=str(out),
        repo_id=repo_id,
        repo_type="model",
        revision=args.revision,
        allow_patterns=UPLOAD_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Upload RETECO SemEval-2027 Sub-track 2b model artifacts",
    )
    log(f"Upload completato: https://huggingface.co/{repo_id}")
    log("README.md e altri file di documentazione esistenti sul repository non sono stati caricati né modificati.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Operazione interrotta dall'utente.")
