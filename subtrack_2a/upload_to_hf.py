#!/usr/bin/env python3
"""Export and upload only model/tokenizer assets for RETECO Sub-track 2a.

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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "subtrack_2a" / "config" / "config.yaml"
DEFAULT_OUTPUT = ROOT_DIR / "hf_release" / "subtrack_2a"

HF_ARTIFACT_NAMES = {
    "config.json", "generation_config.json", "model.safetensors",
    "pytorch_model.bin", "model.safetensors.index.json",
    "pytorch_model.bin.index.json", "adapter_config.json",
    "adapter_model.safetensors", "adapter_model.bin", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.txt", "vocab.json", "merges.txt", "spiece.model",
    "sentencepiece.bpe.model", "tokenizer.model", "chat_template.jinja",
}


def log(message: str) -> None:
    print(f"[RETECO-2a-HF] {message}", flush=True)


def resolve_path(path: str | Path, root: Path = ROOT_DIR) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        log(f"Config YAML non trovato; userò i default per l'export: {path}")
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Dipendenza mancante: installa PyYAML con `pip install pyyaml`.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"La config deve contenere una mappa YAML: {path}")
    return data


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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


def copy_hf_artifacts(source: Path, destination: Path) -> list[str]:
    if not source.is_dir():
        raise FileNotFoundError(f"Directory modello non trovata: {source}")
    copied: list[str] = []
    for item in sorted(source.iterdir()):
        if item.is_file() and is_hf_artifact(item.name):
            shutil.copy2(item, destination / item.name)
            copied.append(item.name)
            log(f"Copiato: {item.name}")
    return copied


def load_torch_checkpoint(path: Path) -> dict[str, Any]:
    import torch
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older PyTorch releases.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError(
            f"Checkpoint non compatibile: {path}. Atteso un dizionario con chiave 'model_state_dict'."
        )
    return payload


def nested_config_value(config: dict[str, Any], section: str, key: str, default: Any) -> Any:
    value = config.get(section, {}) if isinstance(config, dict) else {}
    return value.get(key, default) if isinstance(value, dict) else default


def make_export_from_pt(checkpoint: Path, fallback_config: dict[str, Any], out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild the training wrapper and export its HF backbone or PEFT adapter."""
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Per esportare best_model.pt servono torch e transformers. "
            "Installa le dipendenze del progetto prima di eseguire l'upload."
        ) from exc

    # Import project code from the subtrack directory, independent of cwd.
    subtrack_dir = ROOT_DIR / "subtrack_2a"
    if str(subtrack_dir) not in sys.path:
        sys.path.insert(0, str(subtrack_dir))
    try:
        from models.model import ConversationalBiEncoder
    except Exception as exc:
        raise RuntimeError(
            "Non riesco a importare subtrack_2a/models/model.py. "
            "Esegui lo script dalla repository RETECO completa e verifica il file del modello."
        ) from exc

    payload = load_torch_checkpoint(checkpoint)
    cfg = payload.get("config") if isinstance(payload.get("config"), dict) else fallback_config
    bi_cfg = cfg.get("bi_encoder", {})
    lora_cfg = cfg.get("lora", {})
    data_cfg = cfg.get("data", {})
    domains_cfg = cfg.get("domains", {})
    base_model = str(bi_cfg.get("model_name_or_path", "BAAI/bge-base-en-v1.5"))
    pooling = str(bi_cfg.get("pooling_strategy", "cls"))
    normalize = bool(bi_cfg.get("normalize_embeddings", True))
    temperature = float(bi_cfg.get("temperature", 0.05))

    log(f"Ricostruzione del bi-encoder da checkpoint: {checkpoint}")
    model = ConversationalBiEncoder(
        model_name_or_path=base_model,
        temperature=temperature,
        normalize_embeddings=normalize,
        pooling_strategy=pooling,
        lora_cfg=lora_cfg,
        gradient_checkpointing=False,
        negative_chunk_size=int(cfg.get("training", {}).get("negative_chunk_size", 8))
        if isinstance(cfg.get("training", {}), dict) else 8,
    )
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        # Avoid uploading a subtly wrong model after a config/code mismatch.
        details = (
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        raise RuntimeError(
            "I pesi del checkpoint non corrispondono esattamente all'architettura ricostruita "
            f"({details}). Verifica che config.yaml e models/model.py siano quelli usati nel training."
        )

    model.eval()
    encoder = model.encoder
    if not hasattr(encoder, "save_pretrained"):
        raise RuntimeError("L'encoder non supporta save_pretrained(); export Hugging Face interrotto.")
    encoder.save_pretrained(str(out), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(out))

    try:
        from peft import PeftModel
        is_adapter = isinstance(encoder, PeftModel) or hasattr(encoder, "peft_config")
    except ImportError:
        is_adapter = hasattr(encoder, "peft_config")

    validation = payload.get("val_metrics", {})
    macro = validation.get("macro_average", {}) if isinstance(validation, dict) else {}
    metrics: dict[str, Any] = {}
    for key in ("nDCG@10", "Recall@10", "Recall@50", "MRR"):
        if isinstance(macro, dict) and key in macro:
            metrics[key] = macro[key]
    if payload.get("best_score") is not None:
        metrics["best_checkpoint_score"] = payload["best_score"]
    model_meta = {
        "task": "conversational dense passage retrieval",
        "subtrack": "SemEval 2027 Task 1, Sub-track 2a",
        "export_type": "lora_adapter" if is_adapter else "full_encoder",
        "base_model": base_model,
        "pooling_strategy": pooling,
        "normalize_embeddings": normalize,
        "temperature": temperature,
        "max_query_length": int(data_cfg.get("max_query_length", 256)),
        "max_doc_length": int(data_cfg.get("max_doc_length", 256)),
        "query_strategy": str(data_cfg.get("query_strategy", "history")),
        "query_instruction": bi_cfg.get("query_instruction", {}),
        "active_domains": domains_cfg.get("active_domains", "all"),
        "lora_enabled": bool(lora_cfg.get("enabled", False)),
        "best_epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "internal_validation_metrics": metrics,
        "checkpoint_filename": checkpoint.name,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return model_meta, cfg


def model_is_valid(directory: Path) -> tuple[bool, str]:
    adapter = (directory / "adapter_config.json").is_file() and any(
        (directory / n).is_file() for n in ("adapter_model.safetensors", "adapter_model.bin")
    )
    full = (directory / "config.json").is_file() and any(
        (directory / n).is_file() or any(directory.glob(n))
        for n in ("model.safetensors", "pytorch_model.bin", "model-*.safetensors", "pytorch_model-*.bin")
    )
    if adapter:
        return True, "lora_adapter"
    if full:
        return True, "full_encoder"
    return False, "La cartella deve contenere un adapter PEFT completo oppure config.json e pesi del modello."



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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo-id", help="Hugging Face model repository, e.g. username/reteco-subtrack-2a")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Training config YAML")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoint directory; defaults to paths.checkpoint_dir in config")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Explicit best_model.pt path")
    parser.add_argument("--model-dir", type=Path, default=None, help="Existing HF export directory (adapter or full encoder); takes precedence over .pt")
    parser.add_argument("--base-model-id", default=None, help="Public HF ID for the exact base model, only needed when checkpoint metadata contains a local filesystem path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Local bundle directory")
    parser.add_argument("--revision", default="main", help="Hub branch/revision to upload")
    parser.add_argument("--token", default=None, help="HF token; otherwise uses HF_TOKEN or cached login")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate the local bundle without uploading")
    parser.add_argument("--overwrite-output", action="store_true", help="Replace an existing output directory")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.dry_run and not args.repo_id:
        raise SystemExit("Errore: specifica --repo-id oppure usa --dry-run.")
    repo_id = args.repo_id or "YOUR_USERNAME/reteco-semeval2027-subtrack-2a"
    if args.repo_id and ("/" not in args.repo_id or args.repo_id.startswith("/") or args.repo_id.endswith("/")):
        raise SystemExit("--repo-id deve avere il formato username/nome-repository.")

    config_path = resolve_path(args.config)
    cfg = read_yaml(config_path)
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    default_ckpt = paths_cfg.get("checkpoint_dir", "checkpoints/subtrack_2a/bi_encoder")
    checkpoint_dir = resolve_path(args.checkpoint_dir or default_ckpt)
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint else None
    if checkpoint is None:
        candidates = [
            checkpoint_dir / "best_model.pt",
            checkpoint_dir.parent / "best_model.pt",
            ROOT_DIR / "checkpoints" / "subtrack_2a" / "best_model.pt",
        ]
        checkpoint = next((p for p in candidates if p.is_file()), None)

    model_dir = resolve_path(args.model_dir) if args.model_dir else None
    if model_dir is None and (checkpoint is None or not checkpoint.is_file()):
        export_candidates = [checkpoint_dir / "best_hf_model", checkpoint_dir.parent / "best_hf_model",
                             ROOT_DIR / "checkpoints" / "subtrack_2a" / "best_hf_model"]
        model_dir = next((p for p in export_candidates if p.is_dir()), None)

    out = resolve_path(args.output_dir)
    if out.exists():
        if not args.overwrite_output:
            raise SystemExit(f"La directory di output esiste già: {out}\nUsa --overwrite-output per sostituirla.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=False)

    source_meta: dict[str, Any] = {}
    resolved_cfg = cfg
    if model_dir is not None:
        log(f"Uso export Hugging Face esistente: {model_dir}")
        copied = copy_hf_artifacts(model_dir, out)
        old_meta = read_json(model_dir / "model_metadata.json")
        if isinstance(old_meta, dict):
            source_meta.update(old_meta)
        adapter_cfg = read_json(out / "adapter_config.json")
        full_cfg = read_json(out / "config.json")
        base_model = args.base_model_id or (adapter_cfg or {}).get("base_model_name_or_path") or source_meta.get("base_model")
        if not base_model and isinstance(full_cfg, dict):
            base_model = full_cfg.get("_name_or_path")
        if not base_model:
            base_model = nested_config_value(cfg, "bi_encoder", "model_name_or_path", "BAAI/bge-base-en-v1.5")
        export_type = "lora_adapter" if (out / "adapter_config.json").exists() else "full_encoder"
        meta = {
            **source_meta,
            "task": "conversational dense passage retrieval",
            "subtrack": "SemEval 2027 Task 1, Sub-track 2a",
            "export_type": export_type,
            "base_model": base_model,
            "pooling_strategy": source_meta.get("pooling_strategy", nested_config_value(cfg, "bi_encoder", "pooling_strategy", "cls")),
            "normalize_embeddings": source_meta.get("normalize_embeddings", nested_config_value(cfg, "bi_encoder", "normalize_embeddings", True)),
            "temperature": source_meta.get("temperature", nested_config_value(cfg, "bi_encoder", "temperature", 0.05)),
            "max_query_length": source_meta.get("max_query_length", nested_config_value(cfg, "data", "max_query_length", 256)),
            "max_doc_length": source_meta.get("max_doc_length", nested_config_value(cfg, "data", "max_doc_length", 256)),
            "query_strategy": source_meta.get("query_strategy", nested_config_value(cfg, "data", "query_strategy", "history")),
            "active_domains": source_meta.get("active_domains", nested_config_value(cfg, "domains", "active_domains", "all")),
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if not copied:
            raise RuntimeError(f"Nessun artifact Hugging Face riconosciuto nella cartella: {model_dir}")
        if not any((out / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json", "spiece.model", "tokenizer.model")):
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(base_model))
                tokenizer.save_pretrained(str(out))
                log("Tokenizer salvato a partire dal modello base.")
            except Exception as exc:
                raise RuntimeError(f"Tokenizer assente nell'export e impossibile scaricarlo dal modello base {base_model}: {exc}") from exc
    elif checkpoint is not None and checkpoint.is_file():
        meta, resolved_cfg = make_export_from_pt(checkpoint, cfg, out)
        export_type = meta["export_type"]
    else:
        raise FileNotFoundError(
            "Nessun checkpoint disponibile. Attesi best_model.pt oppure un export HF "
            "in best_hf_model/. Specifica --checkpoint o --model-dir."
        )

    if model_dir is not None:
        export_type = meta["export_type"]

    # Do not publish local filesystem paths from PEFT/config metadata. An adapter
    # needs a resolvable base-model ID; full-model exports do not.
    adapter_config_path = out / "adapter_config.json"
    adapter_json = read_json(adapter_config_path)
    if isinstance(adapter_json, dict):
        stored_base = adapter_json.get("base_model_name_or_path")
        if args.base_model_id:
            adapter_json["base_model_name_or_path"] = args.base_model_id
            write_json(adapter_config_path, adapter_json)
            meta["base_model"] = args.base_model_id
        elif is_local_absolute_model_ref(stored_base):
            raise RuntimeError(
                f"L'adapter fa riferimento a un percorso locale ({stored_base}). "
                "Per evitare di pubblicare un path non portabile, ripeti con --base-model-id ID_HF_DELLO_STESSO_MODELLO."
            )
        elif stored_base:
            meta["base_model"] = stored_base
    else:
        model_config_path = out / "config.json"
        model_json = read_json(model_config_path)
        if isinstance(model_json, dict):
            stored_name = model_json.get("_name_or_path")
            if args.base_model_id:
                meta["base_model"] = args.base_model_id
                model_json["_name_or_path"] = args.base_model_id
                write_json(model_config_path, model_json)
            elif is_local_absolute_model_ref(stored_name):
                # Full weights are included, so their original local source path is
                # not required to load this export and should not be published.
                model_json["_name_or_path"] = repo_id
                write_json(model_config_path, model_json)
                if is_local_absolute_model_ref(meta.get("base_model")):
                    meta["base_model"] = None
        if is_local_absolute_model_ref(meta.get("base_model")):
            meta["base_model"] = None

    valid, description = model_is_valid(out)
    if not valid:
        raise RuntimeError(f"Export non valido: {description}")

    # Keep just the non-sensitive training information needed to interpret embeddings.

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
        raise RuntimeError("Dipendenza mancante: `pip install -U huggingface_hub`. ") from exc
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
        commit_message="Upload RETECO SemEval-2027 Sub-track 2a model artifacts",
    )
    log(f"Upload completato: https://huggingface.co/{repo_id}")
    log("README.md e altri file di documentazione esistenti sul repository non sono stati caricati né modificati.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Operazione interrotta dall'utente.")
