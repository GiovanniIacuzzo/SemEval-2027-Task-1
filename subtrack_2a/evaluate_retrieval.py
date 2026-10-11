#!/usr/bin/env python3
"""Controlled retrieval analysis for RETECO Sub-track 2a.

Evaluates an existing trained checkpoint without running another training loop:
  - dense retrieval with contextual versus reasoned queries;
  - native Lucene BM25 with the raw contextual and raw rewritten queries;
  - RRF combinations on the exact same internal held-out topics, corpus and qrels.

The internal holdout is recreated deterministically from benchmark_train.json by
conversation ID. Rewritten validation queries are loaded from the project cache;
if any are missing, the shared train.py preparation function generates only the
missing rewrites without accessing qrels.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

import train as train_module
from dataset.dataset import (
    TRACK2_DOMAINS,
    ContextAwareQueryFormatter,
    load_track2_domain_data,
    split_conversations_train_val,
)
from models.model import ConversationalBiEncoder
from train import (
    clear_device_cache,
    evaluate_retrieval_comparison_per_domain,
    prepare_query_rewrites,
    resolve_amp_dtype,
    resolve_device,
)
from utils.utils import load_config, save_json, set_seed, setup_logger


def _load_checkpoint_payload(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint non trovato: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError(
            f"Checkpoint non compatibile con il formato LoRA del progetto: {path}"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confronta dense contestualizzato/riscritto, BM25 e RRF usando un checkpoint esistente."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="subtrack_2a/config/config.yaml",
        help="Percorso del config del subtrack 2a.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint LoRA da valutare; default: paths.checkpoint_dir/best_model.pt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File JSON di output; default dentro paths.output_dir.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists() and config_path.parts and config_path.parts[0] == "subtrack_2a":
        config_path = Path(*config_path.parts[1:])
    if not config_path.is_file():
        raise FileNotFoundError(f"Config non trovato: {args.config}")
    config = load_config(config_path)

    general_cfg = config.get("general", {})
    paths_cfg = config.get("paths", {})
    domains_cfg = config.get("domains", {})
    data_cfg = config.get("data", {})
    rewrite_cfg = config.get("query_rewriting", {})
    bi_cfg = config.get("bi_encoder", {})
    lora_cfg = config.get("lora", {})
    evaluation_cfg = config.get("evaluation", {})
    sparse_cfg = config.get("sparse", {})

    seed = int(general_cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(general_cfg.get("device", "auto"))
    use_amp = bool(general_cfg.get("mixed_precision", True) and device.type == "cuda")

    log_dir = Path(paths_cfg.get("log_dir", "outputs/subtrack_2a/logs"))
    logger = setup_logger(
        log_dir=log_dir,
        run_tag=f"retrieval_comparison_{general_cfg.get('run_tag', 'qwen3_reasoning_hybrid_v1')}",
        log_level=str(general_cfg.get("logging_level", "INFO")),
    )
    train_module.logger = logger

    data_mode = str(paths_cfg.get("data_mode", "full")).lower().strip()
    if data_mode not in {"full", "sample"}:
        raise ValueError("paths.data_mode deve essere 'full' oppure 'sample'.")
    data_dir_value = paths_cfg.get("full_data_dir" if data_mode == "full" else "sample_data_dir")
    if not data_dir_value:
        raise ValueError("Directory dati RETECO non specificata nel config.")
    data_dir = Path(data_dir_value)

    requested_domains = domains_cfg.get("active_domains", "all")
    domains = list(TRACK2_DOMAINS) if requested_domains == "all" else list(requested_domains)
    invalid_domains = [domain for domain in domains if domain not in TRACK2_DOMAINS]
    if not domains or invalid_domains:
        raise ValueError(f"Domini non validi: {invalid_domains or domains}")

    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints/subtrack_2a/qwen3_embedding_0.6b_reasoning"))
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else checkpoint_dir / "best_model.pt"
    checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
    logger.info("Checkpoint: %s | epoch=%s | stored best_score=%s", checkpoint_path, checkpoint_payload.get("epoch"), checkpoint_payload.get("best_score"))
    logger.info("Device=%s | AMP=%s | domains=%s", device, use_amp, ", ".join(domains))

    model_name = str(bi_cfg.get("model_name_or_path", "Qwen/Qwen3-Embedding-0.6B"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    max_query_length = int(data_cfg.get("max_query_length", 512))
    max_doc_length = int(data_cfg.get("max_doc_length", 384))
    query_strategy = str(data_cfg.get("query_strategy", "history"))
    val_ratio = float(data_cfg.get("val_ratio", 0.15))

    instruction_cfg = bi_cfg.get("query_instruction", {})
    query_instruction = (
        str(instruction_cfg.get("text", "")).strip()
        if bool(instruction_cfg.get("enabled", False)) else ""
    )
    if "Qwen3-Embedding" in model_name and not query_instruction:
        raise ValueError("Qwen3-Embedding richiede bi_encoder.query_instruction.enabled=true.")

    raw_formatter = ContextAwareQueryFormatter(
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        query_instruction="",
        strategy=query_strategy,
    )
    encoder_formatter = ContextAwareQueryFormatter(
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        query_instruction=query_instruction,
        strategy=query_strategy,
    )

    train_samples_by_domain = {}
    val_samples_by_domain = {}
    corpus_by_domain = {}
    val_qrels_by_domain = {}
    raw_contextual_queries_by_domain = {}

    logger.info("Loading official corpus and recreating the conversation-level internal validation split...")
    for domain in domains:
        corpus, samples, qrels = load_track2_domain_data(
            data_dir=data_dir,
            domain=domain,
            split="train",
            formatter=raw_formatter,
        )
        train_samples, val_samples = split_conversations_train_val(
            samples=samples,
            val_ratio=val_ratio,
            seed=seed,
        )
        if not train_samples or not val_samples:
            raise ValueError(f"Split interno non valido per {domain}.")
        corpus_by_domain[domain] = corpus
        train_samples_by_domain[domain] = train_samples
        val_samples_by_domain[domain] = val_samples
        raw_contextual_queries_by_domain[domain] = {
            sample.topic_id: sample.contextual_query for sample in val_samples
        }
        val_topic_ids = {sample.topic_id for sample in val_samples}
        val_qrels_by_domain[domain] = {
            topic_id: qrels[topic_id]
            for topic_id in val_topic_ids
            if topic_id in qrels
        }
        logger.info(
            "[%s] corpus=%d | internal-val=%d topics | conversations=%d",
            domain, len(corpus), len(val_samples), len({s.conversation_id for s in samples}),
        )

    # Prepare or load the same raw rewrites as training/validation. The rewriter
    # receives only query/history, never qrels or gold document IDs.
    rewrites_by_domain = prepare_query_rewrites(
        domains=domains,
        train_samples_by_domain=train_samples_by_domain,
        val_samples_by_domain=val_samples_by_domain,
        formatter=raw_formatter,
        paths_cfg=paths_cfg,
        rewrite_cfg=rewrite_cfg,
        seed=seed,
        val_ratio=val_ratio,
    )

    for domain in domains:
        rewrite_map = rewrites_by_domain[domain]
        for sample in val_samples_by_domain[domain]:
            sample.contextual_query = encoder_formatter.format(
                query=sample.query,
                history=sample.history,
            )
            raw_rewrite = rewrite_map.get(sample.topic_id, "")
            if not raw_rewrite.strip():
                raise RuntimeError(f"Query riscritta mancante: {domain}/{sample.topic_id}")
            sample.reasoned_query = encoder_formatter.format_rewritten(raw_rewrite)

    model = ConversationalBiEncoder(
        model_name_or_path=model_name,
        temperature=float(bi_cfg.get("temperature", 0.05)),
        normalize_embeddings=bool(bi_cfg.get("normalize_embeddings", True)),
        pooling_strategy=str(bi_cfg.get("pooling_strategy", "last_token")),
        lora_cfg=lora_cfg,
        gradient_checkpointing=False,
        negative_chunk_size=int(config.get("training", {}).get("negative_chunk_size", 8)),
    )

    checkpoint_state = checkpoint_payload["model_state_dict"]
    expected_trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing_adapter_keys = expected_trainable_names - set(checkpoint_state)
    if missing_adapter_keys:
        example = sorted(missing_adapter_keys)[:8]
        raise RuntimeError(
            "Il checkpoint non contiene tutti i parametri LoRA attesi. "
            f"Mancano {len(missing_adapter_keys)} chiavi; esempi: {example}. "
            "Controllare che model/config corrispondano al checkpoint."
        )
    incompatible = model.load_state_dict(checkpoint_state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Chiavi inattese nel checkpoint: {unexpected[:10]}")
    model.to(device)
    model.eval()
    logger.info(
        "Loaded checkpoint adapter weights: %d trainable tensors; expected missing frozen keys=%d",
        len(checkpoint_state), len(incompatible.missing_keys),
    )

    hard_negative_cache_dir = Path(paths_cfg.get("hard_negatives_cache_dir", "data/cache/hard_negatives"))
    output_dir = Path(paths_cfg.get("output_dir", "outputs/subtrack_2a/qwen3_embedding_0.6b_reasoning"))
    output_path = Path(args.output) if args.output else output_dir / "retrieval_comparison_existing_checkpoint.json"

    result = evaluate_retrieval_comparison_per_domain(
        model=model,
        val_samples_by_domain=val_samples_by_domain,
        corpus_by_domain=corpus_by_domain,
        qrels_by_domain=val_qrels_by_domain,
        raw_contextual_queries_by_domain=raw_contextual_queries_by_domain,
        raw_reasoned_queries_by_domain=rewrites_by_domain,
        tokenizer=tokenizer,
        device=device,
        use_amp=use_amp,
        max_query_len=max_query_length,
        max_doc_len=max_doc_length,
        eval_batch_size=int(evaluation_cfg.get("eval_batch_size", 16)),
        document_eval_batch_size=int(evaluation_cfg.get("document_eval_batch_size", 32)),
        retrieval_depth=int(evaluation_cfg.get("retrieval_depth", 1000)),
        bm25_k1=float(sparse_cfg.get("k1", 0.9)),
        bm25_b=float(sparse_cfg.get("b", 0.4)),
        bm25_index_root=hard_negative_cache_dir / "lucene_indexes",
        rrf_k=int(evaluation_cfg.get("rrf_k", 60)),
        compare_query_modes=bool(evaluation_cfg.get("compare_query_modes", True)),
        compare_sparse_hybrid=bool(evaluation_cfg.get("compare_sparse_hybrid", True)),
    )
    result["checkpoint"] = {
        "path": str(checkpoint_path),
        "epoch": checkpoint_payload.get("epoch"),
        "stored_best_score": checkpoint_payload.get("best_score"),
        "model_name": model_name,
        "seed": seed,
        "val_ratio": val_ratio,
    }
    save_json(result, output_path)
    logger.info("Comparison results saved to %s", output_path)
    clear_device_cache(device)


if __name__ == "__main__":
    main()
