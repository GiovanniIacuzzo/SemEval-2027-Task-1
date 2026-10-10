#!/usr/bin/env python3
"""Training pipeline RETECO SemEval-2027 Sub-track 2a.

Nuova configurazione:
    1. legge esclusivamente benchmark_train e il corpus RETECO ufficiale;
    2. crea una validation interna con split a livello di conversazione;
    3. genera/cache-a query autonome solo per il sottoinsieme di training;
    4. mina hard negative BM25 usando query senza istruzione del dense encoder;
    5. addestra Qwen3-Embedding-0.6B con LoRA e InfoNCE;
    6. seleziona il checkpoint tramite nDCG@10 sulla validation interna.

La generazione delle query è separata dal Dataset/DataLoader. Il modello generativo
viene scaricato dalla GPU prima di caricare il bi-encoder.
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import logging
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset.dataset import (
    TRACK2_DOMAINS,
    ConversationalTurnSample,
    ContextAwareQueryFormatter,
    ConversationalCollateFn,
    DomainBalancedBatchSampler,
    RETECO2aTrainDataset,
    load_query_rewrites,
    load_track2_domain_data,
    save_query_rewrites,
    split_conversations_train_val,
)
from models.model import ConversationalBiEncoder, ConversationalQueryRewriter
from retrieval.retrieval import mine_bm25_hard_negatives
from utils.utils import compute_official_ndcg, load_config, save_json, setup_logger, set_seed


logger = logging.getLogger("RETECO_Train")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)-8s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# =============================================================================
# Reproducibility / device helpers
# =============================================================================

def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower().strip()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA richiesto ma non disponibile.")
        return torch.device("cuda")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS richiesto ma non disponibile.")
        return torch.device("mps")
    if requested != "auto":
        raise ValueError("general.device deve essere uno tra auto, cuda, mps, cpu.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def clear_device_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "frozen": int(total - trainable)}


def safe_component(value: str) -> str:
    """Produce una componente filename corta e innocua."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-") or "default"


# =============================================================================
# Query rewriting: precompute/cache, outside DataLoader workers
# =============================================================================

def prepare_query_rewrites(
    *,
    domains: List[str],
    train_samples_by_domain: Dict[str, List[ConversationalTurnSample]],
    formatter: ContextAwareQueryFormatter,
    paths_cfg: Dict[str, Any],
    rewrite_cfg: Dict[str, Any],
    seed: int,
    val_ratio: float,
) -> Dict[str, Dict[str, str]]:
    """Restituisce una cache per dominio e genera solo le riscritture mancanti."""
    cache_root = Path(paths_cfg.get("query_rewrites_cache_dir", "data/cache/query_rewrites/subtrack_2a"))
    cache_root.mkdir(parents=True, exist_ok=True)

    rewrite_model_name = str(rewrite_cfg.get("model_name_or_path", ConversationalQueryRewriter.DEFAULT_MODEL_NAME))
    cache_tag = safe_component(str(rewrite_cfg.get("cache_tag", "query_rewrite_v1")))
    seed_tag = f"seed{seed}_val{int(round(val_ratio * 100)):02d}"
    overwrite = bool(rewrite_cfg.get("overwrite_cache", False))

    rewrites_by_domain: Dict[str, Dict[str, str]] = {}
    rewriter: Optional[ConversationalQueryRewriter] = None

    for domain in domains:
        train_samples = train_samples_by_domain[domain]
        cache_path = cache_root / f"{safe_component(domain)}_{seed_tag}_{cache_tag}.json"

        if cache_path.exists() and not overwrite:
            existing = load_query_rewrites(cache_path)
        else:
            existing = {}

        missing = [sample for sample in train_samples if sample.topic_id not in existing]
        if overwrite:
            missing = list(train_samples)
            existing = {}

        logger.info(
            f"[REWRITE/{domain}] train samples={len(train_samples)} | "
            f"cached={len(train_samples) - len(missing)} | missing={len(missing)}"
        )

        if missing:
            if not bool(rewrite_cfg.get("enabled", True)):
                raise RuntimeError(
                    "query_mode richiede query riscritte ma query_rewriting.enabled=false."
                )
            if rewriter is None:
                logger.info("Loading conversational query rewriter: %s", rewrite_model_name)
                rewriter = ConversationalQueryRewriter(
                    model_name_or_path=rewrite_model_name,
                    torch_dtype=str(rewrite_cfg.get("torch_dtype", "float16")),
                    device=None,
                    device_map=rewrite_cfg.get("device_map", "auto"),
                    load_in_4bit=bool(rewrite_cfg.get("load_in_4bit", True)),
                    max_new_tokens=int(rewrite_cfg.get("max_new_tokens", 64)),
                    max_input_tokens=int(rewrite_cfg.get("max_input_tokens", 4096)),
                    max_history_tokens=int(rewrite_cfg.get("max_history_tokens", 3000)),
                    trust_remote_code=bool(rewrite_cfg.get("trust_remote_code", False)),
                )

            metadata = {
                "purpose": "subtrack_2a_training_query_rewrites",
                "domain": domain,
                "source_split": "benchmark_train.json",
                "internal_train_seed": seed,
                "internal_validation_ratio": val_ratio,
                "generator_model": rewrite_model_name,
                "cache_tag": cache_tag,
                "deterministic_decoding": True,
                "prompt_version": "ConversationalQueryRewriter.SYSTEM_PROMPT/v1",
            }

            def persist_rewrite_checkpoint(
                current_rewrites: Dict[str, str],
            ) -> None:
                save_query_rewrites(
                    cache_path,
                    current_rewrites,
                    metadata=metadata,
                )

            checkpoint_every = int(
                rewrite_cfg.get("checkpoint_every", 8)
            )
            updated = rewriter.rewrite_samples(
                missing,
                existing=existing,
                overwrite=False,
                progress_desc=f"Rewrite [{domain}]",
                checkpoint_every=checkpoint_every,
                checkpoint_callback=persist_rewrite_checkpoint,
            )

            save_query_rewrites(cache_path, updated, metadata=metadata)
            existing = updated
            logger.info("[REWRITE/%s] saved %d rewrites to %s", domain, len(existing), cache_path)

        missing_after = [sample.topic_id for sample in train_samples if not existing.get(sample.topic_id, "").strip()]
        if missing_after:
            raise RuntimeError(
                f"Cache riscritture incompleta per {domain}: {len(missing_after)} query mancanti."
            )

        rewrites_by_domain[domain] = existing

    # Importante per GPU da 16 GB: liberare il generatore prima di caricare l'encoder.
    if rewriter is not None:
        del rewriter
        clear_device_cache(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info("Query rewriter released before loading the dense encoder.")

    return rewrites_by_domain


# =============================================================================
# Dataset diagnostics
# =============================================================================

def summarize_training_datasets(
    domain_datasets: Dict[str, RETECO2aTrainDataset],
    hard_negative_maps: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"domains": {}, "total_samples": 0}
    for domain, dataset in domain_datasets.items():
        samples = dataset.valid_samples
        hard_map = hard_negative_maps.get(domain, {})
        gold_counts = [len(sample.gold_doc_ids) for sample in samples]
        hard_counts = [len(hard_map.get(sample.topic_id, [])) for sample in samples]
        report["domains"][domain] = {
            "num_samples": len(samples),
            "num_unique_conversations": len({sample.conversation_id for sample in samples}),
            "avg_gold_per_sample": float(np.mean(gold_counts)) if gold_counts else 0.0,
            "avg_mined_hard_negatives": float(np.mean(hard_counts)) if hard_counts else 0.0,
            "min_mined_hard_negatives": int(min(hard_counts)) if hard_counts else 0,
            "samples_below_requested_negative_count": int(
                sum(count < dataset.negatives_per_positive for count in hard_counts)
            ),
            **dataset.get_diagnostics(),
        }
        report["total_samples"] += len(samples)
    return report


# =============================================================================
# Validation
# =============================================================================

@torch.inference_mode()
def evaluate_dense_validation_per_domain(
    *,
    model: ConversationalBiEncoder,
    val_samples_by_domain: Dict[str, List[ConversationalTurnSample]],
    corpus_by_domain: Dict[str, Dict[str, str]],
    qrels_by_domain: Dict[str, Dict[str, Dict[str, int]]],
    tokenizer: Any,
    device: torch.device,
    use_amp: bool,
    max_query_len: int,
    max_doc_len: int,
    eval_batch_size: int = 16,
    retrieval_depth: int = 1000,
) -> Dict[str, Any]:
    """Dense validation a macro-media per dominio, con similarity a blocchi."""
    model.eval()
    per_domain: Dict[str, Dict[str, Any]] = {}

    for domain, samples in val_samples_by_domain.items():
        if not samples:
            continue
        corpus = corpus_by_domain[domain]
        qrels = qrels_by_domain.get(domain, {})
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[doc_id] for doc_id in doc_ids]
        logger.info("[VAL/%s] %d queries | %d documents", domain, len(samples), len(doc_texts))

        corpus_chunks: List[torch.Tensor] = []
        for start in range(0, len(doc_texts), eval_batch_size):
            encoded = tokenizer(
                doc_texts[start:start + eval_batch_size],
                padding=True,
                truncation=True,
                max_length=max_doc_len,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with autocast_context(device, use_amp):
                embeddings = model.encode(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    token_type_ids=encoded.get("token_type_ids"),
                )
            corpus_chunks.append(embeddings.float().cpu())
            del encoded, embeddings

        corpus_embeddings = torch.cat(corpus_chunks, dim=0)
        del corpus_chunks
        clear_device_cache(device)

        run: Dict[str, List[Tuple[str, float]]] = {}
        for start in range(0, len(samples), eval_batch_size):
            batch_samples = samples[start:start + eval_batch_size]
            batch_queries = [sample.contextual_query for sample in batch_samples]
            encoded = tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=max_query_len,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with autocast_context(device, use_amp):
                query_embeddings = model.encode(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    token_type_ids=encoded.get("token_type_ids"),
                )
            query_embeddings = query_embeddings.float().cpu()
            scores = query_embeddings @ corpus_embeddings.T
            k = min(int(retrieval_depth), len(doc_ids))
            top_scores, top_indices = torch.topk(scores, k=k, dim=1)
            for row, sample in enumerate(batch_samples):
                run[sample.topic_id] = [
                    (doc_ids[int(top_indices[row, col])], float(top_scores[row, col]))
                    for col in range(k)
                ]
            del encoded, query_embeddings, scores, top_scores, top_indices

        del corpus_embeddings
        clear_device_cache(device)

        corpus_id_set = set(doc_ids)
        valid_qrels: Dict[str, Dict[str, int]] = {}
        for sample in samples:
            filtered = {
                doc_id: int(rel)
                for doc_id, rel in qrels.get(sample.topic_id, {}).items()
                if doc_id in corpus_id_set and int(rel) > 0
            }
            # Fallback diagnostico soltanto: usa i gold disponibili nel sample.
            if not filtered:
                filtered = {doc_id: 1 for doc_id in sample.gold_doc_ids if doc_id in corpus_id_set}
            if filtered:
                valid_qrels[sample.topic_id] = filtered

        trec_run = {topic: {doc_id: score for doc_id, score in ranking} for topic, ranking in run.items()}
        official = compute_official_ndcg(valid_qrels, trec_run, cutoff=10)

        recalls: Dict[int, List[float]] = {10: [], 50: [], 100: [], 500: [], 1000: []}
        reciprocal_ranks: List[float] = []
        for topic_id, gold_map in valid_qrels.items():
            ranked_ids = [doc_id for doc_id, _ in run.get(topic_id, [])]
            golds = set(gold_map)
            for cutoff in recalls:
                recalls[cutoff].append(len(set(ranked_ids[:cutoff]) & golds) / max(1, len(golds)))
            reciprocal_ranks.append(next((1.0 / rank for rank, doc_id in enumerate(ranked_ids, start=1) if doc_id in golds), 0.0))

        result: Dict[str, Any] = {
            "num_topics": len(valid_qrels),
            "num_documents": len(doc_ids),
            "nDCG@10": float(official.get("ndcg_cut_10", 0.0)),
            "MRR": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        }
        for cutoff, values in recalls.items():
            result[f"Recall@{cutoff}"] = float(np.mean(values)) if values else 0.0
        per_domain[domain] = result
        logger.info(
            "[VAL/%s] nDCG@10=%.4f | R@10=%.4f | R@100=%.4f | R@1000=%.4f | MRR=%.4f",
            domain,
            result["nDCG@10"],
            result["Recall@10"],
            result["Recall@100"],
            result["Recall@1000"],
            result["MRR"],
        )

    if not per_domain:
        return {"per_domain": {}, "macro_average": {}}

    metric_names = ["nDCG@10", "Recall@10", "Recall@50", "Recall@100", "Recall@500", "Recall@1000", "MRR"]
    macro: Dict[str, float] = {}
    for name in metric_names:
        values = [metrics[name] for metrics in per_domain.values()]
        macro[name] = float(np.mean(values))
        macro[f"{name}_std"] = float(np.std(values))
    return {"per_domain": per_domain, "macro_average": macro}


# =============================================================================
# Main training pipeline
# =============================================================================

def train_bi_encoder(config: Dict[str, Any]) -> None:
    general_cfg = config.get("general", {})
    paths_cfg = config.get("paths", {})
    domains_cfg = config.get("domains", {})
    data_cfg = config.get("data", {})
    rewrite_cfg = config.get("query_rewriting", {})
    bi_cfg = config.get("bi_encoder", {})
    training_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    evaluation_cfg = config.get("evaluation", {})

    seed = int(general_cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(general_cfg.get("device", "auto"))
    use_amp = bool(general_cfg.get("mixed_precision", True) and device.type == "cuda")

    logger.info("=" * 88)
    logger.info("RETECO SUB-TRACK 2a — QUERY REASONING + QWEN3 EMBEDDING")
    logger.info("Device=%s | AMP FP16=%s | seed=%d", device, use_amp, seed)

    requested_domains = domains_cfg.get("active_domains", "all")
    domains = list(TRACK2_DOMAINS) if requested_domains == "all" else list(requested_domains)
    unknown_domains = [domain for domain in domains if domain not in TRACK2_DOMAINS]
    if not domains or unknown_domains:
        raise ValueError(f"active_domains non valido: {domains}; domini sconosciuti={unknown_domains}")
    logger.info("Domains (%d): %s", len(domains), ", ".join(domains))

    data_mode = str(paths_cfg.get("data_mode", "full")).lower().strip()
    if data_mode not in {"full", "sample"}:
        raise ValueError("paths.data_mode deve essere 'full' oppure 'sample'.")
    base_data_dir_value = paths_cfg.get("full_data_dir" if data_mode == "full" else "sample_data_dir")
    if not base_data_dir_value:
        raise ValueError(f"Manca paths.{ 'full_data_dir' if data_mode == 'full' else 'sample_data_dir' } nel config.")
    base_data_dir = Path(base_data_dir_value)

    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints/subtrack_2a/qwen3_embedding_0.6b_reasoning"))
    output_dir = Path(paths_cfg.get("output_dir", "outputs/subtrack_2a/qwen3_embedding_0.6b_reasoning"))
    hard_negative_cache_dir = Path(paths_cfg.get("hard_negatives_cache_dir", "data/cache/hard_negatives"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    query_mode = str(data_cfg.get("query_mode", "alternate")).lower().strip()
    if query_mode not in RETECO2aTrainDataset.VALID_QUERY_MODES:
        raise ValueError(f"data.query_mode deve essere una tra {sorted(RETECO2aTrainDataset.VALID_QUERY_MODES)}")

    instruction_cfg = bi_cfg.get("query_instruction", {})
    query_instruction = (
        str(instruction_cfg.get("text", "")).strip()
        if bool(instruction_cfg.get("enabled", False))
        else ""
    )
    if "Qwen3-Embedding" in model_name and not query_instruction:
        raise ValueError("Qwen3-Embedding richiede bi_encoder.query_instruction.enabled=true e un'istruzione non vuota.")

    # Il formatter raw è intenzionalmente senza prefisso Qwen: viene usato da BM25.
    raw_formatter = ContextAwareQueryFormatter(
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        query_instruction="",
        strategy=query_strategy,
    )
    # Questo formatter prepara gli input del dense encoder.
    encoder_formatter = ContextAwareQueryFormatter(
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        query_instruction=query_instruction,
        strategy=query_strategy,
    )

    val_ratio = float(data_cfg.get("val_ratio", 0.15))
    train_samples_by_domain: Dict[str, List[ConversationalTurnSample]] = {}
    val_samples_by_domain: Dict[str, List[ConversationalTurnSample]] = {}
    corpus_by_domain: Dict[str, Dict[str, str]] = {}
    val_qrels_by_domain: Dict[str, Dict[str, Dict[str, int]]] = {}

    logger.info("Loading official RETECO training split from %s", base_data_dir)
    for domain in domains:
        corpus, samples, qrels = load_track2_domain_data(
            data_dir=base_data_dir,
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
            raise ValueError(
                f"Split interno vuoto per {domain}: train={len(train_samples)}, val={len(val_samples)}. "
                "Usare più conversazioni o ridurre data.val_ratio."
            )
        corpus_by_domain[domain] = corpus
        train_samples_by_domain[domain] = train_samples
        val_samples_by_domain[domain] = val_samples
        val_topic_ids = {sample.topic_id for sample in val_samples}
        val_qrels_by_domain[domain] = {
            topic_id: qrels[topic_id]
            for topic_id in val_topic_ids
            if topic_id in qrels
        }
        logger.info(
            "[%s] corpus=%d | turns=%d | train=%d | internal-val=%d | conversations=%d",
            domain,
            len(corpus),
            len(samples),
            len(train_samples),
            len(val_samples),
            len({sample.conversation_id for sample in samples}),
        )

    # ----------------------------------------------------------------------
    # Query rewriting cache (only train portion; query rewriting never sees qrels)
    # ----------------------------------------------------------------------
    rewrites_by_domain: Dict[str, Dict[str, str]] = {}
    if query_mode in {"reasoned", "alternate"}:
        if not bool(rewrite_cfg.get("enabled", True)):
            raise ValueError("data.query_mode richiede query_rewriting.enabled=true.")
        rewrites_by_domain = prepare_query_rewrites(
            domains=domains,
            train_samples_by_domain=train_samples_by_domain,
            formatter=raw_formatter,
            paths_cfg=paths_cfg,
            rewrite_cfg=rewrite_cfg,
            seed=seed,
            val_ratio=val_ratio,
        )
    else:
        logger.info("query_mode=contextual: query rewriting cache non richiesta per training.")

    # ----------------------------------------------------------------------
    # Mine BM25 hard negatives before adding the dense encoder instruction.
    # ----------------------------------------------------------------------
    domain_datasets: Dict[str, RETECO2aTrainDataset] = {}
    hard_negative_maps: Dict[str, Dict[str, List[str]]] = {}
    hard_neg_top_k = int(data_cfg.get("hard_negative_pool_size", 40))
    sparse_cfg = config.get("sparse", {})

    logger.info("Mining/loading BM25 hard negatives (unprefixed conversational queries)...")
    for domain in domains:
        corpus = corpus_by_domain[domain]
        train_samples = train_samples_by_domain[domain]

        hard_negatives = mine_bm25_hard_negatives(
            corpus=corpus,
            samples=train_samples,
            top_k=hard_neg_top_k,
            k1=float(sparse_cfg.get("k1", 0.9)),
            b=float(sparse_cfg.get("b", 0.4)),
            cache_dir=hard_negative_cache_dir,
            cache_tag=(
                f"{safe_component(domain)}_train_raw_context_v1_"
                f"{query_strategy}_{max_query_length}_top{hard_neg_top_k}"
            ),
        )
        hard_negative_maps[domain] = hard_negatives

        # Apply Qwen's query instruction only after BM25 mining.
        for sample in train_samples + val_samples_by_domain[domain]:
            sample.contextual_query = encoder_formatter.format(
                query=sample.query,
                history=sample.history,
            )

        if query_mode in {"reasoned", "alternate"}:
            rewrite_map = rewrites_by_domain[domain]
            for sample in train_samples:
                raw_rewrite = rewrite_map.get(sample.topic_id, "")
                sample.reasoned_query = encoder_formatter.format_rewritten(raw_rewrite) if raw_rewrite else ""

        dataset = RETECO2aTrainDataset(
            samples=train_samples,
            corpus=corpus,
            negatives_per_positive=int(data_cfg.get("negatives_per_positive", 2)),
            hard_negatives=hard_negatives,
            sampling_strategy=str(data_cfg.get("negative_sampling_strategy", "bm25_hard")),
            seed=seed,
            query_mode=query_mode,
            require_reasoned_queries=bool(data_cfg.get("require_reasoned_queries", True)) if query_mode != "contextual" else False,
        )
        domain_datasets[domain] = dataset
        logger.info("[DATASET/%s] %s", domain, dataset.get_diagnostics())

    # ----------------------------------------------------------------------
    # Combined dataset and loader
    # ----------------------------------------------------------------------
    concat_train_dataset = ConcatDataset([domain_datasets[domain] for domain in domains])
    sampler_samples: List[ConversationalTurnSample] = []
    for domain in domains:
        sampler_samples.extend(domain_datasets[domain].valid_samples)

    batch_size = int(training_cfg.get("batch_size", 4))
    if batch_size <= 0:
        raise ValueError("training.batch_size deve essere > 0.")
    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps <= 0:
        raise ValueError("training.gradient_accumulation_steps deve essere > 0.")

    collate_fn = ConversationalCollateFn(
        tokenizer=tokenizer,
        max_query_len=max_query_length,
        max_doc_len=max_doc_length,
    )
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", True) and device.type == "cuda")
    domain_balanced = bool(domains_cfg.get("domain_balanced_training", True))
    batch_sampler = None

    if domain_balanced:
        batch_sampler = DomainBalancedBatchSampler(
            samples=sampler_samples,
            batch_size=batch_size,
            seed=seed,
        )
        train_loader = DataLoader(
            concat_train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            concat_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    if len(train_loader) == 0:
        raise ValueError("Il DataLoader non contiene batch. Ridurre batch_size o controllare il dataset.")

    dataset_report = summarize_training_datasets(domain_datasets, hard_negative_maps)
    logger.info("Total train samples=%d | batches/epoch=%d | batch_size=%d | grad_accum=%d", len(concat_train_dataset), len(train_loader), batch_size, gradient_accumulation_steps)
    logger.info("Domain-balanced batches=%s | negatives/query=%d | query_mode=%s", domain_balanced, int(data_cfg.get("negatives_per_positive", 2)), query_mode)
    logger.info("Raw formatter diagnostics: %s", raw_formatter.get_diagnostics())
    logger.info("Encoder formatter diagnostics: %s", encoder_formatter.get_diagnostics())

    # ----------------------------------------------------------------------
    # Load dense encoder only after the query generator has been released.
    # ----------------------------------------------------------------------
    gradient_checkpointing = bool(bi_cfg.get("gradient_checkpointing", True))
    model = ConversationalBiEncoder(
        model_name_or_path=model_name,
        temperature=float(bi_cfg.get("temperature", 0.05)),
        normalize_embeddings=bool(bi_cfg.get("normalize_embeddings", True)),
        pooling_strategy=str(bi_cfg.get("pooling_strategy", "last_token")),
        lora_cfg=lora_cfg,
        gradient_checkpointing=gradient_checkpointing,
        negative_chunk_size=int(training_cfg.get("negative_chunk_size", 8)),
    ).to(device)

    parameter_stats = count_parameters(model)
    logger.info(
        "Model=%s | pooling=%s | temperature=%.4f | LoRA=%s | gradient_checkpointing=%s",
        model_name,
        getattr(model, "pooling_strategy", bi_cfg.get("pooling_strategy", "last_token")),
        float(bi_cfg.get("temperature", 0.05)),
        bool(lora_cfg.get("enabled", False)),
        gradient_checkpointing,
    )
    logger.info("Parameters: total=%s | trainable=%s | frozen=%s", f"{parameter_stats['total']:,}", f"{parameter_stats['trainable']:,}", f"{parameter_stats['frozen']:,}")
    if parameter_stats["trainable"] == 0:
        raise RuntimeError("Il modello non ha parametri trainabili. Controllare la configurazione LoRA.")

    # ----------------------------------------------------------------------
    # Optimizer/scheduler
    # ----------------------------------------------------------------------
    epochs = int(training_cfg.get("epochs", 2))
    if epochs <= 0:
        raise ValueError("training.epochs deve essere > 0.")
    updates_per_epoch = math.ceil(len(train_loader) / gradient_accumulation_steps)
    total_optimizer_steps = updates_per_epoch * epochs
    learning_rate = float(training_cfg.get("learning_rate", 3e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.06))
    warmup_steps = int(total_optimizer_steps * warmup_ratio)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    log_every = max(1, int(training_cfg.get("log_every_n_steps", 20)))
    patience = max(1, int(training_cfg.get("early_stopping_patience", 1)))

    logger.info("Epochs=%d | optimizer updates=%d | LR=%.2e | warmup=%d", epochs, total_optimizer_steps, learning_rate, warmup_steps)

    best_score = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    optimizer_step = 0
    training_history: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------------
    # Epoch loop
    # ----------------------------------------------------------------------
    for epoch in range(epochs):
        logger.info("\n%s\nEPOCH %d/%d\n%s", "=" * 82, epoch + 1, epochs, "=" * 82)
        for dataset in domain_datasets.values():
            dataset.set_epoch(epoch)
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)

        model.train()
        epoch_start = time.time()
        loss_values: List[float] = []
        rank1_values: List[float] = []
        margin_values: List[float] = []
        pos_cos_values: List[float] = []
        hard_cos_values: List[float] = []
        inbatch_cos_values: List[float] = []
        grad_norm_values: List[float] = []
        domain_loss: Dict[str, List[float]] = collections.defaultdict(list)
        non_finite_batches = 0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in progress:
            q_inputs = {key: value.to(device, non_blocking=pin_memory) for key, value in batch["query_inputs"].items()}
            pos_inputs = {key: value.to(device, non_blocking=pin_memory) for key, value in batch["pos_inputs"].items()}
            neg_inputs = {key: value.to(device, non_blocking=pin_memory) for key, value in batch["neg_inputs"].items()}
            current_k_negs = int(batch["k_negs"])

            with autocast_context(device, use_amp):
                output = model(
                    query_inputs=q_inputs,
                    pos_inputs=pos_inputs,
                    neg_inputs=neg_inputs,
                    k_negs=current_k_negs,
                )
                raw_loss = output["loss"]
                scaled_loss = raw_loss / gradient_accumulation_steps

            if not torch.isfinite(raw_loss).item():
                non_finite_batches += 1
                raise FloatingPointError(
                    f"Loss non-finite a epoch={epoch + 1}, batch={batch_idx + 1}: {raw_loss.item()}"
                )

            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            with torch.no_grad():
                logits = output["logits"].detach().float()
                positive_logits = logits[:, 0]
                if logits.shape[1] > 1:
                    max_negative = logits[:, 1:].max(dim=1).values
                    margin = positive_logits - max_negative
                    rank1 = (logits.argmax(dim=1) == 0).float().mean()
                else:
                    max_negative = torch.zeros_like(positive_logits)
                    margin = positive_logits
                    rank1 = torch.ones_like(positive_logits).mean()
                batch_loss_value = float(raw_loss.detach().float().item())
                loss_values.append(batch_loss_value)
                rank1_values.append(float(rank1.item()))
                margin_values.append(float(margin.mean().item()))
                pos_cos_values.append(float(output["positive_cosine"].detach().float().item()))
                hard_cos_values.append(float(output["hard_negative_cosine"].detach().float().item()))
                inbatch_cos_values.append(float(output["in_batch_negative_cosine"].detach().float().item()))
                for domain in set(batch.get("domains", [])):
                    domain_loss[domain].append(batch_loss_value)

            should_update = (
                (batch_idx + 1) % gradient_accumulation_steps == 0
                or (batch_idx + 1) == len(train_loader)
            )
            grad_norm_value: Optional[float] = None
            if should_update:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
                grad_norm_value = float(grad_norm.detach().float().item() if torch.is_tensor(grad_norm) else grad_norm)
                grad_norm_values.append(grad_norm_value)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1

            global_step += 1
            progress.set_postfix(
                loss=f"{batch_loss_value:.4f}",
                rank1=f"{rank1.item():.3f}",
                margin=f"{margin.mean().item():.3f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            if (batch_idx + 1) % log_every == 0:
                logger.info(
                    "[E%d B%04d] loss=%.4f | rank1=%.3f | margin=%.3f | posCos=%.3f | hardCos=%.3f | inBatchCos=%.3f | lr=%.2e%s",
                    epoch + 1,
                    batch_idx + 1,
                    batch_loss_value,
                    float(rank1.item()),
                    float(margin.mean().item()),
                    pos_cos_values[-1],
                    hard_cos_values[-1],
                    inbatch_cos_values[-1],
                    optimizer.param_groups[0]["lr"],
                    f" | grad={grad_norm_value:.3f}" if grad_norm_value is not None else "",
                )

            del q_inputs, pos_inputs, neg_inputs, output, raw_loss, scaled_loss

        progress.close()
        epoch_seconds = time.time() - epoch_start
        train_loss = float(np.mean(loss_values)) if loss_values else float("nan")
        train_loss_std = float(np.std(loss_values)) if loss_values else 0.0
        logger.info(
            "EPOCH %d SUMMARY | train_loss=%.5f ± %.5f | rank1=%.4f | margin=%.4f | posCos=%.4f | hardCos=%.4f | inBatchCos=%.4f | gradNorm=%.4f | seconds=%.1f",
            epoch + 1,
            train_loss,
            train_loss_std,
            float(np.mean(rank1_values)) if rank1_values else 0.0,
            float(np.mean(margin_values)) if margin_values else 0.0,
            float(np.mean(pos_cos_values)) if pos_cos_values else 0.0,
            float(np.mean(hard_cos_values)) if hard_cos_values else 0.0,
            float(np.mean(inbatch_cos_values)) if inbatch_cos_values else 0.0,
            float(np.mean(grad_norm_values)) if grad_norm_values else 0.0,
            epoch_seconds,
        )
        logger.info("Domain training loss: %s", {domain: round(float(np.mean(values)), 5) for domain, values in domain_loss.items() if values})

        # ------------------------------------------------------------------
        # Validation: dense retrieval on internal conversation-level holdout.
        # ------------------------------------------------------------------
        validation = evaluate_dense_validation_per_domain(
            model=model,
            val_samples_by_domain=val_samples_by_domain,
            corpus_by_domain=corpus_by_domain,
            qrels_by_domain=val_qrels_by_domain,
            tokenizer=tokenizer,
            device=device,
            use_amp=use_amp,
            max_query_len=max_query_length,
            max_doc_len=max_doc_length,
            eval_batch_size=int(evaluation_cfg.get("eval_batch_size", 16)),
            retrieval_depth=int(evaluation_cfg.get("retrieval_depth", 1000)),
        )
        macro = validation.get("macro_average", {})
        val_ndcg = float(macro.get("nDCG@10", 0.0))
        logger.info(
            "VALIDATION MACRO | nDCG@10=%.5f | R@10=%.5f | R@50=%.5f | R@100=%.5f | R@1000=%.5f | MRR=%.5f | domain_std=%.5f",
            val_ndcg,
            float(macro.get("Recall@10", 0.0)),
            float(macro.get("Recall@50", 0.0)),
            float(macro.get("Recall@100", 0.0)),
            float(macro.get("Recall@1000", 0.0)),
            float(macro.get("MRR", 0.0)),
            float(macro.get("nDCG@10_std", 0.0)),
        )

        improved = val_ndcg > best_score
        epoch_record = {
            "epoch": epoch + 1,
            "training": {
                "loss": train_loss,
                "loss_std": train_loss_std,
                "rank1": float(np.mean(rank1_values)) if rank1_values else 0.0,
                "margin": float(np.mean(margin_values)) if margin_values else 0.0,
                "positive_cosine": float(np.mean(pos_cos_values)) if pos_cos_values else 0.0,
                "hard_negative_cosine": float(np.mean(hard_cos_values)) if hard_cos_values else 0.0,
                "in_batch_negative_cosine": float(np.mean(inbatch_cos_values)) if inbatch_cos_values else 0.0,
                "gradient_norm": float(np.mean(grad_norm_values)) if grad_norm_values else 0.0,
                "optimizer_steps_total": optimizer_step,
                "global_step": global_step,
                "epoch_seconds": epoch_seconds,
                "non_finite_batches": non_finite_batches,
            },
            "validation": validation,
            "model_selection": {
                "monitor": str(training_cfg.get("monitor_metric", "val_ndcg@10")),
                "score": val_ndcg,
                "best_score_before_epoch": best_score,
                "improved": improved,
            },
        }
        training_history.append(epoch_record)
        save_json(training_history, output_dir / "training_history.json")

        if improved:
            best_score = val_ndcg
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            checkpoint_path = checkpoint_dir / "best_model.pt"
            # Con LoRA salviamo solo i parametri trainabili: il backbone frozen
            # si ricarica dal model_name_or_path e non deve duplicarsi nel checkpoint.
            trainable_names = {
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            trainable_state = {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
                if name in trainable_names
            }
            checkpoint_payload = {
                "checkpoint_format": "trainable_parameters_only",
                "epoch": epoch + 1,
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "model_state_dict": trainable_state,
                "best_score": best_score,
                "val_metrics": validation,
                "dataset_diagnostics": dataset_report,
                "raw_formatter_diagnostics": raw_formatter.get_diagnostics(),
                "encoder_formatter_diagnostics": encoder_formatter.get_diagnostics(),
                "model_parameters": parameter_stats,
                "config": config,
            }
            torch.save(checkpoint_payload, checkpoint_path)
            # Il report JSON esclude i tensori dei pesi per evitare file enormi
            # e problemi di serializzazione JSON.
            checkpoint_report = {
                key: value for key, value in checkpoint_payload.items()
                if key != "model_state_dict"
            }
            save_json(checkpoint_report, output_dir / "best_checkpoint_report.json")
            logger.info("NEW BEST CHECKPOINT | epoch=%d | nDCG@10=%.5f | path=%s", best_epoch, best_score, checkpoint_path)
        else:
            epochs_without_improvement += 1
            logger.info("No validation improvement (%d/%d).", epochs_without_improvement, patience)

        clear_device_cache(device)
        if epochs_without_improvement >= patience:
            logger.info("Early stopping triggered.")
            break

    # ----------------------------------------------------------------------
    # Reports and reproducibility metadata
    # ----------------------------------------------------------------------
    final_report = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_val_ndcg@10": None if best_score == -float("inf") else best_score,
        "epochs_requested": epochs,
        "epochs_completed": len(training_history),
        "domains": domains,
        "model": {
            "name": model_name,
            "pooling": getattr(model, "pooling_strategy", bi_cfg.get("pooling_strategy", "last_token")),
            "temperature": float(bi_cfg.get("temperature", 0.05)),
            "normalize_embeddings": bool(bi_cfg.get("normalize_embeddings", True)),
            "lora_enabled": bool(lora_cfg.get("enabled", False)),
            "gradient_checkpointing": gradient_checkpointing,
            "parameters": parameter_stats,
        },
        "query_rewriting": {
            "enabled": bool(rewrite_cfg.get("enabled", False)) and query_mode != "contextual",
            "model_name_or_path": rewrite_cfg.get("model_name_or_path"),
            "cache_tag": rewrite_cfg.get("cache_tag"),
            "load_in_4bit": bool(rewrite_cfg.get("load_in_4bit", False)),
            "max_new_tokens": int(rewrite_cfg.get("max_new_tokens", 64)),
            "cache_dir": paths_cfg.get("query_rewrites_cache_dir"),
            "query_mode": query_mode,
        },
        "data": {
            "data_mode": data_mode,
            "query_strategy": query_strategy,
            "max_query_length": max_query_length,
            "max_doc_length": max_doc_length,
            "query_mode": query_mode,
            "negatives_per_positive": int(data_cfg.get("negatives_per_positive", 2)),
            "hard_negative_pool_size": hard_neg_top_k,
            "negative_sampling_strategy": data_cfg.get("negative_sampling_strategy", "bm25_hard"),
            "internal_validation_ratio": val_ratio,
            "dataset_diagnostics": dataset_report,
        },
        "training": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "device": str(device),
            "mixed_precision": use_amp,
        },
        "history": training_history,
    }
    save_json(final_report, output_dir / "final_training_report.json")
    logger.info("=" * 88)
    logger.info("TRAINING COMPLETED | best_epoch=%d | best_val_nDCG@10=%.5f", best_epoch, best_score)
    logger.info("History: %s", output_dir / "training_history.json")
    logger.info("Final report: %s", output_dir / "final_training_report.json")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="RETECO Sub-track 2a: query reasoning + Qwen3-Embedding fine-tuning")
    parser.add_argument("--config", type=str, default="subtrack_2a/config/config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists() and config_path.parts and config_path.parts[0] == "subtrack_2a":
        config_path = Path(*config_path.parts[1:])
    if not config_path.exists():
        alternative = Path("config") / config_path.name
        if alternative.exists():
            config_path = alternative
    if not config_path.exists():
        raise FileNotFoundError(f"Config non trovato: {args.config}")

    config = load_config(config_path)
    paths_cfg = config.get("paths", {})
    general_cfg = config.get("general", {})
    try:
        configured_logger = setup_logger(
            log_dir=Path(paths_cfg.get("log_dir", "outputs/subtrack_2a/logs")),
            run_tag=f"train_{general_cfg.get('run_tag', 'qwen3_reasoning_hybrid_v1')}",
            log_level=general_cfg.get("logging_level", "INFO"),
        )
        if configured_logger is not None:
            global logger
            logger = configured_logger
    except Exception as exc:
        logger.warning("File logger non inizializzato; uso il logger standard: %s", exc)

    train_bi_encoder(config)


if __name__ == "__main__":
    main()
