#!/usr/bin/env python3
"""
subtrack_2a/train.py

Training pipeline diagnostica per RETECO SemEval-2027 Sub-track 2a.

Obiettivi:
    - training contrastivo del bi-encoder;
    - hard negatives BM25 separati dal data layer;
    - split train/validation a livello di conversazione;
    - validazione dense per dominio sull'intero corpus del dominio;
    - macro-average nDCG@10;
    - diagnostica dettagliata del training;
    - salvataggio di metriche, configurazione e stato del best checkpoint.

Il dev ufficiale NON viene usato per model selection.
Il training utilizza:
    official train -> internal train/validation split.

Il dev ufficiale viene utilizzato successivamente da inference.py
per la valutazione held-out.
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
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    load_track2_domain_data,
    split_conversations_train_val,
)

from models.model import ConversationalBiEncoder

from retrieval.retrieval import mine_bm25_hard_negatives

from utils.utils import (
    compute_official_ndcg,
    load_config,
    save_json,
    setup_logger,
)


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger("RETECO_Train")

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)-8s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    """Imposta i principali RNG utilizzati dalla pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Device
# =============================================================================

def resolve_device(
    requested: str,
) -> torch.device:
    """
    Risolve il device richiesto.

    Supporta:
        auto
        cuda
        mps
        cpu
    """
    requested = requested.lower().strip()

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Device CUDA richiesto ma CUDA non è disponibile."
            )
        return torch.device("cuda")

    if requested == "mps":
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "Device MPS richiesto ma MPS non è disponibile."
            )
        return torch.device("mps")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def autocast_context(
    device: torch.device,
    enabled: bool,
):
    """
    Context manager AMP.

    Per il nostro workflow:
        CUDA -> FP16
        MPS  -> FP32
        CPU  -> FP32
    """
    if enabled and device.type == "cuda":
        return torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )

    return nullcontext()


def clear_device_cache(device: torch.device) -> None:
    """Libera memoria non più necessaria."""
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    elif device.type == "mps":
        if hasattr(torch, "mps") and hasattr(
            torch.mps,
            "empty_cache",
        ):
            torch.mps.empty_cache()


# =============================================================================
# Model statistics
# =============================================================================

def count_parameters(
    model: torch.nn.Module,
) -> Dict[str, int]:
    """Conta parametri totali e trainabili."""
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = total - trainable

    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(frozen),
    }


# =============================================================================
# Dataset diagnostics
# =============================================================================

def summarize_training_datasets(
    domain_datasets: Dict[str, RETECO2aTrainDataset],
    hard_negative_maps: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    """Produce diagnostica strutturale dei dataset di training."""

    report: Dict[str, Any] = {
        "domains": {},
        "total_samples": 0,
    }

    for domain, dataset in domain_datasets.items():
        samples = dataset.valid_samples
        hard_map = hard_negative_maps.get(domain, {})

        gold_counts = [
            len(sample.gold_doc_ids)
            for sample in samples
        ]

        hard_counts = [
            len(hard_map.get(sample.topic_id, []))
            for sample in samples
        ]

        missing_hard = sum(
            count < dataset.negatives_per_positive
            for count in hard_counts
        )

        report["domains"][domain] = {
            "num_samples": len(samples),
            "num_unique_conversations": len(
                {
                    s.conversation_id
                    for s in samples
                }
            ),
            "avg_gold_per_sample": (
                float(np.mean(gold_counts))
                if gold_counts
                else 0.0
            ),
            "max_gold_per_sample": (
                int(max(gold_counts))
                if gold_counts
                else 0
            ),
            "avg_mined_hard_negatives": (
                float(np.mean(hard_counts))
                if hard_counts
                else 0.0
            ),
            "min_mined_hard_negatives": (
                int(min(hard_counts))
                if hard_counts
                else 0
            ),
            "samples_below_requested_negative_count": int(
                missing_hard
            ),
        }

        report["total_samples"] += len(samples)

    return report


# =============================================================================
# Validation
# =============================================================================

def evaluate_dense_validation_per_domain(
    model: ConversationalBiEncoder,
    val_samples_by_domain: Dict[
        str,
        List[ConversationalTurnSample],
    ],
    corpus_by_domain: Dict[str, Dict[str, str]],
    qrels_by_domain: Dict[
        str,
        Dict[str, Dict[str, int]],
    ],
    tokenizer: AutoTokenizer,
    device: torch.device,
    use_amp: bool,
    max_query_len: int,
    max_doc_len: int,
    eval_batch_size: int = 64,
    retrieval_depth: int = 1000,
) -> Dict[str, Any]:
    """
    Dense retrieval validation.

    Per ogni dominio:
        - encode corpus completo;
        - encode validation queries;
        - retrieve top-1000;
        - calcola nDCG@10, Recall@K, MRR;
        - calcola diagnostica per turno.

    Importante:
        la similarity matrix non viene mai materializzata interamente.
        Le query vengono processate a blocchi.
    """

    model.eval()

    per_domain: Dict[str, Dict[str, Any]] = {}

    for domain, samples in val_samples_by_domain.items():
        if not samples:
            continue

        corpus = corpus_by_domain[domain]
        qrels = qrels_by_domain.get(domain, {})

        doc_ids = list(corpus.keys())
        doc_texts = [
            corpus[doc_id]
            for doc_id in doc_ids
        ]

        logger.info(
            f"[VAL/{domain}] "
            f"{len(samples)} queries | "
            f"{len(doc_texts)} documents"
        )

        # ------------------------------------------------------------------
        # 1. Encode corpus
        # ------------------------------------------------------------------

        corpus_chunks: List[torch.Tensor] = []

        with torch.no_grad():
            for start in range(
                0,
                len(doc_texts),
                eval_batch_size,
            ):
                batch_texts = doc_texts[
                    start:start + eval_batch_size
                ]

                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_doc_len,
                    return_tensors="pt",
                )

                encoded = {
                    key: value.to(device)
                    for key, value in encoded.items()
                }

                with autocast_context(
                    device,
                    use_amp,
                ):
                    embeddings = model.encode(
                        encoded["input_ids"],
                        encoded["attention_mask"],
                        token_type_ids=encoded.get(
                            "token_type_ids"
                        ),
                    )

                corpus_chunks.append(
                    embeddings.float().cpu()
                )

        corpus_embeddings = torch.cat(
            corpus_chunks,
            dim=0,
        )

        del corpus_chunks
        clear_device_cache(device)

        # ------------------------------------------------------------------
        # 2. Encode validation queries + retrieve
        # ------------------------------------------------------------------

        queries = [
            sample.contextual_query
            for sample in samples
        ]

        run: Dict[
            str,
            List[Tuple[str, float]],
        ] = {}

        with torch.no_grad():
            for start in range(
                0,
                len(queries),
                eval_batch_size,
            ):
                batch_samples = samples[
                    start:start + eval_batch_size
                ]

                batch_queries = [
                    sample.contextual_query
                    for sample in batch_samples
                ]

                encoded = tokenizer(
                    batch_queries,
                    padding=True,
                    truncation=True,
                    max_length=max_query_len,
                    return_tensors="pt",
                )

                encoded = {
                    key: value.to(device)
                    for key, value in encoded.items()
                }

                with autocast_context(
                    device,
                    use_amp,
                ):
                    query_embeddings = model.encode(
                        encoded["input_ids"],
                        encoded["attention_mask"],
                        token_type_ids=encoded.get(
                            "token_type_ids"
                        ),
                    )

                query_embeddings = (
                    query_embeddings.float().cpu()
                )

                # Similarity solo per questo mini-batch.
                scores = (
                    query_embeddings
                    @ corpus_embeddings.T
                )

                k = min(
                    retrieval_depth,
                    corpus_embeddings.shape[0],
                )

                top_scores, top_indices = torch.topk(
                    scores,
                    k=k,
                    dim=1,
                )

                for row, sample in enumerate(
                    batch_samples
                ):
                    ranking = []

                    for col in range(k):
                        doc_idx = int(
                            top_indices[row, col]
                        )

                        ranking.append(
                            (
                                doc_ids[doc_idx],
                                float(
                                    top_scores[row, col]
                                ),
                            )
                        )

                    run[sample.topic_id] = ranking

                del scores
                del top_scores
                del top_indices
                del query_embeddings

        del corpus_embeddings
        clear_device_cache(device)

        # ------------------------------------------------------------------
        # 3. Build official qrels intersection
        # ------------------------------------------------------------------

        valid_qrels: Dict[
            str,
            Dict[str, int],
        ] = {}

        corpus_id_set = set(doc_ids)

        for sample in samples:
            official_qrels = qrels.get(
                sample.topic_id,
                {},
            )

            filtered = {
                doc_id: int(rel)
                for doc_id, rel in official_qrels.items()
                if doc_id in corpus_id_set
                and int(rel) > 0
            }

            # Se il qrels non esiste, usiamo i gold del sample
            # solo come fallback diagnostico.
            if not filtered:
                filtered = {
                    doc_id: 1
                    for doc_id in sample.gold_doc_ids
                    if doc_id in corpus_id_set
                }

            if filtered:
                valid_qrels[sample.topic_id] = filtered

        # ------------------------------------------------------------------
        # 4. Metrics
        # ------------------------------------------------------------------

        trec_run = {
            topic_id: {
                doc_id: score
                for doc_id, score in ranking
            }
            for topic_id, ranking in run.items()
        }

        official = compute_official_ndcg(
            valid_qrels,
            trec_run,
            cutoff=10,
        )

        recalls = {
            10: [],
            50: [],
            100: [],
            500: [],
            1000: [],
        }

        reciprocal_ranks = []

        by_turn: Dict[str, List[float]] = (
            collections.defaultdict(list)
        )

        for sample in samples:
            topic_id = sample.topic_id

            if topic_id not in valid_qrels:
                continue

            ranking = run.get(
                topic_id,
                [],
            )

            golds = set(
                valid_qrels[topic_id].keys()
            )

            ranked_ids = [
                doc_id
                for doc_id, _score in ranking
            ]

            for cutoff in recalls:
                hits = len(
                    set(
                        ranked_ids[:cutoff]
                    ).intersection(golds)
                )

                recalls[cutoff].append(
                    hits / max(
                        1,
                        len(golds),
                    )
                )

            rr = 0.0

            for rank, doc_id in enumerate(
                ranked_ids,
                start=1,
            ):
                if doc_id in golds:
                    rr = 1.0 / rank
                    break

            reciprocal_ranks.append(rr)

            # Turn bucket
            if sample.turn_id >= 5:
                turn_key = "T5+"
            else:
                turn_key = f"T{sample.turn_id}"

            # Single-query nDCG for diagnostics
            single_qrel = {
                topic_id: valid_qrels[topic_id]
            }

            single_run = {
                topic_id: trec_run[topic_id]
            }

            single_ndcg = compute_official_ndcg(
                single_qrel,
                single_run,
                cutoff=10,
            ).get(
                "ndcg_cut_10",
                0.0,
            )

            by_turn[turn_key].append(
                float(single_ndcg)
            )

        domain_result: Dict[str, Any] = {
            "num_topics": len(valid_qrels),
            "num_documents": len(doc_ids),
            "nDCG@10": float(
                official.get(
                    "ndcg_cut_10",
                    0.0,
                )
            ),
            "Recall@10": float(
                np.mean(recalls[10])
                if recalls[10]
                else 0.0
            ),
            "Recall@50": float(
                np.mean(recalls[50])
                if recalls[50]
                else 0.0
            ),
            "Recall@100": float(
                np.mean(recalls[100])
                if recalls[100]
                else 0.0
            ),
            "Recall@500": float(
                np.mean(recalls[500])
                if recalls[500]
                else 0.0
            ),
            "Recall@1000": float(
                np.mean(recalls[1000])
                if recalls[1000]
                else 0.0
            ),
            "MRR": float(
                np.mean(reciprocal_ranks)
                if reciprocal_ranks
                else 0.0
            ),
            "nDCG@10_by_turn": {
                turn: float(np.mean(values))
                for turn, values
                in sorted(by_turn.items())
                if values
            },
        }

        per_domain[domain] = domain_result

        logger.info(
            f"[VAL/{domain:<18}] "
            f"nDCG@10={domain_result['nDCG@10']:.4f} | "
            f"R@10={domain_result['Recall@10']:.4f} | "
            f"R@100={domain_result['Recall@100']:.4f} | "
            f"R@1000={domain_result['Recall@1000']:.4f} | "
            f"MRR={domain_result['MRR']:.4f}"
        )

        if domain_result["nDCG@10_by_turn"]:
            turn_string = " | ".join(
                f"{turn}={score:.4f}"
                for turn, score
                in domain_result["nDCG@10_by_turn"].items()
            )

            logger.info(
                f"[VAL/{domain:<18}] "
                f"nDCG@10 by turn: {turn_string}"
            )

    # ----------------------------------------------------------------------
    # Macro average over domains
    # ----------------------------------------------------------------------

    if not per_domain:
        return {
            "per_domain": {},
            "macro_average": {},
        }

    metrics = [
        "nDCG@10",
        "Recall@10",
        "Recall@50",
        "Recall@100",
        "Recall@500",
        "Recall@1000",
        "MRR",
    ]

    macro = {}

    for metric in metrics:
        values = [
            result[metric]
            for result in per_domain.values()
        ]

        macro[metric] = float(
            np.mean(values)
        )

        macro[f"{metric}_std"] = float(
            np.std(values)
        )

    return {
        "per_domain": per_domain,
        "macro_average": macro,
    }


# =============================================================================
# Training
# =============================================================================

def train_bi_encoder(
    config: Dict[str, Any],
) -> None:

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    general_cfg = config.get(
        "general",
        {},
    )

    paths_cfg = config.get(
        "paths",
        {},
    )

    domains_cfg = config.get(
        "domains",
        {},
    )

    data_cfg = config.get(
        "data",
        {},
    )

    bi_cfg = config.get(
        "bi_encoder",
        {},
    )

    training_cfg = config.get(
        "training",
        {},
    )

    lora_cfg = config.get(
        "lora",
        {},
    )

    evaluation_cfg = config.get(
        "evaluation",
        {},
    )

    seed = int(
        general_cfg.get(
            "seed",
            42,
        )
    )

    set_seed(seed)

    device = resolve_device(
        general_cfg.get(
            "device",
            "auto",
        )
    )

    use_amp = bool(
        general_cfg.get(
            "mixed_precision",
            True,
        )
        and device.type == "cuda"
    )

    logger.info(
        "=" * 90
    )
    logger.info(
        "RETECO SUB-TRACK 2a"
    )
    logger.info(
        "Bi-Encoder Training"
    )
    logger.info(
        "=" * 90
    )
    logger.info(
        f"Device        : {device}"
    )
    logger.info(
        f"AMP FP16      : {use_amp}"
    )
    logger.info(
        f"Seed          : {seed}"
    )

    # ------------------------------------------------------------------
    # Domains
    # ------------------------------------------------------------------

    requested_domains = domains_cfg.get(
        "active_domains",
        "all",
    )

    if requested_domains == "all":
        domains = list(TRACK2_DOMAINS)
    else:
        domains = list(requested_domains)

    unknown_domains = [
        domain
        for domain in domains
        if domain not in TRACK2_DOMAINS
    ]

    if unknown_domains:
        raise ValueError(
            f"Domini non validi: {unknown_domains}"
        )

    logger.info(
        f"Domains       : {', '.join(domains)}"
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    data_mode = paths_cfg.get(
        "data_mode",
        "full",
    )

    if data_mode == "full":
        base_data_dir = Path(
            paths_cfg.get(
                "full_data_dir"
            )
        )
    else:
        base_data_dir = Path(
            paths_cfg.get(
                "sample_data_dir"
            )
        )

    checkpoint_dir = Path(
        paths_cfg.get(
            "checkpoint_dir",
            "checkpoints/subtrack_2a/bi_encoder",
        )
    )

    output_dir = Path(
        paths_cfg.get(
            "output_dir",
            "outputs/subtrack_2a",
        )
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hard_negative_cache_dir = Path(
        paths_cfg.get(
            "hard_negatives_cache_dir",
            "data/cache/hard_negatives",
        )
    )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    model_name = bi_cfg.get(
        "model_name_or_path",
        "BAAI/bge-base-en-v1.5",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens(
                {"pad_token": "[PAD]"}
            )

    query_instruction_cfg = bi_cfg.get(
        "query_instruction",
        {},
    )

    query_instruction = (
        query_instruction_cfg.get(
            "text",
            "",
        )
        if query_instruction_cfg.get(
            "enabled",
            False,
        )
        else ""
    )

    max_query_length = int(
        data_cfg.get(
            "max_query_length",
            256,
        )
    )

    max_doc_length = int(
        data_cfg.get(
            "max_doc_length",
            256,
        )
    )

    query_strategy = data_cfg.get(
        "query_strategy",
        "history",
    )

    formatter = ContextAwareQueryFormatter(
        tokenizer=tokenizer,
        max_query_length=max_query_length,
        query_instruction=query_instruction,
        strategy=query_strategy,
    )

    logger.info(
        f"Query strategy        : {query_strategy}"
    )
    logger.info(
        f"Max query tokens      : {max_query_length}"
    )
    logger.info(
        f"Max document tokens   : {max_doc_length}"
    )

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    internal_val_ratio = float(
        data_cfg.get(
            "val_ratio",
            0.15,
        )
    )

    train_samples_by_domain: Dict[
        str,
        List[ConversationalTurnSample],
    ] = {}

    val_samples_by_domain: Dict[
        str,
        List[ConversationalTurnSample],
    ] = {}

    corpus_by_domain: Dict[
        str,
        Dict[str, str],
    ] = {}

    train_qrels_by_domain: Dict[
        str,
        Dict[str, Dict[str, int]],
    ] = {}

    val_qrels_by_domain: Dict[
        str,
        Dict[str, Dict[str, int]],
    ] = {}

    domain_datasets: Dict[
        str,
        RETECO2aTrainDataset,
    ] = {}

    hard_negative_maps: Dict[
        str,
        Dict[str, List[str]],
    ] = {}

    logger.info(
        "Loading datasets..."
    )

    for domain in domains:

        corpus, samples, qrels = (
            load_track2_domain_data(
                data_dir=base_data_dir,
                domain=domain,
                split="train",
                formatter=formatter,
            )
        )

        train_samples, val_samples = (
            split_conversations_train_val(
                samples=samples,
                val_ratio=internal_val_ratio,
                seed=seed,
            )
        )

        corpus_by_domain[domain] = corpus
        train_samples_by_domain[domain] = train_samples
        val_samples_by_domain[domain] = val_samples

        train_qrels_by_domain[domain] = qrels

        # Il qrels del train viene ricostruito perché la validation interna
        # utilizzerà solo i topic appartenenti alla sua porzione.
        val_topic_ids = {
            sample.topic_id
            for sample in val_samples
        }

        val_qrels_by_domain[domain] = {
            topic_id: qrels[topic_id]
            for topic_id in val_topic_ids
            if topic_id in qrels
        }

        logger.info(
            f"[{domain:<18}] "
            f"corpus={len(corpus):>7} | "
            f"turns={len(samples):>4} | "
            f"train={len(train_samples):>4} | "
            f"val={len(val_samples):>4}"
        )

        # --------------------------------------------------------------
        # Hard negatives SOLO sul training split interno.
        # --------------------------------------------------------------

        hard_neg_top_k = int(
            data_cfg.get(
                "hard_negative_pool_size",
                50,
            )
        )

        hard_negatives = mine_bm25_hard_negatives(
            corpus=corpus,
            samples=train_samples,
            top_k=hard_neg_top_k,
            k1=float(
                config.get(
                    "sparse",
                    {},
                ).get(
                    "k1",
                    0.9,
                )
            ),
            b=float(
                config.get(
                    "sparse",
                    {},
                ).get(
                    "b",
                    0.4,
                )
            ),
            cache_dir=hard_negative_cache_dir,
            cache_tag=(
                f"{domain}_train_"
                f"{query_strategy}_"
                f"{max_query_length}"
            ),
        )

        hard_negative_maps[domain] = hard_negatives

        domain_dataset = RETECO2aTrainDataset(
            samples=train_samples,
            corpus=corpus,
            negatives_per_positive=int(
                data_cfg.get(
                    "negatives_per_positive",
                    4,
                )
            ),
            hard_negatives=hard_negatives,
            sampling_strategy=data_cfg.get(
                "negative_sampling_strategy",
                "bm25_hard",
            ),
            seed=seed,
        )

        domain_datasets[domain] = domain_dataset

    # ------------------------------------------------------------------
    # Dataset diagnostics
    # ------------------------------------------------------------------

    dataset_report = summarize_training_datasets(
        domain_datasets=domain_datasets,
        hard_negative_maps=hard_negative_maps,
    )

    logger.info(
        "=" * 90
    )
    logger.info(
        "TRAIN DATASET DIAGNOSTICS"
    )
    logger.info(
        "=" * 90
    )

    for domain, stats in dataset_report[
        "domains"
    ].items():

        logger.info(
            f"{domain:<18} | "
            f"samples={stats['num_samples']:>4} | "
            f"convs={stats['num_unique_conversations']:>3} | "
            f"gold/turn={stats['avg_gold_per_sample']:.2f} | "
            f"hardNeg avg={stats['avg_mined_hard_negatives']:.2f} | "
            f"hardNeg min={stats['min_mined_hard_negatives']:>2} | "
            f"insufficient={stats['samples_below_requested_negative_count']}"
        )

    # ------------------------------------------------------------------
    # ConcatDataset
    # ------------------------------------------------------------------

    concat_train_dataset = ConcatDataset(
        list(
            domain_datasets.values()
        )
    )

    sampler_samples: List[
        ConversationalTurnSample
    ] = []

    for domain in domains:
        sampler_samples.extend(
            domain_datasets[
                domain
            ].valid_samples
        )

    batch_size = int(
        training_cfg.get(
            "batch_size",
            4,
        )
    )

    domain_balanced = bool(
        domains_cfg.get(
            "domain_balanced_training",
            True,
        )
    )

    collate_fn = ConversationalCollateFn(
        tokenizer=tokenizer,
        max_query_len=max_query_length,
        max_doc_len=max_doc_length,
    )

    num_workers = int(
        data_cfg.get(
            "num_workers",
            0,
        )
    )

    pin_memory = bool(
        data_cfg.get(
            "pin_memory",
            True,
        )
        and device.type == "cuda"
    )

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
            drop_last=True,
        )

    logger.info(
        f"Train samples       : {len(concat_train_dataset)}"
    )

    logger.info(
        f"Batch size          : {batch_size}"
    )

    logger.info(
        f"Gradient accumulation: "
        f"{training_cfg.get('gradient_accumulation_steps', 1)}"
    )

    logger.info(
        f"Domain-balanced    : {domain_balanced}"
    )

    # Diagnostica negative.
    requested_k = int(
        data_cfg.get(
            "negatives_per_positive",
            4,
        )
    )

    logger.info(
        f"Explicit negatives/query : {requested_k}"
    )

    logger.info(
        f"In-batch negatives/query  : "
        f"{max(0, batch_size - 1)}"
    )

    logger.info(
        f"Contrastive candidates/query: "
        f"{1 + requested_k + max(0, batch_size - 1)} "
        f"(positive + explicit + in-batch)"
    )

    # ------------------------------------------------------------------
    # Formatter diagnostics
    # ------------------------------------------------------------------

    formatter_diag = formatter.get_diagnostics()

    logger.info(
        "=" * 90
    )
    logger.info(
        "QUERY FORMATTER DIAGNOSTICS"
    )
    logger.info(
        "=" * 90
    )

    for key, value in formatter_diag.items():

        if key.startswith("pct_"):
            logger.info(
                f"{key:<30}: {value:.2f}%"
            )
        else:
            logger.info(
                f"{key:<30}: {value:.2f}"
            )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    gradient_checkpointing = bool(
        bi_cfg.get(
            "gradient_checkpointing",
            False,
        )
    )

    model = ConversationalBiEncoder(
        model_name_or_path=model_name,
        temperature=float(
            bi_cfg.get(
                "temperature",
                0.05,
            )
        ),
        normalize_embeddings=bool(
            bi_cfg.get(
                "normalize_embeddings",
                True,
            )
        ),
        pooling_strategy=bi_cfg.get(
            "pooling_strategy",
            "cls",
        ),
        lora_cfg=lora_cfg,
        gradient_checkpointing=gradient_checkpointing,
        negative_chunk_size=int(
            training_cfg.get(
                "negative_chunk_size",
                8,
            )
        ),
    ).to(device)

    parameter_stats = count_parameters(
        model
    )

    logger.info(
        "=" * 90
    )
    logger.info(
        "MODEL"
    )
    logger.info(
        "=" * 90
    )
    logger.info(
        f"Model                : {model_name}"
    )
    logger.info(
        f"Pooling              : "
        f"{bi_cfg.get('pooling_strategy', 'cls')}"
    )
    logger.info(
        f"Temperature           : "
        f"{bi_cfg.get('temperature', 0.05)}"
    )
    logger.info(
        f"LoRA enabled          : "
        f"{bool(lora_cfg.get('enabled', False))}"
    )
    logger.info(
        f"Gradient checkpoint   : "
        f"{gradient_checkpointing}"
    )
    logger.info(
        f"Total parameters      : "
        f"{parameter_stats['total']:,}"
    )
    logger.info(
        f"Trainable parameters  : "
        f"{parameter_stats['trainable']:,}"
    )
    logger.info(
        f"Frozen parameters     : "
        f"{parameter_stats['frozen']:,}"
    )

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    epochs = int(
        training_cfg.get(
            "epochs",
            3,
        )
    )

    gradient_accumulation_steps = int(
        training_cfg.get(
            "gradient_accumulation_steps",
            1,
        )
    )

    updates_per_epoch = math.ceil(
        len(train_loader)
        / gradient_accumulation_steps
    )

    total_optimizer_steps = (
        updates_per_epoch * epochs
    )

    learning_rate = float(
        training_cfg.get(
            "learning_rate",
            1e-4,
        )
    )

    weight_decay = float(
        training_cfg.get(
            "weight_decay",
            0.01,
        )
    )

    warmup_ratio = float(
        training_cfg.get(
            "warmup_ratio",
            0.1,
        )
    )

    warmup_steps = int(
        total_optimizer_steps
        * warmup_ratio
    )

    trainable_parameters = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )

    scaler = None

    if use_amp:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

    max_grad_norm = float(
        training_cfg.get(
            "max_grad_norm",
            1.0,
        )
    )

    logger.info(
        "=" * 90
    )
    logger.info(
        "OPTIMIZATION"
    )
    logger.info(
        "=" * 90
    )
    logger.info(
        f"Epochs               : {epochs}"
    )
    logger.info(
        f"Batches/epoch        : {len(train_loader)}"
    )
    logger.info(
        f"Optimizer updates    : {total_optimizer_steps}"
    )
    logger.info(
        f"LR                   : {learning_rate:.2e}"
    )
    logger.info(
        f"Weight decay         : {weight_decay:.4f}"
    )
    logger.info(
        f"Warmup steps         : {warmup_steps}"
    )
    logger.info(
        f"Grad clip            : {max_grad_norm:.2f}"
    )

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    monitor_metric = training_cfg.get(
        "monitor_metric",
        "val_ndcg@10",
    )

    best_score = -float("inf")
    best_epoch = -1

    patience = int(
        training_cfg.get(
            "early_stopping_patience",
            2,
        )
    )

    epochs_without_improvement = 0

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    training_history: List[
        Dict[str, Any]
    ] = []

    global_step = 0
    optimizer_step = 0

    # =========================================================================
    # Epoch loop
    # =========================================================================

    for epoch in range(epochs):

        logger.info(
            "\n"
            + "=" * 90
        )
        logger.info(
            f"EPOCH {epoch + 1}/{epochs}"
        )
        logger.info(
            "=" * 90
        )

        # Nuovo shuffle deterministico.
        for domain_dataset in domain_datasets.values():
            domain_dataset.set_epoch(epoch)

        if domain_balanced:
            batch_sampler.set_epoch(epoch)

        model.train()

        epoch_start = time.time()

        running_loss = 0.0
        running_loss_sq = 0.0

        running_rank1 = 0.0
        running_pos_logit = 0.0
        running_max_neg_logit = 0.0
        running_margin = 0.0

        running_grad_norm_sum = 0.0
        running_grad_norm_max = 0.0

        running_positive_cosine = 0.0
        running_hard_negative_cosine = 0.0
        running_in_batch_negative_cosine = 0.0
        running_hard_negative_margin = 0.0
        running_in_batch_margin = 0.0

        domain_loss_sum = collections.defaultdict(float)
        domain_batch_count = collections.defaultdict(int)

        batches_with_multiple_domains = 0
        non_finite_batches = 0

        total_examples = 0

        lr_values: List[float] = []

        optimizer.zero_grad(
            set_to_none=True
        )

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for batch_idx, batch in progress:

            batch_start = time.time()

            batch_domains = batch.get(
                "domains",
                [],
            )

            unique_domains = set(
                batch_domains
            )

            if len(unique_domains) > 1:
                batches_with_multiple_domains += 1

            q_inputs = {
                key: value.to(device)
                for key, value
                in batch["query_inputs"].items()
            }

            pos_inputs = {
                key: value.to(device)
                for key, value
                in batch["pos_inputs"].items()
            }

            neg_inputs = {
                key: value.to(device)
                for key, value
                in batch["neg_inputs"].items()
            }

            current_k_negs = int(
                batch["k_negs"]
            )

            with autocast_context(
                device,
                use_amp,
            ):

                output = model(
                    query_inputs=q_inputs,
                    pos_inputs=pos_inputs,
                    neg_inputs=neg_inputs,
                    k_negs=current_k_negs,
                )

                raw_loss = output["loss"]
                loss = (
                    raw_loss
                    / gradient_accumulation_steps
                )

            if not torch.isfinite(
                raw_loss
            ).item():

                non_finite_batches += 1

                logger.error(
                    f"Non-finite loss at "
                    f"epoch={epoch + 1}, "
                    f"batch={batch_idx + 1}: "
                    f"{raw_loss.item()}"
                )

                raise FloatingPointError(
                    "Training interrotto per loss "
                    "non-finite."
                )

            # -------------------------------------------------------------
            # Backward
            # -------------------------------------------------------------

            if scaler is not None:
                scaler.scale(
                    loss
                ).backward()
            else:
                loss.backward()

            # -------------------------------------------------------------
            # Diagnostics dei logits
            # -------------------------------------------------------------

        with torch.no_grad():

            logits = output["logits"].detach()

            positive_logits = logits[:, 0]

            if logits.shape[1] > 1:

                negative_logits = logits[:, 1:]

                max_negative = (
                    negative_logits.max(
                        dim=1
                    ).values
                )

                margin = (
                    positive_logits
                    - max_negative
                )

                rank1 = (
                    logits.argmax(
                        dim=1
                    ) == 0
                ).float().mean()

            else:

                max_negative = torch.zeros_like(
                    positive_logits
                )

                margin = positive_logits

                rank1 = torch.ones_like(
                    positive_logits
                ).mean()

            # --------------------------------------------------------------
            # Nuove diagnostiche in cosine space
            # --------------------------------------------------------------

            positive_cosine = float(
                output["positive_cosine"].detach().item()
            )

            hard_negative_cosine = float(
                output["hard_negative_cosine"].detach().item()
            )

            in_batch_negative_cosine = float(
                output["in_batch_negative_cosine"].detach().item()
            )

            hard_negative_margin = (
                positive_cosine
                - hard_negative_cosine
            )

            in_batch_margin = (
                positive_cosine
                - in_batch_negative_cosine
            )

            # Accumulo diagnostiche
            running_positive_cosine += positive_cosine
            running_hard_negative_cosine += (
                hard_negative_cosine
            )
            running_in_batch_negative_cosine += (
                in_batch_negative_cosine
            )
            running_hard_negative_margin += (
                hard_negative_margin
            )
            running_in_batch_margin += (
                in_batch_margin
            )

            running_pos_logit += float(
                positive_logits.mean().item()
            )

            running_max_neg_logit += float(
                max_negative.mean().item()
            )

            running_margin += float(
                margin.mean().item()
            )

            running_rank1 += float(
                rank1.item()
            )

            batch_loss = float(
                raw_loss.item()
            )

            running_loss += batch_loss
            running_loss_sq += (
                batch_loss ** 2
            )

            batch_size_actual = len(
                batch["topic_ids"]
            )

            total_examples += (
                batch_size_actual
            )

            # -------------------------------------------------------------
            # Domain loss
            # -------------------------------------------------------------

            if len(unique_domains) == 1:

                domain = next(
                    iter(unique_domains)
                )

                domain_loss_sum[
                    domain
                ] += batch_loss

                domain_batch_count[
                    domain
                ] += 1

            else:

                # Fallback diagnostico
                for domain in unique_domains:
                    domain_loss_sum[
                        domain
                    ] += batch_loss

                    domain_batch_count[
                        domain
                    ] += 1

            # -------------------------------------------------------------
            # Optimizer step
            # -------------------------------------------------------------

            should_update = (
                (batch_idx + 1)
                % gradient_accumulation_steps
                == 0
                or
                (batch_idx + 1)
                == len(train_loader)
            )

            grad_norm_value = None

            if should_update:

                if scaler is not None:
                    scaler.unscale_(
                        optimizer
                    )

                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_grad_norm,
                    )
                )

                grad_norm_value = float(
                    grad_norm.item()
                    if torch.is_tensor(grad_norm)
                    else grad_norm
                )

                running_grad_norm_sum += (
                    grad_norm_value
                )

                running_grad_norm_max = max(
                    running_grad_norm_max,
                    grad_norm_value,
                )

                if scaler is not None:

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                scheduler.step()

                optimizer_step += 1

                current_lr = float(
                    optimizer.param_groups[0][
                        "lr"
                    ]
                )

                lr_values.append(
                    current_lr
                )

            global_step += 1

            # -------------------------------------------------------------
            # Progress bar
            # -------------------------------------------------------------

            step_time = (
                time.time()
                - batch_start
            )

            progress.set_postfix(
                loss=f"{batch_loss:.4f}",
                pos=f"{running_pos_logit / (batch_idx + 1):.2f}",
                margin=f"{running_margin / (batch_idx + 1):.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                sec=f"{step_time:.2f}",
            )

            # -------------------------------------------------------------
            # Periodic detailed log
            # -------------------------------------------------------------

            log_every = int(
                training_cfg.get(
                    "log_every_n_steps",
                    25,
                )
            )

            if (
                (batch_idx + 1)
                % log_every
                == 0
            ):

                logger.info(
                    f"[E{epoch + 1} "
                    f"B{batch_idx + 1:04d}] "
                    f"loss={batch_loss:.4f} | "
                    f"rank1={rank1.item():.3f} | "
                    f"pos={positive_logits.mean().item():.3f} | "
                    f"maxNeg={max_negative.mean().item():.3f} | "
                    f"margin={margin.mean().item():.3f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    f" | posCos={positive_cosine:.3f}"
                    f" | hardCos={hard_negative_cosine:.3f}"
                    f" | inBatchCos={in_batch_negative_cosine:.3f}"
                    + (
                        f" | grad={grad_norm_value:.3f}"
                        if grad_norm_value is not None
                        else ""
                    )
                )

        progress.close()

        # ------------------------------------------------------------------
        # Epoch training statistics
        # ------------------------------------------------------------------

        epoch_time = (
            time.time()
            - epoch_start
        )

        num_batches = max(
            1,
            len(train_loader),
        )

        train_loss = (
            running_loss
            / num_batches
        )

        loss_variance = max(
            0.0,
            (
                running_loss_sq
                / num_batches
            )
            - train_loss ** 2,
        )

        train_loss_std = math.sqrt(
            loss_variance
        )

        average_rank1 = (
            running_rank1
            / num_batches
        )

        average_pos_logit = (
            running_pos_logit
            / num_batches
        )

        average_max_neg = (
            running_max_neg_logit
            / num_batches
        )

        average_margin = (
            running_margin
            / num_batches
        )

        average_positive_cosine = (
            running_positive_cosine
            / num_batches
        )

        average_hard_negative_cosine = (
            running_hard_negative_cosine
            / num_batches
        )

        average_in_batch_negative_cosine = (
            running_in_batch_negative_cosine
            / num_batches
        )

        average_hard_negative_margin = (
            running_hard_negative_margin
            / num_batches
        )

        average_in_batch_margin = (
            running_in_batch_margin
            / num_batches
        )

        average_grad_norm = (
            running_grad_norm_sum
            / max(
                1,
                optimizer_step
                if optimizer_step > 0
                else 1,
            )
        )

        average_lr = (
            float(np.mean(lr_values))
            if lr_values
            else float(
                optimizer.param_groups[0][
                    "lr"
                ]
            )
        )

        examples_per_second = (
            total_examples
            / max(
                epoch_time,
                1e-9,
            )
        )

        # ------------------------------------------------------------------
        # Domain loss diagnostics
        # ------------------------------------------------------------------

        domain_loss_report = {}

        for domain in domains:

            n_batches = domain_batch_count.get(
                domain,
                0,
            )

            domain_loss_report[domain] = (
                domain_loss_sum.get(
                    domain,
                    0.0,
                )
                / max(
                    1,
                    n_batches,
                )
            )

        logger.info(
            "\n"
            + "-" * 90
        )

        logger.info(
            f"EPOCH {epoch + 1} TRAIN SUMMARY"
        )

        logger.info(
            "-" * 90
        )

        logger.info(
            f"Train loss              : "
            f"{train_loss:.5f} ± {train_loss_std:.5f}"
        )

        logger.info(
            f"Contrastive rank@1      : "
            f"{average_rank1:.4f}"
        )

        logger.info(
            f"Positive logit          : "
            f"{average_pos_logit:.4f}"
        )

        logger.info(
            f"Max negative logit      : "
            f"{average_max_neg:.4f}"
        )

        logger.info(
            f"Positive-negative margin: "
            f"{average_margin:.4f}"
        )

        logger.info(
            f"Positive cosine         : "
            f"{average_positive_cosine:.4f}"
        )

        logger.info(
            f"Hard-negative cosine    : "
            f"{average_hard_negative_cosine:.4f}"
        )

        logger.info(
            f"In-batch cosine         : "
            f"{average_in_batch_negative_cosine:.4f}"
        )

        logger.info(
            f"Hard-negative margin    : "
            f"{average_hard_negative_margin:.4f}"
        )

        logger.info(
            f"In-batch margin         : "
            f"{average_in_batch_margin:.4f}"
        )

        logger.info(
            f"Average gradient norm   : "
            f"{average_grad_norm:.4f}"
        )

        logger.info(
            f"Maximum gradient norm   : "
            f"{running_grad_norm_max:.4f}"
        )

        logger.info(
            f"Average learning rate   : "
            f"{average_lr:.3e}"
        )

        logger.info(
            f"Optimizer updates       : "
            f"{optimizer_step}"
        )

        logger.info(
            f"Throughput              : "
            f"{examples_per_second:.2f} examples/s"
        )

        logger.info(
            f"Non-finite batches      : "
            f"{non_finite_batches}"
        )

        logger.info(
            f"Mixed-domain batches    : "
            f"{batches_with_multiple_domains}"
        )

        for domain in domains:

            logger.info(
                f"Train loss [{domain:<18}]: "
                f"{domain_loss_report[domain]:.5f}"
            )

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------

        logger.info(
            "\n"
            + "-" * 90
        )

        logger.info(
            f"EPOCH {epoch + 1} VALIDATION"
        )

        logger.info(
            "-" * 90
        )

        validation = (
            evaluate_dense_validation_per_domain(
                model=model,
                val_samples_by_domain=val_samples_by_domain,
                corpus_by_domain=corpus_by_domain,
                qrels_by_domain=val_qrels_by_domain,
                tokenizer=tokenizer,
                device=device,
                use_amp=use_amp,
                max_query_len=max_query_length,
                max_doc_len=max_doc_length,
                eval_batch_size=int(
                    evaluation_cfg.get(
                        "eval_batch_size",
                        64,
                    )
                ),
                retrieval_depth=int(
                    evaluation_cfg.get(
                        "retrieval_depth",
                        1000,
                    )
                ),
            )
        )

        macro = validation.get(
            "macro_average",
            {},
        )

        val_ndcg = float(
            macro.get(
                "nDCG@10",
                0.0,
            )
        )

        val_recall100 = float(
            macro.get(
                "Recall@100",
                0.0,
            )
        )

        val_recall1000 = float(
            macro.get(
                "Recall@1000",
                0.0,
            )
        )

        val_mrr = float(
            macro.get(
                "MRR",
                0.0,
            )
        )

        logger.info(
            "\n"
            + "=" * 90
        )

        logger.info(
            f"EPOCH {epoch + 1} VALIDATION MACRO"
        )

        logger.info(
            "=" * 90
        )

        logger.info(
            f"nDCG@10               : "
            f"{val_ndcg:.5f}"
        )

        logger.info(
            f"Recall@10             : "
            f"{macro.get('Recall@10', 0.0):.5f}"
        )

        logger.info(
            f"Recall@50             : "
            f"{macro.get('Recall@50', 0.0):.5f}"
        )

        logger.info(
            f"Recall@100            : "
            f"{val_recall100:.5f}"
        )

        logger.info(
            f"Recall@500            : "
            f"{macro.get('Recall@500', 0.0):.5f}"
        )

        logger.info(
            f"Recall@1000           : "
            f"{val_recall1000:.5f}"
        )

        logger.info(
            f"MRR                   : "
            f"{val_mrr:.5f}"
        )

        logger.info(
            f"nDCG domain std       : "
            f"{macro.get('nDCG@10_std', 0.0):.5f}"
        )

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------

        current_score = val_ndcg

        improved = (
            current_score
            > best_score
        )

        epoch_record: Dict[str, Any] = {
            "epoch": epoch + 1,

            "training": {
                "loss": train_loss,
                "loss_std": train_loss_std,
                "rank1": average_rank1,
                "positive_logit": average_pos_logit,
                "max_negative_logit": average_max_neg,
                "positive_negative_margin": average_margin,
                "average_grad_norm": average_grad_norm,
                "max_grad_norm": running_grad_norm_max,
                "average_learning_rate": average_lr,
                "optimizer_steps": optimizer_step,
                "epoch_seconds": epoch_time,
                "examples": total_examples,
                "examples_per_second": examples_per_second,
                "non_finite_batches": non_finite_batches,
                "mixed_domain_batches": batches_with_multiple_domains,
                "domain_loss": domain_loss_report,
            },

            "validation": validation,

            "model_selection": {
                "monitor": monitor_metric,
                "score": current_score,
                "best_score_before_epoch": best_score,
                "improved": improved,
            },
        }

        training_history.append(
            epoch_record
        )

        history_path = (
            output_dir
            / "training_history.json"
        )

        save_json(
            training_history,
            history_path,
        )

        if improved:

            best_score = current_score
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            checkpoint_path = (
                checkpoint_dir
                / "best_model.pt"
            )

            checkpoint_payload = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "model_state_dict": (
                    model.state_dict()
                ),
                "best_score": best_score,

                "val_metrics": validation,

                "dataset_diagnostics": (
                    dataset_report
                ),

                "formatter_diagnostics": (
                    formatter_diag
                ),

                "model_parameters": (
                    parameter_stats
                ),

                "config": config,
            }

            torch.save(
                checkpoint_payload,
                checkpoint_path,
            )

            logger.info(
                f"✓ NEW BEST CHECKPOINT"
            )

            logger.info(
                f"  epoch     = {epoch + 1}"
            )

            logger.info(
                f"  nDCG@10   = {best_score:.5f}"
            )

            logger.info(
                f"  path      = {checkpoint_path}"
            )

            # Salvataggio di un report dedicato al best.
            save_json(
                checkpoint_payload,
                output_dir
                / "best_checkpoint_report.json",
            )

        else:

            epochs_without_improvement += 1

            logger.info(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{patience})"
            )

        # ------------------------------------------------------------------
        # Early stopping
        # ------------------------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            logger.info(
                "\nEarly stopping triggered."
            )

            break

        clear_device_cache(device)

    # =========================================================================
    # Final report
    # =========================================================================

    final_report = {
        "status": "completed",

        "best_epoch": best_epoch,
        "best_val_ndcg@10": (
            None
            if best_score == -float("inf")
            else best_score
        ),

        "epochs_requested": epochs,
        "epochs_completed": len(
            training_history
        ),

        "domains": domains,

        "model": {
            "name": model_name,
            "pooling": bi_cfg.get(
                "pooling_strategy",
                "cls",
            ),
            "temperature": bi_cfg.get(
                "temperature",
                0.05,
            ),
            "normalize_embeddings": bi_cfg.get(
                "normalize_embeddings",
                True,
            ),
            "lora_enabled": bool(
                lora_cfg.get(
                    "enabled",
                    False,
                )
            ),
            "gradient_checkpointing": (
                gradient_checkpointing
            ),
            "parameters": parameter_stats,
        },

        "data": {
            "query_strategy": query_strategy,
            "max_query_length": max_query_length,
            "max_doc_length": max_doc_length,
            "negatives_per_positive": requested_k,
            "hard_negative_pool_size": int(
                data_cfg.get(
                    "hard_negative_pool_size",
                    50,
                )
            ),
            "negative_sampling_strategy": data_cfg.get(
                "negative_sampling_strategy",
                "bm25_hard",
            ),
            "internal_validation_ratio": (
                internal_val_ratio
            ),
            "dataset_diagnostics": dataset_report,
            "formatter_diagnostics": formatter_diag,
        },

        "training": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": (
                gradient_accumulation_steps
            ),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "max_grad_norm": max_grad_norm,
        },

        "history": training_history,
    }

    save_json(
        final_report,
        output_dir
        / "final_training_report.json",
    )

    logger.info(
        "\n"
        + "=" * 90
    )

    logger.info(
        "TRAINING COMPLETED"
    )

    logger.info(
        "=" * 90
    )

    logger.info(
        f"Best epoch       : {best_epoch}"
    )

    logger.info(
        f"Best val nDCG@10  : {best_score:.5f}"
    )

    logger.info(
        f"History           : "
        f"{output_dir / 'training_history.json'}"
    )

    logger.info(
        f"Final report      : "
        f"{output_dir / 'final_training_report.json'}"
    )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "RETECO Sub-track 2a "
            "diagnostic training"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    )

    if not config_path.exists():

        config_path = (
            Path("config")
            / config_path.name
        )

    if not config_path.exists():

        raise FileNotFoundError(
            f"Config non trovato: "
            f"{args.config}"
        )

    config = load_config(
        config_path
    )

    # File logger dopo il caricamento della config.
    paths_cfg = config.get(
        "paths",
        {},
    )

    general_cfg = config.get(
        "general",
        {},
    )

    log_dir = Path(
        paths_cfg.get(
            "log_dir",
            "outputs/subtrack_2a/logs",
        )
    )

    run_tag = general_cfg.get(
        "run_tag",
        "reteco_2a",
    )

    log_level = general_cfg.get(
        "logging_level",
        "INFO",
    )

    global logger

    try:

        logger = setup_logger(
            log_dir=log_dir,
            run_tag=f"train_{run_tag}",
            log_level=log_level,
        )

    except Exception:

        pass

    train_bi_encoder(
        config
    )


if __name__ == "__main__":
    main()