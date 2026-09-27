#!/usr/bin/env python3
"""
subtrack_2a/train.py

Training Pipeline Ufficiale per RETECO Sub-track 2a.
Caratteristiche:
  - Model Selection basato su 'val_ndcg@10' su validazione interna (Held-out conversation split).
  - InfoNCE con K Hard Negatives (default 4) e in-batch negatives.
  - Context-Aware Truncation condiviso con l'inferenza.
  - AMP FP16 e Gradient Accumulation per GPU T4 16GB.
  - Supporto al training sia del Bi-Encoder che del Cross-Encoder.
"""

import os
import sys
import time
import math
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset.dataset import (
    TRACK2_DOMAINS,
    ConversationalTurnSample,
    ContextAwareQueryFormatter,
    RETECO2aTrainDataset,
    ConversationalCollateFn,
    DomainBalancedBatchSampler,
    load_track2_domain_data,
    split_conversations_train_val,
    mine_bm25_hard_negatives,
)
from models.model import ConversationalBiEncoder, ConversationalCrossEncoder
from utils.utils import (
    load_config,
    save_json,
    compute_official_ndcg,
    set_seed,
    setup_logger,
)

logger = logging.getLogger("RETECO_Train")

# ==============================================================================
# 1. Routine di Validazione Interna Basata su Retrieval (nDCG@10, Recall, MRR)
# ==============================================================================

def evaluate_retrieval_validation(
    model: ConversationalBiEncoder,
    val_samples: List[ConversationalTurnSample],
    corpus: Dict[str, str],
    tokenizer: AutoTokenizer,
    device: torch.device,
    use_amp: bool = True,
    max_len: int = 256,
) -> Dict[str, float]:
    """Valuta il retrieval denso sulla split di validazione interna per calcolare nDCG@10 reale."""
    model.eval()
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]

    # Codifica corpus di validazione a batch
    all_doc_embs = []
    with torch.no_grad():
        for i in range(0, len(doc_texts), 128):
            batch = doc_texts[i : i + 128]
            tok = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                embs = model.encode(tok["input_ids"], tok["attention_mask"])
            all_doc_embs.append(embs.cpu())
    corpus_tensor = torch.cat(all_doc_embs, dim=0)

    # Codifica query di validazione
    val_queries = [s.contextual_query for s in val_samples]
    all_q_embs = []
    with torch.no_grad():
        for i in range(0, len(val_queries), 128):
            batch = val_queries[i : i + 128]
            tok = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                embs = model.encode(tok["input_ids"], tok["attention_mask"])
            all_q_embs.append(embs.cpu())
    query_tensor = torch.cat(all_q_embs, dim=0)

    # Dot-product scoring
    sim_scores = torch.matmul(query_tensor, corpus_tensor.T).numpy()

    # Costruzione dizionari run e qrels
    run_dict: Dict[str, Dict[str, float]] = {}
    qrels_dict: Dict[str, Dict[str, int]] = {}

    for idx, s in enumerate(val_samples):
        ranked_idxs = np.argsort(-sim_scores[idx])[:100]
        run_dict[s.topic_id] = {doc_ids[i]: float(sim_scores[idx][i]) for i in ranked_idxs}
        qrels_dict[s.topic_id] = {gid: 1 for gid in s.gold_doc_ids}

    # Calcolo metriche via pytrec_eval
    metrics = compute_official_ndcg(qrels_dict, run_dict, cutoff=10)
    
    # Calcolo Recall e MRR manuale
    recalls = {10: [], 50: [], 100: []}
    mrrs = []
    for s in val_samples:
        ranked = list(run_dict[s.topic_id].keys())
        golds = set(s.gold_doc_ids)
        for k in [10, 50, 100]:
            hits = len(set(ranked[:k]).intersection(golds))
            recalls[k].append(hits / max(1, len(golds)))
        
        # MRR
        rr = 0.0
        for rank_idx, doc_id in enumerate(ranked, 1):
            if doc_id in golds:
                rr = 1.0 / rank_idx
                break
        mrrs.append(rr)

    return {
        "val_ndcg@10": metrics.get("ndcg_cut_10", 0.0),
        "Recall@10": float(np.mean(recalls[10])),
        "Recall@50": float(np.mean(recalls[50])),
        "Recall@100": float(np.mean(recalls[100])),
        "MRR": float(np.mean(mrrs)),
    }


# ==============================================================================
# 2. Pipeline Principale di Addestramento Bi-Encoder
# ==============================================================================

def train_bi_encoder(config: Dict[str, Any]):
    gen_cfg = config.get("general", {})
    paths_cfg = config.get("paths", {})
    dom_cfg = config.get("domains", {})
    data_cfg = config.get("data", {})
    bi_cfg = config.get("bi_encoder", {})
    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    seed = gen_cfg.get("seed", 42)
    set_seed(seed)

    target_dev = gen_cfg.get("device", "auto").lower()
    if target_dev == "cpu":
        device = torch.device("cpu")
    elif target_dev == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif target_dev == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        # Modalità 'auto'
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    use_amp = bool(gen_cfg.get("mixed_precision", True) and device.type == "cuda")
    logger.info(f"Avvio Training Bi-Encoder su Device: {device} | AMP FP16: {use_amp}")

    base_data_dir = Path(paths_cfg.get("full_data_dir" if paths_cfg.get("data_mode") == "full" else "sample_data_dir"))
    active_domains_cfg = dom_cfg.get("active_domains", "all")
    domains = TRACK2_DOMAINS if active_domains_cfg == "all" else active_domains_cfg
    logger.info(f"Domini di addestramento: {domains}")

    # Tokenizer
    model_name = bi_cfg.get("model_name_or_path", "BAAI/bge-base-en-v1.5")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Query Formatter con Context-Aware Truncation
    q_instruction = bi_cfg.get("query_instruction", {}).get("text", "") if bi_cfg.get("query_instruction", {}).get("enabled", False) else ""
    formatter = ContextAwareQueryFormatter(
        tokenizer=tokenizer,
        max_query_length=data_cfg.get("max_query_length", 256),
        query_instruction=q_instruction,
        strategy=data_cfg.get("query_strategy", "budget_context"),
    )

    # Caricamento e aggregazione dati
    all_train_samples: List[ConversationalTurnSample] = []
    all_val_samples: List[ConversationalTurnSample] = []
    combined_corpus: Dict[str, str] = {}
    combined_hard_negs: Dict[str, List[str]] = {}
    hard_negs_cache_dir = Path(paths_cfg.get("hard_negatives_cache_dir", "data/cache/hard_negatives"))

    for d in domains:
        corpus, samples, _ = load_track2_domain_data(base_data_dir, d, split="train", formatter=formatter)
        for doc_id, text in corpus.items():
            combined_corpus[f"{d}_{doc_id}"] = text
        for s in samples:
            s.gold_doc_ids = [f"{d}_{gid}" for gid in s.gold_doc_ids]

        tr_s, val_s = split_conversations_train_val(samples, val_ratio=data_cfg.get("val_ratio", 0.15), seed=seed)
        all_train_samples.extend(tr_s)
        all_val_samples.extend(val_s)

        # Mining Negativi BM25
        negs = mine_bm25_hard_negatives(
            corpus=corpus,
            samples=samples,
            top_k=20,
            domain_prefix=d,
            cache_dir=hard_negs_cache_dir,
        )
        combined_hard_negs.update(negs)

    logger.info(f"Dataset pronto: {len(all_train_samples)} turni di train | {len(all_val_samples)} turni di internal validation")

    # Stampa statistiche diagnostiche sui token della query
    diag = formatter.get_diagnostics()
    logger.info(f"Diagnostica Query Truncation: Troncatura = {diag['pct_truncated']:.1f}% | Domande oltre budget = {diag['pct_question_exceeded']:.1f}% | Token Medi = {diag['avg_query_tokens']:.1f}")

    # Datasets e DataLoaders
    k_negs = data_cfg.get("negatives_per_positive", 4)
    train_dataset = RETECO2aTrainDataset(
        samples=all_train_samples,
        corpus=combined_corpus,
        negatives_per_positive=k_negs,
        hard_negatives=combined_hard_negs,
        sampling_strategy=data_cfg.get("negative_sampling_strategy", "bm25_hard"),
        seed=seed,
    )

    collate_fn = ConversationalCollateFn(
        tokenizer=tokenizer,
        max_query_len=data_cfg.get("max_query_length", 256),
        max_doc_len=data_cfg.get("max_doc_length", 256),
    )

    if dom_cfg.get("domain_balanced_training", True):
        batch_sampler = DomainBalancedBatchSampler(
            samples=all_train_samples,
            batch_size=train_cfg.get("batch_size", 8),
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=data_cfg.get("num_workers", 2),
            pin_memory=data_cfg.get("pin_memory", True),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_cfg.get("batch_size", 8),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=data_cfg.get("num_workers", 2),
            pin_memory=data_cfg.get("pin_memory", True),
        )

    # Inizializzazione Modello
    model = ConversationalBiEncoder(
        model_name_or_path=model_name,
        temperature=bi_cfg.get("temperature", 0.05),
        normalize_embeddings=bi_cfg.get("normalize_embeddings", True),
        pooling_strategy=bi_cfg.get("pooling_strategy", "mean"),
        lora_cfg=lora_cfg,
        device=device,
    ).to(device)

    # Optimizer e Scheduler
    epochs = train_cfg.get("epochs", 3)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-4)), weight_decay=train_cfg.get("weight_decay", 0.01))
    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Selezione metrica per early stopping
    monitor_metric = train_cfg.get("monitor_metric", "val_ndcg@10")
    logger.info(f"Model Selection configurato su: '{monitor_metric}'")

    best_score = -float("inf") if "ndcg" in monitor_metric or "Recall" in monitor_metric or "MRR" in monitor_metric else float("inf")
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints/subtrack_2a/bi_encoder"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Loop di Training
    for epoch in range(epochs):
        model.train()
        train_dataset.set_epoch(epoch)
        total_loss = 0.0
        start_epoch = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            q_inputs = {k: v.to(device) for k, v in batch["query_inputs"].items()}
            pos_inputs = {k: v.to(device) for k, v in batch["pos_inputs"].items()}
            neg_inputs = {k: v.to(device) for k, v in batch["neg_inputs"].items()}

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(
                    query_inputs=q_inputs,
                    pos_inputs=pos_inputs,
                    neg_inputs=neg_inputs,
                    k_negs=batch["k_negs"],
                )
                loss = out["loss"] / grad_accum

            scaler.scale(loss).backward()
            total_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        avg_train_loss = total_loss / len(train_loader)

        # Validazione a fine epoca basata su Retrieval Reale
        val_metrics = evaluate_retrieval_validation(
            model=model,
            val_samples=all_val_samples,
            corpus=combined_corpus,
            tokenizer=tokenizer,
            device=device,
            use_amp=use_amp,
            max_len=data_cfg.get("max_doc_length", 256),
        )

        current_val_score = val_metrics.get(monitor_metric, val_metrics.get("val_ndcg@10", 0.0))
        logger.info(
            f"Epoch {epoch+1:02d}/{epochs:02d} [{time.time() - start_epoch:.1f}s] - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"val_ndcg@10: {val_metrics['val_ndcg@10']:.4f} | "
            f"Recall@10: {val_metrics['Recall@10']:.4f} | "
            f"Recall@50: {val_metrics['Recall@50']:.4f} | "
            f"MRR: {val_metrics['MRR']:.4f}"
        )

        # Controllo Best Checkpoint
        is_improved = current_val_score > best_score if "loss" not in monitor_metric else current_val_score < best_score
        if is_improved:
            best_score = current_val_score
            save_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "monitor_metric": monitor_metric,
                    "best_score": best_score,
                    "val_metrics": val_metrics,
                },
                save_path,
            )
            if lora_cfg.get("enabled", False):
                model.encoder.save_pretrained(checkpoint_dir / "best_hf_model")
            logger.info(f"✓ Nuovo Best Checkpoint salvato ({monitor_metric} = {best_score:.4f}) in: {save_path}")

    logger.info("Training del Bi-Encoder completato con successo.")


def main():
    parser = argparse.ArgumentParser(description="Training Pipeline SemEval Sub-track 2a")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config") / Path(args.config).name
    config = load_config(cfg_path)

    paths_cfg = config.get("paths", {})
    gen_cfg = config.get("general", {})
    
    # Ora i percorsi reali esistono
    log_dir = Path(paths_cfg.get("log_dir", "outputs/subtrack_2a/logs"))
    run_tag = gen_cfg.get("run_tag", "train_run")
    log_level = gen_cfg.get("logging_level", "INFO")

    # Inizializza handler console + file
    global logger
    logger = setup_logger(log_dir=log_dir, run_tag=run_tag, log_level=log_level)

    train_bi_encoder(config)


if __name__ == "__main__":
    main()