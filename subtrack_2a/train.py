#!/usr/bin/env python3
"""
subtrack_2a/train.py

Training Pipeline Ufficiale per RETECO Sub-track 2a:
  - Validazione rigorosa per-dominio con calcolo Macro-Average di nDCG@10.
  - Scheduler con calcolo esatto dei passi (math.ceil).
  - Allineamento perfetto del Sampler su train_dataset.valid_samples.
  - AMP FP16 nativo su CUDA e model selection su 'val_ndcg@10'.
"""

import os
import sys
import time
import math
import logging
import argparse
import collections
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
from models.model import ConversationalBiEncoder
from utils.utils import load_config, compute_official_ndcg

logger = logging.getLogger("RETECO_Train")
if not logger.handlers:
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_ch)
    logger.setLevel(logging.INFO)


# ==============================================================================
# Validazione Per-Dominio Conforme alle Condizioni Ufficiali di Test
# ==============================================================================

def evaluate_retrieval_validation_per_domain(
    model: ConversationalBiEncoder,
    val_samples_by_domain: Dict[str, List[ConversationalTurnSample]],
    corpus_by_domain: Dict[str, Dict[str, str]],
    tokenizer: AutoTokenizer,
    device: torch.device,
    use_amp: bool = True,
    max_len: int = 256,
) -> Dict[str, float]:
    """
    Valuta il retrieval denso per ciascun dominio separatamente,
    rispecchiando le condizioni della valutazione ufficiale.
    """
    model.eval()
    domain_ndcgs = []
    domain_recalls = {10: [], 50: [], 100: []}
    domain_mrrs = []

    for domain, samples in val_samples_by_domain.items():
        if not samples:
            continue
        corpus = corpus_by_domain[domain]
        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did] for did in doc_ids]

        # 1. Codifica del corpus del singolo dominio
        all_doc_embs = []
        with torch.no_grad():
            for i in range(0, len(doc_texts), 128):
                batch = doc_texts[i : i + 128]
                tok = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    embs = model.encode(tok["input_ids"], tok["attention_mask"])
                all_doc_embs.append(embs.cpu())
        corpus_tensor = torch.cat(all_doc_embs, dim=0)

        # 2. Codifica delle query di validazione del singolo dominio
        val_queries = [s.contextual_query for s in samples]
        all_q_embs = []
        with torch.no_grad():
            for i in range(0, len(val_queries), 128):
                batch = val_queries[i : i + 128]
                tok = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    embs = model.encode(tok["input_ids"], tok["attention_mask"])
                all_q_embs.append(embs.cpu())
        query_tensor = torch.cat(all_q_embs, dim=0)

        # 3. Matching matriciale circoscritto al solo dominio
        sim_scores = torch.matmul(query_tensor, corpus_tensor.T).numpy()

        run_dict = {}
        qrels_dict = {}
        for idx, s in enumerate(samples):
            ranked_idxs = np.argsort(-sim_scores[idx])[:100]
            run_dict[s.topic_id] = {doc_ids[i]: float(sim_scores[idx][i]) for i in ranked_idxs}
            qrels_dict[s.topic_id] = {gid: 1 for gid in s.gold_doc_ids}

        dom_res = compute_official_ndcg(qrels_dict, run_dict, cutoff=10)
        domain_ndcgs.append(dom_res.get("ndcg_cut_10", 0.0))

        # Recalls ed MRR di dominio
        for s in samples:
            ranked = list(run_dict[s.topic_id].keys())
            golds = set(s.gold_doc_ids)
            for k in [10, 50, 100]:
                hits = len(set(ranked[:k]).intersection(golds))
                domain_recalls[k].append(hits / max(1, len(golds)))
            rr = 0.0
            for rank_idx, doc_id in enumerate(ranked, 1):
                if doc_id in golds:
                    rr = 1.0 / rank_idx
                    break
            domain_mrrs.append(rr)

    return {
        "val_ndcg@10": float(np.mean(domain_ndcgs)) if domain_ndcgs else 0.0,
        "Recall@10": float(np.mean(domain_recalls[10])) if domain_recalls[10] else 0.0,
        "Recall@50": float(np.mean(domain_recalls[50])) if domain_recalls[50] else 0.0,
        "Recall@100": float(np.mean(domain_recalls[100])) if domain_recalls[100] else 0.0,
        "MRR": float(np.mean(domain_mrrs)) if domain_mrrs else 0.0,
    }


# ==============================================================================
# Pipeline di Addestramento
# ==============================================================================

def train_bi_encoder(config: Dict[str, Any]):
    gen_cfg = config.get("general", {})
    paths_cfg = config.get("paths", {})
    dom_cfg = config.get("domains", {})
    data_cfg = config.get("data", {})
    bi_cfg = config.get("bi_encoder", {})
    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})

    device = torch.device("cuda" if torch.cuda.is_available() and gen_cfg.get("device") != "cpu" else "cpu")
    use_amp = bool(gen_cfg.get("mixed_precision", True) and device.type == "cuda")
    logger.info(f"Avvio Training su Device: {device} | AMP FP16: {use_amp}")

    base_data_dir = Path(paths_cfg.get("full_data_dir" if paths_cfg.get("data_mode") == "full" else "sample_data_dir"))
    domains = TRACK2_DOMAINS if dom_cfg.get("active_domains") == "all" else dom_cfg.get("active_domains")

    tokenizer = AutoTokenizer.from_pretrained(bi_cfg.get("model_name_or_path", "BAAI/bge-base-en-v1.5"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    q_inst = bi_cfg.get("query_instruction", {}).get("text", "") if bi_cfg.get("query_instruction", {}).get("enabled", False) else ""
    formatter = ContextAwareQueryFormatter(
        tokenizer=tokenizer,
        max_query_length=data_cfg.get("max_query_length", 256),
        query_instruction=q_inst,
        strategy=data_cfg.get("query_strategy", "budget_context"),
    )

    all_train_samples = []
    val_samples_by_domain = collections.defaultdict(list)
    corpus_by_domain = {}
    combined_train_corpus = {}
    combined_hard_negs = {}

    hard_negs_cache_dir = Path(paths_cfg.get("hard_negatives_cache_dir", "data/cache/hard_negatives"))

    for d in domains:
        corpus, samples, _ = load_track2_domain_data(base_data_dir, d, split="train", formatter=formatter)
        corpus_by_domain[d] = {f"{d}_{did}": txt for did, txt in corpus.items()}
        combined_train_corpus.update(corpus_by_domain[d])

        for s in samples:
            s.gold_doc_ids = [f"{d}_{gid}" for gid in s.gold_doc_ids]

        tr_s, val_s = split_conversations_train_val(samples, val_ratio=data_cfg.get("val_ratio", 0.15), seed=gen_cfg.get("seed", 42))
        all_train_samples.extend(tr_s)
        val_samples_by_domain[d].extend(val_s)

        negs = mine_bm25_hard_negatives(corpus, samples, top_k=20, domain_prefix=d, cache_dir=hard_negs_cache_dir)
        combined_hard_negs.update(negs)

    # Inizializza Dataset
    train_dataset = RETECO2aTrainDataset(
        samples=all_train_samples,
        corpus=combined_train_corpus,
        negatives_per_positive=data_cfg.get("negatives_per_positive", 4),
        hard_negatives=combined_hard_negs,
        sampling_strategy=data_cfg.get("negative_sampling_strategy", "bm25_hard"),
        seed=gen_cfg.get("seed", 42),
    )

    # Costruisci il Sampler sui campioni validati del Dataset
    batch_size = train_cfg.get("batch_size", 8)
    if dom_cfg.get("domain_balanced_training", True):
        batch_sampler = DomainBalancedBatchSampler(
            samples=train_dataset.valid_samples,
            batch_size=batch_size,
            seed=gen_cfg.get("seed", 42),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=ConversationalCollateFn(tokenizer, data_cfg.get("max_query_length", 256), data_cfg.get("max_doc_length", 256)),
            num_workers=data_cfg.get("num_workers", 2),
            pin_memory=data_cfg.get("pin_memory", True) and device.type == "cuda",
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ConversationalCollateFn(tokenizer, data_cfg.get("max_query_length", 256), data_cfg.get("max_doc_length", 256)),
            num_workers=data_cfg.get("num_workers", 2),
            pin_memory=data_cfg.get("pin_memory", True) and device.type == "cuda",
        )

    diag = formatter.get_diagnostics()
    logger.info(f"Diagnostica Query: Domande Troncate = {diag['pct_question_truncated']:.1f}% | Token Medi Finali = {diag['avg_final_tokens']:.1f}")

    # Modello
    model = ConversationalBiEncoder(
        model_name_or_path=bi_cfg.get("model_name_or_path", "BAAI/bge-base-en-v1.5"),
        temperature=bi_cfg.get("temperature", 0.05),
        normalize_embeddings=bi_cfg.get("normalize_embeddings", True),
        pooling_strategy=bi_cfg.get("pooling_strategy", "mean"),
        lora_cfg=lora_cfg,
        device=device,
    ).to(device)

    # Calcolo esatto dei passi di aggiornamento
    epochs = train_cfg.get("epochs", 3)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 2)
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = updates_per_epoch * epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate", 5e-5)), weight_decay=train_cfg.get("weight_decay", 0.01))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * train_cfg.get("warmup_ratio", 0.1)), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    monitor_metric = train_cfg.get("monitor_metric", "val_ndcg@10")
    best_score = -float("inf")
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints/subtrack_2a/bi_encoder"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_dataset.set_epoch(epoch)
        total_loss = 0.0
        start_time = time.time()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            q_inputs = {k: v.to(device) for k, v in batch["query_inputs"].items()}
            pos_inputs = {k: v.to(device) for k, v in batch["pos_inputs"].items()}
            neg_inputs = {k: v.to(device) for k, v in batch["neg_inputs"].items()}

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(query_inputs=q_inputs, pos_inputs=pos_inputs, neg_inputs=neg_inputs, k_negs=batch["k_negs"])
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

        # Validazione per-dominio
        val_metrics = evaluate_retrieval_validation_per_domain(
            model=model,
            val_samples_by_domain=val_samples_by_domain,
            corpus_by_domain=corpus_by_domain,
            tokenizer=tokenizer,
            device=device,
            use_amp=use_amp,
            max_len=data_cfg.get("max_doc_length", 256),
        )

        current_score = val_metrics.get(monitor_metric, val_metrics.get("val_ndcg@10", 0.0))
        logger.info(
            f"Epoca {epoch+1:02d}/{epochs:02d} [{time.time() - start_time:.1f}s] - "
            f"Train Loss: {total_loss / len(train_loader):.4f} | "
            f"Macro val_ndcg@10: {val_metrics['val_ndcg@10']:.4f} | "
            f"Recall@10: {val_metrics['Recall@10']:.4f} | Recall@50: {val_metrics['Recall@50']:.4f}"
        )

        if current_score > best_score:
            best_score = current_score
            save_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_score": best_score,
                "val_metrics": val_metrics,
            }, save_path)
            if lora_cfg.get("enabled", False):
                model.encoder.save_pretrained(checkpoint_dir / "best_hf_model")
            logger.info(f"✓ Nuovo Best Checkpoint (val_ndcg@10 = {best_score:.4f}) salvato.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config") / Path(args.config).name
    config = load_config(cfg_path)
    train_bi_encoder(config)


if __name__ == "__main__":
    main()