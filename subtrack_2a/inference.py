#!/usr/bin/env python3
"""
subtrack_2a/inference.py

Script di VALUTAZIONE OFFLINE per RETECO Sub-track 2a.
Scopo:
  - Recuperare i documenti per ogni turno dello split con etichette ('dev' o 'train').
  - Confrontare i documenti estratti con i gold labels (qrels_*.txt) tramite pytrec_eval.
  - Stampare a terminale una tabella riassuntiva con nDCG@10 per ciascun dominio e la macro-media.
  - Salvare una bozza dei risultati in formato JSON (evaluation_results.json).
"""

import os
import sys
import time
import math
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer

from dataset.dataset import load_track2_domain_data, TRACK2_DOMAINS
from models.model import ConversationalBiEncoder, ConversationalCrossEncoder
from utils.utils import load_config, save_json, compute_official_ndcg, reciprocal_rank_fusion

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)-8s] %(message)s")
logger = logging.getLogger("RETECO_Eval")


# ==============================================================================
# 1. Motore Lessicale BM25 (Per Retrieval Ibrido)
# ==============================================================================
class SimpleBM25:
    """Calcolo lessicale BM25Okapi conforme ai parametri ufficiali (k1=0.9, b=0.4)."""
    def __init__(self, corpus: Dict[str, str], k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b
        self.doc_ids = list(corpus.keys())
        self.corpus_size = len(self.doc_ids)
        self.doc_len: Dict[str, int] = {}
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: Dict[str, Dict[str, int]] = {}

        total_length = 0
        for doc_id, text in corpus.items():
            tokens = re.findall(r"\b\w+\b", text.lower())
            self.doc_len[doc_id] = len(tokens)
            total_length += len(tokens)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs[doc_id] = tf
            for t in tf.keys():
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = (total_length / self.corpus_size) if self.corpus_size > 0 else 1.0
        self.idf: Dict[str, float] = {
            t: math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
            for t, df in self.doc_freqs.items()
        }

    def get_top_k(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        tokens = re.findall(r"\b\w+\b", query.lower())
        if not tokens:
            return []
        scores: Dict[str, float] = {}
        for token in tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for doc_id in self.doc_ids:
                tf = self.term_freqs[doc_id].get(token, 0)
                if tf > 0:
                    num = tf * (self.k1 + 1.0)
                    den = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[doc_id] / self.avg_doc_len))
                    scores[doc_id] = scores.get(doc_id, 0.0) + (idf_val * (num / den))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


# ==============================================================================
# 2. Caricamento Modelli e Codifica Vettoriale
# ==============================================================================
def load_models(config: Dict[str, Any], device: torch.device):
    """
    Carica i pesi del Bi-Encoder (modello completo, LoRA adapter o base HF)
    e inizializza l'eventuale Cross-Encoder per il re-ranking.
    """
    paths_cfg = config.get("paths", {})
    bi_cfg = config.get("bi_encoder", {})
    cross_cfg = config.get("cross_encoder", {})

    # 1. Risoluzione percorso checkpoint
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints/subtrack_2a"))
    if not checkpoint_dir.exists() and (Path("..") / checkpoint_dir).exists():
        checkpoint_dir = Path("..") / checkpoint_dir

    best_hf_path = checkpoint_dir / "best_hf_model"
    base_model_name = bi_cfg.get("model_name_or_path", "BAAI/bge-base-en-v1.5")

    # 2. Caricamento Tokenizer (predilige la cartella salvata se presente)
    tok_source = str(best_hf_path) if (best_hf_path / "tokenizer_config.json").exists() else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Parametri comuni del Bi-Encoder
    bi_params = {
        "temperature": float(bi_cfg.get("temperature", 0.05)),
        "normalize_embeddings": bool(bi_cfg.get("normalize_embeddings", True)),
        "pooling_strategy": bi_cfg.get("pooling_strategy", "mean"),
    }

    # 3. Caricamento Bi-Encoder
    has_lora = (best_hf_path / "adapter_config.json").exists()
    has_full_weights = (best_hf_path / "model.safetensors").exists() or (best_hf_path / "pytorch_model.bin").exists()

    if has_lora:
        logger.info(f"Caricamento Base Model ({base_model_name}) + LoRA Adapter da {best_hf_path}...")
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("Rilevato adapter LoRA ma 'peft' non è installato. Esegui: pip install peft")

        bi_encoder = ConversationalBiEncoder(
            model_name_or_path=base_model_name,
            **bi_params
        ).to(device)
        
        bi_encoder.encoder = PeftModel.from_pretrained(bi_encoder.encoder, str(best_hf_path))
        # Unione dei pesi LoRA nella backbone per eliminare l'overhead in inferenza
        bi_encoder.encoder = bi_encoder.encoder.merge_and_unload()

    elif has_full_weights:
        logger.info(f"Caricamento Bi-Encoder completo da checkpoint locale: {best_hf_path}")
        bi_encoder = ConversationalBiEncoder(
            model_name_or_path=str(best_hf_path),
            **bi_params
        ).to(device)

    else:
        logger.info(f"Nessun checkpoint valido trovato in {best_hf_path}. Caricamento modello base: {base_model_name}")
        bi_encoder = ConversationalBiEncoder(
            model_name_or_path=base_model_name,
            **bi_params
        ).to(device)

    bi_encoder.eval()

    # 4. Inizializzazione Cross-Encoder (Re-ranking neurale)
    cross_encoder = None
    if cross_cfg.get("enabled", False):
        reranker_name = cross_cfg.get("model_name_or_path", "BAAI/bge-reranker-base")
        logger.info(f"Caricamento Cross-Encoder per Re-ranking: {reranker_name}")
        cross_encoder = ConversationalCrossEncoder(
            model_name_or_path=reranker_name,
            num_labels=int(cross_cfg.get("num_labels", 1)),
        ).to(device)
        cross_encoder.eval()

    return bi_encoder, tokenizer, cross_encoder


def encode_texts(texts: List[str], tokenizer, model, device: torch.device, batch_size: int = 64, max_len: int = 512, use_amp: bool = False) -> torch.Tensor:
    """Codifica a batch vettori densi normalizzati (L2)."""
    model.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                embs = model.encode(inputs["input_ids"], inputs["attention_mask"], inputs.get("token_type_ids"))
            all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


# ==============================================================================
# 3. Pipeline di Valutazione Offline
# ==============================================================================

def evaluate_offline(config: Dict[str, Any], split: str = "dev"):
    gen_cfg = config.get("general", {})
    paths_cfg = config.get("paths", {})
    dom_cfg = config.get("domains", {})
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    sparse_cfg = config.get("sparse", {})
    hybrid_cfg = config.get("hybrid_fusion", {})
    cross_cfg = config.get("cross_encoder", {})

    # Dispositivo
    device_pref = gen_cfg.get("device", "auto")
    if device_pref == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_pref)

    use_amp = bool(gen_cfg.get("mixed_precision", False) and device.type == "cuda")
    logger.info(f"Avvio valutazione offline su device: {device} | Split target: '{split.upper()}'")

    # Percorso dati
    data_mode = paths_cfg.get("data_mode", "full")
    base_data_dir = Path(paths_cfg.get("sample_data_dir" if data_mode == "sample" else "full_data_dir"))

    # Selezione domini
    active_domains_cfg = dom_cfg.get("active_domains", "all")
    domains = TRACK2_DOMAINS if active_domains_cfg == "all" else active_domains_cfg
    logger.info(f"Domini in valutazione ({len(domains)}): {domains}")

    bi_encoder, tokenizer, cross_encoder = load_models(config, device)

    batch_size = eval_cfg.get("eval_batch_size", 32)
    max_doc_len = data_cfg.get("max_doc_length", 128)
    max_q_len = data_cfg.get("max_query_length", 128)
    cutoff_k = eval_cfg.get("cutoff_k", 10)
    top_candidates = hybrid_cfg.get("top_candidates_to_rerank", 50)

    domain_ndcg_scores: Dict[str, float] = {}
    start_total_time = time.time()

    for domain in domains:
        logger.info(f"\n--- Valutazione Dominio: {domain.upper()} ---")
        try:
            corpus, samples, qrels = load_track2_domain_data(
                data_dir=base_data_dir,
                domain=domain,
                split=split,
                query_strategy=data_cfg.get("query_strategy", "concat"),
            )
        except Exception as e:
            logger.warning(f"Salto dominio {domain}: {e}")
            continue

        if not qrels:
            logger.warning(f"File qrels mancante per {domain} nello split '{split}'. Impossibile calcolare nDCG.")
            continue

        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did] for did in doc_ids]

        # 1. Retrieval Denso
        corpus_embs = encode_texts(doc_texts, tokenizer, bi_encoder, device, batch_size, max_doc_len, use_amp)
        queries = [s.contextual_query for s in samples]
        query_embs = encode_texts(queries, tokenizer, bi_encoder, device, batch_size, max_q_len, use_amp)

        scores_matrix = torch.matmul(query_embs, corpus_embs.transpose(0, 1)).numpy()
        dense_run: Dict[str, List[Tuple[str, float]]] = {}
        for q_idx, sample in enumerate(samples):
            q_sc = scores_matrix[q_idx]
            top_idx = np.argsort(-q_sc)[:top_candidates]
            dense_run[sample.topic_id] = [(doc_ids[idx], float(q_sc[idx])) for idx in top_idx]

        # 2. Retrieval Lessicale BM25
        bm25_run: Dict[str, List[Tuple[str, float]]] = {}
        if sparse_cfg.get("enabled", True):
            bm25 = SimpleBM25(corpus, k1=float(sparse_cfg.get("k1", 0.9)), b=float(sparse_cfg.get("b", 0.4)))
            for sample in samples:
                bm25_run[sample.topic_id] = bm25.get_top_k(sample.contextual_query, top_k=top_candidates)

        # 3. Fusione RRF
        if hybrid_cfg.get("enabled", True) and bm25_run:
            candidate_run = reciprocal_rank_fusion([dense_run, bm25_run], k=int(hybrid_cfg.get("rrf_k", 60)), top_n=top_candidates)
        else:
            candidate_run = dense_run

        # 4. Re-ranking Neurale (Opzionale)
        final_run: Dict[str, List[Tuple[str, float]]] = {}
        if cross_cfg.get("enabled", False) and cross_encoder is not None:
            cross_tok = AutoTokenizer.from_pretrained(cross_cfg.get("model_name_or_path", "BAAI/bge-reranker-base"))
            for sample in samples:
                cands = candidate_run.get(sample.topic_id, [])
                if not cands:
                    continue
                q_p = [sample.contextual_query] * len(cands)
                d_p = [corpus[doc_id] for doc_id, _ in cands]
                all_probs = []
                with torch.no_grad():
                    for p_i in range(0, len(q_p), 32):
                        enc = cross_tok(q_p[p_i:p_i+32], d_p[p_i:p_i+32], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                        out = cross_encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                        probs = torch.sigmoid(out["logits"]).cpu().tolist()
                        all_probs.extend(probs if isinstance(probs, list) else [probs])
                reranked = sorted([(cands[idx][0], float(all_probs[idx])) for idx in range(len(cands))], key=lambda x: x[1], reverse=True)
                final_run[sample.topic_id] = reranked[:cutoff_k]
        else:
            for t_id, docs in candidate_run.items():
                final_run[t_id] = docs[:cutoff_k]

        # 5. Calcolo Metrica Ufficiale nDCG@10 tramite pytrec_eval
        trec_format = {t_id: {d_id: sc for d_id, sc in d_list} for t_id, d_list in final_run.items()}
        metrics = compute_official_ndcg(qrels, trec_format, cutoff=cutoff_k)
        score = metrics.get(f"ndcg_cut_{cutoff_k}", 0.0)
        domain_ndcg_scores[domain] = score
        logger.info(f"Risultato {domain}: nDCG@{cutoff_k} = {score:.4f}")

    # ---------------------------------------------------------
    # Stampa a video Report dei Risultati
    # ---------------------------------------------------------
    logger.info("\n" + "=" * 55)
    logger.info(f"TABELLA RISULTATI OFFLINE (SPLIT: {split.upper()})")
    logger.info("=" * 55)
    logger.info(f"{'Dominio':<25} | {'nDCG@10':<10}")
    logger.info("-" * 40)
    for dom, sc in sorted(domain_ndcg_scores.items()):
        logger.info(f"{dom:<25} | {sc:.4f}")

    if domain_ndcg_scores:
        macro_avg = sum(domain_ndcg_scores.values()) / len(domain_ndcg_scores)
        logger.info("-" * 40)
        logger.info(f"{'MACRO-AVERAGE':<25} | {macro_avg:.4f}")
        logger.info("=" * 55)

        # Salvataggio bozza risultati
        out_dir = Path(paths_cfg.get("output_dir", "outputs/subtrack_2a"))
        out_dir.mkdir(parents=True, exist_ok=True)
        results_file = out_dir / f"evaluation_results_{split}.json"
        save_json({"macro_average_ndcg10": macro_avg, "per_domain": domain_ndcg_scores}, results_file)
        logger.info(f"Riepilogo salvato in: {results_file}")

    logger.info(f"Tempo totale di valutazione: {time.time() - start_total_time:.1f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Valutazione diagnostica offline Sub-track 2a")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Percorso al file config.yaml")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "train"], help="Split da valutare (default: dev)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        alt_path = Path("config") / Path(args.config).name
        cfg_path = alt_path if alt_path.exists() else Path("config/config.yaml")

    config = load_config(cfg_path)
    evaluate_offline(config, split=args.split)


if __name__ == "__main__":
    main()