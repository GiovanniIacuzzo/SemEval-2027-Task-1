#!/usr/bin/env python3
"""
subtrack_2a/generate_submission.py

Script UFFICIALE per creare il file di SOTTOMISSIONE per SemEval-2027 Sub-track 2a.
Scopo:
  - Interrogare i turni dello split target (es. 'test' cieco senza gold, oppure 'dev' per prova).
  - Estrarre i migliori 10 passaggi per ciascun turno.
  - Scrivere il file conforme allo standard TREC a 6 colonne:
      <topic_id> Q0 <doc_id> <rank> <score> <tag>
  - Eseguire i controlli bloccanti di validità (assenza duplicati, rank 1..10, score decrescenti).
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
from utils.utils import load_config, write_trec_run, validate_trec_file, reciprocal_rank_fusion

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)-8s] %(message)s")
logger = logging.getLogger("RETECO_Submission")


class SimpleBM25:
    """Motore BM25Okapi standard Lucene (k1=0.9, b=0.4)."""
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


def encode_texts(texts: List[str], tokenizer, model, device: torch.device, batch_size: int = 64, max_len: int = 512, use_amp: bool = False) -> torch.Tensor:
    model.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                embs = model.encode(inputs["input_ids"], inputs["attention_mask"], inputs.get("token_type_ids"))
            all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)


def generate_submission(config: Dict[str, Any], split: str, output_trec_file: Path, run_tag: str):
    gen_cfg = config.get("general", {})
    paths_cfg = config.get("paths", {})
    dom_cfg = config.get("domains", {})
    data_cfg = config.get("data", {})
    sparse_cfg = config.get("sparse", {})
    hybrid_cfg = config.get("hybrid_fusion", {})
    cross_cfg = config.get("cross_encoder", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    use_amp = bool(gen_cfg.get("mixed_precision", False) and device.type == "cuda")

    data_mode = paths_cfg.get("data_mode", "full")
    base_data_dir = Path(paths_cfg.get("sample_data_dir" if data_mode == "sample" else "full_data_dir"))
    active_domains_cfg = dom_cfg.get("active_domains", "all")
    domains = TRACK2_DOMAINS if active_domains_cfg == "all" else active_domains_cfg

    logger.info("=" * 65)
    logger.info("CREAZIONE FILE DI SOTTOMISSIONE PER LA LEADERBOARD")
    logger.info(f"Target Split:   '{split.upper()}' (Nessun bisogno di gold qrels)")
    logger.info(f"Destinazione:   {output_trec_file}")
    logger.info(f"Run Tag:        {run_tag}")
    logger.info(f"Domini ({len(domains)}): {domains}")
    logger.info("=" * 65)

    # Caricamento Bi-Encoder
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints/subtrack_2a"))
    if not checkpoint_dir.exists() and (Path("..") / checkpoint_dir).exists():
        checkpoint_dir = Path("..") / checkpoint_dir
    best_hf_path = checkpoint_dir / "best_hf_model"
    model_source = str(best_hf_path) if best_hf_path.exists() else config.get("bi_encoder", {}).get("model_name_or_path", "BAAI/bge-base-en-v1.5")
    
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    bi_encoder = ConversationalBiEncoder(model_name_or_path=model_source).to(device)
    bi_encoder.eval()

    cross_encoder = None
    if cross_cfg.get("enabled", False):
        reranker_name = cross_cfg.get("model_name_or_path", "BAAI/bge-reranker-base")
        cross_encoder = ConversationalCrossEncoder(model_name_or_path=reranker_name).to(device)
        cross_encoder.eval()

    batch_size = config.get("evaluation", {}).get("eval_batch_size", 32)
    max_doc_len = data_cfg.get("max_doc_length", 128)
    max_q_len = data_cfg.get("max_query_length", 128)
    cutoff_k = 10  # Standard SemEval per la graduatoria finale
    top_candidates = hybrid_cfg.get("top_candidates_to_rerank", 50)

    all_submission_runs: Dict[str, List[Tuple[str, float]]] = {}

    for domain in domains:
        logger.info(f"Elaborazione sottomissione per: {domain.upper()}...")
        try:
            # Qui qrels è None se siamo sul test set, e lo script procede senza problemi
            corpus, samples, _ = load_track2_domain_data(
                data_dir=base_data_dir,
                domain=domain,
                split=split,
                query_strategy=data_cfg.get("query_strategy", "concat"),
            )
        except Exception as e:
            logger.warning(f"Errore caricamento dominio {domain}: {e}")
            continue

        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did] for did in doc_ids]

        # Retrieval Denso
        corpus_embs = encode_texts(doc_texts, tokenizer, bi_encoder, device, batch_size, max_doc_len, use_amp)
        queries = [s.contextual_query for s in samples]
        query_embs = encode_texts(queries, tokenizer, bi_encoder, device, batch_size, max_q_len, use_amp)
        scores_matrix = torch.matmul(query_embs, corpus_embs.transpose(0, 1)).numpy()

        dense_run: Dict[str, List[Tuple[str, float]]] = {}
        for q_idx, sample in enumerate(samples):
            q_sc = scores_matrix[q_idx]
            top_idx = np.argsort(-q_sc)[:top_candidates]
            dense_run[sample.topic_id] = [(doc_ids[idx], float(q_sc[idx])) for idx in top_idx]

        # Retrieval BM25
        bm25_run: Dict[str, List[Tuple[str, float]]] = {}
        if sparse_cfg.get("enabled", True):
            bm25 = SimpleBM25(corpus, k1=float(sparse_cfg.get("k1", 0.9)), b=float(sparse_cfg.get("b", 0.4)))
            for sample in samples:
                bm25_run[sample.topic_id] = bm25.get_top_k(sample.contextual_query, top_k=top_candidates)

        # Fusione RRF
        if hybrid_cfg.get("enabled", True) and bm25_run:
            candidate_run = reciprocal_rank_fusion([dense_run, bm25_run], k=int(hybrid_cfg.get("rrf_k", 60)), top_n=top_candidates)
        else:
            candidate_run = dense_run

        # Re-ranking Neurale
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
                all_submission_runs[sample.topic_id] = reranked[:cutoff_k]
        else:
            for t_id, docs in candidate_run.items():
                all_submission_runs[t_id] = docs[:cutoff_k]

    # Scrittura del file TREC
    output_trec_file.parent.mkdir(parents=True, exist_ok=True)
    write_trec_run(all_submission_runs, output_trec_file, run_tag=run_tag, max_k=cutoff_k)

    # Validazione formale obbligatoria (come da starter kit SemEval format_checker.py)
    is_valid, errors = validate_trec_file(output_trec_file, max_rank=cutoff_k)
    if is_valid:
        num_lines = sum(1 for _ in open(output_trec_file, "r", encoding="utf-8"))
        logger.info("\n" + "=" * 65)
        logger.info("✓ FILE DI SOTTOMISSIONE PRONTO E CONFORME AL 100%!")
        logger.info(f"File generato:   {output_trec_file}")
        logger.info(f"Righe generate:  {num_lines}")
        logger.info(f"Topic coperti:   {len(all_submission_runs)}")
        logger.info("=" * 65 + "\n")
    else:
        logger.error("\n" + "=" * 65)
        logger.error(f"✗ ATTENZIONE: IL FILE PRESENTA {len(errors)} ERRORI FORMALI:")
        for err in errors[:5]:
            logger.error(f"  - {err}")
        logger.error("=" * 65 + "\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generatore ufficiale di sottomissione per la Leaderboard SemEval-2027")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Percorso al file config.yaml")
    parser.add_argument("--split", type=str, default="dev", help="Split target ('test' per la gara, 'dev' per test di consegna)")
    parser.add_argument("--out", type=str, default=None, help="Percorso del file .trec di output")
    parser.add_argument("--tag", type=str, default=None, help="Tag identificativo del team per la sottomissione")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        alt_path = Path("config") / Path(args.config).name
        cfg_path = alt_path if alt_path.exists() else Path("config/config.yaml")

    config = load_config(cfg_path)
    sub_cfg = config.get("submission", {})
    paths_cfg = config.get("paths", {})

    target_split = args.split or sub_cfg.get("target_split", "dev")
    out_file = Path(args.out) if args.out else Path(paths_cfg.get("submission_file", "outputs/subtrack_2a/submission_2a.trec"))
    tag = args.tag or sub_cfg.get("run_tag", "TEAM_2A_RUN")

    generate_submission(config, split=target_split, output_trec_file=out_file, run_tag=tag)


if __name__ == "__main__":
    main()