#!/usr/bin/env python3
"""
subtrack_2a/analyse_data.py

Strumento diagnostico ed esplorativo completo per SemEval-2027 Track 2 (RECOR):
  - Livello 1: Dataset Overview (Corpus, turni, qrels per tutti gli 11 domini).
  - Livello 2: Dinamiche Conversazionali (Token reali, crescita cronologia T1-T5+, overlap lessicale).
  - Livello 3: Retrieval & Complementarità Multi-Dominio:
      * Dense Recall@100 vs BM25 Recall@100 vs RRF vs UNION RECALL.
      * Matrice di Attribuzione Gold: Both, Dense-only, BM25-only, Neither.
      * Performance di Retrieval stratificate per Turn Depth (T1, T2, T3, T4, T5+).
"""

import os
import re
import sys
import json
import math
import logging
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
import torch
import pytrec_eval

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACK2_DOMAINS = [
    "biology",
    "drones",
    "earth_science",
    "economics",
    "hardware",
    "law",
    "medicalsciences",
    "politics",
    "psychology",
    "robotics",
    "sustainable_living",
]

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)-8s] %(message)s")
logger = logging.getLogger("RETECO_EDA")


# ==============================================================================
# Tokenizer e Parser Safe (Zero Warning)
# ==============================================================================

def get_bge_tokenizer():
    """Inizializza il tokenizer BGE rimuovendo il warning di lunghezza per l'EDA."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
        # Disattiva il warning (586 > 512): in EDA misuriamo la lunghezza reale
        tok.model_max_length = int(1e9)
        return tok
    except Exception as e:
        logger.warning(f"Tokenizer BGE non caricabile ({e}). Fallback su regex.")
        class FallbackTokenizer:
            def tokenize(self, text: str) -> List[str]:
                return re.findall(r"\w+", text.lower())
        return FallbackTokenizer()


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        if content.startswith("["):
            return json.loads(content)
        return [json.loads(line) for line in content.splitlines() if line.strip()]


def clean_words(text: str) -> Set[str]:
    stopwords = {
        "i", "me", "my", "we", "our", "you", "your", "he", "him", "she", "her", "it", "its", "they", "them",
        "what", "which", "who", "this", "that", "am", "is", "are", "was", "were", "be", "been", "have", "has",
        "had", "do", "does", "did", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "between", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "in", "out", "on", "off", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should"
    }
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {t for t in tokens if t not in stopwords}


# ==============================================================================
# LIVELLO 1 & 2: Dataset Overview & Dinamiche Conversazionali
# ==============================================================================

def run_dataset_and_conversational_analysis(base_dir: Path, tokenizer: Any) -> Dict[str, Any]:
    logger.info("Elaborazione Livello 1 (Overview) e Livello 2 (Conversational Analysis)...")

    results = {
        "domains": {},
        "token_lengths": {"query": [], "history": [], "contextual": [], "docs_sample": []},
        "turn_depth_stats": defaultdict(lambda: {"count": 0, "hist_tokens": [], "q_tokens": [], "gold_counts": []}),
        "lexical_overlap": {"current_to_gold": [], "history_to_gold": [], "combined_to_gold": []},
        "total_stats": {
            "total_docs": 0, "train_convs": 0, "dev_convs": 0,
            "train_turns": 0, "dev_turns": 0, "qrels_train": 0, "qrels_dev": 0,
            "turns_with_history": 0, "total_turns_analyzed": 0
        }
    }

    for domain in TRACK2_DOMAINS:
        dom_dir = base_dir / domain
        if not dom_dir.exists():
            continue

        docs_file = dom_dir / "documents.jsonl"
        num_docs = 0
        doc_sample_dict = {}
        if docs_file.exists():
            with open(docs_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    num_docs += 1
                    if idx % 20 == 0 and line.strip():
                        item = json.loads(line)
                        did = str(item.get("doc_id") or item.get("id"))
                        txt = (item.get("content") or item.get("text") or "").strip()
                        doc_sample_dict[did] = txt
                        results["token_lengths"]["docs_sample"].append(len(tokenizer.tokenize(txt)))

        results["total_stats"]["total_docs"] += num_docs

        train_convs = load_json_or_jsonl(dom_dir / "benchmark_train.json")
        dev_convs = load_json_or_jsonl(dom_dir / "benchmark_dev.json")

        results["total_stats"]["train_convs"] += len(train_convs)
        results["total_stats"]["dev_convs"] += len(dev_convs)

        qrels_tr = sum(1 for _ in open(dom_dir / "qrels_train.txt")) if (dom_dir / "qrels_train.txt").exists() else 0
        qrels_dv = sum(1 for _ in open(dom_dir / "qrels_dev.txt")) if (dom_dir / "qrels_dev.txt").exists() else 0
        results["total_stats"]["qrels_train"] += qrels_tr
        results["total_stats"]["qrels_dev"] += qrels_dv

        dom_train_turns = sum(len(c.get("turns", [])) for c in train_convs)
        dom_dev_turns = 0

        for conv in dev_convs:
            turns = conv.get("turns", [])
            dom_dev_turns += len(turns)

            for turn in turns:
                results["total_stats"]["total_turns_analyzed"] += 1
                t_depth = turn.get("turn_id", 1)
                depth_key = f"T{t_depth}" if t_depth < 5 else "T5+"

                q_txt = turn.get("query", "").strip()
                h_txt = turn.get("conversation_history", "").strip()
                if h_txt.lower() == "no previous conversation.":
                    h_txt = ""

                g_ids = turn.get("gold_doc_ids", [])

                q_tok_len = len(tokenizer.tokenize(q_txt))
                h_tok_len = len(tokenizer.tokenize(h_txt)) if h_txt else 0

                results["token_lengths"]["query"].append(q_tok_len)
                results["token_lengths"]["history"].append(h_tok_len)
                results["token_lengths"]["contextual"].append(q_tok_len + h_tok_len)

                depth_entry = results["turn_depth_stats"][depth_key]
                depth_entry["count"] += 1
                depth_entry["q_tokens"].append(q_tok_len)
                depth_entry["hist_tokens"].append(h_tok_len)
                depth_entry["gold_counts"].append(len(g_ids))

                if h_txt:
                    results["total_stats"]["turns_with_history"] += 1

                if g_ids and docs_file.exists():
                    gold_texts = [doc_sample_dict[gid] for gid in g_ids if gid in doc_sample_dict]
                    if gold_texts:
                        gold_words = clean_words(" ".join(gold_texts))
                        q_words = clean_words(q_txt)
                        h_words = clean_words(h_txt)
                        if gold_words:
                            results["lexical_overlap"]["current_to_gold"].append(len(q_words & gold_words) / len(gold_words))
                            results["lexical_overlap"]["history_to_gold"].append(len(h_words & gold_words) / len(gold_words))
                            results["lexical_overlap"]["combined_to_gold"].append(len((q_words | h_words) & gold_words) / len(gold_words))

        results["total_stats"]["train_turns"] += dom_train_turns
        results["total_stats"]["dev_turns"] += dom_dev_turns
        results["domains"][domain] = {
            "docs": num_docs, "train_convs": len(train_convs), "dev_convs": len(dev_convs),
            "train_turns": dom_train_turns, "dev_turns": dom_dev_turns,
            "qrels_train": qrels_tr, "qrels_dev": qrels_dv,
        }

    return results


# ==============================================================================
# LIVELLO 3: Retrieval, Union Recall & Turn-Depth Performance (11 Domini)
# ==============================================================================

def run_retrieval_and_turn_depth_analysis(base_dir: Path, cache_dir: Path) -> Dict[str, Any]:
    logger.info("Elaborazione Livello 3: Candidate Pool, Union Recall e Turn-Depth su tutti gli 11 domini...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bge_tok = get_bge_tokenizer()

    # BM25 Stemming leggero
    from dataset.dataset import OfficialCompatibleBM25, load_track2_domain_data, ContextAwareQueryFormatter
    from models.model import ConversationalBiEncoder

    formatter = ContextAwareQueryFormatter(tokenizer=bge_tok, max_query_length=256, strategy="budget_context")

    # Inizializza Bi-Encoder
    bi_encoder = ConversationalBiEncoder("BAAI/bge-base-en-v1.5", pooling_strategy="mean").to(device)
    ckpt_path = Path("checkpoints/subtrack_2a/bi_encoder/best_model.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        bi_encoder.load_state_dict(ckpt["model_state_dict"], strict=False)
    bi_encoder.eval()

    domain_retrieval_results = {}
    turn_depth_metrics = defaultdict(lambda: {"dense_ndcg": [], "bm25_ndcg": [], "rrf_ndcg": [],
                                              "dense_r100": [], "bm25_r100": [], "union_r100": [], "count": 0})

    macro_attribution = {"total_golds": 0, "both": 0, "dense_only": 0, "bm25_only": 0, "neither": 0}

    for domain in TRACK2_DOMAINS:
        try:
            corpus, samples, qrels = load_track2_domain_data(base_dir, domain, split="dev", formatter=formatter)
        except Exception:
            continue

        if not samples or not qrels:
            continue

        doc_ids = list(corpus.keys())

        # 1. Recupero Dense (da Cache su disco)
        cache_files = list(cache_dir.glob(f"embs_{domain}_*.pt"))
        if not cache_files:
            continue
        cached_data = torch.load(cache_files[0], map_location="cpu", weights_only=True)
        corpus_embs = cached_data["embeddings"]
        cached_doc_ids = cached_data["doc_ids"]

        # Codifica query dev
        all_q_embs = []
        prefix = "Represent this sentence for searching relevant passages: "
        with torch.no_grad():
            for i in range(0, len(samples), 64):
                batch_q = [prefix + s.contextual_query for s in samples[i : i + 64]]
                tok = bge_tok(batch_q, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
                embs = bi_encoder.encode(tok["input_ids"], tok["attention_mask"]).cpu()
                all_q_embs.append(embs)
        q_embs = torch.cat(all_q_embs, dim=0)

        sim_mat = torch.matmul(q_embs, corpus_embs.T).numpy()

        # Dense Top-100
        dense_run = {}
        for q_idx, s in enumerate(samples):
            top_idx = np.argsort(-sim_mat[q_idx])[:100]
            dense_run[s.topic_id] = [(cached_doc_ids[idx], float(sim_mat[q_idx][idx])) for idx in top_idx]

        # 2. BM25 Top-100
        bm25 = OfficialCompatibleBM25(corpus, k1=0.9, b=0.4)
        bm25_run = {}
        for s in samples:
            raw_bm25_q = f"{s.history} {s.query}".strip()
            bm25_run[s.topic_id] = bm25.get_top_k(raw_bm25_q, top_k=100)

        # 3. RRF Top-100
        from utils.utils import reciprocal_rank_fusion
        rrf_run = reciprocal_rank_fusion([dense_run, bm25_run], k=30, top_n=100)

        # 4. Calcolo Union Recall & Attribuzione dei Gold (Both / Dense-only / BM25-only / Neither)
        dom_golds = 0
        dom_both = 0
        dom_dense_only = 0
        dom_bm25_only = 0
        dom_neither = 0

        dense_recalls_100 = []
        bm25_recalls_100 = []
        union_recalls_100 = []
        rrf_recalls_100 = []

        # Valutazione per Turn Depth
        dense_trec = {s.topic_id: {d: sc for d, sc in dense_run[s.topic_id][:10]} for s in samples}
        bm25_trec = {s.topic_id: {d: sc for d, sc in bm25_run[s.topic_id][:10]} for s in samples}
        rrf_trec = {s.topic_id: {d: sc for d, sc in rrf_run[s.topic_id][:10]} for s in samples}

        eval_dense = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"}).evaluate(dense_trec)
        eval_bm25 = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"}).evaluate(bm25_trec)
        eval_rrf = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"}).evaluate(rrf_trec)

        for s in samples:
            tid = s.topic_id
            t_depth = f"T{s.turn_id}" if s.turn_id < 5 else "T5+"
            golds = set(qrels.get(tid, {}).keys())
            if not golds:
                continue

            dom_golds += len(golds)
            dense_cand_ids = set(d for d, _ in dense_run[tid])
            bm25_cand_ids = set(d for d, _ in bm25_run[tid])
            union_cand_ids = dense_cand_ids | bm25_cand_ids
            rrf_cand_ids = set(d for d, _ in rrf_run[tid])

            dense_recalls_100.append(len(dense_cand_ids & golds) / len(golds))
            bm25_recalls_100.append(len(bm25_cand_ids & golds) / len(golds))
            union_recalls_100.append(len(union_cand_ids & golds) / len(golds))
            rrf_recalls_100.append(len(rrf_cand_ids & golds) / len(golds))

            # Attribuzione specifica per ogni documento gold
            for gid in golds:
                in_d = gid in dense_cand_ids
                in_b = gid in bm25_cand_ids
                if in_d and in_b:
                    dom_both += 1
                elif in_d and not in_b:
                    dom_dense_only += 1
                elif not in_d and in_b:
                    dom_bm25_only += 1
                else:
                    dom_neither += 1

            # Stratificazione per Turn Depth
            t_stat = turn_depth_metrics[t_depth]
            t_stat["count"] += 1
            t_stat["dense_ndcg"].append(eval_dense.get(tid, {}).get("ndcg_cut_10", 0.0))
            t_stat["bm25_ndcg"].append(eval_bm25.get(tid, {}).get("ndcg_cut_10", 0.0))
            t_stat["rrf_ndcg"].append(eval_rrf.get(tid, {}).get("ndcg_cut_10", 0.0))
            t_stat["dense_r100"].append(len(dense_cand_ids & golds) / len(golds))
            t_stat["bm25_r100"].append(len(bm25_cand_ids & golds) / len(golds))
            t_stat["union_r100"].append(len(union_cand_ids & golds) / len(golds))

        d_r100_mean = float(np.mean(dense_recalls_100))
        b_r100_mean = float(np.mean(bm25_recalls_100))
        u_r100_mean = float(np.mean(union_recalls_100))
        r_r100_mean = float(np.mean(rrf_recalls_100))

        domain_retrieval_results[domain] = {
            "dense_r100": d_r100_mean,
            "bm25_r100": b_r100_mean,
            "union_r100": u_r100_mean,
            "rrf_r100": r_r100_mean,
            "complementarity_delta": u_r100_mean - max(d_r100_mean, b_r100_mean),
            "attribution": {
                "total": dom_golds, "both": dom_both, "dense_only": dom_dense_only,
                "bm25_only": dom_bm25_only, "neither": dom_neither
            }
        }

        macro_attribution["total_golds"] += dom_golds
        macro_attribution["both"] += dom_both
        macro_attribution["dense_only"] += dom_dense_only
        macro_attribution["bm25_only"] += dom_bm25_only
        macro_attribution["neither"] += dom_neither

    return {
        "per_domain": domain_retrieval_results,
        "macro_attribution": macro_attribution,
        "turn_depth_metrics": turn_depth_metrics,
    }


# ==============================================================================
# Generazione Grafici
# ==============================================================================

def generate_diagnostic_plots(data: Dict[str, Any], ret_data: Dict[str, Any], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

    # 1. BGE Token Length Distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lengths = data["token_lengths"]
    ax.hist(lengths["query"], bins=30, alpha=0.7, label=f"Query (Mediana: {int(np.median(lengths['query']))})", color="#1f77b4")
    ax.hist([h for h in lengths["history"] if h > 0], bins=30, alpha=0.6, label=f"History (Mediana: {int(np.median([h for h in lengths['history'] if h > 0])) if lengths['history'] else 0})", color="#ff7f0e")
    ax.axvline(256, color="red", linestyle="--", linewidth=1.5, label="Budget Corrente (256)")
    ax.set_xlabel("Token BGE Reali")
    ax.set_ylabel("Frequenza Turni")
    ax.set_title("Analisi 1: Distribuzione Token Reali (Query vs Cronologia)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(out_dir / "1_bge_token_lengths.png", dpi=300)
    plt.close()

    # 2. History Token Growth per Turn Depth
    fig, ax = plt.subplots(figsize=(7, 4.5))
    depth_keys = sorted(data["turn_depth_stats"].keys())
    avg_hist = [np.mean(data["turn_depth_stats"][k]["hist_tokens"]) for k in depth_keys]
    bars = ax.bar(depth_keys, avg_hist, color="#2ca02c", edgecolor="black", alpha=0.85)
    ax.axhline(256, color="red", linestyle="--", label="Budget 256 token")
    ax.set_xlabel("Profondità di Turno")
    ax.set_ylabel("Media Token Cronologia")
    ax.set_title("Analisi 2: Accumulo della Cronologia per Turn Depth")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    for b in bars:
        ax.annotate(f"{b.get_height():.0f}", xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    plt.savefig(out_dir / "2_history_by_turn_depth.png", dpi=300)
    plt.close()

    # 3. Union Recall vs RRF vs Single Retriever (Tutti gli 11 Domini)
    if ret_data.get("per_domain"):
        fig, ax = plt.subplots(figsize=(11, 4.8))
        doms = sorted(ret_data["per_domain"].keys())
        d_vals = [ret_data["per_domain"][d]["dense_r100"] * 100 for d in doms]
        b_vals = [ret_data["per_domain"][d]["bm25_r100"] * 100 for d in doms]
        u_vals = [ret_data["per_domain"][d]["union_r100"] * 100 for d in doms]
        r_vals = [ret_data["per_domain"][d]["rrf_r100"] * 100 for d in doms]

        x = np.arange(len(doms))
        width = 0.2
        ax.bar(x - 1.5 * width, d_vals, width, label="Dense R@100", color="#1f77b4")
        ax.bar(x - 0.5 * width, b_vals, width, label="BM25 R@100", color="#ff7f0e")
        ax.bar(x + 0.5 * width, r_vals, width, label="RRF R@100", color="#7f7f7f")
        ax.bar(x + 1.5 * width, u_vals, width, label="UNION R@100 (Ceiling)", color="#2ca02c", hatch="//")

        ax.set_ylabel("Recall@100 (%)")
        ax.set_title("Analisi 3: Candidate Pool: Singoli vs RRF vs UNIONE (11 Domini)")
        ax.set_xticks(x)
        ax.set_xticklabels(doms, rotation=35, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        plt.savefig(out_dir / "3_union_vs_rrf_recall.png", dpi=300)
        plt.close()

    # 4. Attribuzione dei Gold (Both, Dense-only, BM25-only, Neither)
    if ret_data.get("macro_attribution"):
        fig, ax = plt.subplots(figsize=(6, 5))
        m = ret_data["macro_attribution"]
        tot = max(1, m["total_golds"])
        labels = [f"Both\n({m['both']/tot*100:.1f}%)", f"Dense Only\n({m['dense_only']/tot*100:.1f}%)",
                  f"BM25 Only\n({m['bm25_only']/tot*100:.1f}%)", f"Persi (Neither)\n({m['neither']/tot*100:.1f}%)"]
        sizes = [m["both"], m["dense_only"], m["bm25_only"], m["neither"]]
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
        ax.pie(sizes, labels=labels, colors=colors, startangle=140, autopct="", wedgeprops=dict(edgecolor="black"))
        ax.set_title("Analisi 4: Attribuzione dei Passaggi Gold (Top-100)")
        plt.savefig(out_dir / "4_gold_attribution_pie.png", dpi=300)
        plt.close()

    # 5. Retrieval Performance per Turn Depth (T1 - T5+)
    if ret_data.get("turn_depth_metrics"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        tds = sorted(ret_data["turn_depth_metrics"].keys())
        d_ndcg = [float(np.mean(ret_data["turn_depth_metrics"][k]["dense_ndcg"])) for k in tds]
        b_ndcg = [float(np.mean(ret_data["turn_depth_metrics"][k]["bm25_ndcg"])) for k in tds]
        r_ndcg = [float(np.mean(ret_data["turn_depth_metrics"][k]["rrf_ndcg"])) for k in tds]

        ax.plot(tds, d_ndcg, marker="o", label="Dense nDCG@10", color="#1f77b4", linewidth=2)
        ax.plot(tds, b_ndcg, marker="s", label="BM25 nDCG@10", color="#ff7f0e", linewidth=2)
        ax.plot(tds, r_ndcg, marker="^", label="RRF nDCG@10", color="#2ca02c", linewidth=2)

        ax.set_xlabel("Profondità di Turno")
        ax.set_ylabel("nDCG@10")
        ax.set_title("Analisi 5: Andamento del Retrieval all'Aumentare della Profondità di Turno")
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.savefig(out_dir / "5_retrieval_by_turn_depth.png", dpi=300)
        plt.close()


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analisi Dati a 3 Livelli SemEval Track 2a")
    parser.add_argument("--data_dir", type=str, default="data/reteco_data/track2_recor")
    parser.add_argument("--cache_dir", type=str, default="data/cache/document_embeddings")
    parser.add_argument("--output_dir", type=str, default="outputs/subtrack_2a")
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.output_dir)

    print("\n" + "=" * 85)
    print(f"{'RETECO SUB-TRACK 2a: DIAGNOSTICA COMPLETA A 3 LIVELLI':^85}")
    print("=" * 85)

    tok = get_bge_tokenizer()
    lvl1_2 = run_dataset_and_conversational_analysis(base_dir, tok)
    lvl3 = run_retrieval_and_turn_depth_analysis(base_dir, cache_dir)

    # -------------------------------------------------------------------------
    # TABELLA 1: Overview
    # -------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print(f"{'LIVELLO 1: PANORAMICA DEL DATASET':^85}")
    print("-" * 85)
    print(f"{'Dominio':<20} | {'Documenti':<10} | {'Conv (Tr/Dv)':<13} | {'Turni (Tr/Dv)':<14} | {'Qrels (Tr/Dv)'}")
    print("-" * 85)
    for dom, d in lvl1_2["domains"].items():
        print(f"{dom:<20} | {d['docs']:<10,d} | {d['train_convs']:>5} / {d['dev_convs']:<5} | {d['train_turns']:>5} / {d['dev_turns']:<6} | {d['qrels_train']:>5} / {d['qrels_dev']:<5}")
    tot = lvl1_2["total_stats"]
    print("-" * 85)
    print(f"{'TOTALE':<20} | {tot['total_docs']:<10,d} | {tot['train_convs']:>5} / {tot['dev_convs']:<5} | {tot['train_turns']:>5} / {tot['dev_turns']:<6} | {tot['qrels_train']:>5} / {tot['qrels_dev']:<5}")

    # -------------------------------------------------------------------------
    # TABELLA 2: Dinamiche Conversazionali
    # -------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print(f"{'LIVELLO 2: DINAMICHE CONVERSAZIONALI & TOKEN REALI':^85}")
    print("-" * 85)
    lens = lvl1_2["token_lengths"]
    h_non_zero = [h for h in lens["history"] if h > 0]
    print(f"Turni con Cronologia Pregressa : {tot['turns_with_history']} / {tot['total_turns_analyzed']} ({(tot['turns_with_history']/max(1, tot['total_turns_analyzed']))*100:.1f}%)")
    print(f"Token BGE Domanda Corrente     : Media {np.mean(lens['query']):.1f} | Mediana {np.median(lens['query']):.0f}")
    print(f"Token BGE Cronologia Pregressa : Media {np.mean(h_non_zero):.1f} | Mediana {np.median(h_non_zero):.0f} | Max {np.max(lens['history'])}")
    print(f"Token BGE Passaggi Corpus      : Media {np.mean(lens['docs_sample']):.1f} | 95° Percentile {np.percentile(lens['docs_sample'], 95):.0f}")

    print("\n" + f"{'Turn Depth':<12} | {'Frequenza':<10} | {'Media Token Query':<18} | {'Media Token History':<20} | {'Media Gold'}")
    print("-" * 85)
    for depth in sorted(lvl1_2["turn_depth_stats"].keys()):
        stat = lvl1_2["turn_depth_stats"][depth]
        print(f"{depth:<12} | {stat['count']:<10} | {np.mean(stat['q_tokens']):<18.1f} | {np.mean(stat['hist_tokens']):<20.1f} | {np.mean(stat['gold_counts']):.2f}")

    # -------------------------------------------------------------------------
    # TABELLA 3: Complementarità e Union Recall (11 Domini)
    # -------------------------------------------------------------------------
    if lvl3.get("per_domain"):
        print("\n" + "-" * 85)
        print(f"{'LIVELLO 3A: CANDIDATE POOL, UNION RECALL & RRF (TOP-100)':^85}")
        print("-" * 85)
        print(f"{'Dominio':<18} | {'Dense R@100':<12} | {'BM25 R@100':<12} | {'RRF R@100':<12} | {'UNION R@100':<13} | {'Δ Union'}")
        print("-" * 85)
        for dom, sc in lvl3["per_domain"].items():
            print(f"{dom:<18} | {sc['dense_r100']*100:>10.1f}% | {sc['bm25_r100']*100:>10.1f}% | {sc['rrf_r100']*100:>10.1f}% | {sc['union_r100']*100:>11.1f}% | {sc['complementarity_delta']*100:>+7.1f}%")
        print("-" * 85)

        # -------------------------------------------------------------------------
        # TABELLA 4: Matrice di Attribuzione Gold Macro
        # -------------------------------------------------------------------------
        m_att = lvl3["macro_attribution"]
        tot_g = max(1, m_att["total_golds"])
        print("\n" + "-" * 85)
        print(f"{'LIVELLO 3B: ATTRIBUZIONE DEI DOCUMENTI GOLD SULL INTERO BENCHMARK':^85}")
        print("-" * 85)
        print(f"Gold Document Totali nel Dev Set : {tot_g}")
        print(f"Recuperati da ENTRAMBI           : {m_att['both']:>5} ({m_att['both']/tot_g*100:5.1f}%)")
        print(f"Recuperati SOLO da DENSE         : {m_att['dense_only']:>5} ({m_att['dense_only']/tot_g*100:5.1f}%)")
        print(f"Recuperati SOLO da BM25          : {m_att['bm25_only']:>5} ({m_att['bm25_only']/tot_g*100:5.1f}%)")
        print(f"PERSI DA ENTRAMBI (Neither)      : {m_att['neither']:>5} ({m_att['neither']/tot_g*100:5.1f}%)")
        print("-" * 85)

        # -------------------------------------------------------------------------
        # TABELLA 5: Stratificazione per Turn Depth
        # -------------------------------------------------------------------------
        print("\n" + "-" * 85)
        print(f"{'LIVELLO 3C: RETRIEVAL ACCURACY PER TURN DEPTH':^85}")
        print("-" * 85)
        print(f"{'Turn Depth':<12} | {'N Turni':<8} | {'Dense nDCG':<12} | {'BM25 nDCG':<12} | {'RRF nDCG':<12} | {'Union R@100'}")
        print("-" * 85)
        for td in sorted(lvl3["turn_depth_metrics"].keys()):
            tm = lvl3["turn_depth_metrics"][td]
            print(f"{td:<12} | {tm['count']:<8} | {np.mean(tm['dense_ndcg']):>10.4f} | {np.mean(tm['bm25_ndcg']):>10.4f} | {np.mean(tm['rrf_ndcg']):>10.4f} | {np.mean(tm['union_r100'])*100:>9.1f}%")
        print("-" * 85)

    # Salvataggio grafici e report JSON
    generate_diagnostic_plots(lvl1_2, lvl3, out_dir / "img")

    json_payload = {
        "overview": lvl1_2["total_stats"],
        "domains": lvl1_2["domains"],
        "retrieval_complementarity": lvl3["per_domain"],
        "macro_attribution": lvl3["macro_attribution"],
        "turn_depth_performance": {k: {m: float(np.mean(v)) for m, v in vals.items() if m != "count"} for k, vals in lvl3["turn_depth_metrics"].items()}
    }
    with open(out_dir / "comprehensive_eda_report.json", "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)
    print(f"\n✓ Report completo esportato in: {out_dir / 'comprehensive_eda_report.json'}")
    print(f"✓ 5 grafici diagnostici salvati in: {out_dir / 'img'}/\n" + "=" * 85)


if __name__ == "__main__":
    main()