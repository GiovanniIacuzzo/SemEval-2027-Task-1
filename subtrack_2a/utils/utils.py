#!/usr/bin/env python3
"""
subtrack_2a/utils/utils.py

Funzioni di utilità per SemEval-2027 Sub-track 2a:
  - Valutazione nDCG ufficiale conforme con penalizzazione topic mancanti (score 0.0).
  - Reciprocal Rank Fusion ponderata.
  - Scrittura e validazione formale TREC a 6 colonne.
"""

import os
import sys
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

import pytrec_eval
import numpy as np


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def compute_official_ndcg(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    cutoff: int = 10,
) -> Dict[str, float]:
    """
    Calcola nDCG@K ufficiale tramite pytrec_eval.
    CONFORMITÀ UFFICIALE: Qualsiasi topic presente nei qrels ma assente
    nella run riceve rigorosamente score 0.0.
    """
    if not qrels:
        return {f"ndcg_cut_{cutoff}": 0.0}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut_{cutoff}"})
    # Valuta solo i topic presenti nella run
    raw_scores = evaluator.evaluate(run)

    # I topic mancanti nei qrels ricevono 0.0
    all_ndcg = []
    for topic_id in qrels.keys():
        if topic_id in raw_scores:
            all_ndcg.append(raw_scores[topic_id].get(f"ndcg_cut_{cutoff}", 0.0))
        else:
            all_ndcg.append(0.0)

    mean_score = float(np.mean(all_ndcg)) if all_ndcg else 0.0
    return {f"ndcg_cut_{cutoff}": mean_score}


def reciprocal_rank_fusion(
    runs: List[Dict[str, List[Tuple[str, float]]]],
    k: int = 30,
    weights: Optional[List[float]] = None,
    top_n: int = 100,
) -> Dict[str, List[Tuple[str, float]]]:
    """Reciprocal Rank Fusion ponderata con k ottimizzato."""
    if weights is None:
        weights = [1.0] * len(runs)
    elif len(weights) != len(runs):
        raise ValueError("I pesi devono corrispondere al numero di liste run.")

    all_topics = set()
    for r in runs:
        all_topics.update(r.keys())

    fused_run: Dict[str, List[Tuple[str, float]]] = {}
    for tid in all_topics:
        scores = {}
        for r_idx, r in enumerate(runs):
            w = weights[r_idx]
            q_list = r.get(tid, [])
            for rank, (doc_id, _) in enumerate(q_list, 1):
                scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (k + rank))

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        fused_run[tid] = sorted_docs

    return fused_run


def write_trec_run(
    run_dict: Dict[str, List[Tuple[str, float]]],
    output_path: Path,
    run_tag: str = "reteco_run",
    max_k: int = 10,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for topic_id in sorted(run_dict.keys()):
            doc_scores = run_dict[topic_id][:max_k]
            for rank, (doc_id, score) in enumerate(doc_scores, 1):
                f.write(f"{topic_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")


def validate_trec_file(trec_path: Path, max_rank: int = 10) -> Tuple[bool, List[str]]:
    errors = []
    if not trec_path.exists():
        return False, ["File TREC non esistente."]

    seen_topics = set()
    topic_ranks = {}

    with open(trec_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 6:
                errors.append(f"Riga {idx}: Formato a 6 colonne violato ({len(parts)} colonne).")
                continue
            t_id, _, d_id, rank_str, _, _ = parts
            rank = int(rank_str)
            if rank > max_rank:
                errors.append(f"Riga {idx}: Rango {rank} > max_rank {max_rank}.")

            topic_ranks.setdefault(t_id, []).append(rank)

    for tid, ranks in topic_ranks.items():
        if ranks != list(range(1, len(ranks) + 1)):
            errors.append(f"Topic {tid}: Ranghi non ordinati sequenzialmente 1..{len(ranks)}.")

    return len(errors) == 0, errors


# =========================================================================
# 4. Fusione Risultati: Reciprocal Rank Fusion (RRF)
# =========================================================================

def reciprocal_rank_fusion(
    runs: List[Dict[str, List[Tuple[str, float]]]],
    k: int = 60,
    top_n: int = 100,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Combina graduatorie multiple (es. BM25 e Bi-Encoder denso) con la formula standard RRF:
        RRF_Score(d) = sum( 1 / (k + rank_m(d)) )
    """
    fused_scores: Dict[str, Dict[str, float]] = {}

    for single_run in runs:
        for topic_id, ranked_list in single_run.items():
            fused_scores.setdefault(topic_id, {})
            # Assicura l'ordine corretto
            sorted_docs = sorted(ranked_list, key=lambda x: x[1], reverse=True)
            for rank_idx, (doc_id, _) in enumerate(sorted_docs, start=1):
                fused_scores[topic_id][doc_id] = fused_scores[topic_id].get(doc_id, 0.0) + (1.0 / (k + rank_idx))

    final_run: Dict[str, List[Tuple[str, float]]] = {}
    for topic_id, doc_dict in fused_scores.items():
        sorted_candidates = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        final_run[topic_id] = sorted_candidates

    return final_run


# =========================================================================
# 5. Visualizzazione Grafica Addestramento
# =========================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Union[str, Path],
    title: str = "Training Progress Sub-track 2a",
) -> None:
    """Traccia e salva su disco l'andamento di Loss ed eventuale Validation nDCG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    if not epochs:
        logger.warning("Nessun dato di loss presente nella cronologia per generare il grafico.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=color)
    ax1.plot(epochs, history["train_loss"], color=color, marker="o", linewidth=2, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.5)

    if "val_ndcg" in history and history["val_ndcg"]:
        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("Validation nDCG@10", color=color)
        ax2.plot(epochs, history["val_ndcg"], color=color, marker="s", linewidth=2, label="Val nDCG@10")
        ax2.tick_params(axis="y", labelcolor=color)

    plt.title(title)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Grafico salvato con successo in: {output_path}")

# ==============================================================================
# 5. Configurazione del Logging Professionale
# ==============================================================================

def setup_logger(log_dir: Path, run_tag: str, log_level: str = "INFO") -> logging.Logger:
    """Inizializza un logger con formattazione dettagliata su console e file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{run_tag}_{timestamp}.log"

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("RETECO_2A_Train")
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Pulisce eventuali handler precedenti
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler su Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler su File
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logger inizializzato. File di log: {log_file}")
    return logger


def set_seed(seed: int = 42) -> None:
    """Garantisce la totale riproducibilità numerica su CPU e GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# =========================================================================
# Test Unitario Diretto
# =========================================================================

if __name__ == "__main__":
    print("Inizializzazione test unitario di utils.py...")

    dummy_qrels = {
        "conv_1_turn_1": {"doc_A": 1, "doc_B": 0},
        "conv_1_turn_2": {"doc_C": 1},
    }
    dummy_run = {
        "conv_1_turn_1": {"doc_A": 12.5, "doc_B": 8.0, "doc_X": 2.1},
        "conv_1_turn_2": {"doc_Y": 15.0, "doc_C": 9.2},
    }

    metrics = compute_official_ndcg(dummy_qrels, dummy_run, cutoff=10)
    print(f"✓ Calcolo metrica completato: {metrics}")

    test_trec_file = Path("test_run.trec")
    converted_run = {
        topic: sorted(docs.items(), key=lambda x: x[1], reverse=True)
        for topic, docs in dummy_run.items()
    }
    write_trec_run(converted_run, test_trec_file, run_tag="test_run", max_k=2)

    valid, errs = validate_trec_file(test_trec_file, max_rank=2)
    print(f"✓ Validazione file TREC: {'Corretto' if valid else 'Fallito'}")
    if errs:
        for err in errs:
            print(f"  [Errore trovato]: {err}")

    if test_trec_file.exists():
        test_trec_file.unlink()
    print("Test completato.")