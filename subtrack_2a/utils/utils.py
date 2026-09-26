#!/usr/bin/env python3
"""
subtrack_2a/utils/utils.py

Modulo di utilità per SemEval-2027 RETECO Sub-track 2a (Conversational Retrieval).
Fornisce:
  - Caricamento e salvataggio configurazioni (YAML / JSON).
  - Calcolo ufficiale della metrica nDCG@10 tramite pytrec_eval (con fallback in puro Python per Mac Air).
  - Scrittura ed esportazione dei run file a 6 colonne nel formato standard TREC.
  - Validazione formale del run file (controllo duplicati, ordinamento score, coerenza rank).
  - Fusione di graduatorie tramite Reciprocal Rank Fusion (RRF) per pipeline ibride (BM25 + Dense).
  - Tracciamento grafico delle curve di addestramento (Loss, nDCG@10).
"""

import os
import torch
from datetime import datetime
import random
import numpy as np
import sys
import json
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pytrec_eval
except ImportError:
    pytrec_eval = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =========================================================================
# 1. Configurazione e I/O Files
# =========================================================================

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Carica un file di configurazione YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")

    if yaml is None:
        raise ImportError("PyYAML non è installato. Esegui: pip install pyyaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, output_path: Union[str, Path], indent: int = 2) -> None:
    """Salva una struttura dati in formato JSON serializzato."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(file_path: Union[str, Path]) -> Any:
    """Carica in modo sicuro un file JSON."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File non trovato: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================================
# 2. Metriche Ufficiali di Retrieval (nDCG@10)
# =========================================================================

def _compute_ndcg_fallback(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    cutoff: int = 10,
) -> float:
    """
    Calcolo deterministico di nDCG@k in puro Python (equivalente a pytrec_eval ndcg_cut_k).
    Utile per test rapidi sul Mac Air se pytrec_eval o compilatori C non sono disponibili.
    """
    all_ndcg = []

    for topic_id, doc_scores in run.items():
        if topic_id not in qrels:
            continue

        topic_qrels = qrels[topic_id]
        
        # Ordina i documenti estratti per punteggio decrescente
        ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:cutoff]

        # Calcolo DCG@k
        dcg = 0.0
        for rank_idx, (doc_id, _) in enumerate(ranked_docs, start=1):
            rel = topic_qrels.get(doc_id, 0)
            if rel > 0:
                dcg += (math.pow(2, rel) - 1.0) / math.log2(rank_idx + 1.0)

        # Calcolo IDCG@k (Ideal DCG)
        ideal_rels = sorted(topic_qrels.values(), reverse=True)[:cutoff]
        idcg = 0.0
        for rank_idx, rel in enumerate(ideal_rels, start=1):
            if rel > 0:
                idcg += (math.pow(2, rel) - 1.0) / math.log2(rank_idx + 1.0)

        ndcg = (dcg / idcg) if idcg > 0.0 else 0.0
        all_ndcg.append(ndcg)

    return sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0.0


def compute_official_ndcg(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    cutoff: int = 10,
) -> Dict[str, float]:
    """
    Calcola nDCG@k ufficiale[cite: 1, 2].
    Se pytrec_eval è installato, utilizza l'implementazione C ufficiale del benchmark[cite: 1, 2].
    In caso contrario, esegue il fallback in puro Python con avviso[cite: 1].
    """
    if not qrels or not run:
        return {f"ndcg_cut_{cutoff}": 0.0}

    metric_name = f"ndcg_cut_{cutoff}"

    if pytrec_eval is not None:
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{cutoff}"})
        eval_scores = evaluator.evaluate(run)
        
        mean_ndcg = sum(
            query_metrics.get(metric_name, 0.0) for query_metrics in eval_scores.values()
        ) / max(len(eval_scores), 1)

        return {metric_name: mean_ndcg}
    else:
        logger.warning("pytrec_eval non rilevato. Viene utilizzato il calcolo fallback in Python.")
        score = _compute_ndcg_fallback(qrels, run, cutoff=cutoff)
        return {metric_name: score}


# =========================================================================
# 3. Formato TREC a 6 Colonne e Validatore Formale
# =========================================================================

def write_trec_run(
    run_dict: Dict[str, List[Tuple[str, float]]],
    output_path: Union[str, Path],
    run_tag: str = "retrieval_2a",
    max_k: int = 10,
) -> None:
    """
    Esporta i risultati nel formato standard TREC a 6 colonne[cite: 2]:
        <topic_id> Q0 <doc_id> <rank> <score> <tag>
    
    Args:
        run_dict: Dizionario {topic_id: [(doc_id, score), ...]}
        output_path: Percorso del file .trec da salvare
        run_tag: Identificativo del sistema
        max_k: Numero massimo di documenti per topic (standard RETECO: 10)[cite: 2, 8]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for topic_id in sorted(run_dict.keys()):
            # Ordina rigorosamente per punteggio decrescente
            ranked_items = sorted(run_dict[topic_id], key=lambda x: x[1], reverse=True)[:max_k]
            
            seen_docs = set()
            rank = 1
            for doc_id, score in ranked_items:
                if doc_id in seen_docs:
                    continue  # Evita duplicati accidentali dello stesso documento[cite: 2]
                seen_docs.add(doc_id)

                f.write(f"{topic_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")
                rank += 1


def validate_trec_file(
    trec_path: Union[str, Path],
    max_rank: int = 10,
) -> Tuple[bool, List[str]]:
    """
    Replica i controlli di integrità di format_checker.py dello starter kit ufficiale[cite: 1, 2]:
      1. Esattamente 6 colonne per riga.
      2. Seconda colonna sempre 'Q0'.
      3. Rango strettamente sequenziale (1, 2, ..., k).
      4. Punteggi non crescenti al crescere del rango.
      5. Nessun documento duplicato per lo stesso topic.
    """
    trec_path = Path(trec_path)
    if not trec_path.exists():
        return False, [f"File {trec_path} non esistente."]

    errors = []
    lines_per_topic: Dict[str, List[Tuple[str, int, float]]] = {}

    with open(trec_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            if not parts:
                continue

            if len(parts) != 6:
                errors.append(f"Riga {line_num}: attese 6 colonne, trovate {len(parts)}.")
                continue

            topic_id, q0, doc_id, rank_str, score_str, _ = parts

            if q0 != "Q0":
                errors.append(f"Riga {line_num}: colonna 2 deve essere 'Q0', trovato '{q0}'.")

            try:
                rank = int(rank_str)
                score = float(score_str)
            except ValueError:
                errors.append(f"Riga {line_num}: rank o score non numerici ({rank_str}, {score_str}).")
                continue

            if rank < 1 or rank > max_rank:
                errors.append(f"Riga {line_num}: rank {rank} non compreso tra 1 e {max_rank}.")

            lines_per_topic.setdefault(topic_id, []).append((doc_id, rank, score))

    # Controllo coerenza di rango e monotonìa score
    for topic_id, records in lines_per_topic.items():
        seen_docs = set()
        prev_rank = 0
        prev_score = float("inf")

        for doc_id, rank, score in records:
            if doc_id in seen_docs:
                errors.append(f"Topic '{topic_id}': documento duplicato '{doc_id}'.")
            seen_docs.add(doc_id)

            if rank != prev_rank + 1:
                errors.append(f"Topic '{topic_id}': salto di rank tra {prev_rank} e {rank}.")
            
            if score > prev_score:
                errors.append(
                    f"Topic '{topic_id}': score non decrescente (rank {rank} ha {score} > {prev_score})."
                )

            prev_rank = rank
            prev_score = score

    is_valid = len(errors) == 0
    return is_valid, errors


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