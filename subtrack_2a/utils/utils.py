#!/usr/bin/env python3
"""Utility functions for RETECO SemEval-2027 Sub-track 2a.

Includes safe configuration/JSON I/O, official nDCG@k evaluation, weighted
Reciprocal Rank Fusion, TREC run writing/validation, reproducible seeds,
logging, and optional training-history plotting.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml


# =============================================================================
# Configuration and JSON I/O
# =============================================================================

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML config and ensure its root is a mapping."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config non trovato: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(f"La root del config YAML deve essere una mappa: {path}")
    return config


def _json_safe(obj: Any) -> Any:
    """Convert common scientific Python values into strict JSON values."""
    if isinstance(obj, torch.Tensor):
        tensor = obj.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    if isinstance(obj, dt.date):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    # Avoid serializing arbitrary objects using an unstable repr by default.
    raise TypeError(f"Tipo non serializzabile in JSON: {type(obj).__name__}")


def save_json(data: Any, path: Union[str, Path], *, indent: int = 2) -> None:
    """Atomically write strict JSON and create parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_data = _json_safe(data)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(safe_data, handle, indent=indent, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# =============================================================================
# Official-style metric
# =============================================================================

def _normalize_qrels(qrels: Mapping[Any, Mapping[Any, Any]]) -> Dict[str, Dict[str, int]]:
    normalized: Dict[str, Dict[str, int]] = {}
    for topic_id, docs in qrels.items():
        topic = str(topic_id)
        normalized[topic] = {str(doc_id): int(rel) for doc_id, rel in docs.items()}
    return normalized


def _normalize_run(run: Mapping[Any, Any]) -> Dict[str, Dict[str, float]]:
    """Accept pytrec_eval score maps or ordered (doc_id, score) rankings."""
    normalized: Dict[str, Dict[str, float]] = {}
    for topic_id, docs in run.items():
        topic = str(topic_id)
        result: Dict[str, float] = {}
        if isinstance(docs, Mapping):
            iterator = docs.items()
        else:
            iterator = docs or []
        for item in iterator:
            try:
                doc_id, score = item
                score_value = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Run malformata per topic {topic}: {item!r}") from exc
            if not math.isfinite(score_value):
                continue
            doc = str(doc_id)
            # If a malformed input repeats the same doc, retain its best score.
            result[doc] = max(result.get(doc, -math.inf), score_value)
        normalized[topic] = result
    return normalized


def compute_official_ndcg(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, Dict[str, float]],
    cutoff: int = 10,
) -> Dict[str, float]:
    """Compute nDCG@cutoff with ``pytrec_eval``.

    All topics in qrels contribute to the mean. A topic absent from the run
    explicitly receives 0.0, preventing a partial run from improving its score
    by omitting difficult queries. qrels should be the judgments for the exact
    split being evaluated; do not replace them with training annotations.
    """
    cutoff = int(cutoff)
    if cutoff <= 0:
        raise ValueError("cutoff deve essere > 0.")
    if not qrels:
        return {f"ndcg_cut_{cutoff}": 0.0}

    normalized_qrels = _normalize_qrels(qrels)
    normalized_run = _normalize_run(run)

    try:
        import pytrec_eval
    except ImportError as exc:
        raise ImportError(
            "compute_official_ndcg richiede pytrec_eval; installa la dipendenza "
            "prevista dal kit ufficiale RETECO."
        ) from exc

    metric_name = f"ndcg_cut_{cutoff}"
    evaluator = pytrec_eval.RelevanceEvaluator(normalized_qrels, {metric_name})
    raw_scores = evaluator.evaluate(normalized_run)

    # TREC evaluator versions can differ on the treatment of topics omitted
    # from run. The explicit qrels-key loop makes our convention deterministic.
    topic_scores: List[float] = []
    for topic_id in normalized_qrels:
        score = raw_scores.get(topic_id, {}).get(metric_name, 0.0)
        score = float(score)
        topic_scores.append(score if math.isfinite(score) else 0.0)

    mean_score = float(np.mean(topic_scores)) if topic_scores else 0.0
    return {metric_name: mean_score}


# =============================================================================
# Reciprocal Rank Fusion
# =============================================================================

def reciprocal_rank_fusion(
    runs: Sequence[Mapping[str, Sequence[Tuple[str, float]]]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
    top_n: int = 100,
) -> Dict[str, List[Tuple[str, float]]]:
    """Fuse ranked lists with (optionally weighted) Reciprocal Rank Fusion.

    ``RRF(d) = sum_m weight_m / (k + rank_m(d))``

    A run is a mapping ``topic_id -> [(doc_id, score), ...]``. Input lists are
    re-sorted by score descending for safety, duplicate doc IDs within one run
    count only at their first rank, and ties in the final score are broken by
    document ID for reproducibility. The ``weights`` argument is preserved
    from the earlier public API; when omitted, all runs receive weight 1.
    """
    if int(k) < 1:
        raise ValueError("k deve essere >= 1.")
    if int(top_n) < 0:
        raise ValueError("top_n deve essere >= 0.")
    if not runs:
        return {}

    if weights is None:
        run_weights = [1.0] * len(runs)
    else:
        if len(weights) != len(runs):
            raise ValueError("I pesi devono corrispondere al numero di run.")
        run_weights = [float(weight) for weight in weights]
        if any(not math.isfinite(weight) or weight < 0 for weight in run_weights):
            raise ValueError("I pesi RRF devono essere finiti e non negativi.")

    all_topics: set[str] = set()
    for run in runs:
        all_topics.update(str(topic_id) for topic_id in run.keys())

    fused: Dict[str, Dict[str, float]] = {topic_id: {} for topic_id in all_topics}
    for run, weight in zip(runs, run_weights):
        if weight == 0:
            continue
        for raw_topic, raw_ranking in run.items():
            topic_id = str(raw_topic)
            if isinstance(raw_ranking, Mapping):
                ranking = list(raw_ranking.items())
            else:
                ranking = list(raw_ranking or [])
            # Inputs are intended as scored rankings. Sorting here preserves
            # compatibility with older callers that didn't pre-sort them.
            normalized: List[Tuple[str, float]] = []
            for entry in ranking:
                if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                    raise ValueError(f"Ranking malformato per topic {topic_id}: {entry!r}")
                doc_id, raw_score = entry
                try:
                    score = float(raw_score)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Score non numerico nel topic {topic_id}: {raw_score!r}") from exc
                if math.isfinite(score):
                    normalized.append((str(doc_id), score))
            normalized.sort(key=lambda item: item[1], reverse=True)

            seen_docs: set[str] = set()
            rank = 0
            for doc_id, _score in normalized:
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                rank += 1
                fused[topic_id][doc_id] = fused[topic_id].get(doc_id, 0.0) + (
                    weight / (int(k) + rank)
                )

    output: Dict[str, List[Tuple[str, float]]] = {}
    for topic_id in sorted(fused):
        ranked = sorted(
            fused[topic_id].items(),
            key=lambda item: (-item[1], item[0]),
        )
        output[topic_id] = ranked[: int(top_n)]
    return output


# =============================================================================
# TREC run output and validation
# =============================================================================

def write_trec_run(
    run_dict: Mapping[str, Sequence[Tuple[str, float]]],
    output_path: Union[str, Path],
    run_tag: str = "reteco_run",
    max_k: int = 10,
) -> None:
    """Write an ordered six-column TREC run file."""
    max_k = int(max_k)
    if max_k < 0:
        raise ValueError("max_k deve essere >= 0.")
    run_tag = str(run_tag).strip()
    if not run_tag or any(character.isspace() for character in run_tag):
        raise ValueError("run_tag deve essere non vuoto e senza spazi.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        ordered_topics = sorted(run_dict.items(), key=lambda pair: str(pair[0]))
        for raw_topic_id, ranking in ordered_topics:
            topic_id = str(raw_topic_id)
            seen: set[str] = set()
            rank = 0
            for item in list(ranking or [])[:max_k]:
                if not isinstance(item, (tuple, list)) or len(item) != 2:
                    raise ValueError(f"Ranking malformato per topic {topic_id}: {item!r}")
                doc_id, raw_score = item
                topic = str(topic_id)
                doc = str(doc_id)
                if not topic or not doc or any(ch.isspace() for ch in topic + doc):
                    raise ValueError("topic_id e doc_id TREC devono essere non vuoti e senza spazi.")
                if doc in seen:
                    continue
                score = float(raw_score)
                if not math.isfinite(score):
                    continue
                seen.add(doc)
                rank += 1
                handle.write(f"{topic} Q0 {doc} {rank} {score:.8f} {run_tag}\n")


def validate_trec_file(
    trec_path: Union[str, Path],
    max_rank: int = 10,
) -> Tuple[bool, List[str]]:
    """Validate six-column TREC rows, finite scores, ranks and duplicate docs."""
    path = Path(trec_path)
    if not path.exists():
        return False, ["File TREC non esistente."]

    errors: List[str] = []
    ranks_by_topic: Dict[str, List[int]] = {}
    seen_docs: Dict[str, set[str]] = {}

    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        return False, [f"Impossibile leggere il file TREC: {exc}"]

    with handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 6:
                errors.append(
                    f"Riga {line_no}: formato a 6 colonne violato ({len(parts)} colonne)."
                )
                continue
            topic_id, q0, doc_id, rank_text, score_text, run_tag = parts
            if q0 != "Q0":
                errors.append(f"Riga {line_no}: la seconda colonna deve essere Q0.")
            if not run_tag:
                errors.append(f"Riga {line_no}: run_tag vuoto.")
            try:
                rank = int(rank_text)
                if rank <= 0:
                    raise ValueError
            except ValueError:
                errors.append(f"Riga {line_no}: rango non valido {rank_text!r}.")
                continue
            try:
                score = float(score_text)
                if not math.isfinite(score):
                    raise ValueError
            except ValueError:
                errors.append(f"Riga {line_no}: score non finito/non numerico {score_text!r}.")
                continue
            if rank > int(max_rank):
                errors.append(f"Riga {line_no}: rango {rank} > max_rank {max_rank}.")

            ranks_by_topic.setdefault(topic_id, []).append(rank)
            docs = seen_docs.setdefault(topic_id, set())
            if doc_id in docs:
                errors.append(f"Riga {line_no}: doc_id duplicato {doc_id!r} per topic {topic_id!r}.")
            docs.add(doc_id)

    for topic_id, ranks in ranks_by_topic.items():
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            errors.append(
                f"Topic {topic_id}: ranghi non sequenziali nell'ordine del file; "
                f"atteso {expected}, trovato {ranks}."
            )
    return not errors, errors


# =============================================================================
# Logging and reproducibility
# =============================================================================

def setup_logger(
    log_dir: Union[str, Path],
    run_tag: str,
    log_level: str = "INFO",
) -> logging.Logger:
    """Configure console + timestamped file logging.

    Fixes the previous ``datetime.now`` error (datetime was imported as a
    module) which was caught by the caller and silently disabled file logging.
    """
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(run_tag)).strip("._-")
    if not safe_tag:
        safe_tag = "reteco_2a"

    level = getattr(logging, str(log_level).upper(), logging.INFO)
    logger = logging.getLogger("RETECO_2A_Train")
    logger.setLevel(level)
    logger.propagate = False

    # Close previous handlers before replacing them, avoiding duplicated logs
    # and leaking file descriptors when this is called more than once.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file = directory / f"{safe_tag}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Logger inizializzato. File di log: %s", log_file)
    return logger


def set_seed(seed: int = 42) -> None:
    """Seed the common Python, NumPy and PyTorch RNGs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# =============================================================================
# Optional training-history visualization
# =============================================================================

def plot_training_history(
    history: Any,
    output_path: Union[str, Path],
    title: str = "Training Progress Sub-track 2a",
) -> None:
    """Plot train loss and validation nDCG from either legacy or new history.

    Supports the legacy mapping ``{'train_loss': [...], 'val_ndcg': [...]}``
    and the current train.py list of per-epoch records with ``training.loss``
    and ``validation.macro_average.nDCG@10``.
    Matplotlib is imported lazily because plotting is optional during training.
    """
    logger = logging.getLogger("RETECO_2A_Train")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(history, Mapping):
        train_losses = list(history.get("train_loss", []))
        val_ndcgs = list(history.get("val_ndcg", []))
    elif isinstance(history, list):
        train_losses = []
        val_ndcgs = []
        for record in history:
            if not isinstance(record, Mapping):
                continue
            train = record.get("training", {})
            validation = record.get("validation", {})
            macro = validation.get("macro_average", {}) if isinstance(validation, Mapping) else {}
            train_losses.append(train.get("loss") if isinstance(train, Mapping) else None)
            val_ndcgs.append(macro.get("nDCG@10") if isinstance(macro, Mapping) else None)
        while train_losses and train_losses[-1] is None:
            train_losses.pop()
        while val_ndcgs and val_ndcgs[-1] is None:
            val_ndcgs.pop()
    else:
        logger.warning("Formato history non supportato; grafico non generato.")
        return

    if not train_losses:
        logger.warning("Nessuna train loss nella cronologia; grafico non generato.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_training_history richiede matplotlib.") from exc

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.plot(epochs, train_losses, marker="o", linewidth=2, label="Train Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)

    valid_vals = [value for value in val_ndcgs if value is not None]
    if valid_vals:
        ax2 = ax1.twinx()
        val_epochs = np.arange(1, len(valid_vals) + 1)
        ax2.set_ylabel("Validation nDCG@10")
        ax2.plot(val_epochs, valid_vals, marker="s", linewidth=2, label="Val nDCG@10")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
    logger.info("Grafico salvato: %s", output)


if __name__ == "__main__":
    print("Smoke test delle utility RETECO Sub-track 2a")
    qrels = {"conv_1_turn_1": {"doc_A": 1}, "conv_1_turn_2": {"doc_C": 1}}
    run = {
        "conv_1_turn_1": {"doc_A": 12.5, "doc_X": 2.1},
        # Deliberatamente manca conv_1_turn_2: deve contribuire con 0.0.
    }
    print("nDCG:", compute_official_ndcg(qrels, run, cutoff=10))

    test_path = Path("test_run.trec")
    try:
        write_trec_run({"conv_1_turn_1": [("doc_A", 1.0), ("doc_B", 0.5)]}, test_path, max_k=2)
        valid, errors = validate_trec_file(test_path, max_rank=2)
        print("TREC validation:", "OK" if valid else "FAILED", errors)
    finally:
        test_path.unlink(missing_ok=True)
