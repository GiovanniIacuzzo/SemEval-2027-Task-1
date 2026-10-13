#!/usr/bin/env python3
"""Controlled candidate reranking evaluation for RETECO Sub-track 2a.

Pipeline:
  1) Reuse official corpus + the same conversation-level internal validation split.
  2) Retrieve BM25 top `--bm25-depth` (cached Lucene index).
  3) Score only the top `--top-k` candidates with Qwen3-Reranker-0.6B.
  4) Keep the original BM25 tail after the reranked top-k, so Recall@1000
     remains comparable and the reranker only changes candidate order.
  5) Save a report after every domain and cache scores after every query.

Run from repository root:
  python subtrack_2a/evaluate_bm25_reranker_aligned.py --config subtrack_2a/config/config.yaml --domains biology,medicalsciences --query-mode contextual

The model follows the official Qwen3-Reranker yes/no-logit scoring template.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import yaml
from tqdm import tqdm

from retrieval.retrieval import OfficialBM25, _corpus_fingerprint

TRACK2_DOMAINS = [
    "biology", "drones", "earth_science", "economics", "hardware", "law",
    "medicalsciences", "politics", "psychology", "robotics", "sustainable_living",
]

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
TASK_INSTRUCTION = (
    "Given a conversational search query and its conversation history, "
    "retrieve passages that best answer the current turn. Use the history "
    "only to interpret the current turn when it is relevant."
)
OFFICIAL_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

SYSTEM_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def setup_logger(log_dir: Path, run_tag: str, level_name: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("RETECO_BM25_FAST")
    logger.setLevel(getattr(logging, level_name.upper(), logging.INFO))
    logger.propagate = False
    for old_handler in list(logger.handlers):
        logger.removeHandler(old_handler)
        old_handler.close()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / f"{run_tag}_{timestamp}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp_path.replace(path)


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError(f"Formato JSON non valido: {path}")
        return value
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL non valido in {path}, riga {line_no}") from exc
    return rows


def load_corpus(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Corpus non trovato: {path}")
    corpus: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON non valido in {path}, riga {line_no}") from exc
            doc_id = item.get("doc_id", item.get("id"))
            text = str(item.get("content", item.get("text", "")) or "").strip()
            if doc_id is None:
                raise ValueError(f"Documento senza ID in {path}, riga {line_no}")
            if text:
                corpus[str(doc_id)] = text
    if not corpus:
        raise ValueError(f"Corpus vuoto: {path}")
    return corpus


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Qrels non trovati: {path}")
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 4:
                raise ValueError(f"Qrels malformati in {path}, riga {line_no}: {line}")
            topic_id, _, doc_id, relevance = parts[:4]
            qrels.setdefault(topic_id, {})[doc_id] = int(relevance)
    return qrels


def format_contextual_query(query: str, history: str, max_query_length: int, strategy: str) -> str:
    """Exact tokenizer-free equivalent of ContextAwareQueryFormatter.

    The project uses whitespace tokens if no tokenizer is supplied. This
    intentionally preserves that behavior for BM25's raw query string.
    """
    query = (query or "").strip()
    history = (history or "").strip()
    if history.lower() == "no previous conversation.":
        history = ""

    if strategy == "query_only" or not history:
        return " ".join(query.split()[:max_query_length]).strip()

    history_header = "\n\nConversation History:\n"
    fixed_text = f"{query}{history_header}"
    fixed_len = len(fixed_text.split())
    if fixed_len > max_query_length:
        header_tokens = len(history_header.split())
        query = " ".join(query.split()[:max(1, max_query_length - header_tokens)])
        fixed_text = f"{query}{history_header}"
        fixed_len = len(fixed_text.split())

    remaining = max(0, max_query_length - fixed_len)
    retained_history = " ".join(history.split()[-remaining:]) if remaining else ""
    return f"{fixed_text}{retained_history}".strip()


def load_domain_samples(benchmark_path: Path, domain: str, max_query_length: int, strategy: str) -> List[Dict[str, Any]]:
    conversations = read_json_or_jsonl(benchmark_path)
    samples: List[Dict[str, Any]] = []
    for conversation in conversations:
        conversation_id = str(conversation["id"])
        for turn in conversation.get("turns", []):
            turn_id = int(turn["turn_id"])
            query = str(turn.get("query", "") or "").strip()
            history = str(turn.get("conversation_history", "") or "").strip()
            gold_doc_ids = [
                str(doc_id)
                for doc_id in (turn.get("gold_doc_ids") or turn.get("supporting_doc_ids") or [])
            ]
            samples.append({
                "domain": domain,
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "topic_id": f"{conversation_id}_turn_{turn_id}",
                "query": query,
                "history": history,
                "contextual_query": format_contextual_query(query, history, max_query_length, strategy),
                "gold_doc_ids": gold_doc_ids,
            })
    return samples


def split_conversations_train_val(samples: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    conversation_ids = sorted({str(sample["conversation_id"]) for sample in samples})
    if len(conversation_ids) < 2:
        raise ValueError("Servono almeno due conversazioni per creare lo split interno.")
    rng = random.Random(seed)
    rng.shuffle(conversation_ids)
    n_val = max(1, int(round(len(conversation_ids) * val_ratio)))
    n_val = min(n_val, len(conversation_ids) - 1)
    val_ids = set(conversation_ids[:n_val])
    return (
        [s for s in samples if str(s["conversation_id"]) not in val_ids],
        [s for s in samples if str(s["conversation_id"]) in val_ids],
    )


def prepare_valid_qrels(samples: Sequence[Dict[str, Any]], qrels: Mapping[str, Mapping[str, int]], corpus_ids: set[str]) -> Tuple[Dict[str, Dict[str, int]], int]:
    valid: Dict[str, Dict[str, int]] = {}
    fallback_count = 0
    for sample in samples:
        topic_id = str(sample["topic_id"])
        judged = {
            str(doc_id): int(rel)
            for doc_id, rel in qrels.get(topic_id, {}).items()
            if str(doc_id) in corpus_ids and int(rel) > 0
        }
        if not judged:
            fallback = {str(doc_id): 1 for doc_id in sample["gold_doc_ids"] if str(doc_id) in corpus_ids}
            if fallback:
                judged = fallback
                fallback_count += 1
        if judged:
            valid[topic_id] = judged
    return valid, fallback_count


def score_run(valid_qrels: Dict[str, Dict[str, int]], run: Dict[str, List[Tuple[str, float]]], corpus_size: int, fallback_topics: int) -> Dict[str, Any]:
    topic_ndcgs: List[float] = []
    recalls: Dict[int, List[float]] = {10: [], 50: [], 100: [], 1000: []}
    reciprocal_ranks: List[float] = []

    for topic_id, judged in valid_qrels.items():
        ranked = run.get(topic_id, [])
        ranked_ids = [str(doc_id) for doc_id, _ in ranked]
        rel_by_doc = {str(doc_id): int(rel) for doc_id, rel in judged.items()}

        dcg = 0.0
        for rank, (doc_id, _) in enumerate(ranked[:10], start=1):
            rel = rel_by_doc.get(str(doc_id), 0)
            if rel > 0:
                dcg += (2.0 ** rel - 1.0) / math.log2(rank + 1)
        ideal_rels = sorted((int(rel) for rel in judged.values()), reverse=True)[:10]
        idcg = sum((2.0 ** rel - 1.0) / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
        topic_ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        gold_ids = set(judged)
        for cutoff in recalls:
            recalls[cutoff].append(len(set(ranked_ids[:cutoff]) & gold_ids) / max(1, len(gold_ids)))
        reciprocal_ranks.append(next((1.0 / rank for rank, doc_id in enumerate(ranked_ids, start=1) if doc_id in gold_ids), 0.0))

    result: Dict[str, Any] = {
        "nDCG@10": sum(topic_ndcgs) / len(topic_ndcgs) if topic_ndcgs else 0.0,
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "num_topics": len(valid_qrels),
        "num_documents": corpus_size,
        "qrels_fallback_topics": fallback_topics,
    }
    for cutoff, values in recalls.items():
        result[f"Recall@{cutoff}"] = sum(values) / len(values) if values else 0.0
    return result


def macro_average(per_domain: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    names = ["nDCG@10", "MRR", "Recall@10", "Recall@50", "Recall@100", "Recall@1000"]
    output: Dict[str, float] = {}
    for name in names:
        values = [float(metrics[name]) for metrics in per_domain.values() if name in metrics]
        mean = sum(values) / len(values) if values else 0.0
        output[name] = mean
        output[f"{name}_std"] = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) if values else 0.0
    return output


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower().strip()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("È stato richiesto CUDA ma torch.cuda.is_available() è False.")
    return torch.device(requested)


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cache(path: Path, expected_meta: Mapping[str, Any]) -> Dict[str, Any]:
    if not path.is_file():
        return {"meta": dict(expected_meta), "topics": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("meta") != dict(expected_meta) or not isinstance(data.get("topics"), dict):
            return {"meta": dict(expected_meta), "topics": {}}
        return data
    except (OSError, json.JSONDecodeError):
        return {"meta": dict(expected_meta), "topics": {}}


def cache_query_signature(
    model_name: str,
    corpus_fingerprint: str,
    sample: Mapping[str, Any],
    candidate_ids: Sequence[str],
    max_length: int,
    current_budget: int,
    history_budget: int,
    document_budget: int,
    instruction: str,
) -> str:
    return sha256_json({
        "model": model_name,
        "corpus_fingerprint": corpus_fingerprint,
        "topic_id": sample["topic_id"],
        "query": sample["query"],
        "history": sample["history"],
        "candidate_ids": list(candidate_ids),
        "max_length": max_length,
        "current_budget": current_budget,
        "history_budget": history_budget,
        "document_budget": document_budget,
        "instruction": instruction,
    })


class QwenReranker:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        max_length: int = 768,
        current_query_tokens: int = 128,
        history_tokens: int = 64,
        document_tokens: int = 400,
        batch_size: int = 8,
        query_mode: str = "contextual",
    ) -> None:
        if max_length < 128:
            raise ValueError("--max-length deve essere almeno 128.")
        if query_mode not in {"contextual", "split_history"}:
            raise ValueError("query_mode deve essere 'contextual' oppure 'split_history'.")
        if min(current_query_tokens, document_tokens, batch_size) <= 0:
            raise ValueError("I budget di query/documento e --batch-size devono essere positivi.")
        if query_mode == "split_history" and history_tokens <= 0:
            raise ValueError("--history-tokens deve essere positivo per split_history.")

        self.model_name = model_name
        self.device = device
        self.max_length = int(max_length)
        self.query_mode = query_mode
        self.current_query_tokens = int(current_query_tokens)
        self.history_tokens = int(history_tokens) if query_mode == "split_history" else 0
        self.document_tokens = int(document_tokens)
        self.batch_size = int(batch_size)

        # Import lazily so --help and static checks work without downloading or
        # initializing Transformers model classes.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logging.info("Loading reranker: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        self.model.config.use_cache = False

        yes_ids = self.tokenizer.encode("yes", add_special_tokens=False)
        no_ids = self.tokenizer.encode("no", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise RuntimeError(
                "Il tokenizer del reranker non codifica 'yes'/'no' in un singolo token; "
                "controllare la versione ufficiale del modello."
            )
        self.yes_token_id = int(yes_ids[0])
        self.no_token_id = int(no_ids[0])
        self.prefix_ids = self.tokenizer.encode(SYSTEM_PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(ASSISTANT_SUFFIX, add_special_tokens=False)
        self.instruction = (
            OFFICIAL_QUERY_INSTRUCTION if query_mode == "contextual" else TASK_INSTRUCTION
        )
        query_prefix = (
            f"<Instruct>: {self.instruction}\n<Query>: "
            if query_mode == "contextual"
            else f"<Instruct>: {self.instruction}\n<Query>: Current turn: "
        )
        self.instruction_query_prefix_ids = self.tokenizer.encode(
            query_prefix, add_special_tokens=False
        )
        self.history_header_ids = (
            self.tokenizer.encode("\nConversation history: ", add_special_tokens=False)
            if query_mode == "split_history" else []
        )
        self.document_header_ids = self.tokenizer.encode(
            "\n<Document>: ", add_special_tokens=False
        )

        # Validate the default budgets against the total token cap. If they do
        # not fit, document tokens are reduced first, then history tokens.
        fixed = (
            len(self.prefix_ids) + len(self.suffix_ids)
            + len(self.instruction_query_prefix_ids)
            + len(self.history_header_ids) + len(self.document_header_ids)
        )
        available = self.max_length - fixed
        wanted = self.current_query_tokens + self.history_tokens + self.document_tokens
        if available < self.current_query_tokens + 16:
            raise ValueError(
                f"max_length={self.max_length} è troppo piccolo per il prompt fisso "
                f"e {self.current_query_tokens} token della query."
            )
        if wanted > available:
            excess = wanted - available
            shrink_doc = min(excess, max(0, self.document_tokens - 64))
            self.document_tokens -= shrink_doc
            excess -= shrink_doc
            shrink_hist = min(excess, max(0, self.history_tokens - 16))
            self.history_tokens -= shrink_hist
            excess -= shrink_hist
            if excess > 0:
                self.current_query_tokens -= excess
        logging.info(
            "Reranker device=%s dtype=%s | query_mode=%s | max_length=%d | budgets query/history/doc=%d/%d/%d | batch=%d",
            device, dtype, self.query_mode, self.max_length, self.current_query_tokens,
            self.history_tokens, self.document_tokens, self.batch_size,
        )

    def _query_tokens(self, query: str, history: str) -> Tuple[List[int], List[int]]:
        query_ids = self.tokenizer.encode(query or "", add_special_tokens=False)
        query_ids = query_ids[: self.current_query_tokens]

        if self.query_mode == "contextual":
            # The whole conversation-aware query is passed as a single query,
            # matching the same representation used by BM25.
            return query_ids, []

        history = (history or "").strip()
        if history.lower() == "no previous conversation.":
            history = ""
        history_ids = self.tokenizer.encode(history, add_special_tokens=False) if history else []
        # Keep the most recent end of the conversation history.
        history_ids = history_ids[-self.history_tokens :] if self.history_tokens else []
        return query_ids, history_ids

    def _format_ids_from_tokens(
        self, query_ids: List[int], history_ids: List[int], document: str
    ) -> List[int]:
        document_ids = self.tokenizer.encode(document or "", add_special_tokens=False)
        document_ids = document_ids[: self.document_tokens]

        ids = (
            self.prefix_ids
            + self.instruction_query_prefix_ids
            + query_ids
            + self.history_header_ids
            + history_ids
            + self.document_header_ids
            + document_ids
            + self.suffix_ids
        )
        if len(ids) > self.max_length:
            # Safety guard for tokenizer/model versions whose prompt token counts
            # differ unexpectedly; retain fixed template and current query.
            overflow = len(ids) - self.max_length
            trim_doc = min(overflow, len(document_ids))
            document_ids = document_ids[:-trim_doc] if trim_doc else document_ids
            overflow -= trim_doc
            if overflow > 0:
                trim_hist = min(overflow, len(history_ids))
                history_ids = history_ids[trim_hist:]
                overflow -= trim_hist
            if overflow > 0:
                trim_query = min(overflow, max(0, len(query_ids) - 16))
                query_ids = query_ids[:-trim_query] if trim_query else query_ids
            ids = (
                self.prefix_ids + self.instruction_query_prefix_ids + query_ids
                + self.history_header_ids + history_ids + self.document_header_ids
                + document_ids + self.suffix_ids
            )
        if len(ids) > self.max_length:
            raise RuntimeError(f"Prompt length {len(ids)} exceeds max_length={self.max_length}.")
        return ids

    def format_ids(self, query: str, history: str, document: str) -> List[int]:
        query_ids, history_ids = self._query_tokens(query, history)
        return self._format_ids_from_tokens(query_ids, history_ids, document)

    @torch.inference_mode()
    def score_pairs(self, pairs: Sequence[Tuple[str, str, str]]) -> List[float]:
        """Pairs are (current_query, history, document); query/history tokenization is cached per query."""
        if not pairs:
            return []
        encoded: List[List[int]] = []
        query_token_cache: Dict[Tuple[str, str], Tuple[List[int], List[int]]] = {}
        for query, history, document in pairs:
            cache_key = (query or "", history or "")
            tokens = query_token_cache.get(cache_key)
            if tokens is None:
                tokens = self._query_tokens(query, history)
                query_token_cache[cache_key] = tokens
            encoded.append(self._format_ids_from_tokens(tokens[0], tokens[1], document))
        all_scores: List[float] = []
        for start in range(0, len(encoded), self.batch_size):
            chunk = encoded[start : start + self.batch_size]
            features = self.tokenizer.pad(
                [{"input_ids": ids} for ids in chunk],
                padding=True,
                return_tensors="pt",
            )
            features = {key: value.to(self.device) for key, value in features.items()}
            try:
                logits = self.model(**features, use_cache=False).logits[:, -1, :]
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or len(chunk) <= 1:
                    raise
                torch.cuda.empty_cache()
                # Retry this chunk recursively with half batches to recover safely.
                midpoint = len(chunk) // 2
                first = self._score_encoded_chunk(chunk[:midpoint])
                second = self._score_encoded_chunk(chunk[midpoint:])
                all_scores.extend(first)
                all_scores.extend(second)
                continue
            scores = logits[:, self.yes_token_id] - logits[:, self.no_token_id]
            all_scores.extend(float(value) for value in scores.float().cpu().tolist())
            del logits, features
        return all_scores

    @torch.inference_mode()
    def _score_encoded_chunk(self, encoded: Sequence[List[int]]) -> List[float]:
        if not encoded:
            return []
        if self.device.type == "cuda" and len(encoded) > 1:
            # Keep OOM recovery bounded and deterministic.
            pass
        features = self.tokenizer.pad(
            [{"input_ids": ids} for ids in encoded], padding=True, return_tensors="pt"
        )
        features = {key: value.to(self.device) for key, value in features.items()}
        try:
            logits = self.model(**features, use_cache=False).logits[:, -1, :]
        except torch.cuda.OutOfMemoryError:
            if self.device.type != "cuda" or len(encoded) <= 1:
                raise
            del features
            torch.cuda.empty_cache()
            midpoint = len(encoded) // 2
            return self._score_encoded_chunk(encoded[:midpoint]) + self._score_encoded_chunk(encoded[midpoint:])
        scores = logits[:, self.yes_token_id] - logits[:, self.no_token_id]
        result = [float(value) for value in scores.float().cpu().tolist()]
        del logits, features
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Valuta reranking Qwen3 sui candidati BM25, senza embedding full-corpus.")
    parser.add_argument("--config", default="subtrack_2a/config/config.yaml")
    parser.add_argument("--domains", default="biology,medicalsciences", help="Domini separati da virgola; usare 'all' per tutti gli 11.")
    parser.add_argument(
        "--query-mode", choices=["contextual", "split_history"], default="contextual",
        help=("contextual: usa esattamente la query contestuale di BM25 nel template standard Qwen; "
              "split_history: passa turno corrente e cronologia in campi separati (variante precedente)."),
    )
    parser.add_argument("--top-k", type=int, default=100, help="Numero di candidati BM25 da riordinare.")
    parser.add_argument("--bm25-depth", type=int, default=1000, help="Profondità BM25 prima del reranking.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch per il cross-encoder; ridurre a 4 se c'è OOM.")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--current-query-tokens", type=int, default=256)
    parser.add_argument("--history-tokens", type=int, default=64)
    parser.add_argument("--document-tokens", type=int, default=400)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.top_k < 10 or args.bm25_depth < args.top_k:
        parser.error("Richiesto --top-k >= 10 e --bm25-depth >= --top-k.")
    if args.batch_size < 1:
        parser.error("--batch-size deve essere >= 1.")

    # Resolve config path in the same way as the fast BM25 evaluator.
    config_path = Path(args.config)
    if not config_path.is_file() and config_path.parts and config_path.parts[0] == "subtrack_2a":
        alt = Path(*config_path.parts[1:])
        if alt.is_file():
            config_path = alt
    if not config_path.is_file():
        parser.error(f"Config non trovato: {args.config}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    general = config.get("general", {})
    paths = config.get("paths", {})
    data_cfg = config.get("data", {})
    domains_cfg = config.get("domains", {})
    sparse_cfg = config.get("sparse", {})

    seed = int(general.get("seed", 42))
    random.seed(seed)
    if args.domains.strip().lower() == "all":
        domains = list(TRACK2_DOMAINS)
    elif args.domains:
        domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    else:
        selected = domains_cfg.get("active_domains", "all")
        domains = list(TRACK2_DOMAINS) if selected == "all" else list(selected)
    invalid = [domain for domain in domains if domain not in TRACK2_DOMAINS]
    if not domains or invalid:
        parser.error(f"Domini non validi: {invalid or domains}")

    data_mode = str(paths.get("data_mode", "full")).lower().strip()
    if data_mode not in {"full", "sample"}:
        raise ValueError("paths.data_mode deve essere 'full' o 'sample'.")
    data_key = "full_data_dir" if data_mode == "full" else "sample_data_dir"
    if data_key not in paths:
        raise ValueError(f"Manca paths.{data_key} nel config.")
    data_dir = Path(paths[data_key])
    output_dir = Path(paths.get("output_dir", "outputs/subtrack_2a"))
    log_dir = Path(paths.get("log_dir", "outputs/subtrack_2a/logs"))
    output_path = Path(args.output) if args.output else output_dir / f"reranker_top{args.top_k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    cache_root = output_dir / "reranker_score_cache" / f"{args.model.replace('/', '_')}_top{args.top_k}_len{args.max_length}_{args.query_mode}"
    logger = setup_logger(log_dir, f"reranker_top{args.top_k}_{general.get('run_tag', 'reteco_2a')}", str(general.get("logging_level", "INFO")))
    logger.info("BM25 candidate reranking: no full-corpus embeddings; model=%s", args.model)

    device = resolve_device(args.device)
    reranker = QwenReranker(
        model_name=args.model,
        device=device,
        max_length=args.max_length,
        current_query_tokens=args.current_query_tokens,
        history_tokens=args.history_tokens,
        document_tokens=args.document_tokens,
        batch_size=args.batch_size,
        query_mode=args.query_mode,
    )

    val_ratio = float(data_cfg.get("val_ratio", 0.15))
    max_query_length = int(data_cfg.get("max_query_length", 512))
    query_strategy = str(data_cfg.get("query_strategy", "history"))
    k1 = float(sparse_cfg.get("k1", 0.9))
    b = float(sparse_cfg.get("b", 0.4))
    index_root = Path(paths.get("hard_negatives_cache_dir", "data/cache/hard_negatives")) / "lucene_indexes"

    report: Dict[str, Any] = {
        "method": "Qwen3-Reranker-0.6B reranking the top BM25 candidates; untouched BM25 tail appended",
        "query_mode": args.query_mode,
        "model": args.model,
        "instruction": reranker.instruction,
        "data_mode": data_mode,
        "data_dir": str(data_dir),
        "domains_requested": domains,
        "validation_split": {"unit": "conversation", "seed": seed, "val_ratio": val_ratio},
        "bm25": {"k1": k1, "b": b, "depth": args.bm25_depth},
        "reranker": {
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "max_length": reranker.max_length,
            "current_query_tokens": reranker.current_query_tokens,
            "history_tokens": reranker.history_tokens,
            "document_tokens": reranker.document_tokens,
            "score": "logit_yes - logit_no",
        },
        "domains": {},
        "macro_average_completed_domains": {},
        "completed_domains": [],
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    completed_bm25: Dict[str, Dict[str, Any]] = {}
    completed_reranked: Dict[str, Dict[str, Any]] = {}
    cache_root.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        domain_started = time.monotonic()
        domain_dir = data_dir / domain
        logger.info("[%s] Loading corpus, benchmark_train and qrels_train...", domain)
        corpus = load_corpus(domain_dir / "documents.jsonl")
        benchmark_path = domain_dir / "benchmark_train.json"
        if not benchmark_path.is_file():
            benchmark_path = domain_dir / "benchmark_train.jsonl"
        samples = load_domain_samples(benchmark_path, domain, max_query_length, query_strategy)
        qrels = load_qrels(domain_dir / "qrels_train.txt")
        _, val_samples = split_conversations_train_val(samples, val_ratio, seed)
        valid_qrels, fallback_count = prepare_valid_qrels(val_samples, qrels, set(corpus))
        if not valid_qrels:
            raise RuntimeError(f"[{domain}] Nessun giudizio positivo valido nel validation split.")
        corpus_fingerprint = _corpus_fingerprint(corpus)
        index_dir = index_root / corpus_fingerprint
        depth = min(args.bm25_depth, len(corpus))
        logger.info("[%s] corpus=%d | validation_topics=%d | scored_topics=%d | top_k=%d | bm25_depth=%d",
                    domain, len(corpus), len(val_samples), len(valid_qrels), args.top_k, depth)

        bm25_run: Dict[str, List[Tuple[str, float]]] = {}
        with OfficialBM25(corpus=corpus, k1=k1, b=b, index_dir=index_dir) as bm25:
            for sample in tqdm(val_samples, desc=f"BM25 [{domain}]", unit="query", leave=False):
                topic_id = str(sample["topic_id"])
                bm25_run[topic_id] = bm25.search_one(sample["contextual_query"], top_k=depth)

        bm25_metrics = score_run(valid_qrels, bm25_run, len(corpus), fallback_count)
        cache_meta = {
            "schema_version": 1,
            "model": args.model,
            "corpus_fingerprint": corpus_fingerprint,
            "top_k": args.top_k,
            "max_length": reranker.max_length,
            "current_query_tokens": reranker.current_query_tokens,
            "history_tokens": reranker.history_tokens,
            "document_tokens": reranker.document_tokens,
            "instruction": reranker.instruction,
            "query_mode": args.query_mode,
        }
        cache_path = cache_root / f"{domain}.json"
        cache = load_cache(cache_path, cache_meta)
        topic_cache: Dict[str, Any] = cache["topics"]
        reranked_run: Dict[str, List[Tuple[str, float]]] = {}
        cache_hits = 0

        for q_idx, sample in enumerate(tqdm(val_samples, desc=f"Rerank [{domain}]", unit="query", leave=False), start=1):
            topic_id = str(sample["topic_id"])
            original_ranking = bm25_run.get(topic_id, [])
            candidate_ranking = original_ranking[:args.top_k]
            candidate_ids = [str(doc_id) for doc_id, _ in candidate_ranking]
            signature = cache_query_signature(
                model_name=args.model,
                corpus_fingerprint=corpus_fingerprint,
                sample=sample,
                candidate_ids=candidate_ids,
                max_length=reranker.max_length,
                current_budget=reranker.current_query_tokens,
                history_budget=reranker.history_tokens,
                document_budget=reranker.document_tokens,
                instruction=reranker.instruction + " [query_mode=" + args.query_mode + "]",
            )
            stored = topic_cache.get(topic_id, {})
            if stored.get("signature") == signature and stored.get("candidate_ids") == candidate_ids and len(stored.get("scores", [])) == len(candidate_ids):
                candidate_scores = [float(value) for value in stored["scores"]]
                cache_hits += 1
            else:
                if args.query_mode == "contextual":
                    # Controlled ablation: rerank using exactly the conversation-aware query
                    # string that was given to BM25, without separately truncating history.
                    pair_inputs = [
                        (str(sample["contextual_query"]), "", corpus[doc_id])
                        for doc_id in candidate_ids
                    ]
                else:
                    # Previous formulation: current turn and capped history as separate fields.
                    pair_inputs = [
                        (str(sample["query"]), str(sample["history"]), corpus[doc_id])
                        for doc_id in candidate_ids
                    ]
                candidate_scores = reranker.score_pairs(pair_inputs)
                topic_cache[topic_id] = {
                    "signature": signature,
                    "candidate_ids": candidate_ids,
                    "scores": candidate_scores,
                }
                cache["topics"] = topic_cache
                save_json(cache, cache_path)

            candidate_with_scores = list(zip(candidate_ids, candidate_scores))
            order = sorted(range(len(candidate_with_scores)), key=lambda idx: (-candidate_with_scores[idx][1], idx))
            reranked_top = [(candidate_with_scores[idx][0], candidate_with_scores[idx][1]) for idx in order]
            # Append BM25 candidates below the reranked pool in their old order.
            untouched_tail = original_ranking[len(candidate_ids):]
            reranked_run[topic_id] = reranked_top + [(str(doc_id), float(score)) for doc_id, score in untouched_tail]

            if q_idx % 5 == 0 or q_idx == len(val_samples):
                logger.info("[%s] reranked %d/%d queries | cache_hits=%d", domain, q_idx, len(val_samples), cache_hits)

        reranked_metrics = score_run(valid_qrels, reranked_run, len(corpus), fallback_count)
        delta = {
            metric: float(reranked_metrics[metric]) - float(bm25_metrics[metric])
            for metric in ("nDCG@10", "MRR", "Recall@10", "Recall@50", "Recall@100", "Recall@1000")
        }
        result = {
            "bm25": bm25_metrics,
            "reranked": reranked_metrics,
            "delta_reranked_minus_bm25": delta,
            "candidate_pool_recall_at_100": bm25_metrics.get("Recall@100"),
            "num_queries": len(valid_qrels),
            "cache_hits": cache_hits,
            "elapsed_seconds": round(time.monotonic() - domain_started, 2),
        }
        report["domains"][domain] = result
        completed_bm25[domain] = bm25_metrics
        completed_reranked[domain] = reranked_metrics
        report["completed_domains"] = list(completed_bm25.keys())
        report["macro_average_completed_domains"] = {
            "bm25": macro_average(completed_bm25),
            "reranked": macro_average(completed_reranked),
            "delta_reranked_minus_bm25": {
                metric: macro_average(completed_reranked)[metric] - macro_average(completed_bm25)[metric]
                for metric in ("nDCG@10", "MRR", "Recall@10", "Recall@50", "Recall@100", "Recall@1000")
            },
        }
        report["last_updated"] = datetime.now().isoformat(timespec="seconds")
        save_json(report, output_path)
        logger.info(
            "[RERANK/%s] BM25 nDCG@10=%.5f | reranked=%.5f | delta=%+.5f | BM25 R@100=%.4f | elapsed=%.1fs | cache_hits=%d/%d",
            domain, bm25_metrics["nDCG@10"], reranked_metrics["nDCG@10"], delta["nDCG@10"],
            bm25_metrics["Recall@100"], result["elapsed_seconds"], cache_hits, len(val_samples),
        )
        logger.info("[RERANK/MACRO-COMPLETED] %s", report["macro_average_completed_domains"])
        logger.info("[RERANK] Partial report saved: %s", output_path)

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_json(report, output_path)
    logger.info("Completed all requested domains. Final report: %s", output_path)
    print(f"\nFINAL_REPORT={output_path}")


if __name__ == "__main__":
    main()
