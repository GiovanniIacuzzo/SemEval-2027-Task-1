#!/usr/bin/env python3
"""
subtrack_2a/dataset/dataset.py

Data pipeline per RETECO SemEval-2027 Sub-track 2a.

Responsabilità:
    1. Rappresentazione dei turni conversazionali.
    2. Formattazione della query contestualizzata e della query riscritta.
    3. Caricamento di corpus, benchmark, qrels e cache delle query riscritte.
    4. PyTorch Dataset per il contrastive training.
    5. Collate function per query/positive/negative.
    6. Domain-balanced batch sampler.

La generazione LLM non viene mai eseguita dentro questo modulo o dentro
__getitem__: le riscritture devono essere precompute con
ConversationalQueryRewriter e salvate tramite save_query_rewrites().

Il corpus di retrieval rimane esclusivamente quello ufficiale RETECO.
Gli ID dei documenti sono conservati esattamente come nel dataset.
"""

from __future__ import annotations

import collections
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizerBase


# =============================================================================
# Costanti
# =============================================================================

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


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ConversationalTurnSample:
    """Singolo turno del benchmark RETECO Track 2a."""

    domain: str
    conversation_id: str
    turn_id: int
    topic_id: str
    query: str
    history: str
    contextual_query: str
    gold_doc_ids: List[str]
    # Query riscritta dal generatore, già formattata con l'eventuale istruzione
    # task-specific dell'encoder. Vuota se non è stata caricata una cache.
    reasoned_query: str = ""


# =============================================================================
# Query formatting
# =============================================================================

class ContextAwareQueryFormatter:
    """
    Format delle query compatibile con BGE e Qwen3-Embedding.

    Strategie contestuali:
        - query_only
        - history
        - budget_context (alias di history)
        - concat (alias di history)

    La query corrente ha priorità sul contesto. Con Qwen3-Embedding passare
    ``query_instruction`` per anteporre l'istruzione richiesta dal modello.
    L'istruzione viene applicata soltanto alle query, mai ai documenti.
    """

    HISTORY_ALIASES = {"history", "budget_context", "concat"}

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase],
        max_query_length: int = 512,
        query_instruction: str = "",
        strategy: str = "history",
    ):
        if max_query_length <= 0:
            raise ValueError("max_query_length deve essere > 0.")

        strategy = strategy.lower().strip()
        if strategy not in {"query_only", *self.HISTORY_ALIASES}:
            raise ValueError(
                f"Strategia query non valida: {strategy}. Usa query_only, "
                "history, budget_context oppure concat."
            )

        self.tokenizer = tokenizer
        self.max_query_length = int(max_query_length)
        self.query_instruction = (query_instruction or "").strip()
        self.strategy = strategy
        self.stats = {
            "total_queries": 0,
            "formatted_rewrites": 0,
            "question_truncated": 0,
            "history_used": 0,
            "history_truncated": 0,
            "raw_query_tokens": 0,
            "raw_history_tokens": 0,
            "retained_history_tokens": 0,
            "final_tokens": 0,
        }

    def _encode(self, text: str) -> List[Any]:
        if self.tokenizer is None:
            # Fallback difensivo per chiamate diagnostiche senza tokenizer.
            return (text or "").split()
        return self.tokenizer.encode(text or "", add_special_tokens=False)

    def _decode(self, ids: Sequence[Any]) -> str:
        if self.tokenizer is None:
            return " ".join(str(x) for x in ids).strip()
        return self.tokenizer.decode(
            list(ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

    def _count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def _truncate_query_to_budget(self, query: str, budget: int) -> str:
        if budget <= 0:
            return ""
        ids = self._encode(query)
        if len(ids) <= budget:
            return query
        self.stats["question_truncated"] += 1
        return self._decode(ids[:budget])

    def _truncate_history_from_left(self, history: str, budget: int) -> str:
        if budget <= 0 or not history:
            return ""
        history_ids = self._encode(history)
        if len(history_ids) <= budget:
            return history
        self.stats["history_truncated"] += 1
        return self._decode(history_ids[-budget:])

    def _instruction_prefix(self) -> str:
        return f"{self.query_instruction} " if self.query_instruction else ""

    def format_rewritten(self, rewritten_query: str) -> str:
        """Applica l'istruzione encoder e il budget a una query già riscritta.

        ``rewritten_query`` deve essere il testo grezzo prodotto dall'LLM; la
        cache deve contenere quel testo grezzo, non il prefisso dell'encoder.
        """
        query = (rewritten_query or "").strip()
        if not query:
            return ""

        prefix = self._instruction_prefix()
        prefix_tokens = self._count_tokens(prefix)
        available = self.max_query_length - prefix_tokens
        if available <= 0:
            raise ValueError(
                "query_instruction occupa da sola il budget max_query_length."
            )
        query = self._truncate_query_to_budget(query, available)
        self.stats["formatted_rewrites"] += 1
        return f"{prefix}{query}".strip()

    def format(self, query: str, history: str) -> str:
        self.stats["total_queries"] += 1
        query = (query or "").strip()
        history = (history or "").strip()
        if history.lower() == "no previous conversation.":
            history = ""

        self.stats["raw_query_tokens"] += self._count_tokens(query)
        if history:
            self.stats["raw_history_tokens"] += self._count_tokens(history)

        prefix = self._instruction_prefix()

        # ------------------------------------------------------------------
        # Query corrente senza cronologia
        # ------------------------------------------------------------------
        if self.strategy == "query_only" or not history:
            prefix_tokens = self._count_tokens(prefix)
            available = max(1, self.max_query_length - prefix_tokens)
            query = self._truncate_query_to_budget(query, available)
            formatted = f"{prefix}{query}".strip()
            self.stats["final_tokens"] += min(
                self._count_tokens(formatted), self.max_query_length
            )
            return formatted

        # ------------------------------------------------------------------
        # Query + history sotto lo stesso budget
        # ------------------------------------------------------------------
        query_part = f"{prefix}{query}".strip()
        history_header = "\n\nConversation History:\n"
        fixed_text = f"{query_part}{history_header}"
        fixed_len = self._count_tokens(fixed_text)

        if fixed_len > self.max_query_length:
            prefix_tokens = self._count_tokens(prefix)
            header_tokens = self._count_tokens(history_header)
            available_for_query = max(
                1,
                self.max_query_length - prefix_tokens - header_tokens,
            )
            query = self._truncate_query_to_budget(query, available_for_query)
            query_part = f"{prefix}{query}".strip()
            fixed_text = f"{query_part}{history_header}"
            fixed_len = self._count_tokens(fixed_text)

        residual_budget = max(0, self.max_query_length - fixed_len)
        retained_history = self._truncate_history_from_left(
            history,
            residual_budget,
        )
        if retained_history:
            self.stats["history_used"] += 1
            self.stats["retained_history_tokens"] += self._count_tokens(
                retained_history
            )

        formatted = f"{fixed_text}{retained_history}".strip()
        self.stats["final_tokens"] += min(
            self._count_tokens(formatted), self.max_query_length
        )
        return formatted

    def get_diagnostics(self) -> Dict[str, float]:
        n = max(1, self.stats["total_queries"])
        return {
            "total_queries": float(self.stats["total_queries"]),
            "formatted_rewrites": float(self.stats["formatted_rewrites"]),
            "pct_question_truncated": (
                100.0 * self.stats["question_truncated"] / n
            ),
            "pct_history_used": 100.0 * self.stats["history_used"] / n,
            "pct_history_truncated": (
                100.0 * self.stats["history_truncated"] / n
            ),
            "avg_raw_query_tokens": self.stats["raw_query_tokens"] / n,
            "avg_raw_history_tokens": self.stats["raw_history_tokens"] / n,
            "avg_retained_history_tokens": (
                self.stats["retained_history_tokens"] / n
            ),
            "avg_final_tokens": self.stats["final_tokens"] / n,
        }


# =============================================================================
# Query rewrite cache
# =============================================================================

def load_query_rewrites(
    rewrites_path: Union[str, Path],
) -> Dict[str, str]:
    """Carica la cache ``topic_id -> query riscritta``.

    Formati accettati:
        1. JSON prodotto da save_query_rewrites(), con chiave ``rewrites``;
        2. JSON semplice ``{topic_id: query}``;
        3. JSONL con record ``{"topic_id": ..., "rewritten_query": ...}``.
    """
    path = Path(rewrites_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache query riscritte non trovata: {path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    rewrites: Dict[str, str] = {}
    if isinstance(parsed, dict):
        content = parsed.get("rewrites", parsed)
        if not isinstance(content, dict):
            raise ValueError(f"Campo 'rewrites' non valido in {path}.")
        for topic_id, query in content.items():
            if isinstance(query, str) and query.strip():
                rewrites[str(topic_id)] = query.strip()
        return rewrites

    # Se non era un singolo JSON, interpreta il file come JSONL.
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSONL non valido in {path}, riga {line_no}."
            ) from exc
        topic_id = record.get("topic_id", record.get("id"))
        query = record.get(
            "rewritten_query",
            record.get("reasoned_query", record.get("query")),
        )
        if topic_id is None or not isinstance(query, str) or not query.strip():
            continue
        rewrites[str(topic_id)] = query.strip()
    return rewrites


def save_query_rewrites(
    rewrites_path: Union[str, Path],
    rewrites: Dict[str, str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Salva la cache in modo atomico, includendo i metadati di riproducibilità."""
    path = Path(rewrites_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata or {}),
        "rewrites": {
            str(topic_id): str(query).strip()
            for topic_id, query in sorted(rewrites.items())
            if str(query).strip()
        },
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


# =============================================================================
# Corpus / benchmark / qrels loaders
# =============================================================================

def load_corpus(documents_path: Union[str, Path]) -> Dict[str, str]:
    """Carica documents.jsonl nel formato ``{doc_id, content}``."""
    path = Path(documents_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus non trovato: {path}")

    corpus: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON non valido in {path}, riga {line_no}"
                ) from exc

            doc_id = item.get("doc_id", item.get("id"))
            text = item.get("content", item.get("text", ""))
            if doc_id is None:
                raise ValueError(
                    f"Documento senza ID in {path}, riga {line_no}"
                )
            doc_id = str(doc_id)
            text = str(text or "").strip()
            if text:
                corpus[doc_id] = text

    if not corpus:
        raise ValueError(f"Corpus vuoto: {path}")
    return corpus


def load_qrels(qrels_path: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """Carica qrels in formato TREC: ``topic_id 0 doc_id relevance``."""
    path = Path(qrels_path)
    if not path.exists():
        raise FileNotFoundError(f"Qrels non trovati: {path}")

    qrels: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 4:
                raise ValueError(
                    f"Qrels malformati in {path}, riga {line_no}: {line}"
                )
            topic_id, _, doc_id, relevance = parts[:4]
            qrels.setdefault(topic_id, {})[doc_id] = int(relevance)
    return qrels


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Supporta sia un singolo array JSON sia JSONL."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Formato JSON inatteso in {path}")
        return data

    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON non valido in {path}, riga {line_no}"
            ) from exc
    return records


def load_benchmark_conversations(
    benchmark_path: Union[str, Path],
    domain: str,
    formatter: ContextAwareQueryFormatter,
    query_rewrites: Optional[Dict[str, str]] = None,
) -> List[ConversationalTurnSample]:
    """Carica benchmark_{split}.json e lo appiattisce in turni."""
    path = Path(benchmark_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark non trovato: {path}")

    conversations = _load_json_or_jsonl(path)
    rewrite_map = query_rewrites or {}
    samples: List[ConversationalTurnSample] = []

    for conversation in conversations:
        conversation_id = str(conversation["id"])
        for turn in conversation.get("turns", []):
            turn_id = int(turn["turn_id"])
            query = str(turn.get("query", "")).strip()
            history = str(turn.get("conversation_history", "") or "").strip()
            gold_doc_ids = [
                str(doc_id)
                for doc_id in (
                    turn.get("gold_doc_ids")
                    or turn.get("supporting_doc_ids")
                    or []
                )
            ]
            topic_id = f"{conversation_id}_turn_{turn_id}"
            contextual_query = formatter.format(query=query, history=history)
            raw_rewrite = rewrite_map.get(topic_id, "")
            reasoned_query = formatter.format_rewritten(raw_rewrite) if raw_rewrite else ""

            samples.append(
                ConversationalTurnSample(
                    domain=domain,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    topic_id=topic_id,
                    query=query,
                    history=history,
                    contextual_query=contextual_query,
                    gold_doc_ids=gold_doc_ids,
                    reasoned_query=reasoned_query,
                )
            )
    return samples


def load_track2_domain_data(
    data_dir: Union[str, Path],
    domain: str,
    split: str = "train",
    formatter: Optional[ContextAwareQueryFormatter] = None,
    query_rewrites: Optional[Dict[str, str]] = None,
) -> Tuple[
    Dict[str, str],
    List[ConversationalTurnSample],
    Dict[str, Dict[str, int]],
]:
    """Carica corpus, benchmark, samples e qrels di un dominio RETECO Track 2a."""
    root = Path(data_dir)
    domain_dir = root / domain
    if not domain_dir.exists():
        raise FileNotFoundError(f"Cartella dominio non trovata: {domain_dir}")

    documents_path = domain_dir / "documents.jsonl"
    benchmark_path = domain_dir / f"benchmark_{split}.json"
    if not benchmark_path.exists():
        benchmark_path = domain_dir / f"benchmark_{split}.jsonl"
    qrels_path = domain_dir / f"qrels_{split}.txt"

    corpus = load_corpus(documents_path)
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark {split} non trovato per il dominio {domain}: {benchmark_path}"
        )
    if not qrels_path.exists():
        raise FileNotFoundError(
            f"Qrels {split} non trovati per il dominio {domain}: {qrels_path}"
        )

    if formatter is None:
        formatter = ContextAwareQueryFormatter(
            tokenizer=None,
            max_query_length=512,
            strategy="history",
        )

    samples = load_benchmark_conversations(
        benchmark_path=benchmark_path,
        domain=domain,
        formatter=formatter,
        query_rewrites=query_rewrites,
    )
    qrels = load_qrels(qrels_path)
    return corpus, samples, qrels


# =============================================================================
# Conversation-level split
# =============================================================================

def split_conversations_train_val(
    samples: List[ConversationalTurnSample],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ConversationalTurnSample], List[ConversationalTurnSample]]:
    """Split train/validation a livello di conversazione per evitare leakage."""
    if not samples:
        return [], []
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio deve essere compreso tra 0 e 1: {val_ratio}")

    conversation_ids = sorted({sample.conversation_id for sample in samples})
    if len(conversation_ids) < 2:
        raise ValueError(
            "Servono almeno due conversazioni distinte per creare train/validation."
        )

    rng = random.Random(seed)
    rng.shuffle(conversation_ids)
    n_val = max(1, int(round(len(conversation_ids) * val_ratio)))
    n_val = min(n_val, len(conversation_ids) - 1)
    val_conversation_ids = set(conversation_ids[:n_val])

    train_samples = [
        sample for sample in samples
        if sample.conversation_id not in val_conversation_ids
    ]
    val_samples = [
        sample for sample in samples
        if sample.conversation_id in val_conversation_ids
    ]
    return train_samples, val_samples


# =============================================================================
# PyTorch training dataset
# =============================================================================

class RETECO2aTrainDataset(Dataset):
    """
    Dataset contrastivo per Sub-track 2a.

    Per ciascun turno vengono mantenuti query contestualizzata e query riscritta.
    ``query_mode`` può essere:
        - ``contextual``: usa sempre query + cronologia;
        - ``reasoned``: usa la riscrittura quando disponibile;
        - ``alternate``: alterna query contestualizzata e riscritta tra epoche.

    ``alternate`` è utile per far apprendere entrambe le formulazioni senza
    duplicare il medesimo topic nello stesso batch: la duplicazione causerebbe
    potenziali falsi negativi nella loss in-batch. Il training loop deve chiamare
    set_epoch(epoch) sul dataset all'inizio di ogni epoca.

    Gli hard negatives sono precomputati esternamente:
        hard_negatives[topic_id] -> [doc_id, ...]
    Il dataset non esegue BM25 né invoca un LLM.
    """

    VALID_STRATEGIES = {"bm25_hard", "mixed", "random"}
    VALID_QUERY_MODES = {"contextual", "reasoned", "alternate"}

    def __init__(
        self,
        samples: List[ConversationalTurnSample],
        corpus: Dict[str, str],
        negatives_per_positive: int = 4,
        hard_negatives: Optional[Dict[str, List[str]]] = None,
        sampling_strategy: str = "bm25_hard",
        seed: int = 42,
        query_mode: str = "alternate",
        require_reasoned_queries: bool = False,
    ):
        super().__init__()
        if not corpus:
            raise ValueError("Il corpus di training è vuoto.")

        sampling_strategy = sampling_strategy.lower().strip()
        if sampling_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Strategia negative sampling non valida: {sampling_strategy}"
            )
        query_mode = query_mode.lower().strip()
        if query_mode not in self.VALID_QUERY_MODES:
            raise ValueError(
                f"query_mode non valido: {query_mode}; usare {sorted(self.VALID_QUERY_MODES)}"
            )

        self.corpus = corpus
        self.corpus_keys = list(corpus.keys())
        self.negatives_per_positive = max(1, int(negatives_per_positive))
        self.hard_negatives = hard_negatives or {}
        self.sampling_strategy = sampling_strategy
        self.seed = int(seed)
        self.epoch = 0
        self.query_mode = query_mode
        self.require_reasoned_queries = bool(require_reasoned_queries)

        self.valid_samples: List[ConversationalTurnSample] = []
        for sample in samples:
            valid_golds = [doc_id for doc_id in sample.gold_doc_ids if doc_id in corpus]
            if not valid_golds:
                continue
            self.valid_samples.append(
                ConversationalTurnSample(
                    domain=sample.domain,
                    conversation_id=sample.conversation_id,
                    turn_id=sample.turn_id,
                    topic_id=sample.topic_id,
                    query=sample.query,
                    history=sample.history,
                    contextual_query=sample.contextual_query,
                    gold_doc_ids=valid_golds,
                    reasoned_query=sample.reasoned_query,
                )
            )

        if not self.valid_samples:
            raise ValueError(
                "Nessun training sample valido: nessun gold document è presente nel corpus."
            )

        self.samples_with_reasoned_query = sum(
            bool(sample.reasoned_query.strip()) for sample in self.valid_samples
        )
        if self.require_reasoned_queries and self.samples_with_reasoned_query < len(self.valid_samples):
            missing = len(self.valid_samples) - self.samples_with_reasoned_query
            raise ValueError(
                f"Mancano query riscritte per {missing}/{len(self.valid_samples)} "
                "training sample, ma require_reasoned_queries=True."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.valid_samples)

    def get_diagnostics(self) -> Dict[str, Union[int, float, str]]:
        total = len(self.valid_samples)
        covered = self.samples_with_reasoned_query
        return {
            "num_samples": total,
            "query_mode": self.query_mode,
            "samples_with_reasoned_query": covered,
            "samples_without_reasoned_query": total - covered,
            "reasoned_query_coverage_pct": 100.0 * covered / max(1, total),
        }

    # ------------------------------------------------------------------
    # Negative sampling
    # ------------------------------------------------------------------

    def _random_negative_ids(
        self,
        gold_ids: set[str],
        excluded: set[str],
        n: int,
        idx: int,
    ) -> List[str]:
        if n <= 0:
            return []
        available = [
            doc_id for doc_id in self.corpus_keys
            if doc_id not in gold_ids and doc_id not in excluded
        ]
        if not available:
            return []
        rng = random.Random(self.seed + self.epoch * 1_000_003 + idx * 9_176)
        if len(available) <= n:
            rng.shuffle(available)
            return available
        return rng.sample(available, n)

    def _select_negatives(
        self,
        sample: ConversationalTurnSample,
        idx: int,
    ) -> List[str]:
        gold_ids = set(sample.gold_doc_ids)
        hard_ids: List[str] = []
        seen = set(gold_ids)
        for doc_id in self.hard_negatives.get(sample.topic_id, []):
            if doc_id not in self.corpus or doc_id in seen:
                continue
            hard_ids.append(doc_id)
            seen.add(doc_id)

        k = self.negatives_per_positive
        if self.sampling_strategy == "bm25_hard":
            selected = hard_ids[:k]
            if len(selected) < k:
                selected.extend(
                    self._random_negative_ids(
                        gold_ids=gold_ids,
                        excluded=set(selected),
                        n=k - len(selected),
                        idx=idx,
                    )
                )
            return selected[:k]

        if self.sampling_strategy == "random":
            return self._random_negative_ids(
                gold_ids=gold_ids,
                excluded=set(),
                n=k,
                idx=idx,
            )[:k]

        # mixed: priorità agli hard negative, completamento con random.
        hard_target = (k + 1) // 2
        selected = hard_ids[:hard_target]
        random_needed = k - len(selected)
        if random_needed > 0:
            selected.extend(
                self._random_negative_ids(
                    gold_ids=gold_ids,
                    excluded=set(selected),
                    n=random_needed,
                    idx=idx,
                )
            )
        if len(selected) < k:
            for doc_id in hard_ids:
                if doc_id not in selected:
                    selected.append(doc_id)
                if len(selected) >= k:
                    break
        return selected[:k]

    def _select_query(self, sample: ConversationalTurnSample, idx: int) -> Tuple[str, str]:
        contextual = sample.contextual_query
        reasoned = sample.reasoned_query
        if self.query_mode == "contextual" or not reasoned:
            return contextual, "contextual"
        if self.query_mode == "reasoned":
            return reasoned, "reasoned"
        # Alternanza per epoca e indice. Ogni sample cambia variante tra epoche
        # successive, ma appare una sola volta nel dataset in ciascuna epoca.
        if (self.epoch + idx) % 2 == 1:
            return reasoned, "reasoned"
        return contextual, "contextual"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.valid_samples[idx]
        gold_ids = sample.gold_doc_ids
        positive_index = (self.epoch + idx) % len(gold_ids)
        positive_id = gold_ids[positive_index]
        positive_text = self.corpus[positive_id]

        negative_ids = self._select_negatives(sample=sample, idx=idx)
        if not negative_ids:
            raise RuntimeError(f"Nessun negative disponibile per topic {sample.topic_id}.")
        negative_texts = [self.corpus[doc_id] for doc_id in negative_ids]
        query_text, query_variant = self._select_query(sample, idx)

        return {
            "topic_id": sample.topic_id,
            "domain": sample.domain,
            "query": query_text,
            "query_variant": query_variant,
            "original_query": sample.query,
            "contextual_query": sample.contextual_query,
            "reasoned_query": sample.reasoned_query,
            "positive": positive_text,
            "positive_id": positive_id,
            "negatives": negative_texts,
            "negative_ids": negative_ids,
        }


# =============================================================================
# Collate
# =============================================================================

class ConversationalCollateFn:
    """Tokenizza query, positivi e negativi per il forward del bi-encoder."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_query_len: int = 512,
        max_doc_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_query_len = int(max_query_len)
        self.max_doc_len = int(max_doc_len)
        if self.max_query_len <= 0 or self.max_doc_len <= 0:
            raise ValueError("max_query_len e max_doc_len devono essere > 0.")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Collate ricevuto un batch vuoto.")
        batch_size = len(batch)
        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]
        k_negs = min(len(item["negatives"]) for item in batch)
        if k_negs <= 0:
            raise ValueError("Almeno un esempio del batch non possiede negativi.")

        flat_negatives: List[str] = []
        for item in batch:
            flat_negatives.extend(item["negatives"][:k_negs])

        query_tokens = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_len,
            return_tensors="pt",
        )
        all_documents = positives + flat_negatives
        all_document_tokens = self.tokenizer(
            all_documents,
            padding=True,
            truncation=True,
            max_length=self.max_doc_len,
            return_tensors="pt",
        )

        positive_tokens = {
            key: value[:batch_size]
            for key, value in all_document_tokens.items()
        }
        negative_tokens = {
            key: value[batch_size:]
            for key, value in all_document_tokens.items()
        }

        return {
            "query_inputs": query_tokens,
            "pos_inputs": positive_tokens,
            "neg_inputs": negative_tokens,
            "batch_size": batch_size,
            "k_negs": k_negs,
            "topic_ids": [item["topic_id"] for item in batch],
            "domains": [item["domain"] for item in batch],
            "query_variants": [item.get("query_variant", "contextual") for item in batch],
            "positive_ids": [item["positive_id"] for item in batch],
            "negative_ids": [item["negative_ids"][:k_negs] for item in batch],
        }


# =============================================================================
# Domain-balanced batch sampler
# =============================================================================

class DomainBalancedBatchSampler(Sampler[List[int]]):
    """Sampler che alterna batch omogenei per dominio in modo deterministico."""

    def __init__(
        self,
        samples: List[ConversationalTurnSample],
        batch_size: int,
        seed: int = 42,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size deve essere > 0.")
        if not samples:
            raise ValueError("Nessun sample fornito al sampler.")

        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0
        self.domain_to_indices: Dict[str, List[int]] = collections.defaultdict(list)
        for idx, sample in enumerate(samples):
            self.domain_to_indices[sample.domain].append(idx)
        self.domains = sorted(self.domain_to_indices.keys())
        if not self.domains:
            raise ValueError("Nessun dominio valido trovato nei sample.")
        self.num_batches = len(samples) // self.batch_size
        if self.num_batches == 0:
            raise ValueError(
                f"Dataset ({len(samples)} sample) più piccolo del batch_size "
                f"({self.batch_size}); ridurre il batch_size."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self.seed + self.epoch * 1_000_003)
        pools = {
            domain: rng.sample(indices, len(indices))
            for domain, indices in self.domain_to_indices.items()
        }
        pointers = {domain: 0 for domain in self.domains}

        for batch_idx in range(self.num_batches):
            domain = self.domains[batch_idx % len(self.domains)]
            pool = pools[domain]
            pointer = pointers[domain]
            batch: List[int] = []

            while len(batch) < self.batch_size:
                remaining = len(pool) - pointer
                if remaining <= 0:
                    pool = rng.sample(
                        self.domain_to_indices[domain],
                        len(self.domain_to_indices[domain]),
                    )
                    pools[domain] = pool
                    pointer = 0
                    remaining = len(pool)

                take = min(self.batch_size - len(batch), remaining)
                batch.extend(pool[pointer:pointer + take])
                pointer += take

            pointers[domain] = pointer
            yield batch

    def __len__(self) -> int:
        return self.num_batches
