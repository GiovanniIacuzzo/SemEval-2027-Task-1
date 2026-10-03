#!/usr/bin/env python3
"""
subtrack_2a/dataset/dataset.py

Data pipeline per RETECO SemEval-2027 Sub-track 2a.

Responsabilità del modulo:
    1. Rappresentazione dei turni conversazionali.
    2. Costruzione deterministica di query + conversation history.
    3. Caricamento di corpus, benchmark e qrels.
    4. PyTorch Dataset per il contrastive training.
    5. Collate function per query/positive/negative.
    6. Domain-balanced batch sampler.

Il retrieval BM25 e il mining degli hard negatives sono implementati
separatamente in:
    subtrack_2a/retrieval/retrieval.py

Nota:
    Gli ID dei documenti vengono mantenuti esattamente come nel dataset.
    Il dominio è memorizzato separatamente e non viene concatenato agli ID.
"""

from __future__ import annotations

import json
import random
import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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
    """
    Rappresenta un singolo turno del benchmark RETECO Track 2a.

    topic_id:
        Identificatore usato nel run TREC e nei qrels.

    contextual_query:
        Testo effettivamente passato al retriever:
        query corrente + history eventualmente formattata/troncata.
    """

    domain: str
    conversation_id: str
    turn_id: int
    topic_id: str

    query: str
    history: str
    contextual_query: str

    gold_doc_ids: List[str]


# =============================================================================
# Query formatting
# =============================================================================

class ContextAwareQueryFormatter:
    """
    Costruisce la query di retrieval rispettando un budget token.

    Strategie supportate:
        - query_only
        - history
        - budget_context   (alias di history)
        - concat           (alias di history)

    La query corrente ha sempre priorità.
    Se la history supera il budget residuo, viene mantenuta la parte finale
    della history, cioè quella più recente.

    Non viene fatto parsing artificiale di "User:", "Assistant:", "Q:", "A:".
    La history viene trattata come testo già formattato dal benchmark.
    """

    HISTORY_ALIASES = {"history", "budget_context", "concat"}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_query_length: int = 512,
        query_instruction: str = "",
        strategy: str = "history",
    ):
        if max_query_length <= 0:
            raise ValueError("max_query_length deve essere > 0.")

        strategy = strategy.lower().strip()

        if strategy not in {"query_only", *self.HISTORY_ALIASES}:
            raise ValueError(
                f"Strategia query non valida: {strategy}. "
                f"Usa query_only, history, budget_context oppure concat."
            )

        self.tokenizer = tokenizer
        self.max_query_length = int(max_query_length)
        self.query_instruction = query_instruction.strip()
        self.strategy = strategy

        self.stats = {
            "total_queries": 0,
            "question_truncated": 0,
            "history_used": 0,
            "history_truncated": 0,
            "raw_query_tokens": 0,
            "raw_history_tokens": 0,
            "retained_history_tokens": 0,
            "final_tokens": 0,
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def _truncate_query_to_budget(self, query: str, budget: int) -> str:
        """
        Trunca la query solo nel caso estremo in cui non entri nemmeno
        senza history.

        Conserviamo i primi `budget` token. Nella pratica RETECO le query
        dovrebbero essere molto più corte di questo limite.
        """
        if budget <= 0:
            return ""

        ids = self.tokenizer.encode(
            query,
            add_special_tokens=False,
        )

        if len(ids) <= budget:
            return query

        self.stats["question_truncated"] += 1

        truncated = self.tokenizer.decode(
            ids[:budget],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        return truncated

    def _truncate_history_from_left(self, history: str, budget: int) -> str:
        """
        Mantiene gli ultimi `budget` token della history.

        In caso di history molto lunga preferiamo sacrificare i turni più
        vecchi piuttosto che interrompere la query corrente.
        """
        if budget <= 0 or not history:
            return ""

        history_ids = self.tokenizer.encode(
            history,
            add_special_tokens=False,
        )

        if len(history_ids) <= budget:
            return history

        self.stats["history_truncated"] += 1

        retained_ids = history_ids[-budget:]

        retained = self.tokenizer.decode(
            retained_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        return retained

    # -------------------------------------------------------------------------
    # Main formatter
    # -------------------------------------------------------------------------

    def format(self, query: str, history: str) -> str:
        self.stats["total_queries"] += 1

        query = (query or "").strip()
        history = (history or "").strip()

        if history.lower() == "no previous conversation.":
            history = ""

        self.stats["raw_query_tokens"] += self._count_tokens(query)

        if history:
            self.stats["raw_history_tokens"] += self._count_tokens(history)

        instruction = self.query_instruction
        prefix = f"{instruction} " if instruction else ""

        # ------------------------------------------------------------------
        # Query only
        # ------------------------------------------------------------------
        if self.strategy == "query_only" or not history:
            raw = f"{prefix}{query}"

            if self._count_tokens(raw) > self.max_query_length:
                available = max(
                    1,
                    self.max_query_length - self._count_tokens(prefix),
                )
                query = self._truncate_query_to_budget(query, available)
                raw = f"{prefix}{query}"

            final_len = self._count_tokens(raw)

            self.stats["final_tokens"] += min(
                final_len,
                self.max_query_length,
            )

            return raw.strip()

        # ------------------------------------------------------------------
        # Query + history
        # ------------------------------------------------------------------

        query_part = f"{prefix}{query}".strip()
        history_header = "\n\nConversation History:\n"

        # Quanto occupano query + header?
        fixed_text = f"{query_part}{history_header}"
        fixed_len = self._count_tokens(fixed_text)

        # Caso estremo: query troppo lunga anche senza history.
        if fixed_len > self.max_query_length:
            available_for_query = max(
                1,
                self.max_query_length
                - self._count_tokens(prefix)
                - self._count_tokens(history_header),
            )

            query = self._truncate_query_to_budget(
                query,
                available_for_query,
            )

            query_part = f"{prefix}{query}".strip()
            fixed_text = f"{query_part}{history_header}"
            fixed_len = self._count_tokens(fixed_text)

        residual_budget = max(
            0,
            self.max_query_length - fixed_len,
        )

        retained_history = self._truncate_history_from_left(
            history,
            residual_budget,
        )

        if retained_history:
            self.stats["history_used"] += 1
            retained_history_tokens = self._count_tokens(retained_history)
            self.stats["retained_history_tokens"] += retained_history_tokens
        else:
            retained_history_tokens = 0

        formatted = f"{fixed_text}{retained_history}".strip()

        final_len = self._count_tokens(formatted)

        self.stats["final_tokens"] += min(
            final_len,
            self.max_query_length,
        )

        return formatted

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, float]:
        n = max(1, self.stats["total_queries"])

        return {
            "total_queries": float(self.stats["total_queries"]),
            "pct_question_truncated": (
                100.0 * self.stats["question_truncated"] / n
            ),
            "pct_history_used": (
                100.0 * self.stats["history_used"] / n
            ),
            "pct_history_truncated": (
                100.0 * self.stats["history_truncated"] / n
            ),
            "avg_raw_query_tokens": (
                self.stats["raw_query_tokens"] / n
            ),
            "avg_raw_history_tokens": (
                self.stats["raw_history_tokens"] / n
            ),
            "avg_retained_history_tokens": (
                self.stats["retained_history_tokens"] / n
            ),
            "avg_final_tokens": (
                self.stats["final_tokens"] / n
            ),
        }


# =============================================================================
# Corpus / benchmark / qrels loaders
# =============================================================================

def load_corpus(
    documents_path: Union[str, Path],
) -> Dict[str, str]:
    """
    Carica documents.jsonl nel formato:
        {"doc_id": "...", "content": "..."}
    """
    path = Path(documents_path)

    if not path.exists():
        raise FileNotFoundError(f"Corpus non trovato: {path}")

    corpus: Dict[str, str] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
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

            if not text:
                continue

            corpus[doc_id] = text

    if not corpus:
        raise ValueError(f"Corpus vuoto: {path}")

    return corpus


def load_qrels(
    qrels_path: Union[str, Path],
) -> Dict[str, Dict[str, int]]:
    """
    Carica qrels in formato TREC:

        topic_id  0  doc_id  relevance
    """
    path = Path(qrels_path)

    if not path.exists():
        raise FileNotFoundError(f"Qrels non trovati: {path}")

    qrels: Dict[str, Dict[str, int]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
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
    """
    Supporta sia un singolo array JSON sia JSONL.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    if raw.startswith("["):
        data = json.loads(raw)

        if not isinstance(data, list):
            raise ValueError(f"Formato JSON inatteso in {path}")

        return data

    records = []

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
) -> List[ConversationalTurnSample]:
    """
    Carica benchmark_{split}.json e lo appiattisce in una lista di turni.
    """
    path = Path(benchmark_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark non trovato: {path}"
        )

    conversations = _load_json_or_jsonl(path)

    samples: List[ConversationalTurnSample] = []

    for conversation in conversations:
        conversation_id = str(conversation["id"])

        for turn in conversation.get("turns", []):
            turn_id = int(turn["turn_id"])

            query = str(turn.get("query", "")).strip()
            history = str(
                turn.get("conversation_history", "") or ""
            ).strip()

            gold_doc_ids = [
                str(doc_id)
                for doc_id in (
                    turn.get("gold_doc_ids")
                    or turn.get("supporting_doc_ids")
                    or []
                )
            ]

            topic_id = f"{conversation_id}_turn_{turn_id}"

            contextual_query = formatter.format(
                query=query,
                history=history,
            )

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
                )
            )

    return samples


def load_track2_domain_data(
    data_dir: Union[str, Path],
    domain: str,
    split: str = "train",
    formatter: Optional[ContextAwareQueryFormatter] = None,
) -> Tuple[
    Dict[str, str],
    List[ConversationalTurnSample],
    Dict[str, Dict[str, int]],
]:
    """
    Carica un dominio RETECO Track 2a.

    Struttura attesa:

        data_dir/
            domain/
                documents.jsonl
                benchmark_train.json
                benchmark_dev.json
                qrels_train.txt
                qrels_dev.txt
    """
    root = Path(data_dir)

    domain_dir = root / domain

    if not domain_dir.exists():
        raise FileNotFoundError(
            f"Cartella dominio non trovata: {domain_dir}"
        )

    documents_path = domain_dir / "documents.jsonl"

    benchmark_path = domain_dir / f"benchmark_{split}.json"
    if not benchmark_path.exists():
        benchmark_path = domain_dir / f"benchmark_{split}.jsonl"

    qrels_path = domain_dir / f"qrels_{split}.txt"

    corpus = load_corpus(documents_path)

    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark {split} non trovato per il dominio "
            f"{domain}: {benchmark_path}"
        )

    if not qrels_path.exists():
        raise FileNotFoundError(
            f"Qrels {split} non trovati per il dominio "
            f"{domain}: {qrels_path}"
        )

    if formatter is None:
        formatter = ContextAwareQueryFormatter(
            tokenizer=None,  # type: ignore[arg-type]
            max_query_length=512,
            strategy="history",
        )

        # Questo ramo esiste solo come fallback difensivo.
        # In produzione passeremo sempre un tokenizer reale.
        def _fallback_format(query: str, history: str) -> str:
            history = history.strip()
            if history.lower() == "no previous conversation.":
                history = ""

            if not history:
                return query.strip()

            return (
                f"{query.strip()}\n\n"
                f"Conversation History:\n"
                f"{history}"
            )

        formatter.format = _fallback_format  # type: ignore[method-assign]

    samples = load_benchmark_conversations(
        benchmark_path=benchmark_path,
        domain=domain,
        formatter=formatter,
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
) -> Tuple[
    List[ConversationalTurnSample],
    List[ConversationalTurnSample],
]:
    """
    Split train/validation a livello di conversazione.

    Nessuna conversazione viene spezzata tra train e validation.
    Questo evita leakage tra turni della stessa conversazione.
    """
    if not samples:
        return [], []

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(
            f"val_ratio deve essere compreso tra 0 e 1: {val_ratio}"
        )

    conversation_ids = sorted(
        {sample.conversation_id for sample in samples}
    )

    rng = random.Random(seed)
    rng.shuffle(conversation_ids)

    n_val = max(
        1,
        int(round(len(conversation_ids) * val_ratio)),
    )

    if n_val >= len(conversation_ids):
        n_val = len(conversation_ids) - 1

    val_conversation_ids = set(
        conversation_ids[:n_val]
    )

    train_samples = [
        sample
        for sample in samples
        if sample.conversation_id not in val_conversation_ids
    ]

    val_samples = [
        sample
        for sample in samples
        if sample.conversation_id in val_conversation_ids
    ]

    return train_samples, val_samples


# =============================================================================
# PyTorch training dataset
# =============================================================================

class RETECO2aTrainDataset(Dataset):
    """
    Dataset contrastivo per Sub-track 2a.

    Per ogni turno:
        query
        positive
        K negatives

    Quando un topic ha più gold documents, il positive viene ruotato
    deterministicamente tra le epoche.

    Gli hard negatives vengono forniti dall'esterno tramite:
        hard_negatives[topic_id] -> [doc_id, ...]

    Il dataset non esegue BM25.
    """

    VALID_STRATEGIES = {
        "bm25_hard",
        "mixed",
        "random",
    }

    def __init__(
        self,
        samples: List[ConversationalTurnSample],
        corpus: Dict[str, str],
        negatives_per_positive: int = 4,
        hard_negatives: Optional[Dict[str, List[str]]] = None,
        sampling_strategy: str = "bm25_hard",
        seed: int = 42,
    ):
        super().__init__()

        if not corpus:
            raise ValueError("Il corpus di training è vuoto.")

        sampling_strategy = sampling_strategy.lower().strip()

        if sampling_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Strategia negative sampling non valida: "
                f"{sampling_strategy}"
            )

        self.corpus = corpus
        self.corpus_keys = list(corpus.keys())

        self.negatives_per_positive = max(
            1,
            int(negatives_per_positive),
        )

        self.hard_negatives = hard_negatives or {}
        self.sampling_strategy = sampling_strategy

        self.seed = int(seed)
        self.epoch = 0

        self.valid_samples: List[ConversationalTurnSample] = []

        for sample in samples:
            valid_golds = [
                doc_id
                for doc_id in sample.gold_doc_ids
                if doc_id in self.corpus
            ]

            if valid_golds:
                # Copia del sample per non modificare l'oggetto originale.
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
                    )
                )

        if not self.valid_samples:
            raise ValueError(
                "Nessun training sample valido: "
                "nessun gold document è presente nel corpus."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.valid_samples)

    # -------------------------------------------------------------------------
    # Negative sampling
    # -------------------------------------------------------------------------

    def _random_negative_ids(
        self,
        gold_ids: set[str],
        excluded: set[str],
        n: int,
        idx: int,
    ) -> List[str]:
        """
        Sampling deterministico ma diverso per esempio/epoca.
        """
        if n <= 0:
            return []

        available = [
            doc_id
            for doc_id in self.corpus_keys
            if doc_id not in gold_ids
            and doc_id not in excluded
        ]

        if not available:
            return []

        rng = random.Random(
            self.seed
            + self.epoch * 1_000_003
            + idx * 9_176
        )

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

        hard_ids = []
        seen = set(gold_ids)

        for doc_id in self.hard_negatives.get(sample.topic_id, []):
            if doc_id not in self.corpus:
                continue

            if doc_id in seen:
                continue

            hard_ids.append(doc_id)
            seen.add(doc_id)

        k = self.negatives_per_positive

        # -------------------------------------------------------------
        # BM25 hard negatives
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # Random negatives
        # -------------------------------------------------------------
        if self.sampling_strategy == "random":
            return self._random_negative_ids(
                gold_ids=gold_ids,
                excluded=set(),
                n=k,
                idx=idx,
            )[:k]

        # -------------------------------------------------------------
        # Mixed:
        # metà hard, metà random.
        # -------------------------------------------------------------
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

        # Nel raro caso in cui i random non siano sufficienti,
        # proviamo a utilizzare ulteriori hard negatives.
        if len(selected) < k:
            for doc_id in hard_ids:
                if doc_id not in selected:
                    selected.append(doc_id)

                if len(selected) >= k:
                    break

        return selected[:k]

    # -------------------------------------------------------------------------
    # Main item
    # -------------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.valid_samples[idx]
        gold_ids = sample.gold_doc_ids

        # Rotazione del positive tra le epoche.
        positive_index = (
            self.epoch + idx
        ) % len(gold_ids)

        positive_id = gold_ids[positive_index]
        positive_text = self.corpus[positive_id]

        negative_ids = self._select_negatives(
            sample=sample,
            idx=idx,
        )

        if not negative_ids:
            raise RuntimeError(
                f"Nessun negative disponibile per topic "
                f"{sample.topic_id}."
            )

        negative_texts = [
            self.corpus[doc_id]
            for doc_id in negative_ids
        ]

        return {
            "topic_id": sample.topic_id,
            "domain": sample.domain,

            "query": sample.contextual_query,

            "positive": positive_text,
            "positive_id": positive_id,

            "negatives": negative_texts,
            "negative_ids": negative_ids,
        }


# =============================================================================
# Collate
# =============================================================================

class ConversationalCollateFn:
    """
    Tokenizza query, positive e negativi.

    Tutti gli esempi del batch vengono portati allo stesso numero di
    negatives K, usando il minimo K disponibile nel batch.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_query_len: int = 512,
        max_doc_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_query_len = int(max_query_len)
        self.max_doc_len = int(max_doc_len)

    def __call__(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not batch:
            raise ValueError("Collate ricevuto un batch vuoto.")

        batch_size = len(batch)

        queries = [
            item["query"]
            for item in batch
        ]

        positives = [
            item["positive"]
            for item in batch
        ]

        k_negs = min(
            len(item["negatives"])
            for item in batch
        )

        if k_negs <= 0:
            raise ValueError(
                "Almeno un esempio del batch non possiede negativi."
            )

        # ------------------------------------------------------------------
        # Flatten dei negativi
        # ------------------------------------------------------------------

        flat_negatives = []

        for item in batch:
            flat_negatives.extend(
                item["negatives"][:k_negs]
            )

        # ------------------------------------------------------------------
        # Query
        # ------------------------------------------------------------------

        query_tokens = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_len,
            return_tensors="pt",
        )

        # ------------------------------------------------------------------
        # DOCUMENTI
        #
        # Positivi + negativi vengono tokenizzati INSIEME.
        # In questo modo hanno necessariamente la stessa sequence length
        # e possono essere concatenati nel forward del bi-encoder.
        # ------------------------------------------------------------------

        all_documents = positives + flat_negatives

        all_document_tokens = self.tokenizer(
            all_documents,
            padding=True,
            truncation=True,
            max_length=self.max_doc_len,
            return_tensors="pt",
        )

        # Prima parte = positivi
        positive_tokens = {
            key: value[:batch_size]
            for key, value in all_document_tokens.items()
        }

        # Seconda parte = negativi
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

            "topic_ids": [
                item["topic_id"]
                for item in batch
            ],

            "domains": [
                item["domain"]
                for item in batch
            ],

            "positive_ids": [
                item["positive_id"]
                for item in batch
            ],

            "negative_ids": [
                item["negative_ids"][:k_negs]
                for item in batch
            ],
        }

# =============================================================================
# Domain-balanced batch sampler
# =============================================================================

class DomainBalancedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler domain-balanced.

    Ogni batch appartiene ad un singolo dominio.

    I domini vengono alternati in modo deterministico:

        biology batch
        drones batch
        earth_science batch
        ...

    Quando un dominio esaurisce i propri esempi, viene reshuffled e
    ricampionato. Questo consente di mantenere lo stesso numero di
    batch per dominio senza costruire batch artificialmente misti.

    Questo comportamento è intenzionale: rende gli in-batch negatives
    più semanticamente difficili rispetto a batch composti da domini
    differenti.

    `set_epoch()` deve essere chiamato dal training loop all'inizio di
    ogni epoca per ottenere un nuovo ordinamento deterministico.
    """

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

        self.domain_to_indices: Dict[str, List[int]] = (
            collections.defaultdict(list)
        )

        for idx, sample in enumerate(samples):
            self.domain_to_indices[sample.domain].append(idx)

        self.domains = sorted(
            self.domain_to_indices.keys()
        )

        if not self.domains:
            raise ValueError(
                "Nessun dominio valido trovato nei sample."
            )

        # Manteniamo esattamente lo stesso numero di batch che avrebbe
        # un DataLoader standard con drop_last=True.
        self.num_batches = len(samples) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(
            self.seed + self.epoch * 1_000_003
        )

        # Una pool separata per ogni dominio.
        pools = {
            domain: rng.sample(
                indices,
                len(indices),
            )
            for domain, indices in self.domain_to_indices.items()
        }

        pointers = {
            domain: 0
            for domain in self.domains
        }

        for batch_idx in range(self.num_batches):
            domain = self.domains[
                batch_idx % len(self.domains)
            ]

            pool = pools[domain]
            pointer = pointers[domain]

            batch: List[int] = []

            while len(batch) < self.batch_size:
                remaining = len(pool) - pointer

                if remaining <= 0:
                    # Ricomincia un nuovo ciclo per il dominio.
                    pool = rng.sample(
                        self.domain_to_indices[domain],
                        len(self.domain_to_indices[domain]),
                    )

                    pools[domain] = pool
                    pointer = 0
                    remaining = len(pool)

                take = min(
                    self.batch_size - len(batch),
                    remaining,
                )

                batch.extend(
                    pool[pointer:pointer + take]
                )

                pointer += take

            pointers[domain] = pointer

            yield batch

    def __len__(self) -> int:
        return self.num_batches
    