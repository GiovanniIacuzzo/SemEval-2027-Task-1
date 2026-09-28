#!/usr/bin/env python3
"""
subtrack_2a/dataset/dataset.py

Modulo di gestione dati robusto per SemEval-2027 Sub-track 2a (RECOR).
Garantisce:
  - Context-Aware Query Truncation: la current question è prioritaria e non viene mai troncata.
  - Multi-positive fix: 1 istanza per turno conversazionale, rotazione deterministica del target positivo.
  - Esclusione rigorosa di TUTTI i gold document del topic dai negativi.
  - Mining di K Hard Negatives tramite BM25 conforme alle baseline ufficiali (k1=0.9, b=0.4).
  - Statistiche diagnostiche su token e lunghezze.
"""

import os
import re
import json
import math
import random
import logging
import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer

logger = logging.getLogger("RETECO_Dataset")

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


@dataclass
class ConversationalTurnSample:
    """Rappresentazione unificata di un turno conversazionale."""
    domain: str
    conversation_id: str
    turn_id: int
    topic_id: str
    query: str
    history: str
    contextual_query: str
    gold_doc_ids: List[str]
    answer: Optional[str] = None


# ==============================================================================
# 1. Context-Aware Query Truncation con Calcolo Esplicito del Budget
# ==============================================================================
class ContextAwareQueryFormatter:
    """
    Costruzione deterministica del contesto a budget.
    La domanda corrente ha priorità assoluta di allocazione token.
    La cronologia viene aggiunta a ritroso per turni interi dal più recente al più vecchio.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_query_length: int = 256,
        query_instruction: str = "",
        strategy: str = "budget_context",
    ):
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.query_instruction = query_instruction.strip()
        self.strategy = strategy

        self.stats = {
            "total_queries": 0,
            "question_truncated": 0,
            "history_fully_retained": 0,
            "history_partially_retained": 0,
            "raw_q_tokens": 0,
            "raw_hist_tokens": 0,
            "retained_hist_tokens": 0,
            "final_tokens": 0,
        }

    def format(self, query: str, history: str) -> str:
        self.stats["total_queries"] += 1
        query_clean = query.strip()
        history_clean = history.strip() if history else ""
        if history_clean.lower() == "no previous conversation.":
            history_clean = ""

        prefix = f"{self.query_instruction} " if self.query_instruction else ""

        # Diagnostica token grezzi
        q_raw_toks = self.tokenizer.tokenize(query_clean)
        self.stats["raw_q_tokens"] += len(q_raw_toks)

        if self.strategy == "query_only" or not history_clean:
            formatted = f"{prefix}Current Question: {query_clean}"
            final_toks = self.tokenizer.tokenize(formatted)
            self.stats["final_tokens"] += min(len(final_toks), self.max_query_length)
            return formatted

        hist_raw_toks = self.tokenizer.tokenize(history_clean)
        self.stats["raw_hist_tokens"] += len(hist_raw_toks)

        header_q = f"{prefix}Current Question: {query_clean}\n\nConversation History:\n"
        header_toks = self.tokenizer.tokenize(header_q)
        header_len = len(header_toks)

        # Se la sola domanda satura o eccede il budget massimo
        if header_len >= self.max_query_length:
            self.stats["question_truncated"] += 1
            self.stats["final_tokens"] += self.max_query_length
            # Troncatura protetta della sola domanda
            return f"{prefix}Current Question: {query_clean}"

        residual_budget = self.max_query_length - header_len

        # Suddivisione della cronologia in turni/blocchi logici
        # Identifica pattern 'User:', 'Assistant:', 'Q:', 'A:' o linee multiple
        turn_delimiters = re.split(r"(?=(?:User:|Assistant:|Q:|A:|\n\n))", history_clean)
        turns = [t.strip() for t in turn_delimiters if t.strip()]

        if not turns:
            turns = [history_clean]

        retained_turns = []
        accumulated_tokens = 0

        # Inclusione a ritroso dal turno più recente verso il più vecchio
        for turn in reversed(turns):
            turn_toks = self.tokenizer.tokenize(turn)
            if accumulated_tokens + len(turn_toks) <= residual_budget:
                retained_turns.append(turn)
                accumulated_tokens += len(turn_toks)
            else:
                break

        if len(retained_turns) == len(turns):
            self.stats["history_fully_retained"] += 1
        elif len(retained_turns) > 0:
            self.stats["history_partially_retained"] += 1

        self.stats["retained_hist_tokens"] += accumulated_tokens
        self.stats["final_tokens"] += header_len + accumulated_tokens

        # Ricomponi la cronologia nell'ordine temporale naturale
        reconstructed_history = "\n".join(reversed(retained_turns))
        return f"{header_q}{reconstructed_history}".strip()

    def get_diagnostics(self) -> Dict[str, float]:
        n = max(1, self.stats["total_queries"])
        return {
            "total_queries": self.stats["total_queries"],
            "pct_question_truncated": (self.stats["question_truncated"] / n) * 100.0,
            "pct_hist_full": (self.stats["history_fully_retained"] / n) * 100.0,
            "pct_hist_partial": (self.stats["history_partially_retained"] / n) * 100.0,
            "avg_raw_q_tokens": self.stats["raw_q_tokens"] / n,
            "avg_raw_hist_tokens": self.stats["raw_hist_tokens"] / n,
            "avg_retained_hist_tokens": self.stats["retained_hist_tokens"] / n,
            "avg_final_tokens": self.stats["final_tokens"] / n,
        }


# ==============================================================================
# 2. Motore BM25 Compatibile con Benchmark Ufficiale
# ==============================================================================
def simple_porter_stem(word: str) -> str:
    """Stemmer di Porter leggero integrato (zero dipendenze esterne)."""
    if len(word) <= 2:
        return word
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    if word.endswith("eed"):
        if len(word) > 4:
            word = word[:-1]
    elif word.endswith("ed") and any(c in "aeiou" for c in word[:-2]):
        word = word[:-2]
    elif word.endswith("ing") and any(c in "aeiou" for c in word[:-3]):
        word = word[:-3]
    return word


BM25_STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "she", "her", "it", "its", "they", "them",
    "what", "which", "who", "this", "that", "am", "is", "are", "was", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "in", "out", "on", "off", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
    "current", "question", "conversation", "history", "previous"
}


def bm25_tokenize(text: str) -> List[str]:
    """Tokenizzazione Lucene-like: regex alfanumerica + stopword removal + stemming."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [simple_porter_stem(w) for w in words if w not in BM25_STOPWORDS and len(w) > 1]


class OfficialCompatibleBM25:
    """BM25 Invertito conforme ai parametri ufficiali del benchmark (k1=0.9, b=0.4)."""

    def __init__(self, corpus: Dict[str, str], k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b
        self.doc_ids = list(corpus.keys())
        self.N = len(self.doc_ids)
        self.doc_len: Dict[int, int] = {}
        self.inv_index = collections.defaultdict(list)
        total_len = 0

        for idx, d_id in enumerate(self.doc_ids):
            tokens = bm25_tokenize(corpus[d_id])
            l = len(tokens)
            self.doc_len[idx] = l
            total_len += l
            tf_map = collections.Counter(tokens)
            for term, freq in tf_map.items():
                self.inv_index[term].append((idx, freq))

        self.avgdl = (total_len / self.N) if self.N > 0 else 1.0
        self.idf = {
            term: math.log((self.N - len(postings) + 0.5) / (len(postings) + 0.5) + 1.0)
            for term, postings in self.inv_index.items()
        }

    def get_top_k(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        q_tokens = bm25_tokenize(query)
        if not q_tokens:
            return []
        scores = collections.defaultdict(float)
        for term in q_tokens:
            if term in self.inv_index:
                term_idf = self.idf[term]
                for doc_idx, tf in self.inv_index[term]:
                    num = tf * (self.k1 + 1.0)
                    den = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_len[doc_idx] / self.avgdl))
                    scores[doc_idx] += term_idf * (num / den)
        if not scores:
            return []
        ranked_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in ranked_indices]

def mine_bm25_hard_negatives(
    corpus: Dict[str, str],
    samples: List[ConversationalTurnSample],
    top_k: int = 20,
    k1: float = 0.9,
    b: float = 0.4,
    domain_prefix: str = "",
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """
    Estrae i negativi BM25 per ciascun turno escludendo rigorosamente
    tutti i gold document del topic.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"hard_negs_bm25_{domain_prefix}_top{top_k}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    bm25 = OfficialCompatibleBM25(corpus, k1=k1, b=b)
    hard_negatives_map: Dict[str, List[str]] = {}

    for s in samples:
        gold_set = set(s.gold_doc_ids)
        if domain_prefix:
            gold_set.update(f"{domain_prefix}_{gid}" for gid in s.gold_doc_ids)

        # Recupera un numero congruo di candidati per compensare i gold filtrati
        candidates = bm25.get_top_k(s.contextual_query, top_k=top_k + len(gold_set) + 10)
        negatives: List[str] = []

        for doc_id, score in candidates:
            target_id = f"{domain_prefix}_{doc_id}" if domain_prefix and not doc_id.startswith(f"{domain_prefix}_") else doc_id
            raw_id = doc_id.split(f"{domain_prefix}_")[-1]

            # Controllo rigoroso anti-false negatives
            if target_id not in gold_set and raw_id not in gold_set and target_id not in negatives:
                negatives.append(target_id)
                if len(negatives) >= top_k:
                    break

        hard_negatives_map[s.topic_id] = negatives

    if cache_dir is not None:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(hard_negatives_map, f)
        except Exception:
            pass

    return hard_negatives_map


# ==============================================================================
# 3. PyTorch Dataset & Collate Function
# ==============================================================================

class RETECO2aTrainDataset(Dataset):
    """
    Dataset contrastivo:
      - 1 sola istanza per target turn (risolve il multi-positive collision).
      - Rotazione deterministica del positivo tra epoche.
      - Supporto a K negativi multipli escludendo rigorosamente tutti i gold.
    """

    def __init__(
        self,
        samples: List[ConversationalTurnSample],
        corpus: Dict[str, str],
        negatives_per_positive: int = 4,
        hard_negatives: Optional[Dict[str, List[str]]] = None,
        sampling_strategy: str = "bm25_hard",
        seed: int = 42,
    ):
        self.corpus = corpus
        self.corpus_keys = list(corpus.keys())
        self.negatives_per_positive = max(1, negatives_per_positive)
        self.hard_negatives = hard_negatives or {}
        self.sampling_strategy = sampling_strategy
        self.epoch = 0
        self.rng = random.Random(seed)

        # Filtra i turni mantenendo 1 campione per turno
        self.valid_samples: List[ConversationalTurnSample] = []
        for s in samples:
            valid_golds = [gid for gid in s.gold_doc_ids if gid in self.corpus]
            if valid_golds:
                s.gold_doc_ids = valid_golds
                self.valid_samples.append(s)

        if not self.valid_samples:
            raise ValueError("Nessun turno valido trovato con documenti presenti nel corpus.")

    def set_epoch(self, epoch: int):
        """Aggiorna l'epoca per la rotazione deterministica dei positivi."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.valid_samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.valid_samples[idx]
        topic_id = sample.topic_id
        golds = sample.gold_doc_ids
        gold_set = set(golds)

        # 1. Rotazione deterministica del positivo: un solo gold alla volta
        pos_idx = (self.epoch + idx) % len(golds)
        pos_id = golds[pos_idx]
        pos_text = self.corpus[pos_id]

        # 2. Selezione di K Hard Negatives escludendo TUTTI i gold document del topic
        selected_neg_ids: List[str] = []
        bm25_candidates = [
            nid for nid in self.hard_negatives.get(topic_id, [])
            if nid in self.corpus and nid not in gold_set
        ]

        if self.sampling_strategy == "bm25_hard" and bm25_candidates:
            # Prende i candidati senza duplicati
            for nid in bm25_candidates:
                if nid not in selected_neg_ids:
                    selected_neg_ids.append(nid)
                if len(selected_neg_ids) >= self.negatives_per_positive:
                    break

        # Fallback deterministico con negativi casuali se i BM25 scarseggiano
        attempts = 0
        while len(selected_neg_ids) < self.negatives_per_positive and attempts < 1000:
            attempts += 1
            rand_id = self.corpus_keys[self.rng.randint(0, len(self.corpus_keys) - 1)]
            if rand_id not in gold_set and rand_id not in selected_neg_ids:
                selected_neg_ids.append(rand_id)

        neg_texts = [self.corpus[nid] for nid in selected_neg_ids]

        return {
            "topic_id": topic_id,
            "domain": sample.domain,
            "query": sample.contextual_query,
            "positive": pos_text,
            "negatives": neg_texts,
            "pos_id": pos_id,
            "neg_ids": selected_neg_ids,
        }


class ConversationalCollateFn:
    """Collate function che tokenizza query, positivo e K negativi multipli."""

    def __init__(self, tokenizer: AutoTokenizer, max_query_len: int = 256, max_doc_len: int = 256):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]
        
        # Gestione negativi multipli: appiattimento per tokenizzazione batch
        batch_size = len(batch)
        k_negs = len(batch[0]["negatives"])
        flat_negatives = [neg for item in batch for neg in item["negatives"]]

        q_tok = self.tokenizer(
            queries, padding=True, truncation=True, max_length=self.max_query_len, return_tensors="pt"
        )
        pos_tok = self.tokenizer(
            positives, padding=True, truncation=True, max_length=self.max_doc_len, return_tensors="pt"
        )
        neg_tok = self.tokenizer(
            flat_negatives, padding=True, truncation=True, max_length=self.max_doc_len, return_tensors="pt"
        )

        return {
            "query_inputs": q_tok,
            "pos_inputs": pos_tok,
            "neg_inputs": neg_tok,
            "batch_size": batch_size,
            "k_negs": k_negs,
        }


# ==============================================================================
# 4. Sampler Bilanciato per Dominio
# ==============================================================================

class DomainBalancedBatchSampler(Sampler):
    """Sampler che garantisce un campionamento bilanciato tra i domini di training."""

    def __init__(self, samples: List[ConversationalTurnSample], batch_size: int, seed: int = 42):
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.domain_to_indices = collections.defaultdict(list)
        for idx, s in enumerate(samples):
            self.domain_to_indices[s.domain].append(idx)
        self.domains = list(self.domain_to_indices.keys())
        self.total_samples = len(samples)

    def __iter__(self):
        # Mescola gli indici all'interno di ogni dominio
        domain_pools = {d: self.rng.sample(idxs, len(idxs)) for d, idxs in self.domain_to_indices.items()}
        domain_ptrs = {d: 0 for d in self.domains}

        batches = []
        current_batch = []
        d_idx = 0

        while len(batches) * self.batch_size < self.total_samples:
            dom = self.domains[d_idx % len(self.domains)]
            d_idx += 1

            if domain_ptrs[dom] >= len(domain_pools[dom]):
                domain_pools[dom] = self.rng.sample(self.domain_to_indices[dom], len(self.domain_to_indices[dom]))
                domain_ptrs[dom] = 0

            sample_idx = domain_pools[dom][domain_ptrs[dom]]
            domain_ptrs[dom] += 1
            current_batch.append(sample_idx)

            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []

        return iter(batches)

    def __len__(self) -> int:
        return self.total_samples // self.batch_size


# ==============================================================================
# 5. Funzioni di Supporto per il Caricamento Dati
# ==============================================================================

def load_corpus(documents_path: Union[str, Path]) -> Dict[str, str]:
    documents_path = Path(documents_path)
    if not documents_path.exists():
        raise FileNotFoundError(f"Corpus non trovato: {documents_path}")
    corpus: Dict[str, str] = {}
    with open(documents_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_id = str(item.get("doc_id") or item.get("id"))
            text = item.get("content") or item.get("text") or ""
            corpus[doc_id] = text.strip()
    return corpus


def load_qrels(qrels_path: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    qrels_path = Path(qrels_path)
    if not qrels_path.exists():
        return {}
    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                t_id, _, d_id, rel = parts[0], parts[1], parts[2], parts[3]
                qrels.setdefault(t_id, {})[d_id] = int(rel)
    return qrels


def load_benchmark_conversations(
    benchmark_path: Union[str, Path],
    domain: str,
    formatter: ContextAwareQueryFormatter,
) -> List[ConversationalTurnSample]:
    benchmark_path = Path(benchmark_path)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark non trovato: {benchmark_path}")

    with open(benchmark_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        raw_data = json.loads(content) if content.startswith("[") else [json.loads(l) for l in content.splitlines() if l.strip()]

    samples: List[ConversationalTurnSample] = []
    for conv in raw_data:
        conv_id = conv.get("id")
        for turn in conv.get("turns", []):
            turn_id = turn.get("turn_id")
            topic_id = f"{conv_id}_turn_{turn_id}"
            query = turn.get("query", "")
            history = turn.get("conversation_history", "")
            gold_doc_ids = turn.get("gold_doc_ids", [])
            answer = turn.get("answer", "")

            # Costruzione con context-aware truncation deterministica
            contextual_q = formatter.format(query=query, history=history)

            samples.append(
                ConversationalTurnSample(
                    domain=domain,
                    conversation_id=conv_id,
                    turn_id=turn_id,
                    topic_id=topic_id,
                    query=query,
                    history=history,
                    contextual_query=contextual_q,
                    gold_doc_ids=gold_doc_ids,
                    answer=answer,
                )
            )
    return samples


def load_track2_domain_data(
    data_dir: Union[str, Path],
    domain: str,
    split: str = "train",
    formatter: Optional[ContextAwareQueryFormatter] = None,
) -> Tuple[Dict[str, str], List[ConversationalTurnSample], Dict[str, Dict[str, int]]]:
    p = Path(data_dir)
    candidates = [
        p / domain,
        p / "track2_recor" / domain,
        Path("..") / p / domain,
        Path("..") / p / "track2_recor" / domain,
    ]
    domain_dir = next((c.resolve() for c in candidates if c.exists() and (c / "documents.jsonl").exists()), None)
    if domain_dir is None:
        raise FileNotFoundError(f"Cartella dominio '{domain}' non trovata in {data_dir}")

    corpus = load_corpus(domain_dir / "documents.jsonl")

    bench_candidates = [
        domain_dir / f"benchmark_{split}.json",
        domain_dir / f"benchmark_{split}.jsonl",
        domain_dir / "benchmark.jsonl",
        domain_dir / "benchmark.json",
    ]
    bench_file = next((b for b in bench_candidates if b.exists()), None)
    if not bench_file:
        raise FileNotFoundError(f"File benchmark non trovato per '{split}' in {domain_dir}")

    if formatter is None:
        # Fallback basico
        class DummyFormatter:
            def format(self, query, history):
                return f"Current Question: {query}\n\nConversation History:\n{history}"
        formatter = DummyFormatter()

    samples = load_benchmark_conversations(bench_file, domain=domain, formatter=formatter)

    qrels_candidates = [domain_dir / f"qrels_{split}.txt", domain_dir / "qrels.txt"]
    qrels_file = next((q for q in qrels_candidates if q.exists()), None)
    qrels = load_qrels(qrels_file) if qrels_file else {}

    return corpus, samples, qrels


def split_conversations_train_val(
    samples: List[ConversationalTurnSample],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ConversationalTurnSample], List[ConversationalTurnSample]]:
    """Splitta per intera conversazione garantendo zero leakage fra i turni."""
    conv_ids = sorted(list(set(s.conversation_id for s in samples)))
    rng = random.Random(seed)
    rng.shuffle(conv_ids)
    n_val = max(1, int(len(conv_ids) * val_ratio))
    val_convs = set(conv_ids[:n_val])
    return [s for s in samples if s.conversation_id not in val_convs], [s for s in samples if s.conversation_id in val_convs]