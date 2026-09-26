#!/usr/bin/env python3
"""
subtrack_2a/dataset/dataset.py

Modulo di gestione dati per SemEval-2027 RETECO Sub-track 2a (Conversational Retrieval).
Funzionalità:
  - Caricamento di benchmark_{train,dev}.json, documents.jsonl e qrels_{train,dev}.txt.
  - Costruzione della query contestuale (cronologia + turno corrente).
  - Indicizzazione lessicale BM25 e mining dei falsi positivi (Hard Negatives).
  - PyTorch Train Dataset con supporto a 'bm25_hard', 'mixed' e 'random' sampling.
  - PyTorch Inference/Corpus Dataset per la codifica vettoriale e la valutazione.
"""

import os
import sys
import math
import re
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

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


# ==============================================================================
# 1. Strutture Dati e Formattazione Query
# ==============================================================================

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


def format_contextual_query(query: str, history: str, strategy: str = "concat") -> str:
    """
    Costruisce la query contestualizzata integrando la cronologia conversazionale.
    """
    if strategy == "query_only":
        return query.strip()

    history_clean = history.strip() if history else ""
    if not history_clean or history_clean.lower() == "no previous conversation.":
        return query.strip()

    return f"{history_clean}\n\nCurrent Question: {query.strip()}"


# ==============================================================================
# 2. Caricamento File Corpus, Qrels e Benchmark
# ==============================================================================

def load_corpus(documents_path: Union[str, Path]) -> Dict[str, str]:
    """Carica documents.jsonl mappando doc_id al testo del passaggio."""
    documents_path = Path(documents_path)
    if not documents_path.exists():
        raise FileNotFoundError(f"File corpus non trovato: {documents_path}")

    corpus: Dict[str, str] = {}
    with open(documents_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_id = str(item.get("doc_id") or item.get("id"))
            text = item.get("content") or item.get("text") or ""
            if not doc_id:
                raise ValueError(f"Record privo di identificativo alla riga {line_idx} in {documents_path}")
            corpus[doc_id] = text.strip()
    return corpus


def load_qrels(qrels_path: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """Legge il file standard TREC qrels a 4 colonne: topic_id 0 doc_id relevance."""
    qrels_path = Path(qrels_path)
    if not qrels_path.exists():
        raise FileNotFoundError(f"File qrels non trovato: {qrels_path}")

    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                topic_id, _, doc_id, rel = parts[0], parts[1], parts[2], parts[3]
                qrels.setdefault(topic_id, {})[doc_id] = int(rel)
    return qrels


def load_benchmark_conversations(
    benchmark_path: Union[str, Path],
    domain: str,
    query_strategy: str = "concat",
) -> List[ConversationalTurnSample]:
    """Estrae i turni conversazionali costruendo il topic_id ufficiale (<conv_id>_turn_<turn_id>)."""
    benchmark_path = Path(benchmark_path)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"File benchmark non trovato: {benchmark_path}")

    raw_data = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            raw_data = json.loads(content)
        else:
            for line in content.splitlines():
                if line.strip():
                    raw_data.append(json.loads(line))

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

            contextual_q = format_contextual_query(
                query=query,
                history=history,
                strategy=query_strategy,
            )

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


# ==============================================================================
# 3. Motore BM25 e Mining degli Hard Negatives
# ==============================================================================

class SimpleBM25:
    """Implementazione BM25Okapi compatta conforme ai parametri ufficiali RETECO."""

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
            tokens = self._tokenize(text)
            t_len = len(tokens)
            self.doc_len[doc_id] = t_len
            total_length += t_len

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs[doc_id] = tf

            for t in tf.keys():
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = (total_length / self.corpus_size) if self.corpus_size > 0 else 1.0

        self.idf: Dict[str, float] = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def get_top_k(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        tokens = self._tokenize(query)
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


def mine_domain_bm25_hard_negatives(
    corpus: Dict[str, str],
    samples: List[ConversationalTurnSample],
    top_k: int = 50,
    k1: float = 0.9,
    b: float = 0.4,
    domain_prefix: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Estrae per ciascun turno i falsi positivi lessicali di BM25 escludendo rigorosamente i gold document.
    Restituisce: {topic_id: [hard_neg_doc_id_1, hard_neg_doc_id_2, ...]}
    """
    bm25 = SimpleBM25(corpus, k1=k1, b=b)
    hard_negatives_map: Dict[str, List[str]] = {}

    for s in samples:
        # Recupera un margine di candidati per compensare i passaggi gold estratti
        candidates = bm25.get_top_k(s.contextual_query, top_k=top_k + len(s.gold_doc_ids) + 5)

        gold_set = set(s.gold_doc_ids)
        if domain_prefix:
            gold_set.update(f"{domain_prefix}_{gid}" for gid in s.gold_doc_ids)

        extracted = []
        for doc_id, score in candidates:
            if score <= 0.0:
                continue

            target_id = f"{domain_prefix}_{doc_id}" if domain_prefix and not doc_id.startswith(f"{domain_prefix}_") else doc_id

            if doc_id not in gold_set and target_id not in gold_set:
                extracted.append(target_id)
                if len(extracted) >= top_k:
                    break

        hard_negatives_map[s.topic_id] = extracted

    return hard_negatives_map


# ==============================================================================
# 4. PyTorch Datasets
# ==============================================================================

class RETECO2aTrainDataset(Dataset):
    """
    Dataset PyTorch per l'addestramento contrastivo con supporto a Hard Negatives.
    """

    def __init__(
        self,
        samples: List[ConversationalTurnSample],
        corpus: Dict[str, str],
        use_triplets: bool = True,
        negatives_per_positive: int = 1,
        hard_negatives: Optional[Dict[str, List[str]]] = None,
        sampling_strategy: str = "bm25_hard",
    ):
        self.corpus = corpus
        self.corpus_keys = list(corpus.keys())
        self.use_triplets = use_triplets
        self.negatives_per_positive = negatives_per_positive
        self.hard_negatives = hard_negatives or {}
        self.sampling_strategy = sampling_strategy

        # Tupla: (topic_id, contextual_query, pos_id, valid_gold_ids)
        self.instances: List[Tuple[str, str, str, List[str]]] = []
        for sample in samples:
            valid_gold_ids = [gid for gid in sample.gold_doc_ids if gid in self.corpus]
            if not valid_gold_ids:
                continue

            for gold_id in valid_gold_ids:
                self.instances.append((sample.topic_id, sample.contextual_query, gold_id, valid_gold_ids))

        if not self.instances:
            raise ValueError("Nessuna istanza di training valida trovata con i documenti presenti nel corpus.")

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        topic_id, contextual_query, pos_id, all_gold_ids = self.instances[idx]
        pos_text = self.corpus[pos_id]

        if not self.use_triplets:
            return {
                "query": contextual_query,
                "positive": pos_text,
                "pos_id": pos_id,
            }

        gold_set = set(all_gold_ids)
        neg_id = None

        # Selezione strategia di campionamento
        use_hard = False
        if self.sampling_strategy == "bm25_hard":
            use_hard = True
        elif self.sampling_strategy == "mixed":
            use_hard = random.random() < 0.5

        if use_hard and topic_id in self.hard_negatives:
            # Considera solo i candidati presenti nel corpus ed esclude i gold
            cands = [nid for nid in self.hard_negatives[topic_id] if nid in self.corpus and nid not in gold_set]
            if cands:
                neg_id = random.choice(cands)

        # Fallback deterministico su negativo casuale
        if neg_id is None:
            while True:
                candidate = random.choice(self.corpus_keys)
                if candidate not in gold_set:
                    neg_id = candidate
                    break

        neg_text = self.corpus[neg_id]

        return {
            "query": contextual_query,
            "positive": pos_text,
            "negative": neg_text,
            "pos_id": pos_id,
            "neg_id": neg_id,
        }


class RETECO2aInferenceDataset(Dataset):
    """Dataset per l'inferenza di retrieval o re-ranking."""

    def __init__(self, samples: List[ConversationalTurnSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, int, List[str]]]:
        sample = self.samples[idx]
        return {
            "topic_id": sample.topic_id,
            "domain": sample.domain,
            "query": sample.query,
            "contextual_query": sample.contextual_query,
            "gold_doc_ids": sample.gold_doc_ids,
        }


class RETECO2aCorpusDataset(Dataset):
    """Dataset per l'indicizzazione densa del corpus dei documenti."""

    def __init__(self, corpus: Dict[str, str]):
        self.doc_ids = list(corpus.keys())
        self.corpus = corpus

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        doc_id = self.doc_ids[idx]
        return {
            "doc_id": doc_id,
            "text": self.corpus[doc_id],
        }


# ==============================================================================
# 5. Helper di Caricamento e Splitting
# ==============================================================================

def load_track2_domain_data(
    data_dir: Union[str, Path],
    domain: str,
    split: str = "train",
    query_strategy: str = "concat",
) -> Tuple[Dict[str, str], List[ConversationalTurnSample], Optional[Dict[str, Dict[str, int]]]]:
    """Carica i dati di un dominio risolvendo automaticamente i percorsi relativi."""
    p = Path(data_dir)

    candidate_paths = [
        p / domain,
        p / "track2_recor" / domain,
        Path("..") / p / domain,
        Path("..") / p / "track2_recor" / domain,
        Path(__file__).resolve().parent.parent.parent / p / domain,
        Path(__file__).resolve().parent.parent.parent / p / "track2_recor" / domain,
    ]

    domain_dir = None
    for cand in candidate_paths:
        if cand.exists() and (cand / "documents.jsonl").exists():
            domain_dir = cand.resolve()
            break

    if domain_dir is None:
        raise FileNotFoundError(
            f"Directory del dominio '{domain}' non trovata. "
            f"Percorsi verificati:\n" + "\n".join(f" - {c}" for c in candidate_paths[:4])
        )

    corpus_path = domain_dir / "documents.jsonl"
    corpus = load_corpus(corpus_path)

    benchmark_candidates = [
        domain_dir / f"benchmark_{split}.json",
        domain_dir / f"benchmark_{split}.jsonl",
        domain_dir / "benchmark.jsonl",
        domain_dir / "benchmark.json",
    ]
    benchmark_path = next((b for b in benchmark_candidates if b.exists()), None)
    if not benchmark_path:
        raise FileNotFoundError(f"File benchmark per lo split '{split}' non trovato in {domain_dir}")

    samples = load_benchmark_conversations(benchmark_path, domain=domain, query_strategy=query_strategy)

    qrels_candidates = [
        domain_dir / f"qrels_{split}.txt",
        domain_dir / "qrels.txt",
    ]
    qrels_path = next((q for q in qrels_candidates if q.exists()), None)
    qrels = load_qrels(qrels_path) if qrels_path else None

    return corpus, samples, qrels


def split_conversations_train_val(
    samples: List[ConversationalTurnSample],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ConversationalTurnSample], List[ConversationalTurnSample]]:
    """Suddivide in train e validation per intera conversazione (zero data leakage)."""
    conv_ids = sorted(list(set(s.conversation_id for s in samples)))
    rng = random.Random(seed)
    rng.shuffle(conv_ids)

    n_val = max(1, int(len(conv_ids) * val_ratio))
    val_conv_set = set(conv_ids[:n_val])

    train_samples = [s for s in samples if s.conversation_id not in val_conv_set]
    val_samples = [s for s in samples if s.conversation_id in val_conv_set]

    return train_samples, val_samples


# ==============================================================================
# 6. Test di Verifica
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test modulo dataset con BM25 Hard Negatives.")
    parser.add_argument("--data_dir", type=str, default="data/reteco_data/track2_recor", help="Directory dei dati")
    parser.add_argument("--domain", type=str, default="drones", help="Dominio di test")
    parser.add_argument("--split", type=str, default="train", help="Split (train o dev)")
    args = parser.parse_args()

    print(f"Test su dominio '{args.domain}' da '{args.data_dir}'...")
    try:
        corpus, samples, qrels = load_track2_domain_data(args.data_dir, args.domain, split=args.split)
        print(f"✓ Corpus: {len(corpus):,} passaggi | Turni: {len(samples)}")

        print("\nEstrazione Hard Negatives tramite BM25...")
        hard_negs = mine_domain_bm25_hard_negatives(corpus, samples, top_k=20)
        avg_negs = sum(len(v) for v in hard_negs.values()) / max(1, len(hard_negs))
        print(f"✓ Hard Negatives minati: media di {avg_negs:.1f} passaggi per query.")

        ds = RETECO2aTrainDataset(
            samples=samples,
            corpus=corpus,
            hard_negatives=hard_negs,
            sampling_strategy="bm25_hard",
        )
        print(f"✓ Dataset PyTorch inizializzato ({len(ds)} istanze).")

        item = ds[0]
        print("\n--- Esempio Triplette Minata ---")
        print(f"Query:        {item['query'][:100]}...")
        print(f"Positivo ID:  {item['pos_id']}")
        print(f"Positivo:     {item['positive'][:90]}...")
        print(f"Negativo ID:  {item['neg_id']}")
        print(f"Hard Neg:     {item['negative'][:90]}...")

    except Exception as e:
        print(f"Errore/Nota: {e}")