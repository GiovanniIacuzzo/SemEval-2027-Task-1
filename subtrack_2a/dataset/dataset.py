#!/usr/bin/env python3
"""
subtrack_2a/dataset/dataset.py

Modulo di gestione dei dati per RETECO Sub-track 2a (Conversational Retrieval).
Supporta:
  - Caricamento dei file benchmark_{train,dev}.json, documents.jsonl e qrels_{train,dev}.txt.
  - Costruzione della query contestuale (risoluzione dello storico multi-turno).
  - PyTorch Dataset per l'addestramento contrastivo (triplette query-positivo-negativo o in-batch negatives).
  - PyTorch Dataset per l'indicizzazione e l'inferenza valutata con pytrec_eval.
"""

import os
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
    Costruisce la rappresentazione della query integrando la cronologia del dialogo.
    
    Args:
        query: Domanda del turno corrente.
        history: Testo della conversazione precedente ('No previous conversation.' per il primo turno).
        strategy: 'concat' (Cronologia + Domanda) o 'query_only' (solo domanda corrente).
    """
    if strategy == "query_only":
        return query.strip()

    history_clean = history.strip() if history else ""
    if not history_clean or history_clean.lower() == "no previous conversation.":
        return query.strip()

    return f"{history_clean}\n\nCurrent Question: {query.strip()}"


def load_corpus(documents_path: Union[str, Path]) -> Dict[str, str]:
    """
    Carica documents.jsonl mappando l'identificativo al testo del passaggio.
    Gestisce le variazioni di chiave ('doc_id' o 'id'; 'content' o 'text').
    """
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
                raise ValueError(f"Record privo di id alla riga {line_idx} in {documents_path}")
            corpus[doc_id] = text.strip()
    return corpus


def load_qrels(qrels_path: Union[str, Path]) -> Dict[str, Dict[str, int]]:
    """
    Legge il file standard TREC qrels a 4 colonne: topic_id 0 doc_id relevance.
    Restituisce: {topic_id: {doc_id: relevance}}
    """
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
    """
    Estrae tutti i singoli turni da benchmark_{train,dev}.json (o .jsonl).
    Costruisce il topic_id ufficiale nel formato: <conversation_id>_turn_<turn_id>.
    """
    benchmark_path = Path(benchmark_path)
    if not benchmark_path.exists():
        raise FileNotFoundError(f"File benchmark non trovato: {benchmark_path}")

    raw_data = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            raw_data = json.loads(content)
        else:
            # Gestione fallback per formati JSONL
            for line in content.splitlines():
                line = line.strip()
                if line:
                    raw_data.append(json.loads(line))

    samples: List[ConversationalTurnSample] = []
    for conv in raw_data:
        conv_id = conv.get("id")
        turns = conv.get("turns", [])
        for turn in turns:
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


class RETECO2aTrainDataset(Dataset):
    """
    Dataset PyTorch per l'addestramento contrastivo su Sub-track 2a.
    Per ogni turno con documenti gold noti genera coppie (query, positivo) o
    triplette (query, positivo, negativo casuale o hard negative).
    """

    def __init__(
        self,
        samples: List[ConversationalTurnSample],
        corpus: Dict[str, str],
        use_triplets: bool = True,
        negatives_per_positive: int = 1,
    ):
        self.corpus = corpus
        self.corpus_keys = list(corpus.keys())
        self.use_triplets = use_triplets
        self.negatives_per_positive = negatives_per_positive

        # Filtra ed espande i campioni validi (che hanno almeno un gold presente nel corpus)
        self.instances: List[Tuple[str, str, List[str]]] = []
        for sample in samples:
            valid_gold_ids = [gid for gid in sample.gold_doc_ids if gid in self.corpus]
            if not valid_gold_ids:
                continue

            for gold_id in valid_gold_ids:
                self.instances.append((sample.contextual_query, gold_id, valid_gold_ids))

        if not self.instances:
            raise ValueError("Nessuna istanza di training valida trovata con i documenti presenti nel corpus.")

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        contextual_query, pos_id, all_gold_ids = self.instances[idx]
        pos_text = self.corpus[pos_id]

        if not self.use_triplets:
            return {
                "query": contextual_query,
                "positive": pos_text,
                "pos_id": pos_id,
            }

        # Campionamento di un negativo casuale dal corpus escludendo i gold noti
        gold_set = set(all_gold_ids)
        while True:
            neg_id = random.choice(self.corpus_keys)
            if neg_id not in gold_set:
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
    """
    Dataset per l'inferenza di retrieval o re-ranking.
    Restituisce le query contestuali e i metadati associati a ciascun turno.
    """

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
    """
    Dataset per l'indicizzazione densa del corpus dei documenti di un dominio.
    """

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


def load_track2_domain_data(
    data_dir: Union[str, Path],
    domain: str,
    split: str = "train",
    query_strategy: str = "concat",
) -> Tuple[Dict[str, str], List[ConversationalTurnSample], Optional[Dict[str, Dict[str, int]]]]:
    """
    Carica corpus, benchmark turni e qrels per un dominio specifico.
    Risolve automaticamente i percorsi sia eseguendo dalla radice che da dentro subtrack_2a/.
    """
    p = Path(data_dir)

    # Lista di possibili percorsi per individuare la cartella del dominio
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
        # Verifica se la cartella esiste e contiene documents.jsonl
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

    # Identificazione file benchmark
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

    # Identificazione file qrels
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
    """
    Suddivide i campioni in train e validation a livello di intera conversazione,
    garantendo che nessun turno pregresso finisca nel validation split.
    """
    conv_ids = sorted(list(set(s.conversation_id for s in samples)))
    rng = random.Random(seed)
    rng.shuffle(conv_ids)

    n_val = max(1, int(len(conv_ids) * val_ratio))
    val_conv_set = set(conv_ids[:n_val])

    train_samples = [s for s in samples if s.conversation_id not in val_conv_set]
    val_samples = [s for s in samples if s.conversation_id in val_conv_set]

    return train_samples, val_samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test rapido del modulo dataset per Sub-track 2a.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sample/track2_recor",
        help="Percorso ai dati di Track 2 (sample o completi).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="drones",
        help="Nome del dominio di prova (es. 'drones', 'biology').",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split da caricare ('train' o 'dev').",
    )

    args = parser.parse_args()
    print(f"Verifica caricamento dataset su dominio '{args.domain}' da {args.data_dir}...")

    try:
        loaded_corpus, loaded_samples, loaded_qrels = load_track2_domain_data(
            data_dir=args.data_dir,
            domain=args.domain,
            split=args.split,
        )

        print(f"✓ Corpus caricato: {len(loaded_corpus)} documenti.")
        print(f"✓ Turni conversazionali estratti: {len(loaded_samples)}.")
        if loaded_qrels:
            print(f"✓ Qrels caricati: {len(loaded_qrels)} topic annotati.")

        train_ds = RETECO2aTrainDataset(samples=loaded_samples, corpus=loaded_corpus, use_triplets=True)
        print(f"✓ Dataset PyTorch istanziato con successo. Numero istanze di training: {len(train_ds)}.")

        first_sample = train_ds[0]
        print("\nEsempio prima istanza:")
        print(f"  [Query Contestuale]: {first_sample['query'][:120]}...")
        print(f"  [Positivo ID]: {first_sample['pos_id']}")
        print(f"  [Positivo Testo]: {first_sample['positive'][:100]}...")
        print(f"  [Negativo ID]: {first_sample['neg_id']}")

    except Exception as e:
        print(f"Nota/Errore: {e}")