#!/usr/bin/env python3
"""
subtrack_2a/retrieval/retrieval.py

Retrieval sparse e hard-negative mining per RETECO Sub-track 2a.

Responsabilità:
    - BM25 compatibile con l'implementazione ufficiale RETECO/RECOR.
    - Retrieval top-k.
    - Mining degli hard negatives.
    - Cache persistente degli hard negatives.

Il modulo NON costruisce le query:
la query viene preparata dal ContextAwareQueryFormatter in dataset.py.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from tqdm import tqdm


# =============================================================================
# Official BM25
# =============================================================================

class OfficialBM25:
    """
    Implementazione BM25 basata sulla stessa pipeline utilizzata dalla
    baseline ufficiale RETECO/RECOR:

        Pyserini Lucene analyzer
        +
        gensim LuceneBM25Model
        +
        SparseMatrixSimilarity

    Parametri ufficiali:
        k1 = 0.9
        b  = 0.4
    """

    def __init__(
        self,
        corpus: Dict[str, str],
        k1: float = 0.9,
        b: float = 0.4,
    ):
        if not corpus:
            raise ValueError("Corpus BM25 vuoto.")

        if k1 <= 0:
            raise ValueError("k1 deve essere > 0.")

        if not (0.0 <= b <= 1.0):
            raise ValueError("b deve essere compreso tra 0 e 1.")

        # Import lazy: evita di imporre Pyserini/Gensim a chi vuole usare
        # soltanto il dense retriever.
        from pyserini import analysis
        from gensim.corpora import Dictionary
        from gensim.models import LuceneBM25Model
        from gensim.similarities import SparseMatrixSimilarity

        self.k1 = float(k1)
        self.b = float(b)

        self.doc_ids = list(corpus.keys())
        self.documents = [
            corpus[doc_id]
            for doc_id in self.doc_ids
        ]

        self.analyzer = analysis.Analyzer(
            analysis.get_lucene_analyzer()
        )

        print(
            f"[BM25] Analysing corpus: "
            f"{len(self.documents)} documents"
        )

        analysed_corpus = [
            self.analyzer.analyze(text)
            for text in tqdm(
                self.documents,
                desc="BM25 analyze corpus",
                unit="doc",
                leave=False,
            )
        ]

        self.dictionary = Dictionary(
            analysed_corpus
        )

        self.model = LuceneBM25Model(
            dictionary=self.dictionary,
            k1=self.k1,
            b=self.b,
        )

        bm25_corpus = self.model[
            list(
                map(
                    self.dictionary.doc2bow,
                    analysed_corpus,
                )
            )
        ]

        self.index = SparseMatrixSimilarity(
            bm25_corpus,
            num_docs=len(self.doc_ids),
            num_terms=len(self.dictionary),
            normalize_queries=False,
            normalize_documents=False,
        )

    # -------------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------------

    def search_one(
        self,
        query: str,
        top_k: int = 1000,
    ) -> List[Tuple[str, float]]:
        """
        Esegue il retrieval BM25 per una singola query.
        """
        if top_k <= 0:
            return []

        query_tokens = self.analyzer.analyze(
            query
        )

        query_bow = self.dictionary.doc2bow(
            query_tokens
        )

        query_vector = self.model[
            query_bow
        ]

        similarities = self.index[
            query_vector
        ].tolist()

        pairs = sorted(
            zip(self.doc_ids, similarities),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            (doc_id, float(score))
            for doc_id, score in pairs[:top_k]
        ]

    def search(
        self,
        queries: Sequence[str],
        query_ids: Optional[Sequence[str]] = None,
        top_k: int = 1000,
        desc: str = "BM25 retrieval",
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieval batch di più query.

        Restituisce:

            {
                topic_id: [
                    (doc_id, score),
                    ...
                ]
            }
        """
        if query_ids is None:
            query_ids = [
                str(i)
                for i in range(len(queries))
            ]

        if len(queries) != len(query_ids):
            raise ValueError(
                "queries e query_ids devono avere la stessa lunghezza."
            )

        results: Dict[str, List[Tuple[str, float]]] = {}

        iterator = zip(query_ids, queries)

        for query_id, query in tqdm(
            iterator,
            total=len(queries),
            desc=desc,
            unit="query",
            leave=False,
        ):
            results[str(query_id)] = self.search_one(
                query=query,
                top_k=top_k,
            )

        return results


# =============================================================================
# Cache utilities
# =============================================================================

def _stable_fingerprint(
    corpus: Dict[str, str],
    samples,
    *,
    top_k: int,
    k1: float,
    b: float,
    cache_tag: str,
) -> str:
    """
    Fingerprint stabile della configurazione che determina gli hard negatives.

    Includiamo:
        - ID del corpus
        - topic ID
        - contextual query
        - gold IDs
        - parametri BM25
        - top_k
        - cache_tag

    In questo modo cambiare il query formatter invalida automaticamente
    la cache precedente.
    """
    payload = {
        "cache_tag": cache_tag,
        "top_k": int(top_k),
        "k1": float(k1),
        "b": float(b),
        "corpus_ids": sorted(corpus.keys()),
        "queries": [
            {
                "topic_id": str(sample.topic_id),
                "query": str(sample.contextual_query),
                "gold": sorted(
                    str(x)
                    for x in sample.gold_doc_ids
                ),
            }
            for sample in samples
        ],
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()[:16]


# =============================================================================
# Hard-negative mining
# =============================================================================

def mine_bm25_hard_negatives(
    corpus: Dict[str, str],
    samples,
    top_k: int = 50,
    k1: float = 0.9,
    b: float = 0.4,
    cache_dir: Optional[Path] = None,
    cache_tag: str = "train",
) -> Dict[str, List[str]]:
    """
    Estrae hard negatives usando BM25.

    Per ogni topic:
        1. Recupera fino a 1000 candidati BM25.
        2. Rimuove tutti i gold documents.
        3. Mantiene i primi `top_k` non-gold documents.

    IMPORTANTE:
        Non viene introdotto nessun prefisso nel document ID.
        Gli ID rimangono identici al corpus RETECO.

    La cache viene invalidata automaticamente quando cambiano:
        - corpus
        - query
        - gold
        - top_k
        - k1/b
        - cache_tag
    """
    if top_k <= 0:
        raise ValueError("top_k deve essere > 0.")

    if not samples:
        return {}

    fingerprint = _stable_fingerprint(
        corpus=corpus,
        samples=samples,
        top_k=top_k,
        k1=k1,
        b=b,
        cache_tag=cache_tag,
    )

    cache_file: Optional[Path] = None

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cache_file = (
            cache_dir
            / f"bm25_hard_negatives_{fingerprint}.json"
        )

        if cache_file.exists():
            try:
                with cache_file.open(
                    "r",
                    encoding="utf-8",
                ) as f:
                    payload = json.load(f)

                # Supportiamo il nuovo formato con metadata.
                if (
                    isinstance(payload, dict)
                    and "negatives" in payload
                ):
                    negatives = payload["negatives"]

                    if isinstance(negatives, dict):
                        print(
                            f"[BM25] Loading cached hard negatives: "
                            f"{cache_file}"
                        )
                        return {
                            str(topic_id): [
                                str(doc_id)
                                for doc_id in doc_ids
                            ]
                            for topic_id, doc_ids
                            in negatives.items()
                        }

                # Compatibilità con vecchie cache semplici.
                if isinstance(payload, dict):
                    return {
                        str(topic_id): [
                            str(doc_id)
                            for doc_id in doc_ids
                        ]
                        for topic_id, doc_ids
                        in payload.items()
                    }

            except Exception:
                # Cache corrotta/incompatibile -> viene rigenerata.
                pass

    print(
        f"[BM25] Mining {top_k} hard negatives "
        f"for {len(samples)} queries..."
    )

    bm25 = OfficialBM25(
        corpus=corpus,
        k1=k1,
        b=b,
    )

    # Il BM25 ufficiale produce top-1000.
    retrieval_depth = min(
        1000,
        len(corpus),
    )

    hard_negatives: Dict[str, List[str]] = {}

    for sample in tqdm(
        samples,
        desc=f"Hard-negative mining [{cache_tag}]",
        unit="query",
    ):
        gold_ids = set(
            str(doc_id)
            for doc_id in sample.gold_doc_ids
        )

        ranked = bm25.search_one(
            query=sample.contextual_query,
            top_k=retrieval_depth,
        )

        negatives: List[str] = []

        for doc_id, _score in ranked:
            doc_id = str(doc_id)

            # Esclusione rigorosa di TUTTI i gold.
            if doc_id in gold_ids:
                continue

            if doc_id in negatives:
                continue

            negatives.append(doc_id)

            if len(negatives) >= top_k:
                break

        hard_negatives[sample.topic_id] = negatives

    if cache_file is not None:
        payload = {
            "fingerprint": fingerprint,
            "cache_tag": cache_tag,
            "top_k": int(top_k),
            "k1": float(k1),
            "b": float(b),
            "num_topics": len(samples),
            "negatives": hard_negatives,
        }

        try:
            with cache_file.open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    payload,
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(
                f"[BM25] Hard-negative cache saved: "
                f"{cache_file}"
            )

        except Exception as exc:
            print(
                f"[BM25] Warning: unable to save hard-negative cache: "
                f"{exc}"
            )

    return hard_negatives
