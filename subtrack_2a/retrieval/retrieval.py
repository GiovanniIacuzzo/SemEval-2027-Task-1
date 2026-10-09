#!/usr/bin/env python3
"""Sparse retrieval and BM25 hard-negative mining for RETECO Sub-track 2a.

The BM25 implementation follows the RETECO starter kit's reference path:
Pyserini's Lucene analyzer + gensim.LuceneBM25Model + SparseMatrixSimilarity,
with k1=0.9 and b=0.4 by default.  The input corpus must be the official
RETECO domain corpus; this module never fetches or augments documents.

Public API retained for compatibility:
    OfficialBM25(corpus, k1=0.9, b=0.4)
    OfficialBM25.search_one(query, top_k=1000) -> [(doc_id, score), ...]
    OfficialBM25.search(queries, query_ids=None, top_k=1000) -> {topic: ranking}
    mine_bm25_hard_negatives(...)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from tqdm import tqdm


_CACHE_SCHEMA_VERSION = 2
_BM25_ALGORITHM_VERSION = "reteco_official_lucene_gensim_v1"
_OFFICIAL_RETRIEVAL_DEPTH = 1000


class OfficialBM25:
    """Official-style BM25 retrieval for one domain corpus.

    Args:
        corpus: Ordered mapping ``doc_id -> document text``. Insertion order is
            intentionally preserved because ties (particularly zero-score ties)
            are resolved stably in corpus order, as in the official starter kit.
        k1: BM25 term-frequency saturation parameter (official default 0.9).
        b: BM25 length-normalization parameter (official default 0.4).

    This class requires ``pyserini`` and a gensim build exposing
    ``LuceneBM25Model``. Imports are lazy so dense-only code can import this
    module without loading the sparse-retrieval dependencies.
    """

    def __init__(
        self,
        corpus: Mapping[str, str],
        k1: float = 0.9,
        b: float = 0.4,
    ) -> None:
        if not corpus:
            raise ValueError("Corpus BM25 vuoto.")
        if not math.isfinite(float(k1)) or float(k1) <= 0:
            raise ValueError("k1 deve essere un numero finito > 0.")
        if not math.isfinite(float(b)) or not 0.0 <= float(b) <= 1.0:
            raise ValueError("b deve essere compreso tra 0 e 1.")

        try:
            from pyserini import analysis
            from gensim.corpora import Dictionary
            from gensim.models import LuceneBM25Model
            from gensim.similarities import SparseMatrixSimilarity
        except Exception as exc:
            raise ImportError(
                "BM25 ufficiale richiede pyserini, gensim con LuceneBM25Model "
                "e le dipendenze Java/JNI previste da Pyserini. Installa le "
                "dipendenze sparse del progetto; non viene usato un fallback "
                "lessicale diverso, perché altererebbe il ranking."
            ) from exc

        self.k1 = float(k1)
        self.b = float(b)
        self.doc_ids = [str(doc_id) for doc_id in corpus.keys()]
        self.documents = [str(text or "") for text in corpus.values()]

        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("Il corpus contiene ID documento duplicati dopo str().")
        if any(not doc_id for doc_id in self.doc_ids):
            raise ValueError("Il corpus contiene un ID documento vuoto.")

        self.analyzer = analysis.Analyzer(analysis.get_lucene_analyzer())

        print(f"[BM25] Analysing corpus: {len(self.documents)} documents")
        analysed_corpus = [
            self.analyzer.analyze(text)
            for text in tqdm(
                self.documents,
                desc="BM25 analyze corpus",
                unit="doc",
                leave=False,
            )
        ]

        self.dictionary = Dictionary(analysed_corpus)
        self.model = LuceneBM25Model(
            dictionary=self.dictionary,
            k1=self.k1,
            b=self.b,
        )
        bm25_corpus = self.model[
            list(map(self.dictionary.doc2bow, analysed_corpus))
        ]
        self.index = SparseMatrixSimilarity(
            bm25_corpus,
            num_docs=len(self.doc_ids),
            num_terms=len(self.dictionary),
            normalize_queries=False,
            normalize_documents=False,
        )

    def search_one(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        """Return the top-k documents ordered by descending BM25 score.

        The full stable sort mirrors the official starter-kit implementation.
        Empty or out-of-vocabulary queries can legitimately return zero-score
        documents, matching that implementation; hard-negative mining below
        discards those zero-score candidates rather than treating them as hard.
        """
        top_k = int(top_k)
        if top_k <= 0:
            return []

        query_tokens = self.analyzer.analyze(str(query or ""))
        query_bow = self.dictionary.doc2bow(query_tokens)
        query_vector = self.model[query_bow]
        similarities = self.index[query_vector].tolist()

        # Stable sorting preserves corpus order for tied scores, as in RETECO's
        # official starter kit. Avoid changing this to argpartition: it can
        # introduce nondeterministic tie ordering around the cutoff.
        pairs = sorted(
            zip(self.doc_ids, similarities),
            key=lambda item: item[1],
            reverse=True,
        )
        return [(doc_id, float(score)) for doc_id, score in pairs[:top_k]]

    def search(
        self,
        queries: Sequence[str],
        query_ids: Optional[Sequence[str]] = None,
        top_k: int = 1000,
        desc: str = "BM25 retrieval",
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Search several queries and return TREC-style ordered rankings."""
        if query_ids is None:
            query_ids = [str(i) for i in range(len(queries))]
        if len(queries) != len(query_ids):
            raise ValueError("queries e query_ids devono avere la stessa lunghezza.")

        results: Dict[str, List[Tuple[str, float]]] = {}
        for query_id, query in tqdm(
            list(zip(query_ids, queries)),
            total=len(queries),
            desc=desc,
            unit="query",
            leave=False,
        ):
            topic_id = str(query_id)
            if topic_id in results:
                raise ValueError(f"query_id duplicato: {topic_id}")
            results[topic_id] = self.search_one(query=str(query or ""), top_k=top_k)
        return results


def _stable_fingerprint(
    corpus: Mapping[str, str],
    samples: Sequence[Any],
    *,
    top_k: int,
    k1: float,
    b: float,
    cache_tag: str,
) -> str:
    """Hash the actual corpus content, order, queries, qrels and BM25 settings.

    Hashing only corpus IDs is not sufficient: changing a document's text while
    preserving its ID would otherwise reuse stale hard negatives. Corpus order
    is included because the official implementation stably resolves tied scores
    according to that order.
    """
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema_version": _CACHE_SCHEMA_VERSION,
                "algorithm_version": _BM25_ALGORITHM_VERSION,
                "cache_tag": str(cache_tag),
                "top_k": int(top_k),
                "k1": float(k1),
                "b": float(b),
                "num_docs": len(corpus),
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\nCORPUS\n")
    for doc_id, text in corpus.items():
        # Length-prefix each field to prevent ambiguous concatenations.
        for value in (str(doc_id), str(text or "")):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    digest.update(b"\nQUERIES\n")
    for sample in samples:
        record = {
            "topic_id": str(getattr(sample, "topic_id", "")),
            "query": str(getattr(sample, "contextual_query", "") or ""),
            "gold": sorted(str(x) for x in (getattr(sample, "gold_doc_ids", []) or [])),
        }
        digest.update(
            json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()[:24]


def _valid_cached_negatives(
    raw: Any,
    *,
    fingerprint: str,
    corpus_ids: set[str],
    samples: Sequence[Any],
    top_k: int,
) -> Optional[Dict[str, List[str]]]:
    """Validate cache metadata and IDs before reusing an on-disk cache."""
    if not isinstance(raw, dict):
        return None
    if raw.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return None
    if raw.get("algorithm_version") != _BM25_ALGORITHM_VERSION:
        return None
    if raw.get("fingerprint") != fingerprint:
        return None
    negatives = raw.get("negatives")
    if not isinstance(negatives, dict):
        return None

    gold_by_topic = {
        str(getattr(sample, "topic_id", "")): {
            str(x) for x in (getattr(sample, "gold_doc_ids", []) or [])
        }
        for sample in samples
    }
    expected_topics = set(gold_by_topic)
    if set(str(k) for k in negatives) != expected_topics:
        return None

    validated: Dict[str, List[str]] = {}
    for topic_id, ids in negatives.items():
        topic_id = str(topic_id)
        if not isinstance(ids, list):
            return None
        clean_ids: List[str] = []
        seen: set[str] = set()
        for doc_id in ids:
            doc_id = str(doc_id)
            if doc_id not in corpus_ids or doc_id in gold_by_topic[topic_id] or doc_id in seen:
                return None
            seen.add(doc_id)
            clean_ids.append(doc_id)
            if len(clean_ids) > top_k:
                return None
        validated[topic_id] = clean_ids
    return validated


def mine_bm25_hard_negatives(
    corpus: Dict[str, str],
    samples: Sequence[Any],
    top_k: int = 50,
    k1: float = 0.9,
    b: float = 0.4,
    cache_dir: Optional[Union[str, Path]] = None,
    cache_tag: str = "train",
) -> Dict[str, List[str]]:
    """Mine BM25 hard negatives from a provided (official) domain corpus.

    For each topic, retrieve up to the official top-1000 candidates, remove all
    gold documents, and retain at most ``top_k`` positive-score candidates.
    Zero-score candidates are not treated as hard negatives; if fewer than the
    requested number remain, RETECO2aTrainDataset can use its random fallback.

    Cache files are versioned and keyed by the actual corpus text/order, query
    text, gold IDs, BM25 parameters, requested pool size and tag. Malformed or
    stale caches are ignored and regenerated.
    """
    top_k = int(top_k)
    if top_k <= 0:
        raise ValueError("top_k deve essere > 0.")
    if not corpus:
        raise ValueError("Corpus per hard-negative mining vuoto.")
    if not samples:
        return {}

    topics = [str(getattr(sample, "topic_id", "")) for sample in samples]
    if any(not topic for topic in topics):
        raise ValueError("Almeno un sample non possiede topic_id valido.")
    if len(set(topics)) != len(topics):
        raise ValueError("topic_id duplicati nei sample di hard-negative mining.")

    fingerprint = _stable_fingerprint(
        corpus,
        samples,
        top_k=top_k,
        k1=k1,
        b=b,
        cache_tag=cache_tag,
    )
    cache_file: Optional[Path] = None
    corpus_ids = {str(doc_id) for doc_id in corpus.keys()}

    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = cache_root / f"bm25_hard_negatives_{fingerprint}.json"
        if cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                cached = _valid_cached_negatives(
                    payload,
                    fingerprint=fingerprint,
                    corpus_ids=corpus_ids,
                    samples=samples,
                    top_k=top_k,
                )
                if cached is not None:
                    print(f"[BM25] Loading validated hard-negative cache: {cache_file}")
                    return cached
                print(f"[BM25] Cache metadata/content mismatch; regenerating: {cache_file}")
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"[BM25] Invalid cache will be regenerated ({exc}).")

    print(f"[BM25] Mining up to {top_k} hard negatives for {len(samples)} queries...")
    bm25 = OfficialBM25(corpus=corpus, k1=k1, b=b)
    retrieval_depth = min(_OFFICIAL_RETRIEVAL_DEPTH, len(corpus))
    hard_negatives: Dict[str, List[str]] = {}

    for sample in tqdm(samples, desc=f"Hard-negative mining [{cache_tag}]", unit="query"):
        topic_id = str(sample.topic_id)
        gold_ids = {str(doc_id) for doc_id in (sample.gold_doc_ids or [])}
        ranked = bm25.search_one(
            query=str(getattr(sample, "contextual_query", "") or ""),
            top_k=retrieval_depth,
        )

        negatives: List[str] = []
        seen: set[str] = set()
        for doc_id, score in ranked:
            doc_id = str(doc_id)
            if doc_id in gold_ids or doc_id in seen:
                continue
            # Zero scores indicate no lexical match. They are not meaningful
            # hard negatives; let the dataset's random fallback fill the pool.
            if not math.isfinite(float(score)) or float(score) <= 0.0:
                continue
            seen.add(doc_id)
            negatives.append(doc_id)
            if len(negatives) >= top_k:
                break
        hard_negatives[topic_id] = negatives

    if cache_file is not None:
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "algorithm_version": _BM25_ALGORITHM_VERSION,
            "fingerprint": fingerprint,
            "cache_tag": str(cache_tag),
            "top_k": top_k,
            "k1": float(k1),
            "b": float(b),
            "num_topics": len(samples),
            "num_documents": len(corpus),
            "negatives": hard_negatives,
        }
        temp_path = cache_file.with_suffix(cache_file.suffix + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            os.replace(temp_path, cache_file)
            print(f"[BM25] Hard-negative cache saved: {cache_file}")
        except OSError as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            print(f"[BM25] Warning: unable to save hard-negative cache: {exc}")

    return hard_negatives
