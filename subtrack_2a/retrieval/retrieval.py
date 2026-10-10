#!/usr/bin/env python3
"""Sparse retrieval and BM25 hard-negative mining for RETECO Sub-track 2a.

Backend:
    Pyserini LuceneIndexer + Pyserini LuceneSearcher (native Lucene BM25).

The previous implementation used a Lucene analyzer with gensim's
LuceneBM25Model and SparseMatrixSimilarity. This version removes the Gensim
dependency, which currently cannot be built in the user's Python 3.14 Studio.
The analyzer and default BM25 parameters are retained, but the backend changes;
rankings are therefore not guaranteed to be numerically identical to the old
Gensim-based implementation and must be validated on the same dev topics.

The input corpus must be the organizer-provided RETECO domain corpus. This
module never downloads, adds, or retrieves documents from external collections.

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
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from tqdm import tqdm


_CACHE_SCHEMA_VERSION = 3
_BM25_ALGORITHM_VERSION = "reteco_pyserini_native_lucene_bm25_v1"
_INDEX_SCHEMA_VERSION = 1
_INDEX_ANALYZER_ID = "anserini_default_english_porter_stemming_stopwords_v1"
_OFFICIAL_RETRIEVAL_DEPTH = 1000
_DEFAULT_INDEX_BATCH_SIZE = 512
_DEFAULT_INDEX_THREADS = 2


def _update_hash_with_string(digest: Any, value: str) -> None:
    """Length-prefix a UTF-8 value to avoid ambiguous hash concatenations."""
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _corpus_fingerprint(corpus: Mapping[str, str]) -> str:
    """Fingerprint document IDs, document text, order, analyzer and index schema."""
    digest = hashlib.sha256()
    header = {
        "index_schema_version": _INDEX_SCHEMA_VERSION,
        "algorithm_version": _BM25_ALGORITHM_VERSION,
        "analyzer_id": _INDEX_ANALYZER_ID,
        "num_docs": len(corpus),
    }
    digest.update(
        json.dumps(
            header,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\nCORPUS\n")
    for doc_id, text in corpus.items():
        _update_hash_with_string(digest, str(doc_id))
        _update_hash_with_string(digest, str(text or ""))
    return digest.hexdigest()[:32]


class OfficialBM25:
    """BM25 retrieval using a native Pyserini/Lucene inverted index.

    Parameters
    ----------
    corpus:
        Ordered mapping ``doc_id -> document text``. Insertion order is kept
        when indexing, which also makes index construction reproducible for a
        fixed Pyserini/Lucene version.
    k1, b:
        BM25 term-frequency saturation and document-length normalization.
        Defaults remain the RETECO project's values (0.9 and 0.4).
    index_dir:
        Optional persistent directory for a Lucene index. When omitted, a
        temporary index is created and removed when this object is closed.
    index_batch_size:
        Number of documents passed to LuceneIndexer in a Python batch.
    index_threads:
        Number of indexer worker threads, where supported by installed Pyserini.

    Note
    ----
    This native Lucene backend removes Gensim. Although the BM25 parameters
    and English analyzer settings are retained, scoring/indexing internals
    differ from gensim.LuceneBM25Model + SparseMatrixSimilarity.
    """

    def __init__(
        self,
        corpus: Mapping[str, str],
        k1: float = 0.9,
        b: float = 0.4,
        index_dir: Optional[Union[str, Path]] = None,
        index_batch_size: int = _DEFAULT_INDEX_BATCH_SIZE,
        index_threads: int = _DEFAULT_INDEX_THREADS,
    ) -> None:
        if not corpus:
            raise ValueError("Corpus BM25 vuoto.")
        if not math.isfinite(float(k1)) or float(k1) <= 0:
            raise ValueError("k1 deve essere un numero finito > 0.")
        if not math.isfinite(float(b)) or not 0.0 <= float(b) <= 1.0:
            raise ValueError("b deve essere un numero finito compreso tra 0 e 1.")
        if int(index_batch_size) <= 0:
            raise ValueError("index_batch_size deve essere > 0.")
        if int(index_threads) <= 0:
            raise ValueError("index_threads deve essere > 0.")

        try:
            from pyserini.analysis import get_lucene_analyzer
            from pyserini.index.lucene import LuceneIndexer
            from pyserini.search.lucene import LuceneSearcher
        except Exception as exc:
            raise ImportError(
                "Il BM25 nativo richiede Pyserini e Java 21. Questa versione "
                "non richiede Gensim. Verifica che pyserini e le sue dipendenze "
                "Java/JNI siano installati nell'ambiente Python attivo."
            ) from exc

        self.k1 = float(k1)
        self.b = float(b)
        self.index_batch_size = int(index_batch_size)
        self.index_threads = int(index_threads)
        self.doc_ids = [str(doc_id) for doc_id in corpus.keys()]
        self.documents = [str(text or "") for text in corpus.values()]
        self._temporary_directory: Optional[tempfile.TemporaryDirectory] = None
        self._closed = False

        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("Il corpus contiene ID documento duplicati dopo str().")
        if any(not doc_id for doc_id in self.doc_ids):
            raise ValueError("Il corpus contiene un ID documento vuoto.")

        self.corpus_fingerprint = _corpus_fingerprint(corpus)

        if index_dir is None:
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="reteco_lucene_bm25_"
            )
            self.index_dir = Path(self._temporary_directory.name) / "index"
            self._build_index(self.index_dir, LuceneIndexer)
            self._write_index_metadata(self.index_dir)
        else:
            self.index_dir = Path(index_dir)
            self.index_dir.parent.mkdir(parents=True, exist_ok=True)
            if not self._is_valid_index(self.index_dir):
                if self.index_dir.exists():
                    shutil.rmtree(self.index_dir, ignore_errors=True)
                self._build_persistent_index(self.index_dir, LuceneIndexer)

        try:
            self.searcher = LuceneSearcher(str(self.index_dir))
            # Explicitly apply the same English analyzer settings as the old
            # pipeline: Porter stemming and English stopword filtering.
            self.searcher.set_analyzer(
                get_lucene_analyzer(
                    language="en",
                    stemming=True,
                    stemmer="porter",
                    stopwords=True,
                )
            )
            self.searcher.set_bm25(self.k1, self.b)
        except Exception:
            self.close()
            raise

    def _index_metadata(self) -> Dict[str, Any]:
        return {
            "index_schema_version": _INDEX_SCHEMA_VERSION,
            "algorithm_version": _BM25_ALGORITHM_VERSION,
            "analyzer_id": _INDEX_ANALYZER_ID,
            "corpus_fingerprint": self.corpus_fingerprint,
            "num_documents": len(self.doc_ids),
        }

    @staticmethod
    def _metadata_path(index_dir: Path) -> Path:
        return index_dir / "_reteco_index_metadata.json"

    def _is_valid_index(self, index_dir: Path) -> bool:
        if not index_dir.is_dir():
            return False
        # Lucene index should include a segments_N file and our matching marker.
        if not any(p.is_file() and p.name.startswith("segments_") for p in index_dir.iterdir()):
            return False
        marker = self._metadata_path(index_dir)
        if not marker.is_file():
            return False
        try:
            with marker.open("r", encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False
        return stored == self._index_metadata()

    def _write_index_metadata(self, index_dir: Path) -> None:
        marker = self._metadata_path(index_dir)
        temp_marker = marker.with_suffix(marker.suffix + ".tmp")
        with temp_marker.open("w", encoding="utf-8") as handle:
            json.dump(self._index_metadata(), handle, ensure_ascii=False, indent=2)
        os.replace(temp_marker, marker)

    def _build_index(self, target_dir: Path, indexer_class: Any) -> None:
        """Build a Lucene index from the in-memory official corpus."""
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)

        try:
            indexer = indexer_class(str(target_dir), threads=self.index_threads)
        except TypeError:
            # Compatibility with Pyserini versions whose embeddable indexer
            # does not expose `threads` as a keyword argument.
            indexer = indexer_class(str(target_dir))

        try:
            starts = range(0, len(self.doc_ids), self.index_batch_size)
            for start in tqdm(
                starts,
                total=(len(self.doc_ids) + self.index_batch_size - 1) // self.index_batch_size,
                desc="Lucene index corpus",
                unit="batch",
                leave=False,
            ):
                stop = min(start + self.index_batch_size, len(self.doc_ids))
                batch = [
                    {"id": self.doc_ids[i], "contents": self.documents[i]}
                    for i in range(start, stop)
                ]
                if hasattr(indexer, "add_batch_dict"):
                    indexer.add_batch_dict(batch)
                else:
                    for record in batch:
                        indexer.add_doc_dict(record)
        finally:
            # close() commits the Lucene index. It is required even when the
            # indexing loop has no more documents.
            indexer.close()

    def _build_persistent_index(self, final_dir: Path, indexer_class: Any) -> None:
        """Build in a sibling temporary directory and publish once complete."""
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_root = Path(
            tempfile.mkdtemp(
                prefix=f".{final_dir.name}.building-",
                dir=str(final_dir.parent),
            )
        )
        staged_index = temp_root / "index"
        try:
            self._build_index(staged_index, indexer_class)
            self._write_index_metadata(staged_index)
            # Another process may have created this same content-addressed
            # index while we were indexing. Reuse it if valid.
            if final_dir.exists() and self._is_valid_index(final_dir):
                return
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            os.replace(staged_index, final_dir)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def search_one(self, query: str, top_k: int = 1000) -> List[Tuple[str, float]]:
        """Return up to top_k Lucene hits in descending BM25 score order.

        Lucene returns matching documents rather than manufacturing a full list
        of zero-score non-matches. Consequently, empty/OOV queries may return an
        empty ranking; this is standard sparse-retrieval behavior.
        """
        top_k = int(top_k)
        if top_k <= 0:
            return []
        query = str(query or "").strip()
        if not query:
            return []

        hits = self.searcher.search(query, k=top_k)
        return [
            (str(hit.docid), float(hit.score))
            for hit in hits
            if math.isfinite(float(hit.score))
        ]

    def search(
        self,
        queries: Sequence[str],
        query_ids: Optional[Sequence[str]] = None,
        top_k: int = 1000,
        desc: str = "BM25 retrieval",
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Search a sequence of queries and return ordered rankings by topic ID."""
        if query_ids is None:
            query_ids = [str(i) for i in range(len(queries))]
        if len(queries) != len(query_ids):
            raise ValueError("queries e query_ids devono avere la stessa lunghezza.")

        results: Dict[str, List[Tuple[str, float]]] = {}
        for query_id, query in tqdm(
            zip(query_ids, queries),
            total=len(queries),
            desc=desc,
            unit="query",
            leave=False,
        ):
            topic_id = str(query_id)
            if topic_id in results:
                raise ValueError(f"query_id duplicato: {topic_id}")
            results[topic_id] = self.search_one(str(query or ""), top_k=top_k)
        return results

    def close(self) -> None:
        """Close the searcher and remove any temporary index directory."""
        if self._closed:
            return
        self._closed = True
        searcher = getattr(self, "searcher", None)
        if searcher is not None:
            try:
                searcher.close()
            except Exception:
                pass
        temporary_directory = self._temporary_directory
        if temporary_directory is not None:
            try:
                temporary_directory.cleanup()
            except Exception:
                pass
            self._temporary_directory = None

    def __enter__(self) -> "OfficialBM25":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _stable_fingerprint(
    corpus: Mapping[str, str],
    samples: Sequence[Any],
    *,
    top_k: int,
    k1: float,
    b: float,
    cache_tag: str,
    corpus_fingerprint: Optional[str] = None,
) -> str:
    """Fingerprint hard negatives from corpus/index version, queries and qrels."""
    corpus_fingerprint = corpus_fingerprint or _corpus_fingerprint(corpus)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "cache_schema_version": _CACHE_SCHEMA_VERSION,
                "algorithm_version": _BM25_ALGORITHM_VERSION,
                "corpus_fingerprint": corpus_fingerprint,
                "cache_tag": str(cache_tag),
                "top_k": int(top_k),
                "k1": float(k1),
                "b": float(b),
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\nQUERIES\n")
    for sample in samples:
        record = {
            "topic_id": str(getattr(sample, "topic_id", "")),
            "query": str(getattr(sample, "contextual_query", "") or ""),
            "gold": sorted(
                str(x) for x in (getattr(sample, "gold_doc_ids", []) or [])
            ),
        }
        digest.update(
            json.dumps(
                record,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
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
    """Validate metadata and IDs before reusing a hard-negative cache."""
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
    if set(str(k) for k in negatives) != set(gold_by_topic):
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
            if (
                doc_id not in corpus_ids
                or doc_id in gold_by_topic[topic_id]
                or doc_id in seen
            ):
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
    """Mine BM25 hard negatives from a provided RETECO domain corpus.

    For every topic, retrieve up to the top 1000 Lucene candidates, remove all
    gold IDs and retain up to ``top_k`` positive-score non-gold documents.
    The persistent Lucene index and hard-negative cache are both content-keyed.
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

    corpus_fp = _corpus_fingerprint(corpus)
    fingerprint = _stable_fingerprint(
        corpus,
        samples,
        top_k=top_k,
        k1=k1,
        b=b,
        cache_tag=cache_tag,
        corpus_fingerprint=corpus_fp,
    )

    cache_root: Optional[Path] = None
    cache_file: Optional[Path] = None
    if cache_dir is not None:
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = cache_root / f"bm25_hard_negatives_{fingerprint}.json"
        if cache_file.is_file():
            try:
                with cache_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                cached = _valid_cached_negatives(
                    payload,
                    fingerprint=fingerprint,
                    corpus_ids={str(doc_id) for doc_id in corpus.keys()},
                    samples=samples,
                    top_k=top_k,
                )
                if cached is not None:
                    print(f"[BM25] Loading validated native-Lucene hard-negative cache: {cache_file}")
                    return cached
                print(f"[BM25] Stale/incompatible hard-negative cache; regenerating: {cache_file}")
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"[BM25] Invalid hard-negative cache will be regenerated ({exc}).")

    persistent_index_dir: Optional[Path] = None
    if cache_root is not None:
        persistent_index_dir = cache_root / "lucene_indexes" / corpus_fp

    print(
        f"[BM25] Mining up to {top_k} hard negatives for {len(samples)} queries "
        f"using native Pyserini/Lucene (k1={float(k1)}, b={float(b)})..."
    )
    bm25 = OfficialBM25(
        corpus=corpus,
        k1=k1,
        b=b,
        index_dir=persistent_index_dir,
    )
    retrieval_depth = min(_OFFICIAL_RETRIEVAL_DEPTH, len(corpus))
    hard_negatives: Dict[str, List[str]] = {}

    try:
        for sample in tqdm(
            samples,
            desc=f"Hard-negative mining [{cache_tag}]",
            unit="query",
        ):
            topic_id = str(sample.topic_id)
            gold_ids = {
                str(doc_id) for doc_id in (getattr(sample, "gold_doc_ids", []) or [])
            }
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
                if not math.isfinite(float(score)) or float(score) <= 0.0:
                    continue
                seen.add(doc_id)
                negatives.append(doc_id)
                if len(negatives) >= top_k:
                    break
            hard_negatives[topic_id] = negatives
    finally:
        bm25.close()

    if cache_file is not None:
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "algorithm_version": _BM25_ALGORITHM_VERSION,
            "fingerprint": fingerprint,
            "corpus_fingerprint": corpus_fp,
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
