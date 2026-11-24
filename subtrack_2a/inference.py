#!/usr/bin/env python3
"""
subtrack_2a/inference.py
"""

import sys
import json
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset.dataset import (
    TRACK2_DOMAINS,
    ContextAwareQueryFormatter,
    load_track2_domain_data,
    load_benchmark_conversations,
)

from retrieval.retrieval import OfficialBM25, _corpus_fingerprint
from models.model import ConversationalBiEncoder, ConversationalCrossEncoder
from utils.utils import (
    load_config,
    save_json,
    compute_official_ndcg,
    ranking_scores,
    set_seed,
    reciprocal_rank_fusion,
    write_trec_run,
    validate_trec_file,
    setup_logger,
)

# Logger base globale a console per evitare errori di import preliminari
logger = logging.getLogger("RETECO_Inference")
if not logger.handlers:
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)-8s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_ch)
    logger.setLevel(logging.INFO)


# ==============================================================================
# 1. Gestore della Cache degli Embedding Documentali
# ==============================================================================
class DocumentEmbeddingCache:
    """Cache degli embedding su disco con fingerprint del checkpoint per evitare collisioni."""

    def __init__(self, cache_dir: Path, model_tag: str, pooling: str, max_len: int, ckpt_path: Optional[Path] = None, normalize: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_tag = model_tag
        self.pooling = pooling
        self.max_len = max_len

        self.normalize = normalize

        # Calcola la fingerprint dei pesi del checkpoint
        self.ckpt_hash = "pretrained"
        if ckpt_path is not None and ckpt_path.exists():
            digest = hashlib.sha256()
            files = [ckpt_path] if ckpt_path.is_file() else sorted(ckpt_path.glob("adapter*"))
            for path in files:
                digest.update(path.name.encode())
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
            self.ckpt_hash = digest.hexdigest()


    def _get_hash_key(self, domain: str, corpus_fingerprint: str) -> str:
        key_str = f"{self.model_tag}_{self.ckpt_hash}_{self.pooling}_{self.max_len}_{domain}_{corpus_fingerprint}_{self.normalize}"
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:16]

    def load(self, domain: str, corpus_size: int, corpus_fingerprint: str) -> Optional[Tuple[torch.Tensor, List[str]]]:
        hash_key = self._get_hash_key(domain, corpus_fingerprint)
        emb_file = self.cache_dir / f"embs_{domain}_{hash_key}.pt"
        meta_file = self.cache_dir / f"meta_{domain}_{hash_key}.json"

        if emb_file.exists() and meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("corpus_size") == corpus_size and meta.get("ckpt_hash") == self.ckpt_hash:
                    logger.info(f"✓ Cache trovata per [{domain}] (Hash: {self.ckpt_hash}). Caricamento immediato...")
                    data = torch.load(emb_file, map_location="cpu", weights_only=True)
                    return data["embeddings"], data["doc_ids"]
            except Exception:
                pass
        return None

    def save(self, domain: str, embeddings: torch.Tensor, doc_ids: List[str], corpus_fingerprint: str):
        hash_key = self._get_hash_key(domain, corpus_fingerprint)
        emb_file = self.cache_dir / f"embs_{domain}_{hash_key}.pt"
        meta_file = self.cache_dir / f"meta_{domain}_{hash_key}.json"
        try:
            torch.save({"embeddings": embeddings.cpu(), "doc_ids": doc_ids}, emb_file)
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump({
                    "domain": domain,
                    "corpus_size": len(doc_ids),
                    "model_tag": self.model_tag,
                    "ckpt_hash": self.ckpt_hash,
                    "pooling": self.pooling,
                    "max_len": self.max_len,
                }, f)
        except Exception as e:
            logger.warning(f"Salvataggio cache fallito per {domain}: {e}")


# ==============================================================================
# 2. Cache dei risultati di retrieval (Dense + BM25)
# ==============================================================================
class RetrievalRunCache:
    """Cache persistente dei ranking Dense/BM25 per rendere le fusioni RRF quasi immediate."""

    VERSION = "retrieval_cache_v2"

    def __init__(
        self,
        cache_dir: Path,
        model_tag: str,
        ckpt_hash: str,
        pooling: str,
        max_query_length: int,
        max_doc_length: int,
        query_strategy: str,
        query_instruction: str,
        sparse_enabled: bool,
        bm25_k1: float,
        bm25_b: float,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Tutti questi parametri entrano nella chiave per evitare di riusare
        # accidentalmente ranking generati con una configurazione diversa.
        self.model_tag = model_tag
        self.ckpt_hash = ckpt_hash
        self.pooling = pooling
        self.max_query_length = int(max_query_length)
        self.max_doc_length = int(max_doc_length)
        self.query_strategy = str(query_strategy)
        self.query_instruction = str(query_instruction)
        self.sparse_enabled = bool(sparse_enabled)
        self.bm25_k1 = float(bm25_k1)
        self.bm25_b = float(bm25_b)

    def _get_hash_key(
        self,
        domain: str,
        split: str,
        top_candidates: int,
        ablation_mode: Optional[str],
    ) -> str:
        payload = {
            "version": self.VERSION,
            "model_tag": self.model_tag,
            "ckpt_hash": self.ckpt_hash,
            "pooling": self.pooling,
            "max_query_length": self.max_query_length,
            "max_doc_length": self.max_doc_length,
            "query_strategy": self.query_strategy,
            "query_instruction": self.query_instruction,
            "sparse_enabled": self.sparse_enabled,
            "bm25_k1": self.bm25_k1,
            "bm25_b": self.bm25_b,
            "domain": domain,
            "split": split,
            "top_candidates": int(top_candidates),
            "ablation_mode": ablation_mode or "main",
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _path(
        self,
        domain: str,
        split: str,
        top_candidates: int,
        ablation_mode: Optional[str],
    ) -> Path:
        key = self._get_hash_key(domain, split, top_candidates, ablation_mode)
        return self.cache_dir / f"retrieval_{domain}_{split}_{key}.json"

    def load(
        self,
        domain: str,
        split: str,
        top_candidates: int,
        ablation_mode: Optional[str],
    ) -> Optional[Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, List[Tuple[str, float]]]]]:
        path = self._path(domain, split, top_candidates, ablation_mode)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            dense_run = {
                str(qid): [(str(doc_id), float(score)) for doc_id, score in docs]
                for qid, docs in payload.get("dense_run", {}).items()
            }
            bm25_run = {
                str(qid): [(str(doc_id), float(score)) for doc_id, score in docs]
                for qid, docs in payload.get("bm25_run", {}).items()
            }

            logger.info(
                f"✓ Retrieval cache trovata per [{domain}] "
                f"(Dense + BM25, {len(dense_run)} query)."
            )
            return dense_run, bm25_run
        except Exception as e:
            logger.warning(f"Cache retrieval non leggibile per {domain}: {e}")
            return None

    def save(
        self,
        domain: str,
        split: str,
        top_candidates: int,
        ablation_mode: Optional[str],
        dense_run: Dict[str, List[Tuple[str, float]]],
        bm25_run: Dict[str, List[Tuple[str, float]]],
        corpus_size: int,
        query_count: int,
    ) -> None:
        path = self._path(domain, split, top_candidates, ablation_mode)

        payload = {
            "version": self.VERSION,
            "domain": domain,
            "split": split,
            "corpus_size": int(corpus_size),
            "query_count": int(query_count),
            "dense_run": dense_run,
            "bm25_run": bm25_run,
        }

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            logger.info(f"✓ Retrieval cache salvata per [{domain}].")
        except Exception as e:
            logger.warning(f"Salvataggio retrieval cache fallito per {domain}: {e}")


# ==============================================================================
# 3. Calcolo Metriche Diagnostiche
# ==============================================================================

def compute_detailed_metrics(
    qrels: Dict[str, Dict[str, int]],
    run: Dict[str, List[Tuple[str, float]]],
    cutoff_k: int = 10,
) -> Dict[str, float]:
    """Calcola nDCG@10, Recall@10, Recall@50, Recall@100 e MRR su una singola run."""
    trec_format = {t_id: ranking_scores(docs) for t_id, docs in run.items()}
    official = compute_official_ndcg(qrels, trec_format, cutoff=cutoff_k)

    recalls = {10: [], 50: [], 100: []}
    mrrs = []

    for t_id, q_golds in qrels.items():
        ranked = list(trec_format.get(t_id, {}))
        golds = {doc_id for doc_id, rel in q_golds.items() if rel > 0}

        for k in [10, 50, 100]:
            hits = len(set(ranked[:k]).intersection(golds))
            recalls[k].append(hits / max(1, len(golds)))

        rr = 0.0
        for rank_idx, d_id in enumerate(ranked, 1):
            if d_id in golds:
                rr = 1.0 / rank_idx
                break
        mrrs.append(rr)

    return {
        f"nDCG@{cutoff_k}": official.get(f"ndcg_cut_{cutoff_k}", 0.0),
        "Recall@10": float(np.mean(recalls[10])) if recalls[10] else 0.0,
        "Recall@50": float(np.mean(recalls[50])) if recalls[50] else 0.0,
        "Recall@100": float(np.mean(recalls[100])) if recalls[100] else 0.0,
        "MRR": float(np.mean(mrrs)) if mrrs else 0.0,
    }


# ==============================================================================
# 4. Pipeline Principale di Inferenza
# ==============================================================================

def run_evaluation(
    config: Dict[str, Any],
    split: str = "dev",
    ablation_mode: Optional[str] = None,
    use_retrieval_cache: bool = True,
    query_mode: Optional[str] = None,
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
):
    gen_cfg = config.get("general", {})
    paths_cfg = config.get("paths", {})
    dom_cfg = config.get("domains", {})
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    bi_cfg = config.get("bi_encoder", {})
    sparse_cfg = config.get("sparse", {})
    hybrid_cfg = config.get("hybrid_fusion", {})
    cross_cfg = config.get("cross_encoder", {})
    lora_cfg = config.get("lora", {})

    # Gestione Ablation Flags (A-G)
    query_strategy = data_cfg.get("query_strategy", "budget_context")
    use_dense = ablation_mode not in {"A", "B"}
    use_sparse = bool(sparse_cfg.get("enabled", True)) and ablation_mode not in {"C", "D"}
    use_fusion = use_dense and use_sparse
    use_dense_checkpoint = use_dense
    use_cross_encoder = bool(cross_cfg.get("enabled", False)) and ablation_mode is None
    use_cross_finetuned = use_cross_encoder

    if ablation_mode == "A":
        logger.info(">>> Modalità Ablation A: BM25 Current-Turn Only <<<")
        query_strategy = "query_only"
        use_cross_encoder = False
    elif ablation_mode == "B":
        logger.info(">>> Modalità Ablation B: BM25 con Cronologia <<<")
        use_cross_encoder = False
    elif ablation_mode == "C":
        logger.info(">>> Modalità Ablation C: Dense Pretrained (No Fine-tuning) <<<")
        use_dense_checkpoint = False
        use_cross_encoder = False
    elif ablation_mode == "D":
        logger.info(">>> Modalità Ablation D: Dense Fine-Tuned Alone <<<")
        use_cross_encoder = False
    elif ablation_mode == "E":
        logger.info(">>> Modalità Ablation E: BM25 + Dense RRF (No Reranker) <<<")
        use_cross_encoder = False
    elif ablation_mode == "F":
        logger.info(">>> Modalità Ablation F: BM25 + Dense RRF + Pretrained Reranker <<<")
        use_cross_encoder = True
        use_cross_finetuned = False
    elif ablation_mode == "G":
        logger.info(">>> Modalità Ablation G: BM25 + Dense RRF + Fine-Tuned Reranker <<<")
        use_cross_encoder = True
        use_cross_finetuned = True

    # Rilevamento Hardware coerente (CUDA per Cloud/T4, MPS per Mac, CPU come fallback)
    target_dev = gen_cfg.get("device", "auto").lower()
    if target_dev == "cpu":
        device = torch.device("cpu")
    elif target_dev == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif target_dev == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    use_amp = bool(gen_cfg.get("mixed_precision", True) and device.type == "cuda")
    logger.info(f"Avvio Valutazione [Split: {split.upper()}] su Device: {device} | AMP FP16: {use_amp}")

    set_seed(int(gen_cfg.get("seed", 42)))
    base_data_dir = data_dir or Path(paths_cfg.get("full_data_dir" if paths_cfg.get("data_mode") == "full" else "sample_data_dir"))
    active = dom_cfg.get("active_domains", "all")
    domains = TRACK2_DOMAINS if active == "all" else [active] if isinstance(active, str) else list(active)
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("Configura almeno un dominio e non ripetere i nomi.")
    query_mode = query_mode or eval_cfg.get("inference_query_mode", "contextual")
    if query_mode not in {"contextual", "reasoned"}:
        raise ValueError(f"Modalità query sconosciuta: {query_mode}")
    if not use_dense and not use_sparse:
        raise ValueError("Le ablazioni A/B richiedono sparse.enabled=true.")

    # Inizializzazione Tokenizer
    model_name = bi_cfg.get("model_name_or_path", "BAAI/bge-base-en-v1.5")
    tokenizer = AutoTokenizer.from_pretrained(model_name) if use_dense else None
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    q_inst = bi_cfg.get("query_instruction", {}).get("text", "") if bi_cfg.get("query_instruction", {}).get("enabled", False) else ""
    formatter = ContextAwareQueryFormatter(
        tokenizer=tokenizer,
        max_query_length=data_cfg.get("max_query_length", 256),
        query_instruction=q_inst,
        strategy=query_strategy,
    )
    rewrites = {}
    if use_dense and query_mode == "reasoned":
        from train import prepare_query_rewrites

        input_samples = {}
        for domain in domains:
            benchmark = base_data_dir / domain / f"benchmark_{split}.json"
            if not benchmark.exists():
                benchmark = benchmark.with_suffix(".jsonl")
            input_samples[domain] = load_benchmark_conversations(benchmark, domain, formatter)
        rewrites = prepare_query_rewrites(
            domains=domains,
            train_samples_by_domain=input_samples,
            formatter=formatter,
            paths_cfg=paths_cfg,
            rewrite_cfg=config.get("query_rewriting", {}),
            seed=int(gen_cfg.get("seed", 42)),
            val_ratio=float(data_cfg.get("val_ratio", 0.15)),
            source_split=split,
        )

    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints/subtrack_2a"))
    best_pt = next((p for p in (
        checkpoint_dir / "best_model.pt", checkpoint_dir / "bi_encoder" / "best_model.pt"
    ) if p.is_file()), None)
    best_hf = next((p for p in (
        checkpoint_dir / "best_hf_model", checkpoint_dir / "bi_encoder" / "best_hf_model"
    ) if (p / "adapter_config.json").is_file()), None)
    loaded_checkpoint = None
    if use_dense:
        if use_dense_checkpoint and best_pt is None and best_hf is None:
            raise FileNotFoundError(f"Checkpoint 2a mancante in {checkpoint_dir}. Usa C per il dense pretrained.")
        bi_encoder = ConversationalBiEncoder(
            model_name_or_path=model_name,
            temperature=bi_cfg.get("temperature", 0.05),
            normalize_embeddings=bi_cfg.get("normalize_embeddings", True),
            pooling_strategy=bi_cfg.get("pooling_strategy", "mean"),
            lora_cfg=lora_cfg if use_dense_checkpoint and best_pt is not None else None,
        )
        if use_dense_checkpoint and best_pt is not None:
            ckpt = torch.load(best_pt, map_location="cpu", weights_only=True)
            stored_model = ckpt.get("config", {}).get("bi_encoder", {}).get("model_name_or_path")
            if stored_model and stored_model != model_name:
                raise ValueError(f"Backbone del checkpoint: {stored_model}; configurato: {model_name}.")
            weights = ckpt["model_state_dict"]
            partial = ckpt.get("checkpoint_format") == "trainable_parameters_only"
            if partial:
                expected = {name for name, parameter in bi_encoder.named_parameters() if parameter.requires_grad}
                if weights.keys() != expected:
                    raise ValueError("Il checkpoint non corrisponde ai parametri trainabili configurati.")
            bi_encoder.load_state_dict(weights, strict=not partial)
            loaded_checkpoint = best_pt
        elif use_dense_checkpoint:
            from peft import PeftConfig, PeftModel
            adapter_config = PeftConfig.from_pretrained(str(best_hf))
            if adapter_config.base_model_name_or_path != model_name:
                raise ValueError("Il backbone dell'adapter non coincide con quello configurato.")
            bi_encoder.encoder = PeftModel.from_pretrained(bi_encoder.encoder, str(best_hf))
            loaded_checkpoint = best_hf
        bi_encoder.to(device).eval()

    # Inizializzazione Cross-Encoder (se richiesto)
    cross_encoder = None
    cross_tok = None
    fine_tuned_pt = None

    if use_cross_encoder and not (ablation_mode in ["A", "B", "C", "D", "E"]):
        reranker_name = cross_cfg.get(
            "model_name_or_path",
            "BAAI/bge-reranker-base"
        )

        cross_ckpt_dir = Path(
            cross_cfg.get(
                "checkpoint_dir",
                "checkpoints/subtrack_2a/cross_encoder"
            )
        )

        fine_tuned_pt = cross_ckpt_dir / "best_reranker.pt"

        # Il pretrained reranker viene usato solo nelle ablation esplicite F.
        # Nel run principale, senza checkpoint fine-tuned, manteniamo RRF.
        use_pretrained_reranker = ablation_mode == "F"

        if use_cross_finetuned and fine_tuned_pt.exists():
            logger.info(
                f"Inizializzazione Cross-Encoder FINE-TUNED: {reranker_name}"
            )

            cross_tok = AutoTokenizer.from_pretrained(reranker_name)

            cross_encoder = ConversationalCrossEncoder(
                model_name_or_path=reranker_name,
                num_labels=1,
                use_lora=cross_cfg.get("use_lora", False),
            ).to(device)

            c_ckpt = torch.load(fine_tuned_pt, map_location=device, weights_only=True)

            cross_encoder.load_state_dict(
                c_ckpt["model_state_dict"],
                strict=True
            )

            cross_encoder.eval()

            logger.info(
                f"✓ Cross-Encoder fine-tuned caricato da: {fine_tuned_pt}"
            )

        elif use_pretrained_reranker:
            logger.info(
                f"Utilizzo Cross-Encoder PRETRAINED per Ablation F: "
                f"{reranker_name}"
            )

            cross_tok = AutoTokenizer.from_pretrained(reranker_name)

            cross_encoder = ConversationalCrossEncoder(
                model_name_or_path=reranker_name,
                num_labels=1,
                use_lora=False,
            ).to(device)

            cross_encoder.eval()

        else:
            raise FileNotFoundError(f"Checkpoint del reranker mancante: {fine_tuned_pt}")

    # Inizializzazione Cache Embeddings
    emb_cache = DocumentEmbeddingCache(
        cache_dir=Path(
            paths_cfg.get(
                "embedding_cache_dir",
                "data/cache/document_embeddings"
            )
        ),
        model_tag=f"{model_name}@{getattr(bi_encoder.encoder.config, '_commit_hash', None)}" if use_dense else model_name,
        pooling=bi_cfg.get("pooling_strategy", "mean"),
        max_len=data_cfg.get("max_doc_length", 256),
        ckpt_path=loaded_checkpoint,
        normalize=bi_cfg.get("normalize_embeddings", True),
    )

    cutoff_k = int(eval_cfg.get("cutoff_k", 10))
    rerank_depth = int(hybrid_cfg.get("top_candidates_to_rerank", 100))
    top_candidates = int(eval_cfg.get("retrieval_depth", 1000))
    if not 10 <= cutoff_k <= top_candidates or not cutoff_k <= rerank_depth:
        raise ValueError("Richiesti cutoff >= 10 e profondità retrieval/rerank >= cutoff.")

    rrf_k_config = int(hybrid_cfg.get("rrf_k", eval_cfg.get("rrf_k", 60)))
    rrf_sweep_values = [10, 30, 60, 100] if split == "train" and use_fusion else [rrf_k_config]

    if rrf_k_config not in rrf_sweep_values:
        rrf_sweep_values.append(rrf_k_config)

    logger.info(
        f"RRF configurato: k={rrf_k_config} | "
        f"Sweep: {sorted(rrf_sweep_values)}"
    )

    # Cache dei ranking Dense/BM25: la fusion RRF resta sempre ricalcolabile
    # senza ripetere encoding del corpus o indicizzazione/search BM25.
    retrieval_cache = RetrievalRunCache(
        cache_dir=Path(
            paths_cfg.get(
                "retrieval_cache_dir",
                "data/cache/retrieval_runs",
            )
        ),
        model_tag=emb_cache.model_tag,
        ckpt_hash=emb_cache.ckpt_hash,
        pooling=bi_cfg.get("pooling_strategy", "mean"),
        max_query_length=data_cfg.get("max_query_length", 256),
        max_doc_length=data_cfg.get("max_doc_length", 256),
        query_strategy=query_strategy,
        query_instruction=q_inst,
        sparse_enabled=sparse_cfg.get("enabled", True),
        bm25_k1=sparse_cfg.get("k1", 0.9),
        bm25_b=sparse_cfg.get("b", 0.4),
    )

    all_reports: Dict[str, Any] = {
        "Dense": {},
        "BM25": {},
        "RRF": {},
        "Final": {},
        "RRF_sweep": {},
    }
    final_trec_run: Dict[str, List[Tuple[str, float]]] = {}
    expected_topics: set[str] = set()

    for domain in domains:
        logger.info(f"\n{'='*25} DOMINIO: {domain.upper()} {'='*25}")
        corpus, samples, qrels = load_track2_domain_data(
            base_data_dir, domain, split=split, formatter=formatter, query_rewrites=rewrites.get(domain)
        )
        domains_topics = [sample.topic_id for sample in samples]
        if not corpus or not domains_topics or len(set(domains_topics)) != len(domains_topics):
            raise ValueError(f"{domain}: corpus/benchmark vuoto o topic duplicati.")
        if split != "test":
            if not qrels or qrels.keys() - set(domains_topics):
                raise ValueError(f"{domain}: qrels vuoti o con turni estranei al benchmark.")
            unjudged = len(domains_topics) - len(qrels)
            if unjudged:
                logger.warning("%s: %d turni senza giudizi ufficiali; esclusi dalle metriche.", domain, unjudged)
        if set(domains_topics) & expected_topics:
            raise ValueError(f"{domain}: topic già presenti in un altro dominio.")
        expected_topics.update(domains_topics)
        queries = [sample.reasoned_query if query_mode == "reasoned" and use_dense
                   else sample.contextual_query for sample in samples]
        if not all(queries):
            raise ValueError(f"{domain}: query vuote o riscritture mancanti.")
        corpus_fp = _corpus_fingerprint(corpus)
        query_signature = [(sample.topic_id, sample.query, sample.history, query)
                           for sample, query in zip(samples, queries)]
        source_hash = hashlib.sha256(json.dumps(
            [corpus_fp, query_signature, query_mode, emb_cache.normalize, use_amp],
            ensure_ascii=False,
        ).encode()).hexdigest()
        cache_split = f"{split}_{source_hash[:20]}"

        doc_ids = list(corpus.keys())
        doc_texts = [corpus[did] for did in doc_ids]

        # 1-2. Retrieval Dense + BM25
        # Prima controlliamo la cache dei ranking completi. Se presente,
        # tutte le fusioni RRF diventano praticamente istantanee.
        cached_retrieval = None
        if use_retrieval_cache:
            cached_retrieval = retrieval_cache.load(
                domain=domain,
                split=cache_split,
                top_candidates=top_candidates,
                ablation_mode=ablation_mode,
            )

        if cached_retrieval is not None:
            dense_run, bm25_run = cached_retrieval

        else:
            # -------------------------
            # 1. Retrieval Denso
            # -------------------------
            dense_run = {}
            if use_dense:
                cached_data = emb_cache.load(domain, len(doc_ids), corpus_fp)

                if cached_data is not None:
                    corpus_embs, cached_doc_ids = cached_data
                    doc_ids = cached_doc_ids
                else:
                    logger.info(f"Codifica densa corpus [{domain}] ({len(doc_texts)} passaggi)...")
                    all_embs = []
                    eval_bs = int(eval_cfg.get("eval_batch_size", 64))
                    with torch.no_grad():
                        for i in range(0, len(doc_texts), eval_bs):
                            batch = doc_texts[i : i + eval_bs]
                            tok = tokenizer(
                                batch,
                                padding=True,
                                truncation=True,
                                max_length=data_cfg.get("max_doc_length", 256),
                                return_tensors="pt",
                            ).to(device)
                            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                                embs = bi_encoder.encode(tok["input_ids"], tok["attention_mask"])
                            all_embs.append(embs.cpu())
                    corpus_embs = torch.cat(all_embs, dim=0)
                    emb_cache.save(domain, corpus_embs, doc_ids, corpus_fp)

                all_q_embs = []
                eval_bs = int(eval_cfg.get("eval_batch_size", 64))
                with torch.no_grad():
                    for i in range(0, len(queries), eval_bs):
                        batch = queries[i : i + eval_bs]
                        tok = tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=data_cfg.get("max_query_length", 256),
                            return_tensors="pt",
                        ).to(device)
                        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                            embs = bi_encoder.encode(tok["input_ids"], tok["attention_mask"])
                        all_q_embs.append(embs.cpu())
                query_embs = torch.cat(all_q_embs, dim=0).float()
                corpus_embs = corpus_embs.float()

                for q_idx, sample in enumerate(samples):
                    scores = torch.mv(corpus_embs, query_embs[q_idx]).float().numpy()
                    top_idx = np.argsort(-scores, kind="stable")[:top_candidates]
                    dense_run[sample.topic_id] = [
                        (doc_ids[idx], float(scores[idx]))
                        for idx in top_idx
                    ]

            # -------------------------
            # 2. Retrieval BM25
            # -------------------------
            bm25_run = {}
            if use_sparse:
                with OfficialBM25(
                    corpus,
                    k1=sparse_cfg.get("k1", 0.9),
                    b=sparse_cfg.get("b", 0.4),
                ) as bm25:
                    for sample in samples:
                        if ablation_mode == "A":
                            # Ablation A: Solo turno corrente
                            raw_bm25_query = sample.query.strip()
                        else:
                            # History completa pulita + Domanda
                            raw_bm25_query = f"{sample.history} {sample.query}".strip()

                        bm25_run[sample.topic_id] = bm25.search_one(
                            query=raw_bm25_query,
                            top_k=top_candidates,
                        )

            if use_retrieval_cache:
                retrieval_cache.save(
                    domain=domain,
                    split=cache_split,
                    top_candidates=top_candidates,
                    ablation_mode=ablation_mode,
                    dense_run=dense_run,
                    bm25_run=bm25_run,
                    corpus_size=len(doc_ids),
                    query_count=len(samples),
                )

        # 3. Reciprocal Rank Fusion sweep
        #
        # Dense e BM25 sono già stati calcolati.
        # Qui cambiamo soltanto k, quindi il costo aggiuntivo è molto basso.

        rrf_runs_by_k: Dict[int, Dict[str, List[Tuple[str, float]]]] = {}
        rrf_metrics_by_k: Dict[int, Dict[str, float]] = {}

        for current_k in sorted(rrf_sweep_values):
            current_rrf_run = reciprocal_rank_fusion(
                [dense_run, bm25_run],
                k=current_k,
                top_n=top_candidates,
            )

            rrf_runs_by_k[current_k] = current_rrf_run

            if qrels and use_fusion:
                rrf_metrics_by_k[current_k] = compute_detailed_metrics(qrels, current_rrf_run, cutoff_k)

        # RRF utilizzato realmente dalla pipeline
        rrf_run = rrf_runs_by_k[rrf_k_config]

        # Salviamo i risultati dello sweep per questo dominio
        all_reports["RRF_sweep"][domain] = {
            str(k): metrics
            for k, metrics in rrf_metrics_by_k.items()
        }

        # 4. Selezione Candidati e Re-ranking Neurale
        if ablation_mode in ["A", "B"]:
            candidate_pool = bm25_run
        elif ablation_mode in ["C", "D"]:
            candidate_pool = dense_run
        else:
            candidate_pool = rrf_run

        final_domain_run: Dict[str, List[Tuple[str, float]]] = {}
        if cross_encoder is not None and cross_tok is not None and not (ablation_mode in ["A", "B", "C", "D", "E"]):
            ce_batch_size = int(cross_cfg.get("batch_size", 16))
            ce_max_len = int(cross_cfg.get("max_seq_length", 384))

            for sample in tqdm(samples, desc=f"Cross-Encoder [{domain}]", leave=False):
                ranking = candidate_pool.get(sample.topic_id, [])
                cands = ranking[:rerank_depth]
                if not cands:
                    continue
                q_list = [sample.contextual_query] * len(cands)
                d_list = [corpus.get(d_id, "") for d_id, _ in cands]

                all_scores = []
                with torch.no_grad():
                    for p_i in range(0, len(q_list), ce_batch_size):
                        enc = cross_tok(
                            q_list[p_i : p_i + ce_batch_size],
                            d_list[p_i : p_i + ce_batch_size],
                            padding=True,
                            truncation=True,
                            max_length=ce_max_len,
                            return_tensors="pt",
                        ).to(device)
                        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                            out = cross_encoder(enc["input_ids"], enc["attention_mask"])
                            scores = out["logits"].squeeze(-1).cpu().tolist()
                        all_scores.extend(scores if isinstance(scores, list) else [scores])

                reranked = sorted([(cands[idx][0], float(all_scores[idx])) for idx in range(len(cands))], key=lambda x: x[1], reverse=True)
                final_domain_run[sample.topic_id] = reranked + ranking[rerank_depth:]
        else:
            for t_id, cands in candidate_pool.items():
                final_domain_run[t_id] = cands

        final_trec_run.update(final_domain_run)
        if not qrels:
            continue

        stages = {"Final": final_domain_run}
        if use_dense:
            stages["Dense"] = dense_run
        if use_sparse:
            stages["BM25"] = bm25_run
        if use_fusion:
            stages["RRF"] = rrf_run
        for stage, run in stages.items():
            scores = compute_detailed_metrics(qrels, run, cutoff_k)
            all_reports[stage][domain] = scores
            logger.info("[%s] %s nDCG@%d: %.4f", domain, stage, cutoff_k, scores[f"nDCG@{cutoff_k}"])

    # ---------------------------------------------------------
    # Report Tabellare e Macro-Average
    # ---------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info(f"{'STADIO RETRIEVAL':<16} | {'nDCG@10':<10} | {'Recall@10':<10} | {'Recall@50':<10} | {'Recall@100':<10} | {'MRR':<10}")
    logger.info("-" * 80)

    summary_json: Dict[str, Any] = {
        "split": split,
        "domains": domains,
        "ablation": ablation_mode,
        "query_mode": query_mode if use_dense else None,
        "checkpoint": str(loaded_checkpoint) if loaded_checkpoint else None,
        "model": model_name if use_dense else None,
        "rrf_k": rrf_k_config,
        "config": config,
        "data_dir": str(base_data_dir),
        "per_domain": all_reports,
        "macro_average": {},
        "rrf_sweep_macro_average": {},
    }
    for stage in ["BM25", "Dense", "RRF", "Final"]:
        scores = all_reports[stage]
        if not scores:
            continue
        macro_ndcg = float(np.mean([m[f"nDCG@{cutoff_k}"] for m in scores.values()]))
        macro_r10 = float(np.mean([m["Recall@10"] for m in scores.values()]))
        macro_r50 = float(np.mean([m["Recall@50"] for m in scores.values()]))
        macro_r100 = float(np.mean([m["Recall@100"] for m in scores.values()]))
        macro_mrr = float(np.mean([m["MRR"] for m in scores.values()]))

        summary_json["macro_average"][stage] = {
            f"nDCG@{cutoff_k}": macro_ndcg,
            "Recall@10": macro_r10,
            "Recall@50": macro_r50,
            "Recall@100": macro_r100,
            "MRR": macro_mrr,
        }
        logger.info(f"{stage:<16} | {macro_ndcg:<10.4f} | {macro_r10:<10.4f} | {macro_r50:<10.4f} | {macro_r100:<10.4f} | {macro_mrr:<10.4f}")

    # ---------------------------------------------------------
    # RRF sweep macro-average
    # ---------------------------------------------------------
    logger.info("")
    logger.info(
        f"{'RRF SWEEP':<16} | "
        f"{'nDCG@10':<10} | "
        f"{'Recall@10':<10} | "
        f"{'Recall@50':<10} | "
        f"{'Recall@100':<10} | "
        f"{'MRR':<10}"
    )

    logger.info("-" * 80)

    for current_k in sorted(rrf_sweep_values):
        domain_metrics = []

        for domain in all_reports["RRF_sweep"]:
            metrics = all_reports["RRF_sweep"][domain].get(
                str(current_k)
            )

            if metrics:
                domain_metrics.append(metrics)

        if not domain_metrics:
            continue

        macro_metrics = {
            f"nDCG@{cutoff_k}": float(
                np.mean([
                    m[f"nDCG@{cutoff_k}"]
                    for m in domain_metrics
                ])
            ),
            "Recall@10": float(
                np.mean([
                    m["Recall@10"]
                    for m in domain_metrics
                ])
            ),
            "Recall@50": float(
                np.mean([
                    m["Recall@50"]
                    for m in domain_metrics
                ])
            ),
            "Recall@100": float(
                np.mean([
                    m["Recall@100"]
                    for m in domain_metrics
                ])
            ),
            "MRR": float(
                np.mean([
                    m["MRR"]
                    for m in domain_metrics
                ])
            ),
        }

        summary_json["rrf_sweep_macro_average"][str(current_k)] = macro_metrics

        logger.info(
            f"k={current_k:<13} | "
            f"{macro_metrics[f'nDCG@{cutoff_k}']:<10.4f} | "
            f"{macro_metrics['Recall@10']:<10.4f} | "
            f"{macro_metrics['Recall@50']:<10.4f} | "
            f"{macro_metrics['Recall@100']:<10.4f} | "
            f"{macro_metrics['MRR']:<10.4f}"
        )

    logger.info("=" * 80)

    # Salvataggio Output Run TREC e JSON
    out_dir = output_dir or Path(paths_cfg.get("output_dir", "outputs/subtrack_2a"))
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{ablation_mode or 'main'}_{query_mode if use_dense else 'lexical'}"
    save_json(summary_json, out_dir / f"evaluation_report_{split}{suffix}.json")

    if split == "test":
        missing = expected_topics - {topic for topic, docs in final_trec_run.items() if ranking_scores(docs)}
        if missing:
            raise ValueError(f"Turni senza ranking: {sorted(missing)[:10]}")
    (out_dir / f"topic_ids_{split}{suffix}.txt").write_text(
        "\n".join(sorted(expected_topics)) + "\n", encoding="utf-8"
    )
    trec_file = out_dir / f"{split}_run{suffix}.trec"
    write_trec_run(final_trec_run, trec_file, run_tag=gen_cfg.get("run_tag", "reteco_2a"), max_k=cutoff_k)
    is_valid, errs = validate_trec_file(trec_file, max_rank=cutoff_k)
    if is_valid:
        logger.info(f"✓ File TREC validato e salvato in: {trec_file}")
    else:
        raise ValueError(f"Errori nel file TREC: {errs[:3]}")


# ==============================================================================
# 5. Entrypoint CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference e Valutazione SemEval Sub-track 2a")
    parser.add_argument("--config", type=str, default=str(Path(__file__).resolve().parent / "config/config.yaml"))
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "train", "test"])
    parser.add_argument("--ablation", type=str, default=None, choices=["A", "B", "C", "D", "E", "F", "G"],
                        help="A=BM25-curr, B=BM25-hist, C=Dense-pre, D=Dense-tuned, E=RRF, F=RRF+pre-rerank, G=RRF+tuned-rerank")
    parser.add_argument(
        "--no-retrieval-cache",
        action="store_true",
        help="Ignora la cache Dense/BM25 e ricalcola il retrieval.",
    )
    parser.add_argument("--query-mode", choices=["contextual", "reasoned"])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.split == "test" and args.data_dir is None:
        parser.error("Per il test specifica --data-dir.")

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        parser.error(f"Config non trovato: {cfg_path}")
    config = load_config(cfg_path)

    # Inizializzazione handler su file con timestamp dopo aver letto il config
    paths_cfg = config.get("paths", {})
    gen_cfg = config.get("general", {})
    log_dir = Path(paths_cfg.get("log_dir", "outputs/subtrack_2a/logs"))
    run_tag = gen_cfg.get("run_tag", "eval")
    log_level = gen_cfg.get("logging_level", "INFO")

    global logger
    try:
        logger = setup_logger(log_dir=log_dir, run_tag=f"eval_{run_tag}", log_level=log_level)
    except Exception:
        pass

    run_evaluation(
        config,
        split=args.split,
        ablation_mode=args.ablation,
        use_retrieval_cache=not args.no_retrieval_cache,
        query_mode=args.query_mode,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()