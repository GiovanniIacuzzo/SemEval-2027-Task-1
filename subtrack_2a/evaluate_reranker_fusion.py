#!/usr/bin/env python3
"""Post-hoc rank-fusion evaluation from cached Qwen3-Reranker scores.

No neural model is loaded and no reranker inference is run. Rebuilds the fast
BM25 run, reads per-query cached reranker scores, and compares BM25, pure
reranking, and Reciprocal Rank Fusion (RRF) variants on the same internal split.
"""
from __future__ import annotations
import argparse, json, math, random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import yaml

# Running this script from the repository root or from its own path is supported.
import sys
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from retrieval.retrieval import OfficialBM25, _corpus_fingerprint

DOMAINS = ["biology", "drones", "earth_science", "economics", "hardware", "law",
           "medicalsciences", "politics", "psychology", "robotics", "sustainable_living"]
MODEL = "Qwen/Qwen3-Reranker-0.6B"
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return value
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_corpus(path: Path) -> Dict[str, str]:
    corpus: Dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = item.get("doc_id", item.get("id"))
            text = str(item.get("content", item.get("text", "")) or "").strip()
            if doc_id is None:
                raise ValueError(f"Document without ID at {path}:{line_no}")
            if text:
                corpus[str(doc_id)] = text
    if not corpus:
        raise ValueError(f"Empty corpus: {path}")
    return corpus


def load_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) < 4:
                raise ValueError(f"Malformed qrels at {path}:{line_no}: {line.rstrip()}")
            topic, _, doc_id, rel = parts[:4]
            qrels.setdefault(topic, {})[doc_id] = int(rel)
    return qrels


def format_query(query: str, history: str, max_len: int, strategy: str) -> str:
    query, history = (query or "").strip(), (history or "").strip()
    if history.lower() == "no previous conversation.":
        history = ""
    if strategy == "query_only" or not history:
        return " ".join(query.split()[:max_len]).strip()
    header = "\n\nConversation History:\n"
    fixed = len(f"{query}{header}".split())
    if fixed > max_len:
        query = " ".join(query.split()[:max(1, max_len - len(header.split()))])
        fixed = len(f"{query}{header}".split())
    remaining = max(0, max_len - fixed)
    kept_history = " ".join(history.split()[-remaining:]) if remaining else ""
    return f"{query}{header}{kept_history}".strip()


def load_samples(path: Path, domain: str, max_query_len: int, strategy: str) -> List[Dict[str, Any]]:
    samples = []
    for conv in read_json_or_jsonl(path):
        conv_id = str(conv["id"])
        for turn in conv.get("turns", []):
            query = str(turn.get("query", "") or "").strip()
            history = str(turn.get("conversation_history", "") or "").strip()
            samples.append({
                "domain": domain,
                "conversation_id": conv_id,
                "turn_id": int(turn["turn_id"]),
                "topic_id": f"{conv_id}_turn_{int(turn['turn_id'])}",
                "query": query,
                "history": history,
                "contextual_query": format_query(query, history, max_query_len, strategy),
                "gold_doc_ids": [str(x) for x in (turn.get("gold_doc_ids") or turn.get("supporting_doc_ids") or [])],
            })
    return samples


def split_samples(samples: List[Dict[str, Any]], ratio: float, seed: int):
    conv_ids = sorted({str(s["conversation_id"]) for s in samples})
    if len(conv_ids) < 2:
        raise ValueError("At least two conversations are required for validation split")
    random.Random(seed).shuffle(conv_ids)
    n_val = min(max(1, int(round(len(conv_ids) * ratio))), len(conv_ids) - 1)
    val_ids = set(conv_ids[:n_val])
    return [s for s in samples if str(s["conversation_id"]) not in val_ids], [s for s in samples if str(s["conversation_id"]) in val_ids]


def valid_qrels_for(samples, qrels, corpus_ids):
    valid: Dict[str, Dict[str, int]] = {}
    fallback = 0
    for s in samples:
        topic = str(s["topic_id"])
        judged = {str(d): int(r) for d, r in qrels.get(topic, {}).items() if str(d) in corpus_ids and int(r) > 0}
        if not judged:
            judged = {str(d): 1 for d in s["gold_doc_ids"] if str(d) in corpus_ids}
            if judged:
                fallback += 1
        if judged:
            valid[topic] = judged
    return valid, fallback


def metrics(qrels: Dict[str, Dict[str, int]], run: Dict[str, List[Tuple[str, float]]], corpus_size: int, fallback: int):
    ndcgs, mrrs = [], []
    recalls: Dict[int, List[float]] = {10: [], 50: [], 100: [], 1000: []}
    for topic, judged in qrels.items():
        ranked = run.get(topic, [])
        ids = [str(d) for d, _ in ranked]
        rels = {str(d): int(r) for d, r in judged.items()}
        dcg = sum((2 ** rels.get(str(d), 0) - 1) / math.log2(i + 2)
                  for i, (d, _) in enumerate(ranked[:10]))
        ideal = sorted(rels.values(), reverse=True)[:10]
        idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        gold = set(rels)
        for k in recalls:
            recalls[k].append(len(set(ids[:k]) & gold) / max(1, len(gold)))
        mrrs.append(next((1.0 / (i + 1) for i, d in enumerate(ids) if d in gold), 0.0))
    out = {
        "nDCG@10": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "MRR": sum(mrrs) / len(mrrs) if mrrs else 0.0,
        "num_topics": len(qrels), "num_documents": corpus_size,
        "qrels_fallback_topics": fallback,
    }
    for k, vals in recalls.items():
        out[f"Recall@{k}"] = sum(vals) / len(vals) if vals else 0.0
    return out


def mean_metrics(per_domain: Dict[str, Dict[str, float]]):
    names = ["nDCG@10", "MRR", "Recall@10", "Recall@50", "Recall@100", "Recall@1000"]
    return {key: sum(float(v[key]) for v in per_domain.values()) / len(per_domain)
            for key in names if per_domain and key in next(iter(per_domain.values()))}


def main():
    p = argparse.ArgumentParser(description="Fast post-hoc BM25/reranker rank-fusion sweep using cached reranker scores; no model inference.")
    p.add_argument("--config", default="subtrack_2a/config/config.yaml")
    p.add_argument("--domains", default="biology,medicalsciences")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--bm25-depth", type=int, default=1000)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--alphas", default="0.25,0.5,1,2,4", help="reranker contribution in RRF; 0.0 equals BM25-only ordering")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    if args.top_k < 10 or args.bm25_depth < args.top_k:
        p.error("Require --top-k >= 10 and --bm25-depth >= --top-k")
    config_path = Path(args.config)
    if not config_path.is_file() and config_path.parts and config_path.parts[0] == "subtrack_2a":
        alt = Path(*config_path.parts[1:])
        if alt.is_file(): config_path = alt
    if not config_path.is_file(): p.error(f"Config not found: {args.config}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    gen, paths = cfg.get("general", {}), cfg.get("paths", {})
    data_cfg, domains_cfg, sparse_cfg = cfg.get("data", {}), cfg.get("domains", {}), cfg.get("sparse", {})
    if args.domains.strip().lower() == "all": domains = DOMAINS[:]
    else: domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    bad = [d for d in domains if d not in DOMAINS]
    if not domains or bad: p.error(f"Invalid domains: {bad or domains}")
    data_mode = str(paths.get("data_mode", "full")).lower()
    data_dir = Path(paths.get("full_data_dir" if data_mode == "full" else "sample_data_dir", ""))
    output_dir = Path(paths.get("output_dir", "outputs/subtrack_2a"))
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "reranker_score_cache" / f"{MODEL.replace('/', '_')}_top{args.top_k}_len768_contextual"
    output_path = Path(args.output) if args.output else output_dir / f"reranker_fusion_top{args.top_k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    seed = int(gen.get("seed", 42)); ratio = float(data_cfg.get("val_ratio", 0.15))
    max_query_len = int(data_cfg.get("max_query_length", 512)); strategy = str(data_cfg.get("query_strategy", "history"))
    k1, b = float(sparse_cfg.get("k1", 0.9)), float(sparse_cfg.get("b", 0.4))
    index_root = Path(paths.get("hard_negatives_cache_dir", "data/cache/hard_negatives")) / "lucene_indexes"
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    systems = ["bm25", "reranked"] + [f"rrf_alpha_{a:g}" for a in alphas if a > 0]
    per_sys_domain = {s: {} for s in systems}
    per_domain_details = {}
    print(f"Fusion evaluation: NO model load / NO inference. cache={cache_dir}")
    print(f"Domains={domains}; top_k={args.top_k}; bm25_depth={args.bm25_depth}; RRF-k={args.rrf_k}; alphas={alphas}")

    for domain in domains:
        ddir = data_dir / domain
        corpus = load_corpus(ddir / "documents.jsonl")
        bench = ddir / "benchmark_train.json"
        if not bench.is_file(): bench = ddir / "benchmark_train.jsonl"
        samples = load_samples(bench, domain, max_query_len, strategy)
        qrels_all = load_qrels(ddir / "qrels_train.txt")
        _, val_samples = split_samples(samples, ratio, seed)
        qrels, fallback = valid_qrels_for(val_samples, qrels_all, set(corpus))
        fingerprint = _corpus_fingerprint(corpus)
        bm25_cache_dir = index_root / fingerprint
        bm25_run = {}
        with OfficialBM25(corpus=corpus, k1=k1, b=b, index_dir=bm25_cache_dir) as bm25:
            for s in val_samples:
                bm25_run[s["topic_id"]] = bm25.search_one(s["contextual_query"], top_k=min(args.bm25_depth, len(corpus)))

        cache_path = cache_dir / f"{domain}.json"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Reranker cache missing for {domain}: {cache_path}. Run the aligned reranker on this domain first.")
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        meta = cache.get("meta", {})
        if meta.get("model") != MODEL or meta.get("top_k") != args.top_k or meta.get("query_mode") != "contextual":
            raise ValueError(f"Cache metadata mismatch for {domain}: {meta}")
        topic_cache = cache.get("topics", {})
        run_rerank, candidate_bm25_ranks, candidate_rerank_ranks = {}, {}, {}
        missing = 0
        for s in val_samples:
            topic = s["topic_id"]
            original = bm25_run.get(topic, [])
            head = original[:args.top_k]
            ids = [str(d) for d, _ in head]
            entry = topic_cache.get(topic, {})
            if entry.get("candidate_ids") != ids or len(entry.get("scores", [])) != len(ids):
                missing += 1
                continue
            scores = [float(x) for x in entry["scores"]]
            rerank_order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
            candidate_bm25_ranks[topic] = {doc_id: rank + 1 for rank, doc_id in enumerate(ids)}
            candidate_rerank_ranks[topic] = {ids[idx]: rank + 1 for rank, idx in enumerate(rerank_order)}
            tail = [(str(d), float(sc)) for d, sc in original[len(ids):]]
            run_rerank[topic] = [(ids[i], scores[i]) for i in rerank_order] + tail
        if missing:
            raise ValueError(f"Cache is missing or mismatched for {missing}/{len(val_samples)} queries in {domain}; run the reranker with identical settings first.")

        sys_runs = {"bm25": bm25_run, "reranked": run_rerank}
        for alpha in alphas:
            if alpha <= 0: continue
            fused_run = {}
            for s in val_samples:
                topic = s["topic_id"]
                original = bm25_run[topic]
                head = original[:args.top_k]
                tail = [(str(d), float(sc)) for d, sc in original[len(head):]]
                bm_rank = candidate_bm25_ranks[topic]
                rr_rank = candidate_rerank_ranks[topic]
                fusion_scores = {doc_id: (1.0 / (args.rrf_k + bm_rank[doc_id])) + alpha * (1.0 / (args.rrf_k + rr_rank[doc_id])) for doc_id in bm_rank}
                fused_ids = sorted(bm_rank, key=lambda doc_id: (-fusion_scores[doc_id], bm_rank[doc_id]))
                fused_run[topic] = [(doc_id, fusion_scores[doc_id]) for doc_id in fused_ids] + tail
            sys_runs[f"rrf_alpha_{alpha:g}"] = fused_run

        per_domain_details[domain] = {"num_queries": len(qrels), "qrels_fallback_topics": fallback}
        for system, run in sys_runs.items():
            per_sys_domain.setdefault(system, {})[domain] = metrics(qrels, run, len(corpus), fallback)
        print(f"\n[{domain}] nDCG@10", end="")
        for system in systems:
            m = per_sys_domain[system][domain]
            print(f" | {system}={m['nDCG@10']:.5f}", end="")
        print()
        print(f"[{domain}] deltas vs BM25", end="")
        base = per_sys_domain["bm25"][domain]["nDCG@10"]
        for system in systems:
            if system == "bm25": continue
            print(f" | {system}={per_sys_domain[system][domain]['nDCG@10']-base:+.5f}", end="")
        print()

    macro = {system: mean_metrics(domains_data) for system, domains_data in per_sys_domain.items()}
    macro_delta = {s: {metric: macro[s][metric] - macro["bm25"][metric] for metric in macro["bm25"]} for s in macro if s != "bm25"}
    report = {
        "method": "post-hoc RRF fusion of BM25 and cached Qwen3-Reranker ranks; no reranker inference",
        "domains": per_domain_details,
        "per_system_per_domain": per_sys_domain,
        "macro_average": macro,
        "macro_delta_vs_bm25": macro_delta,
        "alphas": alphas, "rrf_k": args.rrf_k, "top_k": args.top_k, "bm25_depth": args.bm25_depth,
        "cache_dir": str(cache_dir), "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== MACRO nDCG@10 ===")
    for system, m in macro.items(): print(f"{system:22s} {m['nDCG@10']:.5f}" + (f" ({macro_delta[system]['nDCG@10']:+.5f})" if system in macro_delta else ""))
    print(f"\nFINAL_REPORT={output_path}")

if __name__ == "__main__":
    main()
