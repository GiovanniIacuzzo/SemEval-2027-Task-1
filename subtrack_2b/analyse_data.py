#!/usr/bin/env python3
"""
subtrack_2b/analyse_data.py

Analisi diagnostica strutturata dei dati RETECO Sub-track 2b.

Analizza:
    - conversazioni e turni
    - profondita' T1/T2/T3/T4/T5+
    - lunghezze query/history/answer
    - numero di gold passages per turno
    - dimensione testuale dell'evidenza gold
    - stima della dimensione del prompt
    - integrita' degli ID dei gold passages rispetto al corpus

Il tokenizer Hugging Face e' opzionale: con --tokenizer vengono aggiunte
statistiche token-level esatte per il modello scelto.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


TRACK2_DOMAINS = [
    "biology", "drones", "earth_science", "economics", "hardware",
    "law", "medicalsciences", "politics", "psychology", "robotics",
    "sustainable_living",
]


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"Formato JSON inatteso in {path}")
        return data
    records = []
    for line_no, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON non valido in {path}, riga {line_no}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"Record non-object in {path}, riga {line_no}")
        records.append(item)
    return records


def resolve_benchmark(domain_dir: Path, split: str) -> Path:
    for p in (domain_dir / f"benchmark_{split}.json", domain_dir / f"benchmark_{split}.jsonl"):
        if p.exists():
            return p
    raise FileNotFoundError(f"benchmark_{split} non trovato in {domain_dir}")


def load_corpus(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus non trovato: {path}")
    corpus: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            doc_id = item.get("doc_id", item.get("id"))
            text = item.get("content", item.get("text", ""))
            if doc_id is None:
                raise ValueError(f"Documento senza ID in {path}, riga {line_no}")
            corpus[str(doc_id)] = str(text or "").strip()
    if not corpus:
        raise ValueError(f"Corpus vuoto: {path}")
    return corpus


def nwords(text: str) -> int:
    return len(text.split())


def mean(xs: Sequence[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else 0.0


def pct(xs: Sequence[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    pos = (len(ys) - 1) * p / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(ys[lo])
    w = pos - lo
    return float(ys[lo] * (1 - w) + ys[hi] * w)


def depth_bucket(turn_id: int) -> str:
    return "T1" if turn_id <= 1 else "T2" if turn_id == 2 else "T3" if turn_id == 3 else "T4" if turn_id == 4 else "T5+"


def analyse_domain(domain: str, domain_dir: Path, split: str, tokenizer: Optional[Any] = None) -> Dict[str, Any]:
    benchmark_path = resolve_benchmark(domain_dir, split)
    corpus = load_corpus(domain_dir / "documents.jsonl")
    conversations = load_json_or_jsonl(benchmark_path)

    depth = Counter()
    records: List[Dict[str, Any]] = []
    query_w: List[int] = []
    history_w: List[int] = []
    answer_w: List[int] = []
    gold_n: List[int] = []
    evidence_w: List[int] = []
    prompt_w: List[int] = []
    query_t: List[int] = []
    history_t: List[int] = []
    answer_t: List[int] = []
    evidence_t: List[int] = []
    prompt_t: List[int] = []

    invalid_gold = []
    empty_query = []
    empty_answer = []
    history_present = 0
    multi_gold = 0
    more_than_4_gold = 0

    for conv in conversations:
        cid = str(conv.get("id", "")).strip()
        turns = conv.get("turns", []) or []
        if not isinstance(turns, list):
            raise ValueError(f"turns non e' una lista in {cid}")

        for turn in turns:
            tid = int(turn.get("turn_id", 0))
            topic_id = f"{cid}_turn_{tid}"
            query = str(turn.get("query", "") or "").strip()
            history = str(turn.get("conversation_history", "") or "").strip()
            if history.lower() == "no previous conversation.":
                history = ""
            answer = str(turn.get("answer", "") or "").strip()
            gold_ids = [str(x) for x in (turn.get("gold_doc_ids", []) or [])]

            q = nwords(query); h = nwords(history); a = nwords(answer)
            ew = sum(nwords(corpus.get(doc_id, "")) for doc_id in gold_ids)
            pw = q + h + ew + 32 + 12 * len(gold_ids)

            query_w.append(q); history_w.append(h); answer_w.append(a)
            gold_n.append(len(gold_ids)); evidence_w.append(ew); prompt_w.append(pw)
            depth[depth_bucket(tid)] += 1

            if history:
                history_present += 1
            if len(gold_ids) > 1:
                multi_gold += 1
            if len(gold_ids) > 4:
                more_than_4_gold += 1
            if not query:
                empty_query.append(topic_id)
            if not answer:
                empty_answer.append(topic_id)

            missing = [d for d in gold_ids if d not in corpus]
            if missing:
                invalid_gold.append({"topic_id": topic_id, "missing_doc_ids": missing})

            rec = {
                "domain": domain, "conversation_id": cid, "turn_id": tid,
                "topic_id": topic_id, "query": query, "history": history,
                "answer": answer, "gold_doc_ids": gold_ids,
                "gold_doc_count": len(gold_ids), "query_words": q,
                "history_words": h, "answer_words": a, "evidence_words": ew,
                "rough_prompt_words": pw,
                "subquestion_reasoning_present": bool(turn.get("subquestion_reasoning")),
            }

            if tokenizer is not None:
                qt = len(tokenizer.encode(query, add_special_tokens=False))
                ht = len(tokenizer.encode(history, add_special_tokens=False))
                at = len(tokenizer.encode(answer, add_special_tokens=False))
                et = sum(len(tokenizer.encode(corpus.get(d, ""), add_special_tokens=False)) for d in gold_ids)
                pt = qt + ht + et + 64 + 12 * len(gold_ids)
                rec.update(query_tokens=qt, history_tokens=ht, answer_tokens=at, evidence_tokens=et, rough_prompt_tokens=pt)
                query_t.append(qt); history_t.append(ht); answer_t.append(at); evidence_t.append(et); prompt_t.append(pt)

            if len(records) < 5:
                records.append(rec)

    turns = sum(depth.values())
    out = {
        "domain": domain, "split": split, "benchmark_file": str(benchmark_path),
        "corpus_documents": len(corpus), "conversations": len(conversations), "turns": turns,
        "turns_with_history": history_present,
        "pct_turns_with_history": 100.0 * history_present / max(1, turns),
        "turns_with_multiple_gold": multi_gold,
        "pct_turns_with_multiple_gold": 100.0 * multi_gold / max(1, turns),
        "turns_with_more_than_4_gold": more_than_4_gold,
        "invalid_gold_id_count": len(invalid_gold),
        "invalid_gold_ids": invalid_gold,
        "empty_query_count": len(empty_query), "empty_answer_count": len(empty_answer),
        "depth_distribution": dict(depth),
        "query_words": {"mean": mean(query_w), "median": median(query_w), "p95": pct(query_w, 95), "max": max(query_w) if query_w else 0},
        "history_words": {"mean": mean(history_w), "median": median(history_w), "p95": pct(history_w, 95), "max": max(history_w) if history_w else 0},
        "answer_words": {"mean": mean(answer_w), "median": median(answer_w), "p95": pct(answer_w, 95), "max": max(answer_w) if answer_w else 0},
        "gold_passages_per_turn": {"mean": mean(gold_n), "median": median(gold_n), "p95": pct(gold_n, 95), "max": max(gold_n) if gold_n else 0},
        "evidence_words_per_turn": {"mean": mean(evidence_w), "median": median(evidence_w), "p95": pct(evidence_w, 95), "max": max(evidence_w) if evidence_w else 0},
        "rough_prompt_words": {"mean": mean(prompt_w), "median": median(prompt_w), "p95": pct(prompt_w, 95), "max": max(prompt_w) if prompt_w else 0},
        "samples": records,
        "warnings": [],
    }
    if invalid_gold:
        out["warnings"].append("Gold passage ID non presenti nel corpus.")
    if empty_query:
        out["warnings"].append("Turni con query vuota.")
    if empty_answer:
        out["warnings"].append("Turni con answer vuota.")
    if tokenizer is not None:
        out["query_tokens"] = {"mean": mean(query_t), "median": median(query_t), "p95": pct(query_t, 95), "max": max(query_t) if query_t else 0}
        out["history_tokens"] = {"mean": mean(history_t), "median": median(history_t), "p95": pct(history_t, 95), "max": max(history_t) if history_t else 0}
        out["answer_tokens"] = {"mean": mean(answer_t), "median": median(answer_t), "p95": pct(answer_t, 95), "max": max(answer_t) if answer_t else 0}
        out["evidence_tokens"] = {"mean": mean(evidence_t), "median": median(evidence_t), "p95": pct(evidence_t, 95), "max": max(evidence_t) if evidence_t else 0}
        out["rough_prompt_tokens"] = {"mean": mean(prompt_t), "median": median(prompt_t), "p95": pct(prompt_t, 95), "max": max(prompt_t) if prompt_t else 0}
    return out


def macro_average(reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not reports:
        return {}
    def m(section: str, key: str) -> float:
        return mean([float(r[section][key]) for r in reports if section in r])
    out = {
        "domains": len(reports),
        "macro_query_words_mean": m("query_words", "mean"),
        "macro_history_words_mean": m("history_words", "mean"),
        "macro_answer_words_mean": m("answer_words", "mean"),
        "macro_gold_passages_mean": m("gold_passages_per_turn", "mean"),
        "macro_evidence_words_mean": m("evidence_words_per_turn", "mean"),
        "macro_rough_prompt_words_mean": m("rough_prompt_words", "mean"),
    }
    if any("rough_prompt_tokens" in r for r in reports):
        out["macro_rough_prompt_tokens_mean"] = m("rough_prompt_tokens", "mean")
    return out


def write_csv(reports: Sequence[Dict[str, Any]], path: Path) -> None:
    rows = []
    for r in reports:
        rows.append({
            "domain": r["domain"], "split": r["split"], "corpus_documents": r["corpus_documents"],
            "conversations": r["conversations"], "turns": r["turns"],
            "pct_turns_with_history": r["pct_turns_with_history"],
            "query_words_mean": r["query_words"]["mean"], "query_words_p95": r["query_words"]["p95"],
            "history_words_mean": r["history_words"]["mean"], "history_words_p95": r["history_words"]["p95"],
            "answer_words_mean": r["answer_words"]["mean"], "answer_words_p95": r["answer_words"]["p95"],
            "gold_passages_mean": r["gold_passages_per_turn"]["mean"], "gold_passages_p95": r["gold_passages_per_turn"]["p95"],
            "evidence_words_mean": r["evidence_words_per_turn"]["mean"], "evidence_words_p95": r["evidence_words_per_turn"]["p95"],
            "rough_prompt_words_mean": r["rough_prompt_words"]["mean"], "rough_prompt_words_p95": r["rough_prompt_words"]["p95"],
            "invalid_gold_id_count": r["invalid_gold_id_count"],
        })
        if "rough_prompt_tokens" in r:
            rows[-1].update(rough_prompt_tokens_mean=r["rough_prompt_tokens"]["mean"], rough_prompt_tokens_p95=r["rough_prompt_tokens"]["p95"])
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analisi diagnostica RETECO Sub-track 2b")
    ap.add_argument("--data_dir", default="data/reteco_data/track2_recor")
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--domains", nargs="+", default=TRACK2_DOMAINS)
    ap.add_argument("--output_dir", default="outputs/subtrack_2b")
    ap.add_argument("--tokenizer", default=None, help="Tokenizer HF opzionale")
    args = ap.parse_args()

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        print(f"[INFO] Caricamento tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    data_dir = Path(args.data_dir); output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("       RETECO SUB-TRACK 2b: ANALISI DIAGNOSTICA STRUTTURATA")
    print("=" * 90)
    print(f"Split      : {args.split}")
    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Tokenizer  : {args.tokenizer or 'word-level only'}")
    print()

    reports = []
    for domain in args.domains:
        print("-" * 90)
        print(f"DOMINIO: {domain.upper()}")
        try:
            r = analyse_domain(domain, data_dir / domain, args.split, tokenizer)
        except Exception as exc:
            print(f"[WARNING] Salto {domain}: {exc}")
            continue
        reports.append(r)
        print(f"Corpus={r['corpus_documents']} | Conversazioni={r['conversations']} | Turni={r['turns']}")
        print(f"History presente={r['pct_turns_with_history']:.2f}% | Gold/turno={r['gold_passages_per_turn']['mean']:.2f} | Answer words={r['answer_words']['mean']:.2f}")
        print(f"Prompt stimato words mean/p95={r['rough_prompt_words']['mean']:.1f}/{r['rough_prompt_words']['p95']:.1f}")
        if "rough_prompt_tokens" in r:
            print(f"Prompt stimato tokens mean/p95={r['rough_prompt_tokens']['mean']:.1f}/{r['rough_prompt_tokens']['p95']:.1f}")
        print(f"Profondita': {r['depth_distribution']}")
        for warning in r["warnings"]:
            print(f"[WARNING] {warning}")

    summary = {"split": args.split, "data_dir": str(data_dir), "domains_requested": args.domains, "domains_analysed": [r["domain"] for r in reports], "macro_average": macro_average(reports), "per_domain": reports}
    json_path = output_dir / f"data_analysis_{args.split}.json"
    csv_path = output_dir / f"data_analysis_{args.split}_domains.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(reports, csv_path)

    print(); print("=" * 90); print("MACRO DIAGNOSTICA"); print("=" * 90)
    for key, value in summary["macro_average"].items():
        print(f"{key:<40}: {value:.4f}" if isinstance(value, float) else f"{key:<40}: {value}")
    print(); print(f"✓ JSON salvato: {json_path}"); print(f"✓ CSV salvato: {csv_path}")


if __name__ == "__main__":
    main()
