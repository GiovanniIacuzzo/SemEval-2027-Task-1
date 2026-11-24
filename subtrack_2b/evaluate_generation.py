#!/usr/bin/env python3
"""Prepare blinded paired review and aggregate local RETECO rubric scores."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from generate_submission import read_jsonl

RUBRIC = {
    "correctness": "Are factual claims accurate?",
    "completeness": "Does the answer address all parts of the current request?",
    "relevance": "Does the answer address the current request directly?",
    "conversational_coherence": "Does the answer resolve references and fit the dialogue?",
    "faithfulness": "Are factual claims supported by the supplied gold passages?",
}


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def keyed(rows, key):
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Valori duplicati per {key}.")
    return indexed


def prepare(baseline, candidate, review_path, key_path, seed):
    runs = [keyed(read_jsonl(path), "topic_id") for path in (baseline, candidate)]
    if not runs[0] or runs[0].keys() != runs[1].keys():
        raise ValueError("I file devono avere gli stessi turni e almeno un esempio.")
    cases, key = [], {}
    rng = random.Random(seed)
    for i, topic in enumerate(sorted(runs[0]), 1):
        a, b = (run[topic] for run in runs)
        if any(a[field] != b[field] for field in (
            "query", "conversation_history", "reference", "gold_passages", "domain", "turn_id"
        )):
            raise ValueError(f"Contesto di valutazione discordante: {topic}")
        if not a["gold_passages"] or not a["reference"]:
            raise ValueError(f"Reference o passaggi gold mancanti: {topic}")
        pair = [("baseline", a), ("candidate", b)]
        rng.shuffle(pair)
        for label, (system, row) in zip(("A", "B"), pair):
            case_id = f"{i:04d}-{label}"
            key[case_id] = {"system": system, "topic_id": topic, "domain": row["domain"],
                            "depth": min(int(row["turn_id"]), 5)}
            cases.append({"case_id": case_id, "question": row["query"],
                          "history": row["conversation_history"], "gold_passages": row["gold_passages"],
                          "reference": row["reference"], "answer": row["prediction"],
                          "scores": dict.fromkeys(RUBRIC)})
    rng.shuffle(cases)
    write(review_path, {"instructions": "Score each criterion independently with an integer from 1 (poor) to 5 (excellent). Fill the scores object in every case.",
                        "rubric": RUBRIC, "cases": cases})
    write(key_path, key)


def aggregate(review_path, key_path):
    cases = keyed(json.loads(review_path.read_text(encoding="utf-8"))["cases"], "case_id")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    if not key or cases.keys() != key.keys():
        raise ValueError("La copertura delle schede valutate non coincide con la chiave.")
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for case, metadata in key.items():
        scores = cases[case]["scores"]
        if set(scores) != set(RUBRIC) or any(type(v) is not int or not 1 <= v <= 5 for v in scores.values()):
            raise ValueError(f"Punteggi mancanti o fuori da 1–5: {case}")
        system = metadata["system"]
        for group in ("overall", f"domain/{metadata['domain']}", f"depth/{metadata['depth']}"):
            for criterion, score in scores.items():
                groups[group][system][criterion].append((score - 1) / 4)
    report = {}
    for group, systems in groups.items():
        values = {system: {dim: mean(scores) for dim, scores in dimensions.items()}
                  for system, dimensions in systems.items()}
        report[group] = values | {"delta": {dim: values["candidate"][dim] - values["baseline"][dim] for dim in RUBRIC},
                                  "turns": len(systems["baseline"]["correctness"])}
    domains = [entry for group, entry in report.items() if group.startswith("domain/")]
    report["macro_domains"] = {system: {dim: mean(entry[system][dim] for entry in domains) for dim in RUBRIC}
                               for system in ("baseline", "candidate", "delta")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    for name in ("baseline", "candidate", "review", "key"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    score = subparsers.add_parser("score")
    for name in ("review", "key", "output"):
        score.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.review.resolve() == args.key.resolve():
            parser.error("Scheda e chiave richiedono file distinti.")
        prepare(args.baseline, args.candidate, args.review, args.key, args.seed)
    else:
        write(args.output, aggregate(args.review, args.key))


if __name__ == "__main__":
    main()
