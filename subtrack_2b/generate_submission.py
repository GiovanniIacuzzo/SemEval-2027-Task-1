#!/usr/bin/env python3
"""Validate and export the public RETECO 2b JSONL format."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def validate_predictions(records, expected_ids):
    seen = set()
    for record in records:
        topic = record.get("turn_id")
        answer = record.get("answer")
        if not isinstance(topic, str) or not topic.strip() or topic in seen:
            raise ValueError(f"turn_id non valido o duplicato: {topic!r}")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Risposta vuota o non valida: {topic}")
        seen.add(topic)
    if not seen or seen != expected_ids:
        raise ValueError(
            f"Copertura non valida: mancanti={sorted(expected_ids - seen)[:10]}, "
            f"inattesi={sorted(seen - expected_ids)[:10]}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_jsonl(args.predictions)
    ids = args.expected_ids.read_text(encoding="utf-8").split()
    if len(ids) != len(set(ids)):
        parser.error("Il file degli ID attesi contiene duplicati.")
    validate_predictions(records, set(ids))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as output:
        tmp_path = Path(output.name)
        for record in records:
            row = {"turn_id": record["turn_id"], "answer": record["answer"].strip()}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(args.output)
    print(f"{len(records)} risposte validate: {args.output}")


if __name__ == "__main__":
    main()
