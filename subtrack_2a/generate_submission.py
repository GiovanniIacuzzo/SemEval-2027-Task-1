#!/usr/bin/env python3
"""Validate a 2a TREC ranking and export it in official rank order."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from utils.utils import validate_trec_file, write_trec_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--expected-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", default="reteco_2a")
    args = parser.parse_args()

    valid, errors = validate_trec_file(args.run, max_rank=2**31 - 1)
    if not valid:
        parser.error("; ".join(errors[:5]))
    ids = args.expected_ids.read_text(encoding="utf-8").split()
    expected = set(ids)
    if not expected or len(ids) != len(expected):
        parser.error("Gli ID attesi devono essere non vuoti e univoci.")

    run = {}
    for line in args.run.read_text(encoding="utf-8").splitlines():
        if line.strip():
            topic, _, doc, _, score, _ = line.split()
            run.setdefault(topic, []).append((doc, float(score)))
    if run.keys() != expected:
        parser.error(f"Turni mancanti: {sorted(expected - run.keys())[:10]}. Inattesi: {sorted(run.keys() - expected)[:10]}.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as folder:
        staged = Path(folder) / "submission.trec"
        write_trec_run(run, staged, run_tag=args.run_tag, max_k=10)
        staged.replace(args.output)
    print(f"{len(run)} ranking validati: {args.output}")


if __name__ == "__main__":
    main()
