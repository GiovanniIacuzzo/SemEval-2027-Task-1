#!/usr/bin/env python3
"""
download_data.py

Modulo di download universale e selettivo per SemEval-2027 Task 1 (RETECO).
Permette di selezionare il target (sample, 2a, 2b, 2c, 1a, 1b) modificando
semplicemente la variabile DEFAULT_TASK o tramite flag CLI (--task).
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path
import urllib.request

# ==============================================================================
# CONFIGURAZIONE RAPIDA: MODIFICA SOLO QUESTA VARIABILE
# Opzioni consentite:
#   - "sample"   : Pacchetto rapido da 64 KB (per Mac Air senza librerie)
#   - "2a"       : Solo file per Sub-track 2a (benchmark, corpus, qrels)
#   - "2b"       : Solo file per Sub-track 2b (benchmark, corpus)
#   - "2c"       : Tutti i file di Track 2 (Full Conversational RAG)
#   - "1a"       : File per Sub-track 1a (query temporali e corpus TEMPO)
#   - "1b"       : File per Sub-track 1b (step decomposti e corpus TEMPO)
#   - "all"      : Intero dataset v1.1 (~3.2 GB)
# ==============================================================================
DEFAULT_TASK = "2a"

# Repository e URL ufficiali
REPO_ID = "DataScience-UIBK/RETECO-SemEval2027"
SAMPLE_URL = "https://datascienceuibk.github.io/RETECO/assets/downloads/RETECO_curated_sample_data.zip"

TRACK2_DOMAINS = [
    "biology", "drones", "earth_science", "economics", "hardware",
    "law", "medicalsciences", "politics", "psychology", "robotics",
    "sustainable_living"
]

TRACK1_DOMAINS = [
    "bitcoin", "cardano", "economics", "genealogy", "history", "hsm",
    "iota", "law", "monero", "politics", "quant", "travel", "workplace"
]

# Mappatura dei file minimi indispensabili per ogni sub-track
TASK_SPECS = {
    "2a": {
        "track_folder": "track2_recor",
        "domains": TRACK2_DOMAINS,
        "files": ["documents.jsonl", "benchmark_train.json", "benchmark_dev.json", "qrels_train.txt", "qrels_dev.txt"],
        "description": "Sub-track 2a: Conversational Retrieval"
    },
    "2b": {
        "track_folder": "track2_recor",
        "domains": TRACK2_DOMAINS,
        "files": ["documents.jsonl", "benchmark_train.json", "benchmark_dev.json"],
        "description": "Sub-track 2b: Grounded Generation with Gold Passages"
    },
    "2c": {
        "track_folder": "track2_recor",
        "domains": TRACK2_DOMAINS,
        "files": ["documents.jsonl", "benchmark_train.json", "benchmark_dev.json", "qrels_train.txt", "qrels_dev.txt"],
        "description": "Sub-track 2c: Full Conversational RAG"
    },
    "1a": {
        "track_folder": "track1_tempo",
        "domains": TRACK1_DOMAINS,
        "files": ["documents.jsonl", "examples_train.jsonl", "examples_dev.jsonl", "qrels_train.txt", "qrels_dev.txt", "duplicate_map.json"],
        "description": "Sub-track 1a: Whole-Query Temporal Retrieval"
    },
    "1b": {
        "track_folder": "track1_tempo",
        "domains": TRACK1_DOMAINS,
        "files": ["documents.jsonl", "steps_train.jsonl", "steps_dev.jsonl", "qrels_steps_train.txt", "qrels_steps_dev.txt", "duplicate_map.json"],
        "description": "Sub-track 1b: Step-wise Temporal Retrieval"
    }
}


def download_sample(target_dir: Path):
    """Scarica il pacchetto sample leggero da 64 KB."""
    print(f"\n[1/3] Download del sample curato da: {SAMPLE_URL}")
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "sample_data.zip"

    try:
        urllib.request.urlretrieve(SAMPLE_URL, zip_path)
        print(f"[2/3] Download completato ({zip_path.stat().st_size / 1024:.2f} KB).")
        print(f"[3/3] Estrazione in: {target_dir}")
        
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        if zip_path.exists():
            zip_path.unlink()

        # Riordina cartelle se presenti sotto-directory intermedie
        track2_path = target_dir / "track2_recor"
        if not track2_path.exists():
            found = list(target_dir.glob("**/track2_recor"))
            if found and found[0] != track2_path:
                for item in found[0].iterdir():
                    shutil.move(str(item), str(track2_path / item.name))

        print("\n✓ Sample estratto con successo in:", target_dir)

    except Exception as e:
        print(f"\n[ERRORE] Impossibile scaricare il sample: {e}", file=sys.stderr)
        sys.exit(1)


def download_hf_dataset(target_dir: Path, task_key: str, domains_filter: list = None):
    """Scarica selettivamente i soli file pertinenti al task_key selezionato."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "\n[ERRORE] 'huggingface_hub' non trovato nell'ambiente corrente.\n"
            "Attiva l'ambiente conda ('conda activate reteco') o installalo con 'pip install huggingface_hub'",
            file=sys.stderr
        )
        sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)

    if task_key == "all":
        print(f"\nAvvio download COMPLETO del repository {REPO_ID} (~3.2 GB)...")
        patterns = None
    else:
        spec = TASK_SPECS[task_key]
        folder = spec["track_folder"]
        active_domains = domains_filter if domains_filter else spec["domains"]

        patterns = ["split_manifest.json", "README.md"]
        for d in active_domains:
            for f in spec["files"]:
                patterns.append(f"{folder}/{d}/{f}")

        print("\n" + "=" * 65)
        print(f"DOWNLOAD SELETTIVO: {spec['description']}")
        print(f"Target: {folder} | Domini: {len(active_domains)} attivi")
        print(f"File inclusi: {', '.join(spec['files'])}")
        print("=" * 65)

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_patterns=patterns,
            resume_download=True
        )
        print(f"\n✓ Download per il task '{task_key}' completato con successo in {target_dir}")

    except Exception as e:
        print(f"\n[ERRORE] Durante il download da Hugging Face: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Downloader centralizzato per il benchmark SemEval-2027 RETECO."
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=DEFAULT_TASK,
        choices=["sample", "2a", "2b", "2c", "1a", "1b", "all"],
        help=f"Task da scaricare (default: '{DEFAULT_TASK}')."
    )
    parser.add_argument(
        "--domains", "-d",
        nargs="+",
        default=None,
        help="Opzionale: scarica solo uno o più domini specifici (es. -d drones biology)."
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Percorso di destinazione (default: data/sample o data/reteco_data)."
    )

    args = parser.parse_args()
    selected_task = args.task.lower().strip()
    project_root = Path(__file__).resolve().parent

    if selected_task == "sample":
        out_dir = Path(args.dest) if args.dest else project_root / "data" / "sample"
        download_sample(out_dir)
    else:
        out_dir = Path(args.dest) if args.dest else project_root / "data" / "reteco_data"
        download_hf_dataset(out_dir, task_key=selected_task, domains_filter=args.domains)


if __name__ == "__main__":
    main()