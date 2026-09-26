#!/usr/bin/env python3
"""
analyse_data.py

Modulo di Exploratory Data Analysis (EDA) per SemEval-2027 Task 1 (Track 2: RECOR).
Analizza gli 11 domini e genera grafici pubblicabili in outputs/subtrack_2a/img/.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Rendering headless di Matplotlib (evita problemi su server/Mac)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACK2_DOMAINS = [
    "biology", "drones", "earth_science", "economics", "hardware",
    "law", "medicalsciences", "politics", "psychology", "robotics",
    "sustainable_living"
]

def load_json_or_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    """Carica in modo sicuro file JSON (lista o oggetti JSONL)."""
    if not filepath.exists():
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        if content.startswith("["):
            return json.loads(content)
        return [json.loads(line) for line in content.splitlines() if line.strip()]

def count_lines(filepath: Path) -> int:
    """Conta velocemente le righe di un file senza caricare tutto in memoria."""
    if not filepath.exists():
        return 0
    count = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            count += 1
    return count

def analyze_track2(base_dir: Path, output_img_dir: Path):
    print("=" * 75)
    print("ANALISI DATASET SEMEVAL-2027 TASK 1 - TRACK 2 (RECOR)")
    print(f"Directory sorgente: {base_dir}")
    print(f"Salvataggio grafici: {output_img_dir}")
    print("=" * 75)

    output_img_dir.mkdir(parents=True, exist_ok=True)

    # Strutture di accumulo metriche
    stats = {}
    turn_depths = []
    gold_counts_train = []
    gold_counts_dev = []
    query_lengths = []
    history_lengths = []
    doc_sample_lengths = []

    for domain in TRACK2_DOMAINS:
        domain_path = base_dir / domain
        if not domain_path.exists():
            print(f"⚠️ Dominio non trovato: {domain}")
            continue

        # 1. Corpus passaggi (documents.jsonl)
        docs_file = domain_path / "documents.jsonl"
        num_docs = 0
        if docs_file.exists():
            with open(docs_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    num_docs += 1
                    # Campiona lunghezze di testo (ogni 15 documenti per velocità e leggerezza di RAM)
                    if idx % 15 == 0 and line.strip():
                        try:
                            item = json.loads(line)
                            txt = item.get("content") or item.get("text") or ""
                            doc_sample_lengths.append(len(txt.split()))
                        except Exception:
                            pass

        # 2. Benchmark Train e Dev
        train_convs = load_json_or_jsonl(domain_path / "benchmark_train.json")
        dev_convs = load_json_or_jsonl(domain_path / "benchmark_dev.json")

        num_train_turns = 0
        for conv in train_convs:
            for turn in conv.get("turns", []):
                num_train_turns += 1
                t_id = turn.get("turn_id", 1)
                turn_depths.append(t_id)
                g_docs = turn.get("gold_doc_ids", [])
                gold_counts_train.append(len(g_docs))
                q_text = turn.get("query", "")
                h_text = turn.get("conversation_history", "")
                query_lengths.append(len(q_text.split()))
                if h_text and h_text.lower() != "no previous conversation.":
                    history_lengths.append(len(h_text.split()))

        num_dev_turns = 0
        for conv in dev_convs:
            for turn in conv.get("turns", []):
                num_dev_turns += 1
                g_docs = turn.get("gold_doc_ids", [])
                gold_counts_dev.append(len(g_docs))

        # 3. Qrels count
        qrels_tr = count_lines(domain_path / "qrels_train.txt")
        qrels_dv = count_lines(domain_path / "qrels_dev.txt")

        stats[domain] = {
            "num_docs": num_docs,
            "train_convs": len(train_convs),
            "dev_convs": len(dev_convs),
            "train_turns": num_train_turns,
            "dev_turns": num_dev_turns,
            "qrels_train": qrels_tr,
            "qrels_dev": qrels_dv,
        }

    # Stampa a video tabella consolidata
    print(f"\n{'Dominio':<20} | {'Documenti':<10} | {'Conv (Tr/Dv)':<13} | {'Turni (Tr/Dv)':<14} | {'Qrels (Tr/Dv)'}")
    print("-" * 75)
    total_docs = total_tr_c = total_dv_c = total_tr_t = total_dv_t = total_tr_q = total_dv_q = 0
    for dom, d in stats.items():
        total_docs += d["num_docs"]
        total_tr_c += d["train_convs"]
        total_dv_c += d["dev_convs"]
        total_tr_t += d["train_turns"]
        total_dv_t += d["dev_turns"]
        total_tr_q += d["qrels_train"]
        total_dv_q += d["qrels_dev"]
        print(f"{dom:<20} | {d['num_docs']:<10,d} | {d['train_convs']:>5} / {d['dev_convs']:<5} | {d['train_turns']:>5} / {d['dev_turns']:<6} | {d['qrels_train']:>5} / {d['qrels_dev']:<5}")
    print("-" * 75)
    print(f"{'TOTALE COMPLESSIVO':<20} | {total_docs:<10,d} | {total_tr_c:>5} / {total_dv_c:<5} | {total_tr_t:>5} / {total_dv_t:<6} | {total_tr_q:>5} / {total_dv_q:<5}\n")

    # --------------------------------------------------------------------------
    # GENERAZIONE GRAFICI DIAGNOSTICI
    # --------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # Grafico 1: Corpus Size per Dominio
    fig, ax = plt.subplots(figsize=(10, 5))
    domains_sorted = sorted(stats.keys(), key=lambda x: stats[x]["num_docs"], reverse=True)
    counts = [stats[d]["num_docs"] for d in domains_sorted]
    bars = ax.bar(domains_sorted, counts, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax.set_ylabel("Numero di Documenti nel Corpus")
    ax.set_title("Volume Corpus per Dominio (Track 2 - RECOR)")
    plt.xticks(rotation=40, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    p1 = output_img_dir / "1_corpus_distribution.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"✓ [Grafico 1] Salvato: {p1.name}")

    # Grafico 2: Split Train vs Dev (Turni)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(domains_sorted))
    width = 0.38
    tr_turns = [stats[d]["train_turns"] for d in domains_sorted]
    dv_turns = [stats[d]["dev_turns"] for d in domains_sorted]

    ax.bar(x - width/2, tr_turns, width, label="Train Turns (70%)", color="#2a9d8f", edgecolor="black")
    ax.bar(x + width/2, dv_turns, width, label="Dev Turns (30%)", color="#e76f51", edgecolor="black")
    ax.set_ylabel("Numero di Turni")
    ax.set_title("Distribuzione Turni Conversazionali per Split (Train vs Dev)")
    ax.set_xticks(x)
    ax.set_xticklabels(domains_sorted, rotation=40, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    p2 = output_img_dir / "2_train_vs_dev_turns.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"✓ [Grafico 2] Salvato: {p2.name}")

    # Grafico 3: Distribuzione Profondità di Turno (Turn Position T1 - T5+)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Raggruppa posizioni turni: 1, 2, 3, 4, 5+
    depth_buckets = {"T1": 0, "T2": 0, "T3": 0, "T4": 0, "T5+": 0}
    for td in turn_depths:
        if td == 1:
            depth_buckets["T1"] += 1
        elif td == 2:
            depth_buckets["T2"] += 1
        elif td == 3:
            depth_buckets["T3"] += 1
        elif td == 4:
            depth_buckets["T4"] += 1
        else:
            depth_buckets["T5+"] += 1

    keys = list(depth_buckets.keys())
    vals = list(depth_buckets.values())
    bars = ax.bar(keys, vals, color="#457b9d", edgecolor="black", width=0.55)
    ax.set_xlabel("Posizione del Turno nel Dialogo")
    ax.set_ylabel("Frequenza Turni")
    ax.set_title("Profondità Conversazionale (Metrica di Diagnosi SemEval-2027)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars:
        y = bar.get_height()
        ax.annotate(f"{y} ({y/len(turn_depths)*100:.1f}%)", xy=(bar.get_x() + bar.get_width()/2, y),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    p3 = output_img_dir / "3_turn_depth_distribution.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"✓ [Grafico 3] Salvato: {p3.name}")

    # Grafico 4: Boxplot Gold Passages per Turno (Train vs Dev)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data_to_plot = [gold_counts_train, gold_counts_dev]
    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=["Train Set", "Dev Set"],
                    boxprops=dict(facecolor="#a8dadc", color="#1d3557"),
                    medianprops=dict(color="#e63946", linewidth=2))
    ax.set_ylabel("Passaggi Gold per Turno")
    ax.set_title("Densità di Evidenze Gold per Domanda")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    p4 = output_img_dir / "4_gold_passages_per_turn.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"✓ [Grafico 4] Salvato: {p4.name}")

    # Grafico 5: Distribuzione Lunghezze (Passaggi, Query, Storico)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Lunghezza documenti (passaggi)
    ax1.hist(doc_sample_lengths, bins=40, color="#6a4c93", edgecolor="black", alpha=0.8, range=(0, 600))
    ax1.axvline(np.median(doc_sample_lengths), color="red", linestyle="--", label=f"Mediana: {int(np.median(doc_sample_lengths))} parole")
    ax1.axvline(512, color="orange", linestyle=":", label="Max Token T4 (512)")
    ax1.set_xlabel("Numero di Parole per Documento")
    ax1.set_ylabel("Conteggio")
    ax1.set_title("Lunghezza Passaggi del Corpus")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Lunghezza Query e History
    ax2.hist(query_lengths, bins=25, color="#1982c4", alpha=0.7, label="Turn Query", edgecolor="black", range=(0, 60))
    if history_lengths:
        ax2.hist(history_lengths, bins=25, color="#ff595e", alpha=0.5, label="Conversation History", edgecolor="black", range=(0, 200))
    ax2.set_xlabel("Numero di Parole")
    ax2.set_ylabel("Conteggio")
    ax2.set_title("Lunghezza Query e Storico Pregresso")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)

    p5 = output_img_dir / "5_lengths_distribution.png"
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"✓ [Grafico 5] Salvato: {p5.name}")

    print("\n" + "=" * 75)
    print("Analisi completata con successo! Tutti i grafici sono stati salvati in outputs/subtrack_2a/img/")
    print("=" * 75)

if __name__ == "__main__":
    default_data_dir = Path("data/reteco_data/track2_recor")
    default_out_dir = Path("outputs/subtrack_2a/img")
    
    # Se invocato dalla radice del repo
    if not default_data_dir.exists():
        alt_data = Path("../data/reteco_data/track2_recor")
        if alt_data.exists():
            default_data_dir = alt_data
            default_out_dir = Path("../outputs/subtrack_2a/img")

    analyze_track2(default_data_dir, default_out_dir)