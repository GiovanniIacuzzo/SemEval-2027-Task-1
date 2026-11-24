#!/usr/bin/env python3
"""
subtrack_2b/inference.py

Pipeline di generazione per RETECO Sub-track 2b.

Caratteristiche:
- Supporto split dev/train
- Supporto custom checkpoint
- Supporto CUDA / MPS / CPU
- Utilizzo dello stesso preprocessing del training
- Nessun data leakage
- Supporto direct / refine
- Supporto N-candidates
- Candidate selection senza reference answer
- Output ufficiale JSONL:
      {"turn_id": "...", "answer": "..."}
- Output diagnostico separato
- Metriche offline opzionali
- Breakdown per dominio e turn depth
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List

from tqdm import tqdm


# ==============================================================================
# ROOT PATH
# ==============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ==============================================================================
# LOCAL IMPORTS
# ==============================================================================

from subtrack_2b.dataset.dataset import (
    GroundedGenerationDataset,
    _normalize_active_domains,
    split_conversations_train_val,
    load_all_domains_data,
)
from subtrack_2b.models.model import ConversationalGenerator
from subtrack_2b.generate_submission import validate_predictions
from subtrack_2b.utils.utils import (
    compute_generation_metrics,
    detect_device,
    ensure_dir,
    load_yaml,
    save_json,
    save_jsonl,
    select_best_candidate,
    set_seed,
    setup_logging,
)


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Inference Pipeline RETECO "
            "SemEval-2027 Sub-track 2b"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="subtrack_2b/config/config.yaml",
        help="Percorso config YAML.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "train", "internal-val", "test"],
        help="Split da elaborare.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override del checkpoint.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Elabora solo 5 campioni.",
    )

    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Filtra un singolo dominio.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limita il numero di campioni.",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["direct", "refine"],
        help="Override della generation strategy.",
    )

    parser.add_argument("--zero-shot", action="store_true", help="Usa esplicitamente il modello base.")
    parser.add_argument("--data-dir", type=Path, help="Directory con le sottocartelle dei domini.")
    parser.add_argument("--benchmark-file", help="Nome del benchmark in ciascuna sottocartella.")
    parser.add_argument("--output-dir", type=Path, help="Directory nuova per i risultati della run.")
    parser.add_argument("--topic-ids", type=Path, help="Lista di ID, uno per riga, per confronti appaiati.")
    parser.add_argument("--evidence-strategy", choices=["prefix", "adaptive", "sentences"])
    parser.add_argument("--skip-text-metrics", action="store_true")
    args = parser.parse_args()

    if args.zero_shot and args.checkpoint:
        parser.error("--zero-shot e --checkpoint sono incompatibili.")
    if args.split == "test" and not (args.data_dir and args.benchmark_file):
        parser.error("Per il test specifica --data-dir e --benchmark-file.")
    if args.split == "test" and (args.domain or args.topic_ids or args.smoke_test or args.max_samples):
        parser.error("Per il test configura i domini nel YAML e valuta tutti i loro turni.")

    # ==========================================================================
    # 1. CONFIG
    # ==========================================================================

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config non trovato: {cfg_path}")

    config = load_yaml(
        cfg_path
    )

    general_cfg = config.get(
        "general",
        {},
    )

    paths_cfg = config.get(
        "paths",
        {},
    )

    domains_cfg = config.get(
        "domains",
        {},
    )

    data_cfg = config.get(
        "data",
        {},
    )

    prompt_cfg = config.get(
        "prompt",
        {},
    )

    generation_cfg = config.get(
        "generation",
        {},
    )

    # ==========================================================================
    # 2. SEED
    # ==========================================================================

    seed = int(
        general_cfg.get(
            "seed",
            42,
        )
    )

    set_seed(seed)

    # ==========================================================================
    # 3. DIRECTORIES / LOGGING
    # ==========================================================================

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir or (
        Path(paths_cfg.get("output_dir", "outputs/subtrack_2b")) / f"{args.split}_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    log_dir = ensure_dir(
        paths_cfg.get(
            "log_dir",
            "outputs/subtrack_2b/logs",
        )
    )

    logger = setup_logging(
        log_dir,
        run_tag=f"inference_{args.split}",
        log_level=general_cfg.get(
            "logging_level",
            "INFO",
        ),
    )

    logger.info(
        "=" * 90
    )
    logger.info(
        "RETECO SemEval-2027 Task 1 - Sub-track 2b"
    )
    logger.info(
        "Gold-Passage Grounded Generation"
    )
    logger.info(
        "Inference split: %s",
        args.split.upper(),
    )
    logger.info(
        "=" * 90
    )

    # ==========================================================================
    # 4. DEVICE
    # ==========================================================================

    device, device_name = detect_device(
        general_cfg.get(
            "device",
            "auto",
        )
    )

    logger.info(
        "Dispositivo per generazione: %s",
        device_name.upper(),
    )

    # ==========================================================================
    # 5. DATA DIRECTORY
    # ==========================================================================

    data_mode = str(
        paths_cfg.get(
            "data_mode",
            "full",
        )
    ).lower()

    if data_mode == "full":
        data_dir = paths_cfg.get(
            "full_data_dir",
            "data/reteco_data/track2_recor",
        )

    elif data_mode == "sample":
        data_dir = paths_cfg.get(
            "sample_data_dir",
        )

    else:
        raise ValueError(
            "paths.data_mode deve essere 'full' o 'sample'. "
            f"Ricevuto: {data_mode}"
        )

    data_dir = args.data_dir or data_dir
    if not data_dir:
        raise ValueError(
            "Data directory non configurata."
        )

    data_dir = Path(
        data_dir
    )

    split_filename = (
        paths_cfg.get(
            "dev_file",
            "benchmark_dev.json",
        )
        if args.split == "dev"
        else
        paths_cfg.get(
            "train_file",
            "benchmark_train.json",
        )
    )

    documents_filename = paths_cfg.get(
        "documents_file",
        "documents.jsonl",
    )

    all_domains = domains_cfg.get(
        "all_domains",
        [],
    )

    active_domains = (
        [args.domain]
        if args.domain
        else domains_cfg.get(
            "active_domains",
            "all",
        )
    )

    logger.info(
        "Data mode: %s",
        data_mode,
    )

    logger.info(
        "Data directory: %s",
        data_dir,
    )

    logger.info(
        "Active domains: %s",
        active_domains,
    )

    # ==========================================================================
    # 6. LOAD DATA
    # ==========================================================================

    samples, diagnostics = load_all_domains_data(
        base_dir=data_dir,
        active_domains=active_domains,
        all_domains=all_domains,
        split_filename=split_filename,
        documents_filename=documents_filename,
        require_corpus=args.split != "test",
    )

    expected_domains = set(_normalize_active_domains(active_domains, all_domains))
    missing_domains = expected_domains - set(diagnostics["domains_loaded"])
    if missing_domains or not samples:
        raise ValueError(f"Split vuoto o domini mancanti: {sorted(missing_domains)}")
    for key in ("missing_gold_ids_count", "examples_without_evidence", "empty_query_count"):
        if diagnostics[key]:
            raise ValueError(f"Dati incompleti: {key}={diagnostics[key]}")
    if args.split == "internal-val":
        _, samples = split_conversations_train_val(
            samples, val_ratio=float(config.get("training", {}).get("internal_val_ratio", 0.1)), seed=seed
        )
    if args.topic_ids:
        wanted = [line.strip() for line in args.topic_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_id = {sample.topic_id: sample for sample in samples}
        if not wanted or len(set(wanted)) != len(wanted) or set(wanted) - by_id.keys():
            raise ValueError("--topic-ids contiene ID assenti o duplicati oppure è vuoto.")
        samples = [by_id[topic_id] for topic_id in wanted]
    topics = {sample.topic_id for sample in samples}
    if not topics or len(topics) != len(samples):
        raise ValueError("Split senza turni oppure con identificatori duplicati.")

    if args.smoke_test:
        samples = samples[:5]

        logger.info(
            "SMOKE TEST: limitato a %d campioni.",
            len(samples),
        )

    elif args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError(
                "--max-samples deve essere > 0."
            )

        samples = samples[
            :args.max_samples
        ]

        logger.info(
            "Dataset limitato a %d campioni.",
            len(samples),
        )

    logger.info(
        "Campioni da elaborare: %d",
        len(samples),
    )

    # ==========================================================================
    # 8. CHECKPOINT
    # ==========================================================================

    checkpoint_root = Path(paths_cfg.get("checkpoint_dir", "checkpoints/subtrack_2b"))
    default_checkpoint = checkpoint_root / general_cfg.get("run_tag", "subtrack_2b_run") / "best_model"
    checkpoint_dir = None if args.zero_shot else Path(args.checkpoint or default_checkpoint)
    if checkpoint_dir is not None and not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint mancante: {checkpoint_dir}. Per il modello base usa --zero-shot.")
    if args.split == "test" and (
        prompt_cfg.get("ablation") == "E" or prompt_cfg.get("use_subquestion_reasoning")
    ):
        parser.error("L'input di test non deve usare annotazioni di reasoning.")
    generator = ConversationalGenerator.from_pretrained(config, device=device, is_training=False)
    if checkpoint_dir is not None:
        generator.load_checkpoint(checkpoint_dir)

    # ==========================================================================
    # 10. GENERATION CONFIG
    # ==========================================================================

    strategy = (
        args.strategy
        or generation_cfg.get(
            "strategy",
            "direct",
        )
    )

    max_new_tokens = int(
        generation_cfg.get(
            "max_new_tokens",
            512,
        )
    )

    do_sample = bool(
        generation_cfg.get(
            "do_sample",
            False,
        )
    )

    temperature = float(
        generation_cfg.get(
            "temperature",
            0.7,
        )
    )

    top_p = float(
        generation_cfg.get(
            "top_p",
            0.9,
        )
    )

    top_k = int(
        generation_cfg.get(
            "top_k",
            50,
        )
    )

    repetition_penalty = float(
        generation_cfg.get(
            "repetition_penalty",
            1.05,
        )
    )

    num_candidates = int(
        generation_cfg.get(
            "num_candidates",
            1,
        )
    )

    use_grounding_check = bool(
        generation_cfg.get(
            "grounding_check",
            False,
        )
    )

    if num_candidates <= 0:
        raise ValueError(
            "generation.num_candidates deve essere >= 1."
        )

    logger.info(
        "Generation config:"
    )

    logger.info(
        "  strategy=%s",
        strategy,
    )

    logger.info(
        "  do_sample=%s",
        do_sample,
    )

    logger.info(
        "  max_new_tokens=%d",
        max_new_tokens,
    )

    logger.info(
        "  num_candidates=%d",
        num_candidates,
    )

    logger.info(
        "  grounding_check=%s",
        use_grounding_check,
    )

    # ==========================================================================
    # 11. BUILD INFERENCE DATASET
    # ==========================================================================

    max_input_tokens = int(
        data_cfg.get(
            "max_input_tokens",
            3584,
        )
    )

    max_output_tokens = int(
        data_cfg.get(
            "max_output_tokens",
            512,
        )
    )

    max_history_turns = data_cfg.get(
        "max_history_turns"
    )

    if max_history_turns is not None:
        max_history_turns = int(
            max_history_turns
        )

    # Usiamo lo STESSO Dataset del training, ma in modalità inference.
    # Questo garantisce la stessa:
    # - truncation history
    # - evidence budget
    # - chat template tokenization
    # - ablation
    evidence_strategy = args.evidence_strategy or prompt_cfg.get("evidence_strategy", "adaptive")
    generation_dataset = GroundedGenerationDataset(
        samples=samples,
        tokenizer=generator.tokenizer,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_history_turns=max_history_turns,
        ablation=prompt_cfg.get(
            "ablation",
            "D",
        ),
        include_domain=bool(
            prompt_cfg.get(
                "include_domain",
                True,
            )
        ),
        use_subquestion_reasoning=bool(
            prompt_cfg.get(
                "use_subquestion_reasoning",
                False,
            )
        ),
        is_training=False,
        evidence_strategy=evidence_strategy,
    )

    # ==========================================================================
    # 12. OUTPUT STRUCTURES
    # ==========================================================================

    official_predictions: List[
        Dict[str, Any]
    ] = []

    diagnostic_predictions: List[
        Dict[str, Any]
    ] = []

    domain_preds = defaultdict(list)
    domain_refs = defaultdict(list)

    depth_preds = defaultdict(list)
    depth_refs = defaultdict(list)

    # ==========================================================================
    # 13. INFERENCE LOOP
    # ==========================================================================

    start_infer = time.time()

    for idx in tqdm(
        range(len(generation_dataset)),
        desc=f"Generazione ({args.split})",
    ):
        item = generation_dataset[idx]

        sample = item["sample"]
        prompt_ids = item["prompt_ids"]
        refine_diagnostics = None

        # ------------------------------------------------------------------
        # Candidate generation
        # ------------------------------------------------------------------
        if num_candidates > 1:
            candidates: List[str] = []

            for _ in range(
                num_candidates
            ):
                generated = (
                    generator.generate_single(
                        prompt=prompt_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                    )
                )

                candidates.append(
                    generated
                )

            generated_answer = (
                select_best_candidate(
                    candidates=candidates,
                    query=sample.query,
                    history=sample.conversation_history,
                    evidence_passages=sample.gold_passages,
                    use_grounding=use_grounding_check,
                )
            )

        else:
            generated_answer = (
                generator.generate_single(
                    prompt=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
            )

        # ------------------------------------------------------------------
        # Optional refinement
        # ------------------------------------------------------------------
        if strategy == "refine":
            _, refine_ids, refine_diagnostics = generation_dataset.prepare_prompt(
                sample, draft_answer=generated_answer
            )

            generated_answer = (
                generator.generate_single(
                    prompt=refine_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                )
            )

        generated_answer = (
            generated_answer.strip()
        )

        # ==========================================================================
        # 14. OFFICIAL OUTPUT
        # ==========================================================================

        official_predictions.append(
            {
                # Official RETECO Track 2 turn identifier:
                # <conversation_id>_turn_<turn_id>
                "turn_id": sample.topic_id,
                "answer": generated_answer,
            }
        )

        # ==========================================================================
        # 15. DIAGNOSTIC OUTPUT
        # ==========================================================================

        diagnostic_record = {
            "conversation_id": sample.conversation_id,
            "turn_id": sample.turn_id,
            "topic_id": sample.topic_id,
            "domain": sample.domain,
            "query": sample.query,
            "conversation_history": sample.conversation_history,
            "prediction": generated_answer,
            "reference": sample.answer if args.split != "test" else None,
            "prompt": item["prompt_diagnostics"],
            "refine_prompt": refine_diagnostics,
            "gold_passages": sample.gold_passages if args.split != "test" else None,
            "gold_doc_ids": sample.gold_doc_ids,
            "num_evidence_passages": len(
                sample.gold_passages
            ),
            "prediction_words": len(
                generated_answer.split()
            ),
        }

        diagnostic_predictions.append(
            diagnostic_record
        )

        # ------------------------------------------------------------------
        # Metrics grouping
        # ------------------------------------------------------------------
        if sample.answer and args.split != "test" and not args.skip_text_metrics:
            domain_preds[
                sample.domain
            ].append(
                generated_answer
            )

            domain_refs[
                sample.domain
            ].append(
                sample.answer
            )

            if sample.turn_id < 5:
                depth_bucket = (
                    f"turn_{sample.turn_id}"
                )
            else:
                depth_bucket = "turn_5_plus"

            depth_preds[
                depth_bucket
            ].append(
                generated_answer
            )

            depth_refs[
                depth_bucket
            ].append(
                sample.answer
            )

    # ==========================================================================
    # 16. TIMING
    # ==========================================================================

    elapsed_time = (
        time.time()
        - start_infer
    )

    throughput = (
        len(samples)
        / max(
            elapsed_time,
            1e-4,
        )
    )

    logger.info(
        "Generazione completata in %.1f s (%.3f turni/s).",
        elapsed_time,
        throughput,
    )

    # ==========================================================================
    # 17. SAVE OFFICIAL OUTPUT
    # ==========================================================================

    generation_file = (
        output_dir
        / f"generation_{args.split}.jsonl"
    )

    metadata = {
        "model": config.get("model", {}).get("name"),
        "model_revision": getattr(generator.model.config, "_commit_hash", None)
        or config.get("model", {}).get("revision"),
        "device": str(device),
        "loaded_in_4bit": bool(getattr(generator.model, "is_loaded_in_4bit", False)),
        "checkpoint": str(checkpoint_dir.resolve()) if checkpoint_dir is not None else None,
        "split": args.split,
        "config": config,
        "cli": vars(args) | {
            "data_dir": str(args.data_dir) if args.data_dir else None,
            "topic_ids": str(args.topic_ids) if args.topic_ids else None,
            "output_dir": str(output_dir),
        },
        "evidence_strategy": evidence_strategy,
        "resolved_strategy": strategy,
        "documented_missing_gold_ids": diagnostics.get("documented_missing_gold_ids_count", 0),
        "topic_ids": [sample.topic_id for sample in samples],
    }
    save_json(metadata, output_dir / "run.json")
    (output_dir / "topic_ids.txt").write_text(
        "\n".join(metadata["topic_ids"]) + "\n", encoding="utf-8"
    )
    if args.split == "test":
        validate_predictions(official_predictions, set(metadata["topic_ids"]))
    save_jsonl(
        official_predictions,
        generation_file,
    )

    logger.info(
        "Output ufficiale salvato: %s",
        generation_file,
    )

    # ==========================================================================
    # 18. SAVE DIAGNOSTIC OUTPUT
    # ==========================================================================

    prediction_file = (
        output_dir
        / f"{args.split}_predictions.jsonl"
    )

    save_jsonl(
        diagnostic_predictions,
        prediction_file,
    )

    logger.info(
        "Output diagnostico salvato: %s",
        prediction_file,
    )

    # ==========================================================================
    # 19. OFFLINE METRICS
    # ==========================================================================

    predictions_with_refs = [
        record
        for record in diagnostic_predictions
        if record.get("reference")
    ]

    if args.skip_text_metrics:
        logger.info("Metriche testuali disabilitate per questa run.")
    elif predictions_with_refs:
        all_predictions = [
            record["prediction"]
            for record in predictions_with_refs
        ]

        all_references = [
            record["reference"]
            for record in predictions_with_refs
        ]

        logger.info(
            "Calcolo metriche offline..."
        )

        overall_metrics = (
            compute_generation_metrics(
                all_predictions,
                all_references,
            )
        )

        # ------------------------------------------------------------------
        # Per domain
        # ------------------------------------------------------------------
        per_domain_metrics = {}

        for domain, pred_list in domain_preds.items():
            ref_list = domain_refs[
                domain
            ]

            per_domain_metrics[
                domain
            ] = compute_generation_metrics(
                pred_list,
                ref_list,
            )

        # ------------------------------------------------------------------
        # Per depth
        # ------------------------------------------------------------------
        per_depth_metrics = {}

        for bucket, pred_list in depth_preds.items():
            ref_list = depth_refs[
                bucket
            ]

            per_depth_metrics[
                bucket
            ] = compute_generation_metrics(
                pred_list,
                ref_list,
            )

        evaluation_summary = {
            "overall": overall_metrics,
            "per_domain": per_domain_metrics,
            "per_depth": per_depth_metrics,
            "samples_evaluated": len(
                all_predictions
            ),
            "checkpoint": metadata["checkpoint"],
            "strategy": strategy,
            "num_candidates": num_candidates,
            "ablation": prompt_cfg.get(
                "ablation",
                "D",
            ),
        }

        evaluation_file = (
            output_dir
            / f"{args.split}_evaluation_summary.json"
        )

        save_json(
            evaluation_summary,
            evaluation_file,
        )

        logger.info(
            "=" * 80
        )
        logger.info(
            "RISULTATI OFFLINE - %s",
            args.split.upper(),
        )

        for metric_name, value in (
            overall_metrics.items()
        ):
            logger.info(
                "  %-20s %.4f",
                metric_name,
                value,
            )

        logger.info(
            "Evaluation summary: %s",
            evaluation_file,
        )

        logger.info(
            "=" * 80
        )

    else:
        logger.info(
            "Nessuna reference disponibile: "
            "metriche offline non calcolate."
        )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()