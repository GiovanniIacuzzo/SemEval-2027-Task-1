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
    load_all_domains_data,
)
from subtrack_2b.models.model import ConversationalGenerator
from subtrack_2b.utils.utils import (
    build_prompt,
    build_refinement_prompt,
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
        choices=["dev", "train"],
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

    args = parser.parse_args()

    # ==========================================================================
    # 1. CONFIG
    # ==========================================================================

    cfg_path = Path(args.config)

    if not cfg_path.exists():
        cfg_path = (
            Path(__file__).resolve().parent
            / "config"
            / "config.yaml"
        )

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config non trovato: {args.config}"
        )

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

    output_dir = ensure_dir(
        paths_cfg.get(
            "output_dir",
            "outputs/subtrack_2b",
        )
    )

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
    )

    if not samples:
        logger.error(
            "Nessun dato trovato per split '%s'.",
            args.split,
        )
        sys.exit(1)

    logger.info(
        "Campioni totali caricati: %d",
        len(samples),
    )

    logger.info(
        "Domini caricati: %s",
        diagnostics.get(
            "domains_loaded",
            [],
        ),
    )

    missing_ids = diagnostics.get(
        "missing_gold_ids_count",
        0,
    )

    if missing_ids > 0:
        logger.warning(
            "Gold document IDs mancanti: %d",
            missing_ids,
        )

    # ==========================================================================
    # 7. LIMITS
    # ==========================================================================

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

    default_checkpoint = (
        Path(
            paths_cfg.get(
                "checkpoint_dir",
                "checkpoints/subtrack_2b",
            )
        )
        / "best_model"
    )

    checkpoint_dir = (
        Path(args.checkpoint)
        if args.checkpoint
        else default_checkpoint
    )

    logger.info(
        "Checkpoint selezionato: %s",
        checkpoint_dir,
    )

    # ==========================================================================
    # 9. MODEL
    # ==========================================================================

    generator = ConversationalGenerator.from_pretrained(
        config,
        device=device,
        is_training=False,
    )

    if checkpoint_dir.exists():
        generator.load_checkpoint(
            checkpoint_dir
        )

        logger.info(
            "Checkpoint caricato correttamente."
        )

    else:
        logger.warning(
            "Checkpoint '%s' non trovato. "
            "Procedo in zero-shot.",
            checkpoint_dir,
        )

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
        prompt_str = item["prompt_text"]

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
                        prompt=prompt_str,
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
                    prompt=prompt_str,
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
            refine_prompt = (
                build_refinement_prompt(
                    query=sample.query,
                    conversation_history=sample.conversation_history,
                    evidence_passages=sample.gold_passages,
                    draft_answer=generated_answer,
                    domain=sample.domain,
                )
            )

            generated_answer = (
                generator.generate_single(
                    prompt=refine_prompt,
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
            "reference": sample.answer,
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
        if sample.answer:
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

    if predictions_with_refs:
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
            "checkpoint": str(
                checkpoint_dir
            ),
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