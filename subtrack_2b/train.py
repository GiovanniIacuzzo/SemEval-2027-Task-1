#!/usr/bin/env python3
"""
subtrack_2b/train.py

Pipeline di addestramento Supervised Fine-Tuning (SFT)
per RETECO SemEval-2027 Sub-track 2b.

Caratteristiche:
- Caricamento configurabile di Causal LM instruction-tuned + LoRA
- Supporto opzionale 4-bit / QLoRA
- Isolamento del dev ufficiale
- Internal validation split a livello di conversazione
- Loss calcolata esclusivamente sui token della gold answer
- Domain-balanced sampling opzionale
- Supporto CUDA / MPS / CPU
- Mixed precision su CUDA
- Gradient accumulation
- Gradient clipping
- Cosine scheduler con warmup
- Early stopping su internal validation
- Checkpoint best_model / last_model
- Resume reale con optimizer/scheduler/scaler state
- Smoke test end-to-end
"""

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


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
    GroundedDataCollator,
    GroundedGenerationDataset,
    GroundedTurnSample,
    apply_domain_balanced_sampling,
    load_all_domains_data,
    split_conversations_train_val,
)
from subtrack_2b.models.model import ConversationalGenerator
from subtrack_2b.utils.utils import (
    detect_device,
    ensure_dir,
    load_yaml,
    save_json,
    set_seed,
    setup_logging,
)


# ==============================================================================
# CHECKPOINT / RESUME HELPERS
# ==============================================================================

def _resume_state_path(checkpoint_dir: Path) -> Path:
    """Percorso dello stato completo di training."""
    return checkpoint_dir / "last_model" / "training_state.pt"


def _save_training_state(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    config: Dict[str, Any],
) -> None:
    """
    Salva lo stato necessario per un resume reale.

    Non salva i pesi del modello: quelli vengono gestiti separatamente da
    generator.save_checkpoint().
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
    }

    torch.save(state, path)


def _load_training_state(
    path: Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """
    Carica lo stato di training salvato.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training state non trovato: {path}"
        )

    state = torch.load(
        path,
        map_location=map_location,
    )

    optimizer.load_state_dict(
        state["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        state["scheduler_state_dict"]
    )

    scaler_state = state.get(
        "scaler_state_dict"
    )

    if scaler_state:
        scaler.load_state_dict(
            scaler_state
        )

    return state


def _load_model_for_resume(
    generator: ConversationalGenerator,
    checkpoint_dir: Path,
) -> None:
    """
    Carica il last checkpoint del modello prima di creare optimizer/scheduler.

    Per PEFT/LoRA:
        base model -> sostituzione dell'adapter default con quello salvato

    In questo modo i parametri sui quali verrà costruito l'optimizer sono
    effettivamente quelli del checkpoint caricato.
    """
    last_dir = checkpoint_dir / "last_model"

    if not last_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint last_model non trovato: {last_dir}"
        )

    model = generator.model

    if model is None:
        raise RuntimeError(
            "Modello non inizializzato."
        )

    # --------------------------------------------------------------------------
    # Caso PEFT
    # --------------------------------------------------------------------------
    is_peft = model.__class__.__name__.startswith(
        "Peft"
    )

    if is_peft:
        try:
            # Import locale per non rendere PEFT obbligatorio in full FT.
            from peft import PeftModel  # noqa: F401

            adapter_files = [
                last_dir / "adapter_model.safetensors",
                last_dir / "adapter_model.bin",
            ]

            has_adapter = any(
                path.exists()
                for path in adapter_files
            )

            if not has_adapter:
                raise FileNotFoundError(
                    "last_model non contiene un adapter PEFT riconosciuto."
                )

            # Rimuovi l'adapter default vuoto creato all'inizializzazione.
            if (
                hasattr(model, "peft_config")
                and "default" in model.peft_config
                and hasattr(model, "delete_adapter")
            ):
                model.delete_adapter("default")

            # Carica il checkpoint come nuovo adapter "default".
            # is_trainable=True perché stiamo effettuando un resume di training.
            model.load_adapter(
                str(last_dir),
                adapter_name="default",
                is_trainable=True,
            )

            model.set_adapter("default")

            generator.model = model

        except Exception as exc:
            raise RuntimeError(
                f"Impossibile caricare il checkpoint LoRA per resume: {exc}"
            ) from exc

        return

    # --------------------------------------------------------------------------
    # Caso full model
    # --------------------------------------------------------------------------
    try:
        generator.load_checkpoint(
            last_dir
        )
    except Exception as exc:
        raise RuntimeError(
            f"Impossibile caricare il full-model checkpoint per resume: {exc}"
        ) from exc


# ==============================================================================
# DATA HELPERS
# ==============================================================================

def _resolve_data_dir(
    paths_cfg: Dict[str, Any],
) -> Path:
    """
    Seleziona full o sample data directory sulla base del config.
    """

    data_mode = str(
        paths_cfg.get(
            "data_mode",
            "full",
        )
    ).lower()

    if data_mode == "sample":
        data_dir = paths_cfg.get(
            "sample_data_dir"
        )
    elif data_mode == "full":
        data_dir = paths_cfg.get(
            "full_data_dir"
        )
    else:
        raise ValueError(
            "paths.data_mode deve essere 'full' o 'sample'. "
            f"Ricevuto: {data_mode}"
        )

    if not data_dir:
        raise ValueError(
            f"Nessuna directory dati configurata per data_mode='{data_mode}'."
        )

    return Path(data_dir)


def _move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Sposta il batch sul device.

    Usa non_blocking=True solo quando ha senso per CUDA.
    """
    non_blocking = device.type == "cuda"

    return {
        key: value.to(
            device,
            non_blocking=non_blocking,
        )
        for key, value in batch.items()
    }


# ==============================================================================
# EVALUATION
# ==============================================================================

def run_evaluation_epoch(
    model: ConversationalGenerator,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, int]:
    """
    Calcola la loss media sull'internal validation split.

    Restituisce:
        (mean_loss, valid_steps)
    """

    if len(dataloader) == 0:
        return float("inf"), 0

    model.model.eval()

    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(
                batch,
                device,
            )

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            steps += 1

    if steps == 0:
        return float("inf"), 0

    return total_loss / steps, steps


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Training Pipeline RETECO SemEval-2027 "
            "Sub-track 2b"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default="subtrack_2b/config/config.yaml",
        help="Percorso del file config YAML.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Riprende dal last_model insieme a optimizer, "
            "scheduler e scaler."
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Esegue un training end-to-end minimo.",
    )

    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Addestra esclusivamente sul dominio specificato.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limita il numero di esempi di training.",
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

    config = load_yaml(cfg_path)

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

    training_cfg = config.get(
        "training",
        {},
    )

    # ==========================================================================
    # 2. SEED / RUN TAG
    # ==========================================================================

    seed = int(
        general_cfg.get(
            "seed",
            42,
        )
    )

    set_seed(seed)

    run_tag = str(
        general_cfg.get(
            "run_tag",
            "subtrack_2b_run",
        )
    )

    if args.smoke_test:
        run_tag += "_smoke_test"

    # ==========================================================================
    # 3. DIRECTORIES
    # ==========================================================================

    log_dir = ensure_dir(
        paths_cfg.get(
            "log_dir",
            "outputs/subtrack_2b/logs",
        )
    )

    checkpoint_dir = ensure_dir(
        paths_cfg.get(
            "checkpoint_dir",
            "checkpoints/subtrack_2b",
        )
    )

    output_dir = ensure_dir(
        paths_cfg.get(
            "output_dir",
            "outputs/subtrack_2b",
        )
    )

    logger = setup_logging(
        log_dir,
        run_tag=run_tag,
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
        "Run Tag: %s",
        run_tag,
    )
    logger.info(
        "=" * 90
    )

    logger.info(
        "Config utilizzato: %s",
        cfg_path,
    )

    # Salva una copia JSON della config effettivamente utilizzata.
    save_json(
        config,
        checkpoint_dir / "training_config.json",
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
        "Dispositivo selezionato: %s",
        device_name.upper(),
    )

    # ==========================================================================
    # 5. DATASET PATHS
    # ==========================================================================

    data_dir = _resolve_data_dir(
        paths_cfg
    )

    train_file = paths_cfg.get(
        "train_file",
        "benchmark_train.json",
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
        paths_cfg.get(
            "data_mode",
            "full",
        ),
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

    logger.info(
        "Caricamento dataset di training e risoluzione gold evidence..."
    )

    raw_samples, diagnostics = load_all_domains_data(
        base_dir=data_dir,
        active_domains=active_domains,
        all_domains=all_domains,
        split_filename=train_file,
        documents_filename=paths_cfg.get(
            "documents_file",
            "documents.jsonl",
        ),
    )

    if not raw_samples:
        logger.error(
            "Nessun dato caricato."
        )

        logger.error(
            "Verificare:"
        )

        logger.error(
            "  data_dir=%s",
            data_dir,
        )

        logger.error(
            "  train_file=%s",
            train_file,
        )

        logger.error(
            "  active_domains=%s",
            active_domains,
        )

        sys.exit(1)

    logger.info(
        "Campioni totali caricati: %d",
        len(raw_samples),
    )

    logger.info(
        "Domini caricati: %s",
        diagnostics.get(
            "domains_loaded",
            [],
        ),
    )

    missing_ids_count = diagnostics.get(
        "missing_gold_ids_count",
        0,
    )

    missing_examples = diagnostics.get(
        "examples_with_missing_evidence",
        0,
    )

    if missing_ids_count > 0:
        logger.warning(
            "Gold evidence: %d document IDs mancanti in %d esempi.",
            missing_ids_count,
            missing_examples,
        )

    examples_without_evidence = diagnostics.get(
        "examples_without_evidence",
        0,
    )

    if examples_without_evidence > 0:
        logger.warning(
            "Esempi senza evidence risolta: %d",
            examples_without_evidence,
        )

    empty_answers = diagnostics.get(
        "empty_answer_count",
        0,
    )

    if empty_answers > 0:
        logger.warning(
            "Esempi con answer vuota: %d",
            empty_answers,
        )

    # ==========================================================================
    # 7. INTERNAL VALIDATION SPLIT
    # ==========================================================================

    val_ratio = float(
        training_cfg.get(
            "internal_val_ratio",
            0.1,
        )
    )

    if args.smoke_test:
        val_ratio = 0.2

    train_samples, internal_val_samples = (
        split_conversations_train_val(
            raw_samples,
            val_ratio=val_ratio,
            seed=seed,
        )
    )

    logger.info(
        "Conversation-level split completato:"
    )

    logger.info(
        "  Train: %d turni",
        len(train_samples),
    )

    logger.info(
        "  Internal validation: %d turni",
        len(internal_val_samples),
    )

    if not internal_val_samples:
        logger.warning(
            "Internal validation vuota. "
            "Il best checkpoint verrà gestito senza "
            "valutazione valida."
        )

    # ==========================================================================
    # 8. DOMAIN BALANCING
    # ==========================================================================

    if (
        training_cfg.get(
            "domain_balanced_sampling",
            True,
        )
        and not args.smoke_test
        and len({
            sample.domain
            for sample in train_samples
        }) > 1
    ):
        temperature = float(
            training_cfg.get(
                "domain_sampling_temperature",
                0.5,
            )
        )

        train_samples = apply_domain_balanced_sampling(
            train_samples,
            temperature=temperature,
            seed=seed,
        )

        logger.info(
            "Domain-balanced sampling applicato "
            "(temperature=%.3f).",
            temperature,
        )

    # ==========================================================================
    # 9. SAMPLE LIMITS / SMOKE TEST
    # ==========================================================================

    if args.smoke_test:
        train_samples = train_samples[:8]
        internal_val_samples = internal_val_samples[:4]

        logger.info(
            "SMOKE TEST: %d train / %d val",
            len(train_samples),
            len(internal_val_samples),
        )

    elif args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError(
                "--max-samples deve essere > 0."
            )

        train_samples = train_samples[
            :args.max_samples
        ]

        logger.info(
            "Training limitato a %d esempi.",
            len(train_samples),
        )

    if not train_samples:
        raise RuntimeError(
            "Il training dataset è vuoto dopo "
            "split/filtri/smoke-test."
        )

    # ==========================================================================
    # 10. MODEL
    # ==========================================================================

    logger.info(
        "Caricamento modello e tokenizer..."
    )

    generator = ConversationalGenerator.from_pretrained(
        config,
        device=device,
        is_training=True,
    )

    trainable_params, total_params = (
        generator.get_nb_trainable_parameters()
    )

    logger.info(
        "Parametri trainable: %s / %s (%.2f%%)",
        f"{trainable_params:,}",
        f"{total_params:,}",
        (
            100.0
            * trainable_params
            / max(total_params, 1)
        ),
    )

    # ==========================================================================
    # 11. DATASET CONFIG
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

    ablation = prompt_cfg.get(
        "ablation",
        "D",
    )

    include_domain = bool(
        prompt_cfg.get(
            "include_domain",
            True,
        )
    )

    use_subquestion_reasoning = bool(
        prompt_cfg.get(
            "use_subquestion_reasoning",
            False,
        )
    )

    # ==========================================================================
    # 12. TRAIN DATASET
    # ==========================================================================

    train_ds = GroundedGenerationDataset(
        samples=train_samples,
        tokenizer=generator.tokenizer,
        max_input_tokens=(
            512
            if args.smoke_test
            else max_input_tokens
        ),
        max_output_tokens=(
            128
            if args.smoke_test
            else max_output_tokens
        ),
        max_history_turns=max_history_turns,
        ablation=ablation,
        include_domain=include_domain,
        use_subquestion_reasoning=(
            use_subquestion_reasoning
        ),
        is_training=True,
    )

    # ==========================================================================
    # 13. INTERNAL VALIDATION DATASET
    # ==========================================================================

    val_ds = GroundedGenerationDataset(
        samples=internal_val_samples,
        tokenizer=generator.tokenizer,
        max_input_tokens=(
            512
            if args.smoke_test
            else max_input_tokens
        ),
        max_output_tokens=(
            128
            if args.smoke_test
            else max_output_tokens
        ),
        max_history_turns=max_history_turns,
        ablation=ablation,
        include_domain=include_domain,
        use_subquestion_reasoning=(
            use_subquestion_reasoning
        ),
        is_training=True,
    )

    # ==========================================================================
    # 14. COLLATOR
    # ==========================================================================

    pad_token_id = generator.tokenizer.pad_token_id

    if pad_token_id is None:
        raise ValueError(
            "Tokenizer privo di pad_token_id."
        )

    collator = GroundedDataCollator(
        pad_token_id=pad_token_id
    )

    # ==========================================================================
    # 15. DATALOADERS
    # ==========================================================================

    train_batch_size = int(
        training_cfg.get(
            "train_batch_size",
            1,
        )
    )

    eval_batch_size = int(
        training_cfg.get(
            "eval_batch_size",
            1,
        )
    )

    if train_batch_size <= 0:
        raise ValueError(
            "train_batch_size deve essere > 0."
        )

    if eval_batch_size <= 0:
        raise ValueError(
            "eval_batch_size deve essere > 0."
        )

    num_workers = 0 if args.smoke_test else int(
        data_cfg.get("num_workers", 0)
    )

    pin_memory_cfg = bool(
        data_cfg.get(
            "pin_memory",
            False,
        )
    )

    # pin_memory è utile principalmente su CUDA.
    pin_memory = (
        pin_memory_cfg
        and device.type == "cuda"
    )

    # persistent_workers richiede num_workers > 0.
    persistent_workers = (
        num_workers > 0
    )

    logger.info(
        "DataLoader: batch=%d/%d | workers=%d | pin_memory=%s",
        train_batch_size,
        eval_batch_size,
        num_workers,
        pin_memory,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collator,
    )

    # ==========================================================================
    # 16. TRAINING HYPERPARAMETERS
    # ==========================================================================

    epochs = int(
        training_cfg.get(
            "epochs",
            2,
        )
    )

    if args.smoke_test:
        epochs = 1

    learning_rate = float(
        training_cfg.get(
            "learning_rate",
            2e-4,
        )
    )

    weight_decay = float(
        training_cfg.get(
            "weight_decay",
            0.01,
        )
    )

    grad_accum_steps = int(
        training_cfg.get(
            "gradient_accumulation_steps",
            8,
        )
    )

    if args.smoke_test:
        grad_accum_steps = 1

    grad_accum_steps = max(
        1,
        grad_accum_steps,
    )

    max_grad_norm = float(
        training_cfg.get(
            "max_grad_norm",
            1.0,
        )
    )

    warmup_ratio = float(
        training_cfg.get(
            "warmup_ratio",
            0.05,
        )
    )

    early_stopping_patience = int(
        training_cfg.get(
            "early_stopping_patience",
            2,
        )
    )

    # ==========================================================================
    # 17. OPTIMIZER
    # ==========================================================================

    no_decay_terms = [
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
        "norm.weight",
    ]

    optimizer_grouped_parameters = [
        {
            "params": [
                param
                for name, param
                in generator.model.named_parameters()
                if (
                    param.requires_grad
                    and not any(
                        term in name
                        for term in no_decay_terms
                    )
                )
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                param
                for name, param
                in generator.model.named_parameters()
                if (
                    param.requires_grad
                    and any(
                        term in name
                        for term in no_decay_terms
                    )
                )
            ],
            "weight_decay": 0.0,
        },
    ]

    if not any(
        group["params"]
        for group in optimizer_grouped_parameters
    ):
        raise RuntimeError(
            "Nessun parametro trainable trovato. "
            "Controllare LoRA/PEFT configuration."
        )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
    )

    # ==========================================================================
    # 18. SCHEDULER
    # ==========================================================================

    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / grad_accum_steps
    )

    total_training_steps = (
        optimizer_steps_per_epoch
        * epochs
    )

    warmup_steps = int(
        total_training_steps
        * warmup_ratio
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    logger.info(
        "Training steps totali: %d | warmup: %d | optimizer steps/epoch: %d",
        total_training_steps,
        warmup_steps,
        optimizer_steps_per_epoch,
    )

    # ==========================================================================
    # 19. MIXED PRECISION
    # ==========================================================================

    use_fp16 = bool(
        training_cfg.get(
            "fp16",
            True,
        )
    )

    use_bf16 = bool(
        training_cfg.get(
            "bf16",
            False,
        )
    )

    use_amp = (
        device.type == "cuda"
        and (use_fp16 or use_bf16)
    )

    amp_dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
    )

    use_scaler = (
        use_amp
        and amp_dtype == torch.float16
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_scaler,
    )

    logger.info(
        "AMP: enabled=%s | dtype=%s | GradScaler=%s",
        use_amp,
        str(amp_dtype),
        use_scaler,
    )

    # ==========================================================================
    # 20. CHECKPOINT DIRECTORIES
    # ==========================================================================

    best_dir = checkpoint_dir / "best_model"
    last_dir = checkpoint_dir / "last_model"
    tokenizer_dir = checkpoint_dir / "tokenizer"

    state_path = _resume_state_path(
        checkpoint_dir
    )

    # ==========================================================================
    # 21. RESUME
    # ==========================================================================

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if args.resume:
        if args.smoke_test:
            raise ValueError(
                "--resume e --smoke-test non possono essere usati insieme."
            )

        if not last_dir.exists():
            raise FileNotFoundError(
                f"Resume richiesto ma last_model non esiste: {last_dir}"
            )

        logger.info(
            "Resume richiesto: caricamento last_model..."
        )

        # Il modello viene caricato PRIMA della creazione dello optimizer
        # per fare in modo che l'optimizer punti ai parametri del checkpoint.
        _load_model_for_resume(
            generator,
            checkpoint_dir,
        )

        # Ricostruzione dello optimizer dopo il load del modello.
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        param
                        for name, param
                        in generator.model.named_parameters()
                        if (
                            param.requires_grad
                            and not any(
                                term in name
                                for term in no_decay_terms
                            )
                        )
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        param
                        for name, param
                        in generator.model.named_parameters()
                        if (
                            param.requires_grad
                            and any(
                                term in name
                                for term in no_decay_terms
                            )
                        )
                    ],
                    "weight_decay": 0.0,
                },
            ],
            lr=learning_rate,
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        state = _load_training_state(
            state_path,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location="cpu",
        )

        completed_epoch = int(
            state.get(
                "epoch",
                0,
            )
        )

        start_epoch = completed_epoch + 1

        global_step = int(
            state.get(
                "global_step",
                0,
            )
        )

        best_val_loss = float(
            state.get(
                "best_val_loss",
                float("inf"),
            )
        )

        logger.info(
            "Resume completato:"
        )

        logger.info(
            "  ultimo epoch completato: %d",
            completed_epoch,
        )

        logger.info(
            "  prossimo epoch: %d",
            start_epoch,
        )

        logger.info(
            "  global step: %d",
            global_step,
        )

        logger.info(
            "  best val loss: %.6f",
            best_val_loss,
        )

        if start_epoch > epochs:
            logger.info(
                "Il checkpoint contiene già tutte le epoche richieste. "
                "Nessun ulteriore training necessario."
            )

            generator.tokenizer.save_pretrained(
                tokenizer_dir
            )

            return

    # ==========================================================================
    # 22. TRAINING LOOP
    # ==========================================================================

    logger.info(
        "Avvio training SFT..."
    )

    start_time = time.time()

    for epoch in range(
        start_epoch,
        epochs + 1,
    ):
        generator.model.train()

        running_loss = 0.0
        epoch_steps = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        pbar = tqdm(
            train_loader,
            desc=(
                f"Epoca {epoch:02d}/{epochs:02d}"
            ),
        )

        num_batches = len(
            train_loader
        )

        remainder = (
            num_batches % grad_accum_steps
        )

        for step, batch in enumerate(
            pbar,
            start=1,
        ):
            batch = _move_batch_to_device(
                batch,
                device,
            )

            # Numero reale di micro-batch nel gruppo corrente.
            if (
                remainder != 0
                and step > num_batches - remainder
            ):
                current_accumulation = remainder
            else:
                current_accumulation = (
                    grad_accum_steps
                )

            if use_amp:
                with torch.amp.autocast(
                    "cuda",
                    dtype=amp_dtype,
                ):
                    outputs = generator(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )

                    raw_loss = outputs.loss
                    loss = (
                        raw_loss
                        / current_accumulation
                    )

                scaler.scale(
                    loss
                ).backward()

            else:
                outputs = generator(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )

                raw_loss = outputs.loss

                loss = (
                    raw_loss
                    / current_accumulation
                )

                loss.backward()

            raw_loss_value = raw_loss.item()

            running_loss += raw_loss_value
            epoch_steps += 1

            # --------------------------------------------------------------
            # Optimizer step
            # --------------------------------------------------------------
            is_accumulation_end = (
                step % grad_accum_steps == 0
                or step == num_batches
            )

            if is_accumulation_end:
                if use_amp:
                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        generator.model.parameters(),
                        max_grad_norm,
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:
                    torch.nn.utils.clip_grad_norm_(
                        generator.model.parameters(),
                        max_grad_norm,
                    )

                    optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

            current_avg_loss = (
                running_loss
                / max(
                    epoch_steps,
                    1,
                )
            )

            current_lr = (
                scheduler.get_last_lr()[0]
            )

            pbar.set_postfix(
                {
                    "loss": f"{current_avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

        # ==========================================================================
        # 23. INTERNAL VALIDATION
        # ==========================================================================

        train_loss = (
            running_loss
            / max(
                epoch_steps,
                1,
            )
        )

        if len(val_loader) > 0:
            val_loss, valid_val_steps = (
                run_evaluation_epoch(
                    generator,
                    val_loader,
                    device,
                )
            )

            if math.isfinite(val_loss):
                val_ppl = math.exp(
                    min(
                        val_loss,
                        20.0,
                    )
                )
            else:
                val_ppl = float("inf")

        else:
            val_loss = float("inf")
            valid_val_steps = 0
            val_ppl = float("inf")

        logger.info(
            "Epoca %02d/%02d completata | "
            "Train Loss: %.4f | "
            "Val Loss: %.4f | "
            "Val PPL: %.2f | "
            "Global Step: %d",
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_ppl,
            global_step,
        )

        # ==========================================================================
        # 24. BEST CHECKPOINT
        # ==========================================================================

        improved = (
            valid_val_steps > 0
            and val_loss < best_val_loss
        )

        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            logger.info(
                "Nuovo best checkpoint "
                "(Internal Val Loss: %.6f).",
                val_loss,
            )

            generator.save_checkpoint(
                best_dir
            )

        else:
            epochs_without_improvement += 1

            logger.info(
                "Nessun miglioramento della validation "
                "(patience: %d/%d).",
                epochs_without_improvement,
                early_stopping_patience,
            )

        # Se non abbiamo una validation valida, salva comunque il primo
        # checkpoint come best per rendere disponibile un modello.
        if (
            valid_val_steps == 0
            and not best_dir.exists()
        ):
            logger.warning(
                "Internal validation non disponibile: "
                "salvo l'epoch corrente come best_model."
            )

            generator.save_checkpoint(
                best_dir
            )

        # ==========================================================================
        # 25. LAST CHECKPOINT
        # ==========================================================================

        generator.save_checkpoint(
            last_dir
        )

        # ==========================================================================
        # 26. FULL TRAINING STATE
        # ==========================================================================

        _save_training_state(
            state_path,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
        )

        # ==========================================================================
        # 27. EARLY STOPPING
        # ==========================================================================

        if (
            early_stopping_patience > 0
            and epochs_without_improvement
            >= early_stopping_patience
            and valid_val_steps > 0
        ):
            logger.info(
                "Early stopping attivato dopo %d epoche senza miglioramento.",
                epochs_without_improvement,
            )
            break

    # ==========================================================================
    # 28. FINALIZATION
    # ==========================================================================

    total_time = (
        time.time()
        - start_time
    )

    tokenizer_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generator.tokenizer.save_pretrained(
        tokenizer_dir
    )

    # --------------------------------------------------------------------------
    # Training diagnostics
    # --------------------------------------------------------------------------

    training_summary: Dict[str, Any] = {
        "status": "completed",
        "run_tag": run_tag,
        "total_time_seconds": total_time,
        "epochs_requested": epochs,
        "last_completed_epoch": (
            epoch if "epoch" in locals() else 0
        ),
        "best_val_loss": best_val_loss,
        "train_samples": len(
            train_samples
        ),
        "val_samples": len(
            internal_val_samples
        ),
        "device": device_name,
        "model_name": generator.model_name,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "ablation": ablation,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "max_history_turns": max_history_turns,
        "gradient_accumulation_steps": grad_accum_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "domain_balanced_sampling": bool(
            training_cfg.get(
                "domain_balanced_sampling",
                True,
            )
        ),
        "data_mode": paths_cfg.get(
            "data_mode",
            "full",
        ),
        "domains_loaded": diagnostics.get(
            "domains_loaded",
            [],
        ),
        "missing_gold_ids_count": diagnostics.get(
            "missing_gold_ids_count",
            0,
        ),
        "examples_with_missing_evidence": diagnostics.get(
            "examples_with_missing_evidence",
            0,
        ),
        "examples_without_evidence": diagnostics.get(
            "examples_without_evidence",
            0,
        ),
        "empty_answer_count": diagnostics.get(
            "empty_answer_count",
            0,
        ),
        "resume": args.resume,
    }

    save_json(
        training_summary,
        checkpoint_dir / "training_summary.json",
    )

    logger.info(
        "=" * 90
    )

    logger.info(
        "Training terminato."
    )

    logger.info(
        "Tempo totale: %.1f secondi (%.2f minuti)",
        total_time,
        total_time / 60.0,
    )

    logger.info(
        "Best validation loss: %.6f",
        best_val_loss,
    )

    logger.info(
        "Best checkpoint: %s",
        best_dir,
    )

    logger.info(
        "Last checkpoint: %s",
        last_dir,
    )

    logger.info(
        "Training state: %s",
        state_path,
    )

    logger.info(
        "Output directory: %s",
        output_dir,
    )

    logger.info(
        "=" * 90
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()