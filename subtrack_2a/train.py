#!/usr/bin/env python3
"""
subtrack_2a/train.py

Pipeline di addestramento modulare per RETECO SemEval-2027 Sub-track 2a (Conversational Retrieval).
Include:
  - Funzione dedicata train_one_epoch con barra di avanzamento tqdm dettagliata.
  - Funzione dedicata validate_one_epoch per la valutazione periodica.
  - Monitoraggio in tempo reale: Loss di batch, Running Loss, Learning Rate e VRAM.
  - Cosine Annealing Learning Rate Scheduler con Warmup.
  - Automatic Mixed Precision (AMP FP16) con GradScaler.
  - Checkpointing automatico (PyTorch State Dict + formato Hugging Face).
"""

import os
import sys
import time
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm

# Import dei componenti modulari interni
from dataset.dataset import (
    ConversationalTurnSample,
    RETECO2aTrainDataset,
    load_track2_domain_data,
    split_conversations_train_val,
    TRACK2_DOMAINS,
)
from models.model import ConversationalBiEncoder
from utils.utils import (
    load_config,
    save_json,
    plot_training_history,
    setup_logger,
    set_seed,
)


# ==============================================================================
# 1. Collate Function con Tokenizzazione Dinamica a Batch
# ==============================================================================

class ConversationalCollateFn:
    """Tokenizza a batch dinamici le query contestualizzate, i positivi e i negativi."""

    def __init__(self, tokenizer: AutoTokenizer, max_query_len: int = 256, max_doc_len: int = 512):
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, Dict[str, torch.Tensor]]:
        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]
        negatives = [item["negative"] for item in batch] if "negative" in batch[0] else None

        q_tok = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_len,
            return_tensors="pt",
        )
        pos_tok = self.tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=self.max_doc_len,
            return_tensors="pt",
        )

        batch_dict = {
            "query_inputs": q_tok,
            "pos_inputs": pos_tok,
        }

        if negatives is not None:
            neg_tok = self.tokenizer(
                negatives,
                padding=True,
                truncation=True,
                max_length=self.max_doc_len,
                return_tensors="pt",
            )
            batch_dict["neg_inputs"] = neg_tok

        return batch_dict


# ==============================================================================
# 2. Ottimizzazione e Parametri Grouped
# ==============================================================================

def build_optimizer_grouped_parameters(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Esclude bias e LayerNorm dal decadimento del peso per evitare instabilità."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    return [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
            "lr": learning_rate,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]


# ==============================================================================
# 3. Metodi di Esecuzione Epoca (Train & Val)
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = False,
    global_step: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, int]:
    """
    Esegue l'addestramento per una singola epoca con barra di avanzamento tqdm interattiva.
    """
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    # Barra di avanzamento per l'epoca corrente
    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]",
        dynamic_ncols=True,
        leave=True,
    )

    for step, batch in enumerate(progress_bar, start=1):
        # Spostamento dei tensori sul device designato
        query_inputs = {k: v.to(device) for k, v in batch["query_inputs"].items()}
        pos_inputs = {k: v.to(device) for k, v in batch["pos_inputs"].items()}
        neg_inputs = (
            {k: v.to(device) for k, v in batch["neg_inputs"].items()}
            if "neg_inputs" in batch
            else None
        )

        # Forward pass in precisione mista (AMP)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(query_inputs=query_inputs, pos_inputs=pos_inputs, neg_inputs=neg_inputs)
            loss = outputs["loss"] / grad_accum_steps

        # Backward pass con GradScaler
        scaler.scale(loss).backward()
        batch_loss_val = loss.item() * grad_accum_steps
        running_loss += batch_loss_val

        # Aggiornamento dei pesi a intervalli di accumulation
        if step % grad_accum_steps == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # Metriche live per la barra di avanzamento
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = running_loss / step

        postfix_metrics = {
            "loss": f"{batch_loss_val:.4f}",
            "avg_loss": f"{avg_loss:.4f}",
            "lr": f"{current_lr:.2e}",
        }

        if device.type == "cuda":
            vram_gb = torch.cuda.memory_allocated() / 1e9
            postfix_metrics["vram"] = f"{vram_gb:.2f}GB"

        progress_bar.set_postfix(postfix_metrics)

    epoch_avg_loss = running_loss / max(1, len(train_loader))
    return epoch_avg_loss, global_step


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    use_amp: bool = False,
) -> float:
    """
    Esegue la validazione sull'held-out validation split per l'epoca corrente.
    """
    model.eval()
    running_val_loss = 0.0

    progress_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Valid]",
        dynamic_ncols=True,
        leave=False,
    )

    with torch.no_grad():
        for step, batch in enumerate(progress_bar, start=1):
            query_inputs = {k: v.to(device) for k, v in batch["query_inputs"].items()}
            pos_inputs = {k: v.to(device) for k, v in batch["pos_inputs"].items()}
            neg_inputs = (
                {k: v.to(device) for k, v in batch["neg_inputs"].items()}
                if "neg_inputs" in batch
                else None
            )

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                val_out = model(query_inputs=query_inputs, pos_inputs=pos_inputs, neg_inputs=neg_inputs)
                loss = val_out["loss"]
                running_val_loss += loss.item()

            progress_bar.set_postfix({"val_loss": f"{running_val_loss / step:.4f}"})

    return running_val_loss / max(1, len(val_loader))


# ==============================================================================
# 4. Pipeline Principale di Addestramento
# ==============================================================================

def run_training(config: Dict[str, Any], logger: logging.Logger) -> None:
    # ---------------------------------------------------------
    # Setup Hardware e Riproducibilità
    # ---------------------------------------------------------
    gen_cfg = config.get("general", {})
    paths_cfg = config.get("paths", {})
    dom_cfg = config.get("domains", {})
    data_cfg = config.get("data", {})
    bi_cfg = config.get("bi_encoder", {})
    train_cfg = config.get("training", {})

    seed = gen_cfg.get("seed", 42)
    set_seed(seed)

    device_pref = gen_cfg.get("device", "auto")
    if device_pref == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_pref)

    logger.info(f"Dispositivo di calcolo in uso: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM Totale: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    use_amp = bool(gen_cfg.get("mixed_precision", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info(f"Mixed Precision FP16 nativo: {use_amp}")

    # ---------------------------------------------------------
    # Caricamento e Preparazione Dati
    # ---------------------------------------------------------
    data_mode = paths_cfg.get("data_mode", "sample")
    base_data_dir = Path(paths_cfg.get("sample_data_dir" if data_mode == "sample" else "full_data_dir"))
    logger.info(f"Modalità dati: '{data_mode}' | Percorso base: {base_data_dir}")

    active_domains_cfg = dom_cfg.get("active_domains", "all")
    domains_to_load = TRACK2_DOMAINS if active_domains_cfg == "all" else active_domains_cfg
    logger.info(f"Domini da addestrare ({len(domains_to_load)}): {domains_to_load}")

    all_train_samples: List[ConversationalTurnSample] = []
    combined_corpus: Dict[str, str] = {}

    for domain in domains_to_load:
        try:
            corpus, samples, _ = load_track2_domain_data(
                data_dir=base_data_dir,
                domain=domain,
                split="train",
                query_strategy=data_cfg.get("query_strategy", "concat"),
            )
            for doc_id, text in corpus.items():
                combined_corpus[f"{domain}_{doc_id}"] = text
            for s in samples:
                s.gold_doc_ids = [f"{domain}_{gid}" for gid in s.gold_doc_ids]
                all_train_samples.append(s)

            logger.info(f"  [{domain:<18}] Corpus: {len(corpus):>6} docs | Turni: {len(samples):>4}")
        except Exception as e:
            logger.error(f"Errore caricamento dominio {domain}: {e}")

    if not all_train_samples:
        logger.error("Nessun dato caricato. Interruzione.")
        sys.exit(1)

    # Splitting a livello di conversazione (garantisce zero data leakage)
    train_samples, val_samples = split_conversations_train_val(all_train_samples, val_ratio=0.15, seed=seed)
    logger.info(f"Campioni suddivisi: Train {len(train_samples)} | Val {len(val_samples)}")

    train_dataset = RETECO2aTrainDataset(
        samples=train_samples,
        corpus=combined_corpus,
        use_triplets=data_cfg.get("use_triplets", True),
        negatives_per_positive=data_cfg.get("negatives_per_positive", 1),
    )
    val_dataset = RETECO2aTrainDataset(
        samples=val_samples,
        corpus=combined_corpus,
        use_triplets=data_cfg.get("use_triplets", True),
        negatives_per_positive=1,
    )
    logger.info(f"Istanze PyTorch create -> Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ---------------------------------------------------------
    # Tokenizer, DataLoaders e Modello
    # ---------------------------------------------------------
    model_name = bi_cfg.get("model_name_or_path", "BAAI/bge-base-en-v1.5")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    collate_fn = ConversationalCollateFn(
        tokenizer=tokenizer,
        max_query_len=data_cfg.get("max_query_length", 256),
        max_doc_len=data_cfg.get("max_doc_length", 512),
    )

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 0)
    pin_mem = bool(data_cfg.get("pin_memory", True) and device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    logger.info(f"Inizializzazione Modello Bi-Encoder: {model_name}")
    model = ConversationalBiEncoder(
        model_name_or_path=model_name,
        temperature=bi_cfg.get("temperature", 0.05),
        normalize_embeddings=bi_cfg.get("normalize_embeddings", True),
        pooling_strategy=bi_cfg.get("pooling_strategy", "mean"),
    ).to(device)

    # ---------------------------------------------------------
    # Ottimizzatore e Cosine Scheduler
    # ---------------------------------------------------------
    lr = float(train_cfg.get("learning_rate", 2.0e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    epochs = int(train_cfg.get("epochs", 5))
    grad_accum_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    optimizer_grouped_params = build_optimizer_grouped_parameters(model, lr, weight_decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_params, eps=1e-8)

    total_update_steps = math.ceil(len(train_loader) / grad_accum_steps) * epochs
    warmup_steps = int(total_update_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    logger.info(f"Configurazione Scheduler: Warmup Steps {warmup_steps} | Update Steps Totali {total_update_steps}")

    # ---------------------------------------------------------
    # Ciclo Principale di Addestramento
    # ---------------------------------------------------------
    checkpoint_dir = Path(paths_cfg.get("checkpoint_dir", "checkpoints/subtrack_2a"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / "subtrack_2a_best.pt"
    last_model_path = checkpoint_dir / "subtrack_2a_last.pt"

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience = int(train_cfg.get("early_stopping_patience", 2))
    patience_counter = 0
    global_step = 0

    logger.info("\n" + "=" * 70)
    logger.info("INIZIO ADDESTRAMENTO NEURALE")
    logger.info("=" * 70)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        # 1. Addestramento con barra tqdm
        train_loss, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            grad_accum_steps=grad_accum_steps,
            max_grad_norm=max_grad_norm,
            use_amp=use_amp,
            global_step=global_step,
            logger=logger,
        )
        history["train_loss"].append(train_loss)

        # 2. Validazione
        val_loss = validate_one_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            use_amp=use_amp,
        )
        history["val_loss"].append(val_loss)

        epoch_duration = time.time() - epoch_start_time
        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] completata in {epoch_duration:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # 3. Checkpointing & Early Stopping
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": config,
            },
            last_model_path,
        )

        if val_loss < best_val_loss:
            diff = best_val_loss - val_loss
            best_val_loss = val_loss
            patience_counter = 0

            hf_export_dir = checkpoint_dir / "best_hf_model"
            hf_export_dir.mkdir(parents=True, exist_ok=True)
            
            model.save_pretrained(str(hf_export_dir), safe_serialization=True)
            tokenizer.save_pretrained(str(hf_export_dir))

            checkpoint_meta = {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "timestamp": datetime.now().isoformat(),
                "config_run_tag": gen_cfg.get("run_tag", "2a_run")
            }
            save_json(checkpoint_meta, checkpoint_dir / "best_meta.json")

            logger.info(f"  Nuovo Record Val Loss: {best_val_loss:.4f} (-{diff:.4f}) -> Modello HF salvato in {hf_export_dir}")
        else:
            patience_counter += 1
            logger.info(f"Pazienza early stopping: {patience - patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info(f"  Early stopping attivato all'epoca {epoch}.")
                break

    # ---------------------------------------------------------
    # Finalizzazione ed Esportazione Metriche
    # ---------------------------------------------------------
    output_dir = Path(paths_cfg.get("output_dir", "outputs/subtrack_2a"))
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "epochs_completed": epoch,
        "best_val_loss": best_val_loss,
        "history": history,
        "completed_at": datetime.now().isoformat(),
    }
    save_json(summary_payload, output_dir / "training_history.json")

    curve_path = output_dir / "loss_curves.png"
    plot_training_history(history, curve_path, title=f"Training Loss - {gen_cfg.get('run_tag', 'Model')}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE DI TRAINING TERMINATA")
    logger.info(f"Miglior Val Loss raggiunta: {best_val_loss:.4f}")
    logger.info(f"Grafico salvato in: {curve_path}")
    logger.info("=" * 70 + "\n")


# ==============================================================================
# 5. Entrypoint CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Training Pipeline per SemEval-2027 Task 1 (RETECO) Sub-track 2a")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Percorso al file config.yaml",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        alt_path = Path(__file__).resolve().parent / "config" / "config.yaml"
        if alt_path.exists():
            cfg_path = alt_path
        else:
            raise FileNotFoundError(f"Config YAML non trovato in {args.config} o {alt_path}")

    config = load_config(cfg_path)

    paths_cfg = config.get("paths", {})
    gen_cfg = config.get("general", {})
    log_dir = Path(paths_cfg.get("log_dir", "outputs/subtrack_2a/logs"))
    run_tag = gen_cfg.get("run_tag", "2a_run")
    log_level = gen_cfg.get("logging_level", "INFO")

    logger = setup_logger(log_dir=log_dir, run_tag=run_tag, log_level=log_level)
    logger.info(f"Configurazione caricata da: {cfg_path}")

    try:
        run_training(config, logger)
    except KeyboardInterrupt:
        logger.warning("Addestramento interrotto manualmente.")
    except Exception as e:
        logger.exception(f"Errore critico durante il training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()