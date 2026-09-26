#!/usr/bin/env python3
"""
subtrack_2a/upload_to_hf.py

Script professionale per il caricamento controllato del checkpoint addestrato su Hugging Face Hub.
Funzionalità:
  - Lettura prioritaria dei parametri da config/config.yaml con override da CLI.
  - Verifica automatica dell'autenticazione utente (HF_TOKEN o huggingface-cli login).
  - Convalida dell'integrità dei pesi (controllo safe_serialization model.safetensors).
  - Arricchimento del Model Card (README.md) integrando i metadati di training (best_val_loss, run_tag).
  - Upload asincrono e sicuro della cartella escludendo file spuri (.DS_Store, log).
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from huggingface_hub import HfApi, create_repo
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("[ERRORE] 'huggingface_hub' non è installato. Esegui: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)

from utils.utils import load_config, setup_logger


# ==============================================================================
# 1. Generatore Model Card Strutturato
# ==============================================================================

def build_model_card(
    repo_id: str,
    base_model: str,
    meta_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Genera la scheda documentale del modello in formato Markdown conforme allo standard HF."""
    best_loss_str = "N/A"
    epoch_str = "N/A"
    run_tag = "RETECO-2a"

    if meta_info:
        best_loss_str = f"{meta_info.get('best_val_loss', 'N/A'):.4f}" if isinstance(meta_info.get('best_val_loss'), float) else "N/A"
        epoch_str = str(meta_info.get("epoch", "N/A"))
        run_tag = meta_info.get("config_run_tag", run_tag)

    return f"""---
            language:
            - en
            license: cc-by-sa-4.0
            tags:
            - semeval-2027
            - information-retrieval
            - conversational-retrieval
            - dense-retrieval
            - recor
            - text-embeddings
            pipeline_tag: feature-extraction
            ---
            """

