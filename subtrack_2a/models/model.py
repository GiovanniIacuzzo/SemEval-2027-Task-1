#!/usr/bin/env python3
"""
subtrack_2a/models/model.py

Modelli neurali per RETECO Sub-track 2a (Conversational Retrieval).
Supporta:
  - Architettura Bi-Encoder con contrastive loss InfoNCE (in-batch ed hard negatives).
  - Parameter-Efficient Fine-Tuning tramite LoRA e QLoRA a 4-bit (PEFT).
  - Rilevamento automatico dei target modules per modelli Encoder (BGE/BERT) e Decoder (Qwen/Llama).
  - Fallback sicuro: se QLoRA 4-bit è richiesto su macOS (MPS/CPU), passa automaticamente
    a LoRA standard senza mandare in crash l'esecuzione.
  - Cross-Encoder per il re-ranking fine della lista dei candidati.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

logger = logging.getLogger("RETECO_Models")

# Verifica disponibilità moduli PEFT e Quantizzazione
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


# ==============================================================================
# 1. Pooling Layers
# ==============================================================================

class MeanPooling(nn.Module):
    """Esegue il pooling pesato sulla attention mask ignorando i token di padding."""

    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


# ==============================================================================
# 2. Conversational Bi-Encoder (Full / LoRA / QLoRA)
# ==============================================================================

class ConversationalBiEncoder(nn.Module):
    """
    Bi-Encoder neurale per dense retrieval conversazionale.
    Compatibile con BGE, RoBERTa e modelli di classe Qwen / LLaMA.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-base-en-v1.5",
        temperature: float = 0.05,
        normalize_embeddings: bool = True,
        pooling_strategy: str = "mean",
        lora_cfg: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.pooling_strategy = pooling_strategy
        self.lora_cfg = lora_cfg or {}
        self.is_peft = False

        target_device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        self.config = AutoConfig.from_pretrained(model_name_or_path)

        # -------------------------------------------------------------
        # Configurazione Quantizzazione 4-bit (QLoRA)
        # -------------------------------------------------------------
        lora_enabled = self.lora_cfg.get("enabled", False)
        use_qlora_4bit = self.lora_cfg.get("use_qlora_4bit", False)
        bnb_config = None

        if lora_enabled and use_qlora_4bit:
            if target_device.type != "cuda":
                logger.warning(
                    "[QLoRA Warning] La quantizzazione a 4-bit con bitsandbytes è supportata solo su GPU NVIDIA (CUDA). "
                    "Rilevato device non-CUDA (%s): fallback automatico a LoRA 16-bit/FP32.",
                    target_device.type
                )
            elif not BNB_AVAILABLE:
                logger.warning("[QLoRA Warning] 'bitsandbytes' non è installato. Procedo con LoRA standard senza quantizzazione.")
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("✓ Quantizzazione QLoRA 4-bit (NF4) abilitata per %s", model_name_or_path)

        # Caricamento del modello base
        if bnb_config is not None:
            self.encoder = AutoModel.from_pretrained(
                model_name_or_path,
                config=self.config,
                quantization_config=bnb_config,
                device_map={"": target_device},
            )
        else:
            self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        # -------------------------------------------------------------
        # Inizializzazione Adapter LoRA (PEFT)
        # -------------------------------------------------------------
        if lora_enabled:
            if not PEFT_AVAILABLE:
                raise ImportError("Per usare LoRA/QLoRA è necessario installare peft: pip install peft")

            if bnb_config is not None:
                self.encoder = prepare_model_for_kbit_training(self.encoder)

            target_modules = self._resolve_target_modules(self.lora_cfg.get("target_modules"))
            r = int(self.lora_cfg.get("r", 16))
            lora_alpha = int(self.lora_cfg.get("lora_alpha", 32))
            lora_dropout = float(self.lora_cfg.get("lora_dropout", 0.05))

            peft_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )

            self.encoder = get_peft_model(self.encoder, peft_config)
            self.is_peft = True

            logger.info("✓ Modulo LoRA agganciato con successo:")
            logger.info(f"   - Rank (r): {r} | Alpha: {lora_alpha} | Dropout: {lora_dropout}")
            logger.info(f"   - Target Modules: {target_modules}")
            self.encoder.print_trainable_parameters()

        self.pooler = MeanPooling() if pooling_strategy == "mean" else None

    def _resolve_target_modules(self, configured_modules: Optional[List[str]]) -> List[str]:
        """Rileva automaticamente i layer lineari corretti per BGE/BERT o Qwen/Llama."""
        if configured_modules and configured_modules != "auto":
            return configured_modules

        # Rilevamento basato sui layer presenti nel modello caricato
        module_names = set()
        for name, _ in self.encoder.named_modules():
            parts = name.split(".")
            module_names.add(parts[-1])

        # Se architettura stile Qwen / LLaMA
        if "q_proj" in module_names or "k_proj" in module_names:
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        
        # Se architettura stile BERT / RoBERTa / BGE
        if "query" in module_names or "key" in module_names:
            return ["query", "key", "value", "dense"]

        # Default fallback generico per modelli lineari di attenzione
        return ["query", "value"]

    def _pool(self, model_output: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "cls":
            return model_output[0][:, 0]
        # Se decoder-only (come Qwen), usiamo l'ultimo hidden state con mask
        return self.pooler(model_output[0], attention_mask)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Codifica un batch di testi in vettori densi normalizzati (L2)."""
        extra_args = {}
        # Alcuni modelli (come Qwen o RoBERTa) non accettano token_type_ids
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            extra_args["token_type_ids"] = token_type_ids

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **extra_args)
        embeddings = self._pool(outputs, attention_mask)

        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def compute_similarity(self, query_embs: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """Calcola la similarità dot-product tra matrici di embedding normalizzate."""
        return torch.matmul(query_embs, doc_embs.transpose(-2, -1))

    def forward(
        self,
        query_inputs: Dict[str, torch.Tensor],
        pos_inputs: Dict[str, torch.Tensor],
        neg_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass con InfoNCE Loss per in-batch e hard negatives."""
        query_embs = self.encode(
            input_ids=query_inputs["input_ids"],
            attention_mask=query_inputs["attention_mask"],
            token_type_ids=query_inputs.get("token_type_ids"),
        )
        pos_embs = self.encode(
            input_ids=pos_inputs["input_ids"],
            attention_mask=pos_inputs["attention_mask"],
            token_type_ids=pos_inputs.get("token_type_ids"),
        )

        batch_size = query_embs.size(0)

        if neg_inputs is None:
            scores = self.compute_similarity(query_embs, pos_embs) / self.temperature
            labels = torch.arange(batch_size, device=query_embs.device, dtype=torch.long)
            loss = F.cross_entropy(scores, labels)
            return {"loss": loss, "scores": scores, "query_embs": query_embs, "pos_embs": pos_embs}

        neg_embs = self.encode(
            input_ids=neg_inputs["input_ids"],
            attention_mask=neg_inputs["attention_mask"],
            token_type_ids=neg_inputs.get("token_type_ids"),
        )

        all_doc_embs = torch.cat([pos_embs, neg_embs], dim=0)
        all_scores = self.compute_similarity(query_embs, all_doc_embs) / self.temperature

        labels = torch.arange(batch_size, device=query_embs.device, dtype=torch.long)
        loss = F.cross_entropy(all_scores, labels)

        return {
            "loss": loss,
            "scores": all_scores,
            "query_embs": query_embs,
            "pos_embs": pos_embs,
            "neg_embs": neg_embs,
        }

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        """
        Salva i pesi in modo compatto.
        Se LoRA è attivo, salva solo l'adapter (~15-30 MB) e la configurazione.
        """
        os.makedirs(save_directory, exist_ok=True)
        if self.is_peft:
            self.encoder.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)
        else:
            self.encoder.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)
            self.config.save_pretrained(save_directory)


# ==============================================================================
# 3. Conversational Cross-Encoder
# ==============================================================================

class ConversationalCrossEncoder(nn.Module):
    """Cross-Encoder per il re-ranking della lista dei migliori candidati."""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-base",
        num_labels: int = 1,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        extra_args = {}
        if token_type_ids is not None and "token_type_ids" in self.model.forward.__code__.co_varnames:
            extra_args["token_type_ids"] = token_type_ids

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            **extra_args,
        )

        logits = outputs.logits.squeeze(-1) if self.num_labels == 1 else outputs.logits
        output_dict = {"logits": logits}

        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss() if self.num_labels == 1 else nn.CrossEntropyLoss()
            target = labels.float() if self.num_labels == 1 else labels.long()
            output_dict["loss"] = loss_fn(logits, target)

        return output_dict

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory, safe_serialization=safe_serialization, **kwargs)
        self.config.save_pretrained(save_directory)