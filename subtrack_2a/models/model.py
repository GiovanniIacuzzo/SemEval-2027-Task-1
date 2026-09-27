#!/usr/bin/env python3
"""
subtrack_2a/models/model.py

Modelli neurali per RETECO Sub-track 2a:
  - ConversationalBiEncoder: BGE Base con CLS o Mean Pooling, LoRA e InfoNCE multi-negativo.
  - ConversationalCrossEncoder: BGE Reranker con supporto LoRA e fine-tuning supervisionato.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification


class DensePooling(nn.Module):
    """Pooling configurabile: 'mean' (mascherato) o 'cls'."""

    def __init__(self, strategy: str = "mean"):
        super().__init__()
        self.strategy = strategy.lower()
        if self.strategy not in ["mean", "cls"]:
            raise ValueError(f"Strategia pooling non valida: {strategy}. Usa 'mean' o 'cls'.")

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.strategy == "cls":
            return last_hidden_state[:, 0]

        # Mean pooling con esclusione dei token di padding
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class ConversationalBiEncoder(nn.Module):
    """Bi-Encoder per retrieval denso conversazionale con gestione parsimoniosa della memoria."""

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
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.pooling = DensePooling(strategy=pooling_strategy)

        # Inizializzazione backbone HF
        self.encoder = AutoModel.from_pretrained(model_name_or_path)

        # Risparmio drastico memoria: Gradient Checkpointing
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        # Configurazione LoRA se richiesta
        self.is_lora_enabled = bool(lora_cfg and lora_cfg.get("enabled", False))
        if self.is_lora_enabled:
            from peft import LoraConfig, get_peft_model
            target_mods = lora_cfg.get("target_modules", "auto")
            if target_mods == "auto":
                target_mods = ["query", "key", "value", "dense"]

            peft_config = LoraConfig(
                r=lora_cfg.get("r", 16),
                lora_alpha=lora_cfg.get("lora_alpha", 32),
                lora_dropout=lora_cfg.get("lora_dropout", 0.05),
                target_modules=target_mods,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.encoder = get_peft_model(self.encoder, peft_config)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        embs = self.pooling(outputs.last_hidden_state, attention_mask)
        if self.normalize_embeddings:
            embs = F.normalize(embs, p=2, dim=-1)
        return embs

    def forward(
        self,
        query_inputs: Dict[str, torch.Tensor],
        pos_inputs: Dict[str, torch.Tensor],
        neg_inputs: Optional[Dict[str, torch.Tensor]] = None,
        k_negs: int = 1,
    ) -> Dict[str, torch.Tensor]:
        q_embs = self.encode(**query_inputs)       # [B, D]
        pos_embs = self.encode(**pos_inputs)       # [B, D]
        B, D = q_embs.shape

        if neg_inputs is None:
            logits = torch.matmul(q_embs, pos_embs.T) / self.temperature
            labels = torch.arange(B, device=q_embs.device)
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}

        # CHUNKING ANTI-OOM: codifica i negativi a fette di massimo 8 alla volta
        total_negs = B * k_negs
        chunk_size = 8
        neg_embs_list = []

        for i in range(0, total_negs, chunk_size):
            chunk_slice = {
                k: v[i : i + chunk_size] for k, v in neg_inputs.items()
            }
            neg_embs_list.append(self.encode(**chunk_slice))

        flat_neg_embs = torch.cat(neg_embs_list, dim=0)
        neg_embs = flat_neg_embs.view(B, k_negs, D)

        # 1. Similarità con il positivo per ogni elemento: [B, 1]
        pos_sim = torch.sum(q_embs * pos_embs, dim=-1, keepdim=True) / self.temperature

        # 2. Similarità con i K hard negatives espliciti: [B, K]
        hard_neg_sim = torch.sum(q_embs.unsqueeze(1) * neg_embs, dim=-1) / self.temperature

        # 3. In-batch negatives derivanti dagli altri positivi nel batch: [B, B]
        in_batch_sim = torch.matmul(q_embs, pos_embs.T) / self.temperature
        mask = torch.eye(B, dtype=torch.bool, device=q_embs.device)
        in_batch_neg_sim = in_batch_sim.masked_fill(mask, -1e9)

        # Concatenazione finale
        logits = torch.cat([pos_sim, hard_neg_sim, in_batch_neg_sim], dim=-1)
        labels = torch.zeros(B, dtype=torch.long, device=q_embs.device)
        loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}

class ConversationalCrossEncoder(nn.Module):
    """Cross-Encoder per il re-ranking congiunto (Query, Passaggio)."""

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-base",
        num_labels: int = 1,
        use_lora: bool = False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)

        if use_lora:
            from peft import LoraConfig, get_peft_model
            peft_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["query", "key", "value", "dense"],
                task_type="SEQ_CLS",
            )
            self.model = get_peft_model(self.model, peft_cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = out.logits
        loss = out.loss if labels is not None else None
        return {"loss": loss, "logits": logits}