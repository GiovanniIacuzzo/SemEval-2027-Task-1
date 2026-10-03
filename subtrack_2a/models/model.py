#!/usr/bin/env python3
"""
subtrack_2a/models/model.py

Modelli neurali per RETECO Sub-track 2a:

    - ConversationalBiEncoder
        BGE encoder + configurable pooling + optional LoRA
        + multi-negative InfoNCE.

    - ConversationalCrossEncoder
        BGE reranker per il successivo stadio di re-ranking.

Il modello non contiene logica di dataset, BM25 o evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
)


# =============================================================================
# Pooling
# =============================================================================

class DensePooling(nn.Module):
    """
    Pooling configurabile.

    Strategie:
        - cls
        - mean

    Per BGE v1.5 il default del nostro progetto è CLS.
    """

    def __init__(
        self,
        strategy: str = "cls",
    ):
        super().__init__()

        self.strategy = (
            strategy
            .lower()
            .strip()
        )

        if self.strategy not in {
            "cls",
            "mean",
        }:
            raise ValueError(
                f"Pooling non valido: "
                f"{strategy}. "
                f"Usa 'cls' oppure 'mean'."
            )

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        if self.strategy == "cls":
            return last_hidden_state[:, 0]

        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(
                last_hidden_state.size()
            )
            .float()
        )

        masked_embeddings = (
            last_hidden_state
            * mask
        )

        sum_embeddings = (
            masked_embeddings.sum(
                dim=1
            )
        )

        sum_mask = (
            mask.sum(
                dim=1
            )
            .clamp(min=1e-9)
        )

        return (
            sum_embeddings
            / sum_mask
        )


# =============================================================================
# Bi-Encoder
# =============================================================================

class ConversationalBiEncoder(nn.Module):
    """
    Dense bi-encoder per RETECO Sub-track 2a.

    Training objective:

        positive
        +
        explicit hard negatives
        +
        in-batch negatives

    tramite cross-entropy / InfoNCE.

    Quando LoRA è abilitata:
        il backbone viene congelato e vengono ottimizzati
        solamente i parametri LoRA.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-base-en-v1.5",
        temperature: float = 0.05,
        normalize_embeddings: bool = True,
        pooling_strategy: str = "cls",
        lora_cfg: Optional[
            Dict[str, Any]
        ] = None,
        device: Optional[
            torch.device
        ] = None,
        gradient_checkpointing: bool = False,
        negative_chunk_size: int = 8,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError(
                "temperature deve essere > 0."
            )

        if negative_chunk_size <= 0:
            raise ValueError(
                "negative_chunk_size deve essere > 0."
            )

        self.temperature = float(
            temperature
        )

        self.normalize_embeddings = bool(
            normalize_embeddings
        )

        self.pooling = DensePooling(
            strategy=pooling_strategy
        )

        self.negative_chunk_size = int(
            negative_chunk_size
        )

        self.encoder = AutoModel.from_pretrained(
            model_name_or_path
        )

        # ------------------------------------------------------------------
        # LoRA
        # ------------------------------------------------------------------

        self.is_lora_enabled = bool(
            lora_cfg
            and lora_cfg.get(
                "enabled",
                False,
            )
        )

        if self.is_lora_enabled:

            from peft import (
                LoraConfig,
                get_peft_model,
            )

            target_modules = (
                lora_cfg.get(
                    "target_modules",
                    "auto",
                )
            )

            if target_modules == "auto":

                target_modules = [
                    "query",
                    "key",
                    "value",
                ]

            if isinstance(
                target_modules,
                str,
            ):
                target_modules = [
                    target_modules
                ]

            peft_config = LoraConfig(
                r=int(
                    lora_cfg.get(
                        "r",
                        16,
                    )
                ),
                lora_alpha=int(
                    lora_cfg.get(
                        "lora_alpha",
                        32,
                    )
                ),
                lora_dropout=float(
                    lora_cfg.get(
                        "lora_dropout",
                        0.05,
                    )
                ),
                target_modules=target_modules,
                bias=lora_cfg.get(
                    "bias",
                    "none",
                ),
                task_type="FEATURE_EXTRACTION",
            )

            self.encoder = get_peft_model(
                self.encoder,
                peft_config,
            )

        # ------------------------------------------------------------------
        # Gradient checkpointing
        # ------------------------------------------------------------------

        self.gradient_checkpointing = bool(
            gradient_checkpointing
        )

        if self.gradient_checkpointing:

            if hasattr(
                self.encoder,
                "gradient_checkpointing_enable",
            ):
                self.encoder.gradient_checkpointing_enable()

            # Necessario per alcune combinazioni
            # Transformers + PEFT + gradient checkpointing.
            if hasattr(
                self.encoder,
                "enable_input_require_grads",
            ):
                self.encoder.enable_input_require_grads()

            if hasattr(
                self.encoder,
                "config",
            ):
                self.encoder.config.use_cache = False

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def parameter_statistics(
        self,
    ) -> Dict[str, int]:

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return {
            "total": int(total),
            "trainable": int(trainable),
            "frozen": int(
                total - trainable
            ),
        }

    # -------------------------------------------------------------------------
    # Encode
    # -------------------------------------------------------------------------

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[
            torch.Tensor
        ] = None,
    ) -> torch.Tensor:

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            model_inputs[
                "token_type_ids"
            ] = token_type_ids

        outputs = self.encoder(
            **model_inputs
        )

        embeddings = self.pooling(
            outputs.last_hidden_state,
            attention_mask,
        )

        if self.normalize_embeddings:

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=-1,
            )

        return embeddings

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------
    def forward(
        self,
        query_inputs: Dict[str, torch.Tensor],
        pos_inputs: Dict[str, torch.Tensor],
        neg_inputs: Optional[Dict[str, torch.Tensor]] = None,
        k_negs: int = 1,
    ) -> Dict[str, torch.Tensor]:

        # ------------------------------------------------------------------
        # Query embedding
        # ------------------------------------------------------------------
        q_embs = self.encode(**query_inputs)

        # ------------------------------------------------------------------
        # Standard in-batch InfoNCE
        # ------------------------------------------------------------------
        if neg_inputs is None:

            pos_embs = self.encode(**pos_inputs)

            cosine_matrix = q_embs @ pos_embs.T
            logits = cosine_matrix / self.temperature

            labels = torch.arange(
                q_embs.size(0),
                device=q_embs.device,
            )

            loss = F.cross_entropy(
                logits,
                labels,
            )

            diagonal = torch.diagonal(
                cosine_matrix
            )

            if q_embs.size(0) > 1:

                mask = ~torch.eye(
                    q_embs.size(0),
                    dtype=torch.bool,
                    device=q_embs.device,
                )

                in_batch_values = cosine_matrix[mask]

                mean_in_batch = in_batch_values.mean()

            else:

                mean_in_batch = torch.tensor(
                    0.0,
                    device=q_embs.device,
                )

            return {
                "loss": loss,
                "logits": logits,
                "positive_cosine": diagonal.mean(),
                "hard_negative_cosine": torch.tensor(
                    0.0,
                    device=q_embs.device,
                ),
                "in_batch_negative_cosine": mean_in_batch,
            }

        if k_negs <= 0:
            raise ValueError(
                f"k_negs deve essere > 0, ricevuto {k_negs}"
            )

        batch_size = q_embs.size(0)

        # ------------------------------------------------------------------
        # Positive + explicit negatives in UN SOLO forward
        # ------------------------------------------------------------------
        total_negatives = batch_size * k_negs

        document_inputs = {}

        for key in pos_inputs.keys():

            positive_part = pos_inputs[key]

            negative_part = neg_inputs[key]

            document_inputs[key] = torch.cat(
                [
                    positive_part,
                    negative_part,
                ],
                dim=0,
            )

        document_embs = self.encode(
            **document_inputs
        )

        # First B embeddings = positives
        pos_embs = document_embs[:batch_size]

        # Remaining B*K embeddings = explicit negatives
        neg_embs_flat = document_embs[batch_size:]

        neg_embs = neg_embs_flat.view(
            batch_size,
            k_negs,
            -1,
        )

        # ------------------------------------------------------------------
        # Cosine similarities
        # ------------------------------------------------------------------
        positive_cosine = (
            q_embs * pos_embs
        ).sum(dim=-1)

        hard_negative_cosine = (
            q_embs.unsqueeze(1)
            * neg_embs
        ).sum(dim=-1)

        all_positive_cosine = (
            q_embs @ pos_embs.T
        )

        diagonal_mask = torch.eye(
            batch_size,
            dtype=torch.bool,
            device=q_embs.device,
        )

        if batch_size > 1:

            in_batch_values = (
                all_positive_cosine[
                    ~diagonal_mask
                ]
            )

            mean_in_batch_cosine = (
                in_batch_values.mean()
            )

        else:

            mean_in_batch_cosine = torch.tensor(
                0.0,
                device=q_embs.device,
            )

        # ------------------------------------------------------------------
        # Temperature-scaled logits
        # ------------------------------------------------------------------

        positive_logits = (
            positive_cosine
            / self.temperature
        ).unsqueeze(1)

        hard_negative_logits = (
            hard_negative_cosine
            / self.temperature
        )

        in_batch_cosine_for_logits = (
            all_positive_cosine
            .masked_fill(
                diagonal_mask,
                -1e4,
            )
        )

        in_batch_negative_logits = (
            in_batch_cosine_for_logits
            / self.temperature
        )

        logits = torch.cat(
            [
                positive_logits,
                hard_negative_logits,
                in_batch_negative_logits,
            ],
            dim=-1,
        )

        labels = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=q_embs.device,
        )

        loss = F.cross_entropy(
            logits,
            labels,
        )

        return {
            "loss": loss,
            "logits": logits,

            "positive_cosine": (
                positive_cosine.mean()
            ),

            "hard_negative_cosine": (
                hard_negative_cosine.mean()
            ),

            "in_batch_negative_cosine": (
                mean_in_batch_cosine
            ),
        }

# =============================================================================
# Cross-Encoder
# =============================================================================

class ConversationalCrossEncoder(nn.Module):
    """
    Cross-encoder per il reranking.

    Viene mantenuto separato dal training del bi-encoder.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-base",
        num_labels: int = 1,
        use_lora: bool = False,
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
        )

        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                config=self.config,
            )
        )

        if use_lora:

            from peft import (
                LoraConfig,
                get_peft_model,
            )

            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=[
                    "query",
                    "key",
                    "value",
                ],
                task_type="SEQ_CLS",
            )

            self.model = get_peft_model(
                self.model,
                peft_config,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[
            torch.Tensor
        ] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        return {
            "loss": output.loss,
            "logits": output.logits,
        }