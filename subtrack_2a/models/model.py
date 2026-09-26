#!/usr/bin/env python3
"""
subtrack_2a/models/model.py

Modelli neurali per RETECO Sub-track 2a (Conversational Retrieval):
  1. ConversationalBiEncoder:
     - Architettura a due torri (Bi-Encoder) basata su Hugging Face Transformers.
     - Mean pooling con attenzione alla maschera e normalizzazione vettoriale L2.
     - Supporto per loss contrastiva InfoNCE (in-batch negatives e hard negatives).
  2. ConversationalCrossEncoder:
     - Architettura Cross-Encoder per il re-ranking fine dei top candidati.
     - Valutazione congiunta di query contestualizzata e documento candidato.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class MeanPooling(nn.Module):
    """Esegue il pooling pesato sulla attention mask ignorando i token di padding."""
    
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask


class ConversationalBiEncoder(nn.Module):
    """
    Bi-Encoder neurale per dense retrieval conversazionale.
    Mappa query contestuali e passaggi in uno spazio vettoriale condiviso.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-base-en-v1.5",
        temperature: float = 0.05,
        normalize_embeddings: bool = True,
        pooling_strategy: str = "mean",
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.pooling_strategy = pooling_strategy

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config)
        self.pooler = MeanPooling() if pooling_strategy == "mean" else None

    def _pool(self, model_output: Tuple[torch.Tensor, ...], attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling_strategy == "cls":
            return model_output[0][:, 0]
        return self.pooler(model_output[0], attention_mask)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Codifica un batch di testi in vettori densi normalizzati.
        """
        extra_args = {}
        if token_type_ids is not None and "token_type_ids" in self.encoder.forward.__code__.co_varnames:
            extra_args["token_type_ids"] = token_type_ids

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **extra_args)
        embeddings = self._pool(outputs, attention_mask)

        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def compute_similarity(self, query_embs: torch.Tensor, doc_embs: torch.Tensor) -> torch.Tensor:
        """
        Calcola la similarità coseno / dot-product tra matrici di embedding normalizzate.
        """
        return torch.matmul(query_embs, doc_embs.transpose(-2, -1))

    def forward(
        self,
        query_inputs: Dict[str, torch.Tensor],
        pos_inputs: Dict[str, torch.Tensor],
        neg_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass per l'addestramento con loss contrastiva InfoNCE.
        
        Supporta:
          1. In-batch negatives se neg_inputs è None.
          2. Hard negatives dedicati se neg_inputs è fornito.
        """
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
            # InfoNCE simmetrica/in-batch
            scores = self.compute_similarity(query_embs, pos_embs) / self.temperature
            labels = torch.arange(batch_size, device=query_embs.device, dtype=torch.long)
            loss = F.cross_entropy(scores, labels)
            return {"loss": loss, "scores": scores, "query_embs": query_embs, "pos_embs": pos_embs}

        # Con negativi espliciti / hard negatives
        neg_embs = self.encode(
            input_ids=neg_inputs["input_ids"],
            attention_mask=neg_inputs["attention_mask"],
            token_type_ids=neg_inputs.get("token_type_ids"),
        )

        # pos_sim: [batch_size, 1]
        pos_sim = torch.sum(query_embs * pos_embs, dim=-1, keepdim=True) / self.temperature
        
        # in-batch + hard negative scores:
        # Per ciascuna query calcoliamo la similarità sia con i negativi campionati che con tutti i positivi del batch
        all_doc_embs = torch.cat([pos_embs, neg_embs], dim=0)
        all_scores = self.compute_similarity(query_embs, all_doc_embs) / self.temperature
        
        # Il target corretto per la i-esima query corrisponde alla colonna i-esima (positivo associato)
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
        Salva i pesi dell'encoder in formato safetensors compatto e la relativa configurazione.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.encoder.save_pretrained(
            save_directory,
            safe_serialization=safe_serialization,
            **kwargs
        )
        self.config.save_pretrained(save_directory)


class ConversationalCrossEncoder(nn.Module):
    """
    Cross-Encoder per il re-ranking della lista dei migliori candidati.
    Accetta in input sequenze congiunte [CLS] Query Contestuale [SEP] Documento [SEP].
    """

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
        """
        Forward pass per training o scoring.
        Se labels è fornito calcola BCEWithLogitsLoss.
        """
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
            if self.num_labels == 1:
                loss_fn = nn.BCEWithLogitsLoss()
                output_dict["loss"] = loss_fn(logits, labels.float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                output_dict["loss"] = loss_fn(logits, labels.long())

        return output_dict

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        """
        Salva i pesi del cross-encoder in formato safetensors compatto e la configurazione.
        """
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(
            save_directory,
            safe_serialization=safe_serialization,
            **kwargs
        )
        self.config.save_pretrained(save_directory)


class RETECOModelWrapper:
    """
    Interfaccia ad alto livello per gestire tokenizzazione, inferenza a lotti
    e codifica rapida su GPU T4 (o CPU/MPS per Mac).
    """

    def __init__(
        self,
        model_type: str = "bi_encoder",
        model_name_or_path: str = "BAAI/bge-base-en-v1.5",
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 512,
    ):
        self.max_length = max_length
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if model_type == "bi_encoder":
            self.model = ConversationalBiEncoder(model_name_or_path=model_name_or_path).to(self.device)
        elif model_type == "cross_encoder":
            self.model = ConversationalCrossEncoder(model_name_or_path=model_name_or_path).to(self.device)
        else:
            raise ValueError(f"Tipo modello '{model_type}' non riconosciuto. Usa 'bi_encoder' o 'cross_encoder'.")

    def encode_texts(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> torch.Tensor:
        """
        Codifica una lista arbitraria di testi (passaggi o query) a lotti con precisione mista (FP16).
        """
        self.model.eval()
        all_embeddings = []

        total = len(texts)
        indices = range(0, total, batch_size)

        with torch.no_grad():
            for i in indices:
                batch_texts = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                use_amp = self.device.type == "cuda"
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    embs = self.model.encode(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        token_type_ids=encoded.get("token_type_ids"),
                    )

                all_embeddings.append(embs.cpu())

        return torch.cat(all_embeddings, dim=0)

    def score_pairs(self, queries: List[str], passages: List[str], batch_size: int = 32) -> List[float]:
        """
        Calcola i punteggi di rilevanza tra coppie (query, passaggio) tramite Cross-Encoder.
        """
        if not isinstance(self.model, ConversationalCrossEncoder):
            raise TypeError("Il calcolo tramite score_pairs richiede che il wrapper sia istanziato con 'cross_encoder'.")

        self.model.eval()
        scores: List[float] = []

        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                b_queries = queries[i : i + batch_size]
                b_passages = passages[i : i + batch_size]

                encoded = self.tokenizer(
                    b_queries,
                    b_passages,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                use_amp = self.device.type == "cuda"
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    out = self.model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                        token_type_ids=encoded.get("token_type_ids"),
                    )
                    logits = out["logits"]
                    probs = torch.sigmoid(logits).cpu().tolist() if logits.dim() == 1 else logits[:, 1].cpu().tolist()

                scores.extend(probs if isinstance(probs, list) else [probs])

        return scores


if __name__ == "__main__":
    print("Inizializzazione test unitario di model.py...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device rilevato: {device}")

    # Test rapido istanziazione Bi-Encoder
    bi_encoder = ConversationalBiEncoder("BAAI/bge-base-en-v1.5", temperature=0.05).to(device)
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

    dummy_queries = ["What causes drone motor overheating?", "How to calibrate ESCs?"]
    dummy_docs = ["High ambient temperatures and bad lubrication degrade motors.", "ESCs require PWM calibration."]

    q_tok = tokenizer(dummy_queries, padding=True, truncation=True, return_tensors="pt").to(device)
    d_tok = tokenizer(dummy_docs, padding=True, truncation=True, return_tensors="pt").to(device)

    out = bi_encoder(query_inputs=q_tok, pos_inputs=d_tok)
    print("✓ Bi-Encoder forward completato con successo.")
    print(f"  Loss contrastiva calcolata: {out['loss'].item():.4f}")
    print(f"  Shape query embeddings: {out['query_embs'].shape}")
    print(f"  Shape pos embeddings: {out['pos_embs'].shape}")