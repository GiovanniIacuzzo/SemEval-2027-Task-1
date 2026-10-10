#!/usr/bin/env python3
"""
subtrack_2a/models/model.py

Modelli neurali per RETECO SemEval-2027 Sub-track 2a.

Componenti:
    - ConversationalBiEncoder:
        bi-encoder compatibile con BGE e Qwen3-Embedding, pooling configurabile,
        LoRA opzionale e contrastive InfoNCE con hard/in-batch negatives.
    - ConversationalQueryRewriter:
        riscrittura deterministica delle query conversazionali con un LLM
        instruction-tuned. La generazione va eseguita in una fase separata e
        memorizzata in cache; non deve avvenire dentro un PyTorch Dataset.
    - ConversationalCrossEncoder:
        wrapper compatibile con il precedente reranker cross-encoder.

Nota Qwen3-Embedding:
    Il modello usa last-token pooling e un'istruzione task-specific per le query.
    L'istruzione va applicata alle query, non ai documenti. La formattazione è
    gestita da ContextAwareQueryFormatter in dataset.py.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# Istruzione consigliata da usare con Qwen3-Embedding per RETECO.
# Passarla a ContextAwareQueryFormatter(query_instruction=...).
QWEN3_RETRIEVAL_INSTRUCTION = (
    "Instruct: Given a conversational information-seeking query, retrieve "
    "relevant documents from the provided corpus that satisfy the user's "
    "information need.\nQuery:"
)


# =============================================================================
# Pooling
# =============================================================================

class DensePooling(nn.Module):
    """Pooling configurabile: ``cls``, ``mean`` oppure ``last_token``."""

    VALID_STRATEGIES = {"cls", "mean", "last_token"}

    def __init__(self, strategy: str = "cls"):
        super().__init__()
        self.strategy = str(strategy).lower().strip()
        if self.strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Pooling non valido: {strategy}. Usa una tra "
                f"{sorted(self.VALID_STRATEGIES)}."
            )

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.strategy == "cls":
            return last_hidden_state[:, 0]

        if self.strategy == "last_token":
            # Funziona sia con padding sinistro sia con padding destro.
            # Prende l'ultima posizione non mascherata per ogni sequenza.
            valid_positions = torch.arange(
                attention_mask.size(1),
                device=attention_mask.device,
            ).unsqueeze(0).expand_as(attention_mask)
            last_positions = valid_positions.masked_fill(
                attention_mask == 0,
                -1,
            ).max(dim=1).values
            if torch.any(last_positions < 0):
                raise ValueError(
                    "Una sequenza ha attention_mask completamente vuota."
                )
            batch_positions = torch.arange(
                last_hidden_state.size(0),
                device=last_hidden_state.device,
            )
            return last_hidden_state[batch_positions, last_positions]

        mask = attention_mask.unsqueeze(-1).to(
            dtype=last_hidden_state.dtype
        )
        summed = (last_hidden_state * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denominator


# =============================================================================
# Bi-Encoder
# =============================================================================

class ConversationalBiEncoder(nn.Module):
    """
    Dense bi-encoder per RETECO Sub-track 2a.

    Compatibilità:
        - BAAI/bge-base-en-v1.5: pooling ``cls`` (default automatico).
        - Qwen/Qwen3-Embedding-* : pooling ``last_token`` (default automatico).

    ``pooling_strategy="auto"`` sceglie il pooling in base al modello. Query e
    documenti devono comunque essere preformattati in modo coerente dal layer
    dati; l'istruzione Qwen3 va applicata alle query, non ai documenti.

    Obiettivo di training:
        positivo + hard negatives espliciti + in-batch negatives, tramite
        cross-entropy / InfoNCE.
    """

    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-base-en-v1.5",
        temperature: float = 0.05,
        normalize_embeddings: bool = True,
        pooling_strategy: str = "auto",
        lora_cfg: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        gradient_checkpointing: bool = False,
        negative_chunk_size: int = 8,
        trust_remote_code: Optional[bool] = None,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature deve essere > 0.")
        if negative_chunk_size <= 0:
            raise ValueError("negative_chunk_size deve essere > 0.")

        self.model_name_or_path = str(model_name_or_path)
        self.temperature = float(temperature)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.negative_chunk_size = int(negative_chunk_size)
        self.requested_device = device

        # Alcune revisioni del repository Qwen3-Embedding usano custom code
        # dichiarato nel model config; è configurabile dal chiamante.
        if trust_remote_code is None:
            trust_remote_code = "qwen3-embedding" in self.model_name_or_path.lower()

        self.encoder = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=bool(trust_remote_code),
        )

        model_type = str(
            getattr(getattr(self.encoder, "config", None), "model_type", "")
        ).lower()
        is_qwen3_embedding = (
            "qwen3-embedding" in self.model_name_or_path.lower()
            or model_type == "qwen3"
            or model_type.startswith("qwen3_")
        )

        pooling_strategy = str(pooling_strategy).lower().strip()
        if is_qwen3_embedding:
            # Qwen3-Embedding è progettato per last-token pooling. Se il vecchio
            # config BGE passa esplicitamente "cls", falliamo subito anziché
            # produrre embedding silenziosamente incompatibili con il model card.
            if pooling_strategy in {"auto", "last_token"}:
                pooling_strategy = "last_token"
            else:
                raise ValueError(
                    "Qwen3-Embedding richiede pooling_strategy='last_token' "
                    "(oppure 'auto'); il valore ricevuto è "
                    f"{pooling_strategy!r}. Aggiornare config/config.yaml."
                )
        elif pooling_strategy == "auto":
            pooling_strategy = "cls"
        self.pooling_strategy = pooling_strategy
        self.pooling = DensePooling(strategy=self.pooling_strategy)

        # ------------------------------------------------------------------
        # LoRA: selezione dei moduli compatibile con BGE e Qwen3.
        # ------------------------------------------------------------------
        self.is_lora_enabled = bool(
            lora_cfg and lora_cfg.get("enabled", False)
        )

        if self.is_lora_enabled:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError(
                    "LoRA è stata richiesta ma PEFT non è installato. "
                    "Installare peft prima di avviare il training."
                ) from exc

            target_modules = lora_cfg.get("target_modules", "auto")
            if target_modules == "auto":
                module_suffixes = {
                    name.rsplit(".", 1)[-1]
                    for name, _ in self.encoder.named_modules()
                    if name
                }
                if is_qwen3_embedding:
                    candidates = ["q_proj", "k_proj", "v_proj", "o_proj"]
                else:
                    candidates = ["query", "key", "value"]
                target_modules = [
                    name for name in candidates if name in module_suffixes
                ]
                if not target_modules:
                    raise ValueError(
                        "Impossibile individuare automaticamente i moduli LoRA "
                        f"per {self.model_name_or_path}. Specificare "
                        "lora_cfg.target_modules nel config."
                    )
            elif isinstance(target_modules, str):
                target_modules = [target_modules]

            peft_config = LoraConfig(
                r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
                lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
                target_modules=list(target_modules),
                bias=str(lora_cfg.get("bias", "none")),
                task_type="FEATURE_EXTRACTION",
            )
            self.encoder = get_peft_model(self.encoder, peft_config)

        # ------------------------------------------------------------------
        # Gradient checkpointing
        # ------------------------------------------------------------------
        self.gradient_checkpointing = bool(gradient_checkpointing)
        if self.gradient_checkpointing:
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()
            if hasattr(self.encoder, "enable_input_require_grads"):
                self.encoder.enable_input_require_grads()
            config = getattr(self.encoder, "config", None)
            if config is not None and hasattr(config, "use_cache"):
                config.use_cache = False

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def parameter_statistics(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": int(total),
            "trainable": int(trainable),
            "frozen": int(total - trainable),
        }

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model_inputs: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            # Alcuni backbone (ad es. Qwen3) non accettano token_type_ids.
            # Il tokenizer del relativo modello normalmente non li produce;
            # se presenti, li passiamo e lasciamo a HF la validazione.
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**model_inputs)
        embeddings = self.pooling(outputs.last_hidden_state, attention_mask)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    # ------------------------------------------------------------------
    # Forward / InfoNCE
    # ------------------------------------------------------------------

    def forward(
        self,
        query_inputs: Dict[str, torch.Tensor],
        pos_inputs: Dict[str, torch.Tensor],
        neg_inputs: Optional[Dict[str, torch.Tensor]] = None,
        k_negs: int = 1,
    ) -> Dict[str, torch.Tensor]:
        q_embs = self.encode(**query_inputs)

        # Modalità senza hard negatives espliciti: in-batch InfoNCE standard.
        if neg_inputs is None:
            pos_embs = self.encode(**pos_inputs)
            cosine_matrix = q_embs @ pos_embs.T
            logits = cosine_matrix / self.temperature
            labels = torch.arange(q_embs.size(0), device=q_embs.device)
            loss = F.cross_entropy(logits, labels)
            diagonal = torch.diagonal(cosine_matrix)

            if q_embs.size(0) > 1:
                mask = ~torch.eye(
                    q_embs.size(0), dtype=torch.bool, device=q_embs.device
                )
                mean_in_batch = cosine_matrix[mask].mean()
            else:
                mean_in_batch = cosine_matrix.new_tensor(0.0)

            return {
                "loss": loss,
                "logits": logits,
                "positive_cosine": diagonal.mean(),
                "hard_negative_cosine": cosine_matrix.new_tensor(0.0),
                "in_batch_negative_cosine": mean_in_batch,
            }

        if k_negs <= 0:
            raise ValueError(f"k_negs deve essere > 0, ricevuto {k_negs}")

        batch_size = q_embs.size(0)
        document_inputs: Dict[str, torch.Tensor] = {}
        for key, positive_part in pos_inputs.items():
            if key not in neg_inputs:
                raise KeyError(f"Campo {key!r} assente in neg_inputs.")
            document_inputs[key] = torch.cat(
                [positive_part, neg_inputs[key]], dim=0
            )

        document_embs = self.encode(**document_inputs)
        pos_embs = document_embs[:batch_size]
        neg_embs = document_embs[batch_size:].view(batch_size, k_negs, -1)

        positive_cosine = (q_embs * pos_embs).sum(dim=-1)
        hard_negative_cosine = (q_embs.unsqueeze(1) * neg_embs).sum(dim=-1)
        all_positive_cosine = q_embs @ pos_embs.T

        diagonal_mask = torch.eye(
            batch_size, dtype=torch.bool, device=q_embs.device
        )
        if batch_size > 1:
            mean_in_batch_cosine = all_positive_cosine[~diagonal_mask].mean()
        else:
            mean_in_batch_cosine = q_embs.new_tensor(0.0)

        positive_logits = (positive_cosine / self.temperature).unsqueeze(1)
        hard_negative_logits = hard_negative_cosine / self.temperature

        all_pos_scaled = all_positive_cosine / self.temperature
        in_batch_negative_logits = all_pos_scaled.masked_fill(
            diagonal_mask, -1000.0
        )
        
        logits = torch.cat(
            [positive_logits, hard_negative_logits, in_batch_negative_logits],
            dim=-1,
        )
        labels = torch.zeros(batch_size, dtype=torch.long, device=q_embs.device)
        loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "positive_cosine": positive_cosine.mean(),
            "hard_negative_cosine": hard_negative_cosine.mean(),
            "in_batch_negative_cosine": mean_in_batch_cosine,
        }


# =============================================================================
# Conversational query reasoning / rewriting
# =============================================================================

class ConversationalQueryRewriter:
    """
    LLM wrapper che converte un turno conversazionale in una query autonoma.

    La procedura è deliberatamente separata dai Dataset: eseguirla una volta,
    salvare i risultati con save_query_rewrites() in dataset.py e riutilizzare
    la cache in training e inference.

    Il prompt chiede di risolvere riferimenti ed esplicitare relazioni utili al
    retrieval, ma non chiede di rivelare una chain-of-thought né di rispondere
    alla domanda. L'output deve essere soltanto la query riscritta.

    Per una GPU da 16 GB è possibile usare load_in_4bit=True e device_map="auto".
    La quantizzazione 4-bit richiede CUDA, accelerate e bitsandbytes.
    """

    DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

    SYSTEM_PROMPT = """You are a query-rewriting component for conversational document retrieval.
Your task is to transform the current user turn into one concise, self-contained search query that retrieves documents from a fixed corpus.
Use the conversation history only to resolve references, ellipsis, omitted entities, and relationships needed to understand the current information need.
Make the target entities, concepts, comparison, and requested information explicit when they are supported by the conversation.
Do not answer the question. Do not invent facts, entities, constraints, or assumptions. Do not add information from outside the supplied conversation.
Prefer preserving the user's terminology and technical vocabulary. If the context does not resolve an ambiguity, preserve the ambiguity rather than guessing.
Return only the rewritten search query, on one line, with no explanation, heading, quotation marks, bullet points, or hidden reasoning."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        *,
        torch_dtype: Union[str, torch.dtype] = "auto",
        device: Optional[Union[str, torch.device]] = None,
        device_map: Optional[str] = None,
        load_in_4bit: bool = False,
        max_new_tokens: int = 96,
        max_input_tokens: int = 8192,
        max_history_tokens: int = 6000,
        trust_remote_code: bool = False,
    ):
        if max_new_tokens <= 0 or max_input_tokens <= 0 or max_history_tokens <= 0:
            raise ValueError("I limiti di token devono essere tutti > 0.")

        self.model_name_or_path = str(model_name_or_path)
        self.device = torch.device(device) if device is not None else None
        self.max_new_tokens = int(max_new_tokens)
        self.max_input_tokens = int(max_input_tokens)
        self.max_history_tokens = int(max_history_tokens)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if isinstance(torch_dtype, torch.dtype):
            model_kwargs["torch_dtype"] = torch_dtype
        elif str(torch_dtype).lower() != "auto":
            dtype_key = str(torch_dtype).lower()
            if dtype_key not in dtype_map:
                raise ValueError(
                    "torch_dtype deve essere 'auto', 'float16', 'bfloat16' "
                    "oppure 'float32'."
                )
            model_kwargs["torch_dtype"] = dtype_map[dtype_key]
        else:
            model_kwargs["torch_dtype"] = "auto"

        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "load_in_4bit=True richiede una GPU CUDA; disabilitare "
                    "la quantizzazione su CPU/MPS."
                )
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "BitsAndBytesConfig non disponibile. Aggiornare transformers "
                    "e installare bitsandbytes."
                ) from exc

            compute_dtype = (
                torch.bfloat16
                if isinstance(torch_dtype, torch.dtype) and torch_dtype == torch.bfloat16
                else dtype_map.get(str(torch_dtype).lower(), torch.float16)
            )
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = device_map or "auto"
        elif device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kwargs,
        )
        if self.device is not None and not load_in_4bit and device_map is None:
            self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _clean_history(self, history: str) -> str:
        history = (history or "").strip()
        if history.lower() == "no previous conversation.":
            return ""
        if not history:
            return ""

        token_ids = self.tokenizer.encode(history, add_special_tokens=False)
        if len(token_ids) > self.max_history_tokens:
            token_ids = token_ids[-self.max_history_tokens:]
            history = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
        return history

    def _build_prompt(self, query: str, history: str) -> str:
        history = self._clean_history(history)
        history_text = history if history else "(No previous conversation is available.)"
        user_content = (
            "Conversation history (may be empty):\n"
            f"{history_text}\n\n"
            "Current user turn to retrieve information for:\n"
            f"{(query or '').strip()}\n\n"
            "Produce the self-contained retrieval query now."
        )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        # Qwen3-Instruct-2507 uses its official chat template; unlike earlier
        # Qwen3 Thinking releases, this model does not require enable_thinking.
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(prompt)

    @staticmethod
    def _normalize_output(text: str, fallback_query: str) -> str:
        text = (text or "").strip()
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = text.replace("```", " ").strip()
        text = re.sub(
            r"^(?:rewritten query|standalone query|search query|query)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        text = text.strip(" \t\r\n\"'`“”‘’")
        # The prompt requests one line. If the model nevertheless emits an
        # explanation on later lines, keep the first non-empty line only.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            text = lines[0]
        return text or (fallback_query or "").strip()

    @torch.inference_mode()
    def rewrite(self, query: str, history: str = "") -> str:
        """Restituisce una query di retrieval autonoma e riproducibile."""
        query = (query or "").strip()
        if not query:
            return ""

        prompt = self._build_prompt(query=query, history=history)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        # Per modelli con device_map='auto', model.device indica normalmente
        # il device d'ingresso. In caso contrario usiamo il device richiesto.
        input_device = self.device
        if input_device is None:
            try:
                input_device = self.model.device
            except Exception:
                input_device = next(self.model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        decoded = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return self._normalize_output(decoded, fallback_query=query)


    @torch.inference_mode()
    def rewrite_batch(
        self,
        samples: Iterable[Any],
        existing: Optional[Dict[str, str]] = None,
        overwrite: bool = False,
        *,
        batch_size: int = 4,
        progress_desc: str = "Rewriting conversational queries",
        checkpoint_every: int = 8,
        checkpoint_callback: Optional[
            Callable[[Dict[str, str]], None]
        ] = None,
    ) -> Dict[str, str]:
        """
        Genera query riscritte in batch.

        Ogni sample deve esporre topic_id, query e history come
        attributi oppure come chiavi di dizionario.

        Le riscritture già presenti nella cache vengono riutilizzate,
        salvo overwrite=True. Il callback permette di salvare la cache
        periodicamente, anche durante la generazione.
        """
        if batch_size <= 0:
            raise ValueError("batch_size deve essere > 0.")

        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every deve essere > 0.")

        sample_list = list(samples)
        rewrites = dict(existing or {})

        pending = []

        for sample in sample_list:
            if isinstance(sample, dict):
                topic_id = str(sample["topic_id"])
                query = str(sample.get("query", "") or "").strip()
                history = str(sample.get("history", "") or "")
            else:
                topic_id = str(sample.topic_id)
                query = str(sample.query or "").strip()
                history = str(sample.history or "")

            if not overwrite and topic_id in rewrites:
                continue

            pending.append((topic_id, query, history))

        # Nessuna generazione necessaria: conserva la cache esistente.
        if not pending:
            if checkpoint_callback is not None:
                checkpoint_callback(dict(rewrites))
            return rewrites

        from tqdm.auto import tqdm

        # Per la generazione batch di modelli decoder-only è importante
        # utilizzare il padding a sinistra.
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        input_device = self.device
        if input_device is None:
            try:
                input_device = self.model.device
            except Exception:
                input_device = next(self.model.parameters()).device

        generated_since_checkpoint = 0

        try:
            with tqdm(
                total=len(pending),
                desc=progress_desc,
                unit="query",
                dynamic_ncols=True,
                leave=True,
            ) as progress:

                for start in range(0, len(pending), batch_size):
                    batch = pending[start:start + batch_size]

                    # L'ordine dei prompt e quello delle risposte
                    # rimangono allineati tramite topic_id.
                    prompts = [
                        self._build_prompt(query=query, history=history)
                        for _, query, history in batch
                    ]

                    # Una sola tokenizzazione batch per questo gruppo.
                    encoded = self.tokenizer(
                        prompts,
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_tokens,
                        return_tensors="pt",
                    )

                    input_width = encoded["input_ids"].shape[1]

                    # Passiamo soltanto gli input standard del modello
                    # causale e spostiamo i tensori sul device d'ingresso.
                    model_inputs = {
                        key: value.to(input_device)
                        for key, value in encoded.items()
                        if key in {"input_ids", "attention_mask"}
                    }

                    output_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                    # generate() restituisce prompt + continuazione.
                    # input_width include anche il padding sinistro.
                    generated_ids = output_ids[:, input_width:]

                    decoded_batch = self.tokenizer.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    for (topic_id, query, _), decoded in zip(
                        batch, decoded_batch
                    ):
                        rewrites[topic_id] = self._normalize_output(
                            decoded,
                            fallback_query=query,
                        ) or query

                    completed = len(batch)
                    generated_since_checkpoint += completed

                    progress.update(completed)
                    progress.set_postfix(
                        batch=len(batch),
                        cached=len(rewrites) - len(batch),
                    )

                    # Salvataggio periodico della cache.
                    # Con batch_size=4 e checkpoint_every=8 si salva
                    # normalmente ogni due batch.
                    if (
                        checkpoint_callback is not None
                        and generated_since_checkpoint >= checkpoint_every
                    ):
                        checkpoint_callback(dict(rewrites))
                        generated_since_checkpoint = 0
                        progress.set_postfix(
                            batch=len(batch),
                            cache="saved",
                        )

        finally:
            self.tokenizer.padding_side = original_padding_side

        # Salvataggio finale, inclusa l'ultima frazione del checkpoint.
        if checkpoint_callback is not None:
            checkpoint_callback(dict(rewrites))

        return rewrites

# =============================================================================
# Cross-Encoder
# =============================================================================

class ConversationalCrossEncoder(nn.Module):
    """Cross-encoder per un eventuale stadio separato di reranking."""

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
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            config=self.config,
        )

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError("use_lora=True richiede il pacchetto peft.") from exc
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["query", "key", "value"],
                task_type="SEQ_CLS",
            )
            self.model = get_peft_model(self.model, peft_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return {"loss": output.loss, "logits": output.logits}
