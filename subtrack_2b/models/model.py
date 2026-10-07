#!/usr/bin/env python3
"""
subtrack_2b/models/model.py

Wrapper generativo per causal language models instruction-tuned.

Supporto a:
- LoRA / PEFT
- QLoRA / quantizzazione 4-bit su CUDA
- Gradient Checkpointing
- FP16 / BF16 / FP32
- Chat template per modelli instruction-tuned
- Generazione greedy e nucleus sampling
- Salvataggio checkpoint completo / adapter-only
- Caricamento robusto di checkpoint LoRA
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

logger = logging.getLogger("subtrack_2b")


class ConversationalGenerator(nn.Module):
    """
    Classe modulare per la gestione del generatore causale
    instruction-tuned.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
    ) -> None:
        super().__init__()

        self.config = config
        self.device = device

        self.model_name = (
            config.get("model", {})
            .get(
                "name",
                "Qwen/Qwen2.5-7B-Instruct",
            )
        )

        self.tokenizer: Any = None
        self.model: Any = None

    # ==========================================================================
    # DEVICE / DTYPE HELPERS
    # ==========================================================================

    @staticmethod
    def _resolve_torch_dtype(
        config: Dict[str, Any],
        device: torch.device,
    ) -> torch.dtype:
        """
        Determina il dtype corretto in funzione del device.
        """

        training_cfg = config.get(
            "training",
            {},
        )

        if device.type == "cuda":
            if training_cfg.get("bf16", False):
                return torch.bfloat16

            if training_cfg.get("fp16", True):
                return torch.float16

            return torch.float32

        if device.type == "mps":
            # FP16 è generalmente il compromesso più pratico su MPS.
            return torch.float16

        # CPU -> evitare FP16.
        return torch.float32

    def _get_model_input_device(self) -> torch.device:
        """
        Restituisce il device effettivo sul quale si trovano i parametri
        del modello.

        È particolarmente importante con device_map='auto' e quantizzazione.
        """

        if self.model is not None:
            try:
                first_param = next(
                    self.model.parameters()
                )

                return first_param.device

            except StopIteration:
                pass

        return self.device

    # ==========================================================================
    # CHAT TEMPLATE
    # ==========================================================================

    @staticmethod
    def _build_chat_messages(
        prompt: str,
    ) -> list:
        """
        Costruisce i messaggi passati al chat template.

        Il prompt costruito dal data layer contiene già il contesto,
        l'evidence e le istruzioni specifiche del task, quindi viene
        mantenuto come singolo user message.

        Questo evita di duplicare o alterare arbitrariamente il formato
        costruito da utils.build_prompt().
        """

        return [
            {
                "role": "user",
                "content": prompt,
            }
        ]

    def _encode_prompt(
        self,
        prompt: str,
        return_tensors: Optional[str] = None,
    ) -> Any:
        """
        Tokenizza il prompt usando il chat template quando disponibile.

        Fallback robusto a tokenizer() quando il tokenizer non dispone
        di chat template.
        """

        messages = self._build_chat_messages(
            prompt
        )

        if getattr(
            self.tokenizer,
            "chat_template",
            None,
        ):
            try:
                kwargs: Dict[str, Any] = {
                    "tokenize": True,
                    "add_generation_prompt": True,
                }

                if return_tensors is not None:
                    kwargs["return_tensors"] = return_tensors

                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    **kwargs,
                )

                return encoded

            except Exception as exc:
                logger.warning(
                    "Errore apply_chat_template(); "
                    "fallback su tokenizer standard. Errore: %s",
                    exc,
                )

        if return_tensors is not None:
            return self.tokenizer(
                prompt,
                return_tensors=return_tensors,
                add_special_tokens=True,
            )

        return self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
        )

    # ==========================================================================
    # PRETRAINED MODEL
    # ==========================================================================

    @classmethod
    def from_pretrained(
        cls,
        config: Dict[str, Any],
        device: torch.device,
        is_training: bool = True,
    ) -> "ConversationalGenerator":
        """
        Inizializza tokenizer e modello.

        In training:
            - eventualmente prepara QLoRA
            - abilita gradient checkpointing
            - applica LoRA

        In inference:
            - carica il base model
            - NON applica un adapter LoRA vuoto
            - l'adapter viene successivamente caricato da load_checkpoint()
        """

        instance = cls(
            config=config,
            device=device,
        )

        model_cfg = config.get(
            "model",
            {},
        )

        model_name = model_cfg.get(
            "name",
            "Qwen/Qwen2.5-7B-Instruct",
        )

        trust_remote = model_cfg.get(
            "trust_remote_code",
            True,
        )

        # ----------------------------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------------------------
        logger.info(
            "Caricamento tokenizer per '%s'...",
            model_name,
        )

        instance.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote,
            padding_side=(
                "right"
                if is_training
                else "left"
            ),
        )

        if instance.tokenizer.pad_token is None:
            if instance.tokenizer.eos_token is not None:
                instance.tokenizer.pad_token = (
                    instance.tokenizer.eos_token
                )
                instance.tokenizer.pad_token_id = (
                    instance.tokenizer.eos_token_id
                )
            else:
                raise ValueError(
                    "Tokenizer privo sia di pad_token sia di eos_token."
                )

        # ----------------------------------------------------------------------
        # Dtype
        # ----------------------------------------------------------------------
        torch_dtype = cls._resolve_torch_dtype(
            config=config,
            device=device,
        )

        logger.info(
            "Dtype selezionato: %s",
            torch_dtype,
        )

        # ----------------------------------------------------------------------
        # Quantizzazione 4-bit
        # ----------------------------------------------------------------------
        load_4bit = bool(
            model_cfg.get(
                "load_in_4bit",
                False,
            )
        )

        quantization_config = None

        if load_4bit:
            if device.type != "cuda":
                logger.warning(
                    "load_in_4bit=True ma device=%s. "
                    "Disabilito la quantizzazione 4-bit.",
                    device.type,
                )
                load_4bit = False

            else:
                try:
                    import bitsandbytes  # noqa: F401

                    from transformers import BitsAndBytesConfig

                    compute_dtype = (
                        torch.bfloat16
                        if config.get(
                            "training",
                            {},
                        ).get("bf16", False)
                        else torch.float16
                    )

                    quantization_config = (
                        BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=compute_dtype,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                    )

                    logger.info(
                        "Quantizzazione 4-bit "
                        "(bitsandbytes/NF4) abilitata."
                    )

                except ImportError:
                    logger.warning(
                        "bitsandbytes non disponibile: "
                        "disabilito load_in_4bit."
                    )
                    load_4bit = False

        # ----------------------------------------------------------------------
        # Model kwargs
        # ----------------------------------------------------------------------
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": trust_remote,
            "dtype": torch_dtype,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = (
                quantization_config
            )
            model_kwargs["device_map"] = "auto"

        logger.info(
            "Caricamento base model: %s",
            model_name,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        # ----------------------------------------------------------------------
        # Non-quantized model -> device esplicito
        # ----------------------------------------------------------------------
        if (
            quantization_config is None
            and device.type != "cpu"
        ):
            base_model = base_model.to(device)

        # ----------------------------------------------------------------------
        # Config model
        # ----------------------------------------------------------------------
        if getattr(
            base_model,
            "config",
            None,
        ) is not None:

            if (
                instance.tokenizer.pad_token_id
                is not None
            ):
                base_model.config.pad_token_id = (
                    instance.tokenizer.pad_token_id
                )

            if is_training:
                # Fondamentale per gradient checkpointing.
                base_model.config.use_cache = False
            else:
                # Per inference si può riattivare.
                base_model.config.use_cache = True

        # ----------------------------------------------------------------------
        # LoRA config
        # ----------------------------------------------------------------------
        lora_cfg = model_cfg.get(
            "lora",
            {},
        )

        lora_enabled = bool(
            lora_cfg.get(
                "enabled",
                True,
            )
        )

        # ==========================================================================
        # TRAINING PREPARATION
        # ==========================================================================

        if is_training:
            # ------------------------------------------------------------------
            # QLoRA preparation
            # ------------------------------------------------------------------
            if quantization_config is not None:
                try:
                    from peft import (
                        prepare_model_for_kbit_training,
                    )
                except ImportError as exc:
                    raise RuntimeError(
                        "load_in_4bit=True richiede PEFT."
                    ) from exc

                logger.info(
                    "Preparazione del modello quantizzato "
                    "per k-bit training..."
                )

                base_model = prepare_model_for_kbit_training(
                    base_model,
                    use_gradient_checkpointing=bool(
                        model_cfg.get(
                            "gradient_checkpointing",
                            True,
                        )
                    ),
                )

            # ------------------------------------------------------------------
            # Gradient checkpointing
            # ------------------------------------------------------------------
            if model_cfg.get(
                "gradient_checkpointing",
                True,
            ):
                base_model.gradient_checkpointing_enable()

                if hasattr(
                    base_model,
                    "enable_input_require_grads",
                ):
                    base_model.enable_input_require_grads()

                logger.info(
                    "Gradient checkpointing abilitato."
                )

            # ------------------------------------------------------------------
            # LoRA / PEFT
            # ------------------------------------------------------------------
            if lora_enabled:
                try:
                    from peft import (
                        LoraConfig,
                        TaskType,
                        get_peft_model,
                    )

                    target_modules = lora_cfg.get(
                        "target_modules",
                        [
                            "q_proj",
                            "v_proj",
                        ],
                    )

                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=int(
                            lora_cfg.get(
                                "r",
                                16,
                            )
                        ),
                        lora_alpha=int(
                            lora_cfg.get(
                                "alpha",
                                32,
                            )
                        ),
                        lora_dropout=float(
                            lora_cfg.get(
                                "dropout",
                                0.05,
                            )
                        ),
                        target_modules=target_modules,
                        bias="none",
                    )

                    instance.model = get_peft_model(
                        base_model,
                        peft_config,
                    )

                    trainable_params, all_params = (
                        instance.get_nb_trainable_parameters()
                    )

                    pct = (
                        100.0
                        * trainable_params
                        / max(all_params, 1)
                    )

                    logger.info(
                        "LoRA agganciato con successo: "
                        "%s / %s parametri addestrabili "
                        "(%.2f%%).",
                        f"{trainable_params:,}",
                        f"{all_params:,}",
                        pct,
                    )

                except ImportError:
                    if quantization_config is not None:
                        raise RuntimeError(
                            "PEFT è necessario quando "
                            "load_in_4bit=True."
                        )

                    logger.warning(
                        "PEFT non installato. "
                        "Fallback a full fine-tuning."
                    )

                    instance.model = base_model

            else:
                if quantization_config is not None:
                    raise RuntimeError(
                        "Non è consigliabile/compatibile eseguire "
                        "full fine-tuning del modello 4-bit senza LoRA."
                    )

                instance.model = base_model

        # ==========================================================================
        # INFERENCE
        # ==========================================================================

        else:
            # Non applichiamo un adapter LoRA vuoto in inference.
            # load_checkpoint() potrà poi fare:
            #
            # base model
            #     +
            # saved adapter
            #
            # Questo evita PEFT annidato.
            instance.model = base_model

        return instance

    # ==========================================================================
    # PARAMETERS
    # ==========================================================================

    def get_nb_trainable_parameters(
        self,
    ) -> Tuple[int, int]:
        """
        Restituisce:
            (parametri addestrabili, parametri totali)
        """

        if self.model is None:
            return 0, 0

        trainable = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

        total = sum(
            p.numel()
            for p in self.model.parameters()
        )

        return trainable, total

    # ==========================================================================
    # FORWARD
    # ==========================================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Forward pass del causal LM.
        """

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    # ==========================================================================
    # GENERATION
    # ==========================================================================

    @torch.inference_mode()
    def generate_single(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ) -> str:
        """
        Genera una risposta da un singolo prompt.

        Utilizza il chat template del tokenizer quando disponibile.
        """

        if self.model is None:
            raise RuntimeError(
                "Il modello non è stato inizializzato."
            )

        if self.tokenizer is None:
            raise RuntimeError(
                "Il tokenizer non è stato inizializzato."
            )

        self.model.eval()

        model_input_device = (
            self._get_model_input_device()
        )

        encoded = self._encode_prompt(
            prompt,
            return_tensors="pt",
        )

        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
            attention_mask = encoded.get(
                "attention_mask"
            )

        else:
            if isinstance(encoded, torch.Tensor):
                input_ids = encoded
            else:
                input_ids = torch.tensor(
                    [encoded],
                    dtype=torch.long,
                )

            attention_mask = torch.ones_like(
                input_ids
            )

        input_ids = input_ids.to(
            model_input_device
        )

        if attention_mask is None:
            attention_mask = torch.ones_like(
                input_ids
            )
        else:
            attention_mask = attention_mask.to(
                model_input_device
            )

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(
                max_new_tokens
            ),
            "do_sample": bool(
                do_sample
            ),
            "repetition_penalty": float(
                repetition_penalty
            ),
            "pad_token_id": (
                self.tokenizer.pad_token_id
            ),
            "eos_token_id": (
                self.tokenizer.eos_token_id
            ),
            "use_cache": True,
        }

        if do_sample:
            if temperature <= 0:
                raise ValueError(
                    "temperature deve essere > 0 "
                    "quando do_sample=True."
                )

            generation_kwargs.update(
                {
                    "temperature": float(
                        temperature
                    ),
                    "top_p": float(
                        top_p
                    ),
                    "top_k": int(
                        top_k
                    ),
                }
            )

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

        prompt_len = input_ids.shape[1]

        generated_tokens = outputs[
            0,
            prompt_len:,
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        return answer

    # ==========================================================================
    # CHECKPOINT SAVING
    # ==========================================================================

    def save_checkpoint(
        self,
        output_dir: Union[str, Path],
    ) -> None:
        """
        Salva:
        - adapter LoRA se il modello è PEFT;
        - modello completo altrimenti;
        - tokenizer.
        """

        if self.model is None:
            raise RuntimeError(
                "Nessun modello da salvare."
            )

        path = Path(output_dir)
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_pretrained(
            path
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(
                path
            )

        # Metadata minima utile per il debugging.
        metadata = {
            "base_model_name": self.model_name,
            "device": str(self.device),
            "is_peft_model": bool(
                self.model.__class__.__name__.startswith(
                    "Peft"
                )
            ),
        }

        try:
            import json

            with open(
                path / "checkpoint_metadata.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    metadata,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        except Exception as exc:
            logger.warning(
                "Impossibile salvare checkpoint metadata: %s",
                exc,
            )

        logger.info(
            "Checkpoint salvato correttamente in: %s",
            path,
        )

    # ==========================================================================
    # CHECKPOINT LOADING
    # ==========================================================================

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
    ) -> None:
        """
        Carica:
        - adapter PEFT/LoRA;
        - oppure checkpoint completo Hugging Face;
        - oppure model.pt legacy.

        Importante:
        quando il checkpoint è LoRA, il metodo evita di creare un
        PeftModel annidato.
        """

        path = Path(
            checkpoint_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint non trovato: {path}"
            )

        # ----------------------------------------------------------------------
        # Rilevamento PEFT adapter
        # ----------------------------------------------------------------------
        adapter_exists = (
            (path / "adapter_model.bin").exists()
            or (path / "adapter_model.safetensors").exists()
        )

        if adapter_exists:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "Il checkpoint è un adapter PEFT/LoRA "
                    "ma PEFT non è installato."
                ) from exc

            # ------------------------------------------------------------------
            # Caso 1: base model già PeftModel
            # ------------------------------------------------------------------
            if self.model.__class__.__name__.startswith(
                "Peft"
            ):
                logger.info(
                    "Modello già PEFT: caricamento adapter "
                    "dal checkpoint %s",
                    path,
                )

                if hasattr(
                    self.model,
                    "load_adapter",
                ):
                    adapter_name = "checkpoint"

                    try:
                        self.model.load_adapter(
                            str(path),
                            adapter_name=adapter_name,
                            is_trainable=False,
                        )
                    except TypeError:
                        # Compatibilità con versioni PEFT più vecchie.
                        self.model.load_adapter(
                            str(path),
                            adapter_name=adapter_name,
                        )

                    if hasattr(
                        self.model,
                        "set_adapter",
                    ):
                        self.model.set_adapter(
                            adapter_name
                        )

                else:
                    raise RuntimeError(
                        "PeftModel non espone load_adapter(). "
                        "Aggiorna PEFT."
                    )

            # ------------------------------------------------------------------
            # Caso 2: base CausalLM -> PeftModel
            # ------------------------------------------------------------------
            else:
                logger.info(
                    "Creazione PeftModel da base model + adapter: %s",
                    path,
                )

                self.model = PeftModel.from_pretrained(
                    self.model,
                    str(path),
                    is_trainable=False,
                )

            # In inference vogliamo eval.
            self.model.eval()

            logger.info(
                "Adapter PEFT caricato correttamente."
            )

            return

        # ----------------------------------------------------------------------
        # Full model Hugging Face checkpoint
        # ----------------------------------------------------------------------
        full_model_files = [
            path / "model.safetensors",
            path / "pytorch_model.bin",
        ]

        has_full_hf_checkpoint = (
            (path / "config.json").exists()
            and any(
                file.exists()
                for file in full_model_files
            )
        )

        if has_full_hf_checkpoint:
            logger.info(
                "Caricamento full model Hugging Face da: %s",
                path,
            )

            torch_dtype = self._resolve_torch_dtype(
                self.config,
                self.device,
            )

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": self.config.get(
                    "model",
                    {},
                ).get(
                    "trust_remote_code",
                    True,
                ),
                "dtype": torch_dtype,
            }

            loaded_model = (
                AutoModelForCausalLM.from_pretrained(
                    str(path),
                    **model_kwargs,
                )
            )

            if self.device.type != "cpu":
                loaded_model = loaded_model.to(
                    self.device
                )

            loaded_model.config.use_cache = True

            self.model = loaded_model

            logger.info(
                "Full model caricato correttamente."
            )

            return

        # ----------------------------------------------------------------------
        # Legacy model.pt
        # ----------------------------------------------------------------------
        state_dict_pt = path / "model.pt"

        if state_dict_pt.exists():
            logger.info(
                "Caricamento state dict legacy: %s",
                state_dict_pt,
            )

            state_dict = torch.load(
                state_dict_pt,
                map_location=self._get_model_input_device(),
            )

            self.model.load_state_dict(
                state_dict
            )

            logger.info(
                "State dict caricato correttamente."
            )

            return

        raise FileNotFoundError(
            f"Nessun file checkpoint riconosciuto in {path}. "
            f"Attesi adapter_model.*, model.safetensors, "
            f"pytorch_model.bin oppure model.pt."
        )
        