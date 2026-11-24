"""Inference-only depth recurrence over a pretrained decoder layer."""

from contextlib import contextmanager

import torch
from torch import nn


SUPPORTED_DECODERS = frozenset({"qwen2", "qwen3"})


def resolve_recurrent_layer(model: nn.Module, index: int | None) -> tuple[int, nn.Module]:
    """Resolve the decoder layer on a Hugging Face model, including a PEFT wrapper."""
    unwrap = getattr(model, "get_base_model", None)
    base = unwrap() if callable(unwrap) else model
    model_type = getattr(base.config, "model_type", None)
    layers = getattr(getattr(base, "model", None), "layers", None)
    if model_type not in SUPPORTED_DECODERS or not isinstance(layers, nn.ModuleList) or not layers:
        raise ValueError(f"Profondità ricorrente supportata per Qwen2/Qwen2.5 e Qwen3 densi. Trovato: {model_type}.")
    index = (len(layers) - 1) // 2 if index is None else index
    if not 0 <= index < len(layers):
        raise ValueError(f"Indice del layer {index} fuori intervallo [0, {len(layers) - 1}].")
    return index, layers[index]


@contextmanager
def repeat_decoder_layer(layer: nn.Module, passes: int):
    """Repeat the same layer on its own output with unchanged masks and positions."""
    if passes < 1:
        raise ValueError("Il numero di passaggi deve essere almeno 1.")
    if passes == 1:
        yield
        return

    def repeat(module, args, kwargs, output):
        if torch.is_grad_enabled() or module.training:
            raise RuntimeError("La ricorrenza sperimentale richiede inferenza in modalità eval.")
        if kwargs.get("use_cache") or any(
            kwargs.get(name) is not None for name in ("past_key_value", "past_key_values")
        ):
            raise RuntimeError("La ricorrenza richiede use_cache=False e nessuna KV cache preesistente.")
        for _ in range(passes - 1):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("Il primo output del layer ricorrente deve essere un tensore.")
            # Bypass this module's own hook so the extra passes cannot recurse.
            if args:
                output = module.forward(hidden, *args[1:], **kwargs)
            else:
                output = module.forward(**(kwargs | {"hidden_states": hidden}))
        return output

    handle = layer.register_forward_hook(repeat, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
