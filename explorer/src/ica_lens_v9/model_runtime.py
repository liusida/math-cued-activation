from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import torch
from gb10_load_llm import from_pretrained_to_cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

from .capture.runtime import make_capture_hook, transformer_layers
from .layers import layer_index


@dataclass
class Runtime:
    model_id: str
    tokenizer: object
    model: object
    device: torch.device


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def dtype_from_name(name: str | None) -> torch.dtype | None:
    if name is None or name == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


_runtime_cache: dict[tuple[str, str, str], Runtime] = {}
_runtime_locks: dict[tuple[str, str, str], Lock] = {}
_runtime_cache_lock = Lock()
_RUNTIME_CACHE_MAX_SIZE = 3


def load_runtime(model_id: str, device_name: str = "auto", dtype_name: str = "auto") -> Runtime:
    key = (model_id, device_name, dtype_name)
    with _runtime_cache_lock:
        cached = _runtime_cache.get(key)
        if cached is not None:
            return cached
        load_lock = _runtime_locks.setdefault(key, Lock())

    # Only duplicate misses for the same runtime key wait on each other. Cache
    # hits and loads for other keys are not blocked by this lock.
    with load_lock:
        with _runtime_cache_lock:
            cached = _runtime_cache.get(key)
            if cached is not None:
                return cached
        runtime = _load_runtime_uncached(model_id, device_name, dtype_name)
        with _runtime_cache_lock:
            if key not in _runtime_cache and len(_runtime_cache) >= _RUNTIME_CACHE_MAX_SIZE:
                old_key = next(iter(_runtime_cache))
                _runtime_cache.pop(old_key, None)
                _runtime_locks.pop(old_key, None)
            _runtime_cache[key] = runtime
        return runtime


def _load_runtime_uncached(model_id: str, device_name: str = "auto", dtype_name: str = "auto") -> Runtime:
    device = resolve_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    dtype = dtype_from_name(dtype_name)
    kwargs = {"local_files_only": True}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device.type == "cuda":
        model = from_pretrained_to_cuda(AutoModelForCausalLM, model_id, device=device, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return Runtime(model_id=model_id, tokenizer=tokenizer, model=model, device=device)


def hidden_states_for_layer(model: object, layer: str, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the activation site used by v9 capture for one prompt.

    HuggingFace ``output_hidden_states`` can apply a final model norm to the last
    layer. v9 captures transformer block outputs with hooks before that norm, so
    live probing must use the same hook site.
    """
    hidden, _ = hidden_states_and_logits_for_layer(model, layer, inputs)
    return hidden


def hidden_states_and_logits_for_layer(
    model: object, layer: str, inputs: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the captured residual-post stream and actual final logits."""
    index = layer_index(str(layer))
    if index is None:
        raise ValueError(f"Unsupported layer name: {layer!r}")
    if index < 0:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        return outputs.hidden_states[0][0].detach().to(torch.float32), outputs.logits[0].detach().to(torch.float32)

    layers = transformer_layers(model)
    if index >= len(layers):
        raise ValueError(f"Layer {layer!r} is outside model layer range 0..{len(layers) - 1}")
    captured: dict[int, torch.Tensor] = {}
    handle = layers[index].register_forward_hook(make_capture_hook(index, captured))
    try:
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        handle.remove()
    if index not in captured:
        raise RuntimeError(f"Forward hook did not capture {layer}.")
    return captured[index][0].detach().to(torch.float32), outputs.logits[0].detach().to(torch.float32)
