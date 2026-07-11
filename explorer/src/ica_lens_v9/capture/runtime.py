from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..torch_utils import torch_dtype


def make_capture_hook(layer_index: int, captured: dict[int, torch.Tensor]):
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(f"Layer {layer_index} hook expected tensor output, got {type(hidden).__name__}.")
        captured[layer_index] = hidden.detach()

    return hook


def transformer_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    candidates = [
        ("model.layers", lambda m: getattr(getattr(m, "model", None), "layers", None)),
        ("transformer.h", lambda m: getattr(getattr(m, "transformer", None), "h", None)),
        ("gpt_neox.layers", lambda m: getattr(getattr(m, "gpt_neox", None), "layers", None)),
    ]
    for _name, getter in candidates:
        layers = getter(model)
        if layers is not None:
            if len(layers) == 0:
                raise RuntimeError("Transformer layer container is empty.")
            return layers
    raise RuntimeError(
        "Could not find transformer layers. Add this model architecture to transformer_layers()."
    )


def iter_dataset_texts(*, path: str, name: str | None, split: str, text_column: str, streaming: bool) -> Iterable[str]:
    dataset = load_dataset(path, name, split=split, streaming=streaming)
    for row in dataset:
        text = row.get(text_column)
        if isinstance(text, str):
            yield text


def load_model_and_tokenizer(model_id: str, *, device: str, dtype: str) -> tuple[torch.nn.Module, Any]:
    torch_device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ensure_pad_token(tokenizer)

    model_dtype = torch_dtype(dtype)
    if torch_device.type == "cpu" and model_dtype == torch.float16:
        model_dtype = torch.float32

    kwargs: dict[str, Any] = {"dtype": model_dtype, "low_cpu_mem_usage": True}
    if torch_device.type == "cuda":
        kwargs["device_map"] = {"": torch_device.index or 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if torch_device.type != "cuda":
        model.to(torch_device)
    model.eval()
    return model, tokenizer


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


def ensure_pad_token(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
