from __future__ import annotations

from typing import Any


def activation_layers(manifest: dict[str, Any]) -> list[str]:
    layers = manifest["capture"]["layers"]
    if not isinstance(layers, list) or not all(isinstance(layer, str) for layer in layers):
        raise ValueError("Manifest does not contain a valid capture.layers list.")
    return layers


def layer_shard_records(manifest: dict[str, Any], layer: str) -> list[dict[str, Any]]:
    layer_shards = manifest.get("layer_shards")
    if isinstance(layer_shards, dict) and isinstance(layer_shards.get(layer), list):
        return [dict(shard) for shard in layer_shards[layer]]
    return [dict(shard) for shard in manifest["shards"]]


def layer_index(layer: str) -> int | None:
    if layer == "embedding":
        return -1
    if layer.startswith("layer_"):
        try:
            return int(layer.removeprefix("layer_"))
        except ValueError:
            return None
    return None


def layer_to_hidden_index(layer: str) -> int:
    if layer == "embedding":
        return 0
    if layer.startswith("layer_"):
        return int(layer.removeprefix("layer_")) + 1
    raise ValueError(f"Unsupported layer name: {layer}")
