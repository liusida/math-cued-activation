from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..torch_utils import torch_dtype


def flush_shard(
    *,
    capture_dir: Path,
    shard_index: int,
    buffers: dict[str, list[torch.Tensor]],
    input_id_buffer: list[torch.Tensor],
    doc_id_buffer: list[torch.Tensor],
    position_buffer: list[torch.Tensor],
    shards: list[dict[str, object]],
) -> int:
    if not input_id_buffer:
        return shard_index

    hidden_states = {key: torch.cat(chunks, dim=0) for key, chunks in buffers.items() if chunks}
    input_ids = torch.cat(input_id_buffer, dim=0)
    doc_ids = torch.cat(doc_id_buffer, dim=0)
    positions = torch.cat(position_buffer, dim=0)
    n_tokens = int(input_ids.shape[0])

    shard_name = f"shard_{shard_index:05d}.pt"
    metadata_paths = {
        "input_ids": save_tensor(capture_dir, "input_ids", shard_name, input_ids),
        "doc_ids": save_tensor(capture_dir, "doc_ids", shard_name, doc_ids),
        "positions": save_tensor(capture_dir, "positions", shard_name, positions),
    }
    layer_paths = {key: save_tensor(capture_dir, key, shard_name, tensor) for key, tensor in hidden_states.items()}
    shards.append(
        {
            "index": shard_index,
            "tokens": n_tokens,
            **metadata_paths,
            "layers": layer_paths,
            "hidden_size": int(next(iter(hidden_states.values())).shape[1]),
        }
    )

    for chunks in buffers.values():
        chunks.clear()
    input_id_buffer.clear()
    doc_id_buffer.clear()
    position_buffer.clear()
    return shard_index + 1


def store_input_embedding_layer(*, manifest_path: Path, model: torch.nn.Module, storage_dtype: str, shard_token_budget: int) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    capture_dir = manifest_path.parent
    dtype = torch_dtype(storage_dtype)
    embedding = model.get_input_embeddings().weight.detach().to(dtype=dtype, device="cpu")
    hidden_size = int(manifest["model"]["hidden_size"])
    if int(embedding.shape[1]) != hidden_size:
        raise RuntimeError(f"Embedding width {int(embedding.shape[1])} does not match hidden_size {hidden_size}.")

    layer_shards = []
    for shard_index, start in enumerate(range(0, int(embedding.shape[0]), shard_token_budget)):
        stop = min(int(embedding.shape[0]), start + shard_token_budget)
        shard_name = f"shard_{shard_index:05d}.pt"
        token_ids = torch.arange(start, stop, dtype=torch.long)
        layer_shards.append(
            {
                "index": shard_index,
                "tokens": int(stop - start),
                "input_ids": save_tensor(capture_dir, "embedding_input_ids", shard_name, token_ids),
                "doc_ids": save_tensor(capture_dir, "embedding_doc_ids", shard_name, torch.full_like(token_ids, -1)),
                "positions": save_tensor(capture_dir, "embedding_positions", shard_name, token_ids),
                "layers": {"embedding": save_tensor(capture_dir, "embedding", shard_name, embedding[start:stop].contiguous())},
                "hidden_size": hidden_size,
                "row_source": "input_embedding_matrix",
            }
        )

    layers = list(manifest["capture"].get("layers") or [])
    manifest["capture"]["layers"] = ["embedding", *[layer for layer in layers if layer != "embedding"]]
    manifest["capture"]["embedding_layer_source"] = "input_embedding_matrix"
    manifest["capture"]["embedding_rows"] = int(embedding.shape[0])
    manifest["capture"]["embedding_updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("layer_shards", {})["embedding"] = layer_shards
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def save_tensor(capture_dir: Path, subdir: str, shard_name: str, tensor: torch.Tensor) -> str:
    output_dir = capture_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_dir / shard_name)
    return f"{subdir}/{shard_name}"
