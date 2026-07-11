from __future__ import annotations

import json
import platform
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ..torch_utils import torch_dtype
from .runtime import make_capture_hook
from .sampling import sample_positions_by_doc
from .storage import flush_shard


def capture_post_block_activations(
    *,
    texts: Sequence[str],
    model: torch.nn.Module,
    tokenizer: Any,
    layer_modules: Sequence[torch.nn.Module],
    selected_layer_indices: Sequence[int],
    output_dir: Path,
    run_name: str,
    model_id: str,
    model_short_name: str,
    dataset_manifest: dict[str, object],
    context_length: int,
    token_budget: int,
    activation_dtype: str,
    shard_token_budget: int,
    seed: int,
) -> Path:
    if shard_token_budget <= 0:
        raise ValueError("shard_token_budget must be positive.")
    if not selected_layer_indices:
        raise ValueError("At least one layer must be selected.")

    device = next(model.parameters()).device
    storage_dtype = torch_dtype(activation_dtype)
    selected_layer_names = [f"layer_{idx:02d}" for idx in selected_layer_indices]

    tokenized_docs: list[torch.Tensor] = []
    doc_lengths: list[int] = []
    for text in tqdm(texts, dynamic_ncols=True, desc="tokenize docs"):
        if not isinstance(text, str) or not text.strip():
            continue
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=context_length)
        input_ids = encoded["input_ids"][0].to(dtype=torch.long, device="cpu")
        if int(input_ids.shape[0]) > 0:
            tokenized_docs.append(input_ids)
            doc_lengths.append(int(input_ids.shape[0]))

    if not tokenized_docs:
        raise RuntimeError("No tokenized documents available.")

    selected_by_doc = sample_positions_by_doc(doc_lengths, token_budget=token_budget, seed=seed)
    selected_docs = sorted(selected_by_doc)

    buffers: dict[str, list[torch.Tensor]] = {name: [] for name in selected_layer_names}
    input_id_buffer: list[torch.Tensor] = []
    doc_id_buffer: list[torch.Tensor] = []
    position_buffer: list[torch.Tensor] = []
    shards: list[dict[str, object]] = []
    token_count = 0
    shard_index = 0
    shard_tokens = 0
    started_at = time.time()

    captured: dict[int, torch.Tensor] = {}
    handles = []
    try:
        for layer_index in selected_layer_indices:
            handles.append(layer_modules[layer_index].register_forward_hook(make_capture_hook(layer_index, captured)))

        pbar = tqdm(total=token_budget, unit="tok", dynamic_ncols=True, desc="capture post-block activations")
        try:
            for doc_id in selected_docs:
                positions = selected_by_doc[doc_id]
                input_ids_cpu = tokenized_docs[doc_id]
                input_ids = input_ids_cpu.unsqueeze(0).to(device)
                attention_mask = torch.ones_like(input_ids, device=device)
                position_tensor = torch.tensor(positions, dtype=torch.long, device=device)

                captured.clear()
                with torch.inference_mode():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                missing = [idx for idx in selected_layer_indices if idx not in captured]
                if missing:
                    raise RuntimeError(f"Forward hooks did not capture layers: {missing}")

                take = int(position_tensor.shape[0])
                for layer_index, layer_name in zip(selected_layer_indices, selected_layer_names, strict=True):
                    hidden = captured[layer_index]
                    selected = hidden[0].index_select(0, position_tensor)
                    buffers[layer_name].append(selected.detach().to(dtype=storage_dtype).cpu())

                input_id_buffer.append(input_ids_cpu.index_select(0, position_tensor.cpu()))
                doc_id_buffer.append(torch.full((take,), doc_id, dtype=torch.long))
                position_buffer.append(position_tensor.cpu())

                token_count += take
                shard_tokens += take
                pbar.update(take)
                pbar.set_postfix(docs=len(selected_docs), shard_tokens=shard_tokens)

                if shard_tokens >= shard_token_budget:
                    shard_index = flush_shard(
                        capture_dir=output_dir,
                        shard_index=shard_index,
                        buffers=buffers,
                        input_id_buffer=input_id_buffer,
                        doc_id_buffer=doc_id_buffer,
                        position_buffer=position_buffer,
                        shards=shards,
                    )
                    shard_tokens = 0
        finally:
            pbar.close()
    finally:
        for handle in handles:
            handle.remove()

    if token_count != token_budget:
        raise RuntimeError(f"Expected {token_budget} sampled tokens, captured {token_count}.")

    if shard_tokens > 0:
        flush_shard(
            capture_dir=output_dir,
            shard_index=shard_index,
            buffers=buffers,
            input_id_buffer=input_id_buffer,
            doc_id_buffer=doc_id_buffer,
            position_buffer=position_buffer,
            shards=shards,
        )

    manifest = {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": model_id,
            "short_name": model_short_name,
            "hidden_size": int(model.config.hidden_size),
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "vocab_size": int(model.config.vocab_size),
        },
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        },
        "dataset": {
            **dataset_manifest,
            "candidate_documents": len(tokenized_docs),
            "candidate_tokens": int(sum(doc_lengths)),
        },
        "capture": {
            "context_length": context_length,
            "requested_tokens": token_budget,
            "captured_tokens": token_count,
            "documents": len(selected_docs),
            "activation_dtype": activation_dtype,
            "shard_token_budget": shard_token_budget,
            "layers": selected_layer_names,
            "activation_site": "post_transformer_block_output_before_final_model_norm",
            "site_note": "Captured with forward hooks on each transformer block/layer output. This avoids the final-layer output_hidden_states/last_hidden_state normalization mismatch.",
            "storage_layout": "per_layer_shards",
            "sampling_policy": "random_token_positions_without_exclusion",
            "seed": seed,
            "elapsed_seconds": round(time.time() - started_at, 3),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "shards": shards,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
