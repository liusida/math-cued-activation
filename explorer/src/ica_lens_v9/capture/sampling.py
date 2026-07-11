from __future__ import annotations

from collections.abc import Sequence

import torch


def sample_positions_by_doc(doc_lengths: list[int], *, token_budget: int, seed: int) -> dict[int, list[int]]:
    total_tokens = int(sum(doc_lengths))
    if token_budget > total_tokens:
        raise ValueError(f"Requested {token_budget} tokens, but only {total_tokens} candidates exist.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sampled = torch.randperm(total_tokens, generator=generator)[:token_budget]
    sampled, _ = sampled.sort()
    cumulative = torch.tensor(doc_lengths, dtype=torch.long).cumsum(dim=0)
    doc_indices = torch.bucketize(sampled, cumulative, right=True)
    starts = torch.zeros_like(sampled)
    nonzero = doc_indices > 0
    starts[nonzero] = cumulative[doc_indices[nonzero] - 1]
    positions = sampled - starts

    lengths = torch.tensor(doc_lengths, dtype=torch.long)
    if bool((positions < 0).any()) or bool((positions >= lengths[doc_indices]).any()):
        raise RuntimeError("Sampled an out-of-bounds token position.")

    selected: dict[int, list[int]] = {}
    for doc_id, position in zip(doc_indices.tolist(), positions.tolist(), strict=True):
        selected.setdefault(int(doc_id), []).append(int(position))
    return selected


def resolve_layers(requested: Sequence[str] | None, layer_names: Sequence[str]) -> list[str]:
    if not requested:
        return list(layer_names)
    resolved = []
    for item in requested:
        if item.isdigit():
            name = f"layer_{int(item):02d}"
        elif item.startswith("layer_"):
            name = item
        else:
            raise ValueError(f"Unsupported layer selector {item!r}; use an index or layer_XX.")
        if name not in layer_names:
            raise ValueError(f"Layer {name!r} is not available. Valid range: {layer_names[0]}..{layer_names[-1]}.")
        resolved.append(name)
    return resolved
