#!/usr/bin/env python3
"""Fit FastICA on Qwen and/or VibeThinker IMO activations."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


DATASET_SLUG = "OpenEvals__IMO-AnswerBench"
QWEN_SLUG = "Qwen__Qwen2.5-Coder-3B-Instruct"
VIBETHINKER_SLUG = "WeiboAI__VibeThinker-3B"
DEFAULT_VIBETHINKER_ONLY_ACTIVATIONS = 1_000_000


@dataclass(frozen=True)
class ActivationFile:
    path: Path
    rows: int
    hidden_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load saved activation rows, L2-normalize them, and fit c=d FastICA. "
            "Default fits 1M sampled VibeThinker activation rows."
        )
    )
    parser.add_argument(
        "--activation-root",
        type=Path,
        default=Path("~/data/ICA-data/math-cued-activation"),
    )
    parser.add_argument("--dataset-slug", default=DATASET_SLUG)
    parser.add_argument("--qwen-slug", default=QWEN_SLUG)
    parser.add_argument("--vibethinker-slug", default=VIBETHINKER_SLUG)
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument(
        "--source",
        choices=["vibethinker", "qwen", "mixed"],
        default="vibethinker",
        help="Activation source to fit. mixed keeps the old Qwen/VibeThinker balanced path.",
    )
    parser.add_argument(
        "--max-qwen-activations",
        "--max-qwen-activation",
        dest="max_qwen_activations",
        type=int,
        help="Maximum Qwen activation rows/tokens to use. Default: all Qwen rows.",
    )
    parser.add_argument(
        "--max-vibethinker-activations",
        "--max-vibethinker-activation",
        dest="max_vibethinker_activations",
        type=int,
        help=(
            "Maximum VibeThinker activation rows/tokens to sample. "
            "Default: 1,000,000 for --source vibethinker; same count as selected Qwen rows for --source mixed."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--fun", choices=["logcosh", "exp", "cube"], default="logcosh")
    parser.add_argument("--whiten-solver", choices=["svd", "eigh"], default="eigh")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a torch device string such as cuda:0.",
    )
    parser.add_argument(
        "--fit-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Dtype for normalized training matrix and FastICA fit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .pt artifact. Default is under results/ica/.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars, including FastICA iteration progress.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Index files and report selected row counts without loading tensors or fitting ICA.",
    )
    parser.add_argument(
        "--write-manifest-only",
        type=Path,
        help="Load an existing .pt artifact and write its .json manifest without fitting ICA.",
    )
    return parser.parse_args()


def layer_dir(root: Path, dataset_slug: str, model_slug: str, layer: int) -> Path:
    return root.expanduser() / dataset_slug / model_slug / f"layer_{layer:02d}"


def load_activation_shape(path: Path) -> tuple[int, int]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if "activation_shape" in bundle:
        shape = tuple(int(dim) for dim in bundle["activation_shape"])
    else:
        shape = tuple(int(dim) for dim in bundle["activations"].shape)
    if len(shape) != 2:
        raise ValueError(f"Expected 2D activations in {path}, got shape {shape}")
    return shape


def list_activation_files(folder: Path, *, desc: str, progress: bool) -> list[ActivationFile]:
    paths = sorted(folder.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No activation .pt files found in {folder}")

    files: list[ActivationFile] = []
    iterator: Iterable[Path] = paths
    if progress:
        iterator = tqdm(paths, desc=f"Index {desc}", unit="file")

    hidden_size: int | None = None
    for path in iterator:
        rows, width = load_activation_shape(path)
        if hidden_size is None:
            hidden_size = width
        elif width != hidden_size:
            raise ValueError(f"Hidden size mismatch in {path}: {width} != {hidden_size}")
        files.append(ActivationFile(path=path, rows=rows, hidden_size=width))
    return files


def choose_global_rows(total_rows: int, max_rows: int | None, rng: random.Random) -> list[int]:
    if max_rows is None or max_rows >= total_rows:
        return list(range(total_rows))
    if max_rows <= 0:
        raise ValueError("Activation caps must be positive.")
    return sorted(rng.sample(range(total_rows), max_rows))


def rows_by_file(files: list[ActivationFile], selected_rows: list[int]) -> dict[Path, list[int]]:
    out: dict[Path, list[int]] = {}
    cursor = 0
    selected_cursor = 0
    n_selected = len(selected_rows)
    for file in files:
        start = cursor
        end = cursor + file.rows
        local_rows: list[int] = []
        while selected_cursor < n_selected and selected_rows[selected_cursor] < end:
            global_row = selected_rows[selected_cursor]
            if global_row >= start:
                local_rows.append(global_row - start)
            selected_cursor += 1
        if local_rows:
            out[file.path] = local_rows
        cursor = end
    return out


def load_selected_activations(
    files: list[ActivationFile],
    selected_rows: list[int],
    *,
    label: str,
    dtype: torch.dtype,
    progress: bool,
) -> tuple[torch.Tensor, list[dict]]:
    file_rows = rows_by_file(files, selected_rows)
    chunks: list[torch.Tensor] = []
    manifest: list[dict] = []
    iterator: Iterable[tuple[Path, list[int]]] = file_rows.items()
    if progress:
        iterator = tqdm(list(file_rows.items()), desc=f"Load {label}", unit="file")

    for path, local_rows in iterator:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        activations = bundle["activations"]
        row_index = torch.tensor(local_rows, dtype=torch.long)
        chunk = activations.index_select(0, row_index).to(dtype=dtype)
        chunk = F.normalize(chunk, p=2, dim=1, eps=1e-12)
        chunks.append(chunk)
        manifest.append(
            {
                "path": str(path),
                "problem_id": bundle.get("problem", {}).get("problem_id"),
                "selected_rows": len(local_rows),
                "activation_shape": tuple(int(dim) for dim in activations.shape),
            }
        )

    if not chunks:
        raise ValueError(f"No selected activation rows for {label}.")
    return torch.cat(chunks, dim=0), manifest


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def default_output_path(source: str, layer: int, hidden_size: int, max_iter: int) -> Path:
    source_slug = {
        "vibethinker": "vibethinker_only",
        "qwen": "qwen_only",
        "mixed": "qwen_vibethinker_mixed",
    }[source]
    return Path("results") / "ica" / (
        f"{source_slug}_layer{layer:02d}_c{hidden_size}_iter{max_iter}.pt"
    )


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_manifest(artifact: dict, output: Path) -> Path:
    manifest_path = output.with_suffix(".json")
    manifest = {
        "schema": artifact["schema"],
        "artifact": str(output),
        "n_iter": int(artifact["n_iter"]),
        "lim_history_tail": [float(x) for x in artifact.get("lim_history", [])[-10:]],
        "config": json_safe(artifact["config"]),
        "data": {
            key: {
                subkey: value
                for subkey, value in section.items()
                if subkey != "files"
            }
            for key, section in artifact["data"].items()
            if isinstance(section, dict)
        },
    }
    data = artifact["data"]
    manifest["data"]["dataset_slug"] = data["dataset_slug"]
    manifest["data"]["layer"] = data["layer"]
    manifest["data"]["hidden_size"] = data["hidden_size"]
    manifest["data"]["normalization"] = data["normalization"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.write_manifest_only is not None:
        output = args.write_manifest_only.expanduser()
        artifact = torch.load(output, map_location="cpu", weights_only=False)
        manifest_path = write_manifest(artifact, output)
        print(f"Saved manifest: {manifest_path}")
        return

    progress = not args.no_progress
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    fit_dtype = {"float32": torch.float32, "float64": torch.float64}[args.fit_dtype]
    qwen_dir = layer_dir(args.activation_root, args.dataset_slug, args.qwen_slug, args.layer)
    vibethinker_dir = layer_dir(
        args.activation_root, args.dataset_slug, args.vibethinker_slug, args.layer
    )

    qwen_files: list[ActivationFile] = []
    vibethinker_files: list[ActivationFile] = []
    if args.source in {"qwen", "mixed"}:
        qwen_files = list_activation_files(qwen_dir, desc="Qwen", progress=progress)
    if args.source in {"vibethinker", "mixed"}:
        vibethinker_files = list_activation_files(
            vibethinker_dir, desc="VibeThinker", progress=progress
        )

    hidden_sizes = {
        files[0].hidden_size
        for files in (qwen_files, vibethinker_files)
        if files
    }
    if len(hidden_sizes) != 1:
        raise ValueError(f"Hidden size mismatch across selected sources: {sorted(hidden_sizes)}")
    hidden_size = hidden_sizes.pop()

    qwen_total = sum(file.rows for file in qwen_files)
    vibethinker_total = sum(file.rows for file in vibethinker_files)
    qwen_target = (
        min(qwen_total, args.max_qwen_activations or qwen_total)
        if qwen_files
        else 0
    )
    if args.source == "vibethinker":
        vibethinker_cap = args.max_vibethinker_activations or DEFAULT_VIBETHINKER_ONLY_ACTIVATIONS
    elif args.source == "mixed":
        vibethinker_cap = args.max_vibethinker_activations or qwen_target
    else:
        vibethinker_cap = 0
    vibethinker_target = min(vibethinker_total, vibethinker_cap) if vibethinker_files else 0

    print(f"Source: {args.source}")
    if qwen_files:
        print(f"Qwen activation rows available: {qwen_total:,}")
        print(f"Selected Qwen rows: {qwen_target:,}")
    if vibethinker_files:
        print(f"VibeThinker activation rows available: {vibethinker_total:,}")
        print(f"Selected VibeThinker rows: {vibethinker_target:,}")
    print(f"Hidden size / ICA components: {hidden_size}")

    if args.dry_run:
        return

    selected_total = qwen_target + vibethinker_target
    if selected_total < hidden_size:
        raise SystemExit(
            f"Need at least {hidden_size:,} total selected rows for c=d FastICA, "
            f"got {selected_total:,}. Increase the activation caps."
        )

    chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    data_sections: dict[str, dict] = {}

    if qwen_target:
        qwen_rows = choose_global_rows(qwen_total, qwen_target, rng)
        qwen_X, qwen_manifest = load_selected_activations(
            qwen_files, qwen_rows, label="Qwen", dtype=fit_dtype, progress=progress
        )
        chunks.append(qwen_X)
        label_chunks.append(torch.zeros(qwen_X.shape[0], dtype=torch.long))
        data_sections["qwen"] = {
            "model_slug": args.qwen_slug,
            "available_rows": qwen_total,
            "selected_rows": int(qwen_X.shape[0]),
            "files": qwen_manifest,
        }

    if vibethinker_target:
        vibethinker_rows = choose_global_rows(vibethinker_total, vibethinker_target, rng)
        vibethinker_X, vibethinker_manifest = load_selected_activations(
            vibethinker_files,
            vibethinker_rows,
            label="VibeThinker",
            dtype=fit_dtype,
            progress=progress,
        )
        chunks.append(vibethinker_X)
        label_chunks.append(torch.ones(vibethinker_X.shape[0], dtype=torch.long))
        data_sections["vibethinker"] = {
            "model_slug": args.vibethinker_slug,
            "available_rows": vibethinker_total,
            "selected_rows": int(vibethinker_X.shape[0]),
            "files": vibethinker_manifest,
        }

    X = torch.cat(chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)

    device = choose_device(args.device)
    print(f"Training matrix: {tuple(X.shape)} {X.dtype}")
    print(f"FastICA device: {device}")
    X = X.to(device)

    from fastica_torch import FastICA

    ica = FastICA(
        n_components=hidden_size,
        algorithm="parallel",
        whiten="unit-variance",
        fun=args.fun,
        max_iter=args.max_iter,
        tol=args.tol,
        whiten_solver=args.whiten_solver,
        random_state=args.seed,
        progress=progress,
    )
    ica.fit(X)

    output = args.output or default_output_path(args.source, args.layer, hidden_size, args.max_iter)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema": "math_cued_fastica_v1",
        "components": ica.components_.detach().cpu(),
        "mixing": ica.mixing_.detach().cpu(),
        "mean": None if ica.mean_ is None else ica.mean_.detach().cpu(),
        "whitening": None if ica.whitening_ is None else ica.whitening_.detach().cpu(),
        "unmixing": None if ica._unmixing is None else ica._unmixing.detach().cpu(),
        "n_iter": int(ica.n_iter_),
        "lim_history": [float(x) for x in getattr(ica, "lim_history_", [])],
        "labels": labels,
        "config": vars(args),
        "data": {
            "dataset_slug": args.dataset_slug,
            "source": args.source,
            "layer": args.layer,
            "hidden_size": hidden_size,
            "normalization": "per-row L2 before FastICA whitening",
            **data_sections,
        },
    }
    torch.save(artifact, output)

    manifest_path = write_manifest(artifact, output)

    print(f"Saved ICA artifact: {output}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
