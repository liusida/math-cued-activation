from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from ..io_utils import load_json
from ..layers import activation_layers
from ..paths import REPO_ROOT
from ..torch_utils import resolve_device, summary as tensor_summary, torch_dtype
from .artifacts import environment_report, save_fastica_artifact, vendor_report, write_metric_history_csv
from .fastica import fit_fastica_with_metrics
from .inputs import activation_manifest_path, load_activation_config, load_layer_activations, resolve_fit_layers
from .wandb import init_wandb


DEFAULT_ACTIVATION_ROOT = Path("/home/liusida/research/ICA-paper/data/activations_v9")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "v9" / "artifacts" / "ica"
DEFAULT_WANDB_TOKEN_FILE = REPO_ROOT / "v9" / "API_tokens" / ".wandb_token"
DEFAULT_WANDB_ENTITY = "liusida"
DEFAULT_WANDB_PROJECT = "ICA-paper"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit full-rank FastICA (c=d) after row-normalization, centering, and whitening. "
            "Records per-iteration log-cosh and excess-kurtosis summaries."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Activation or fit config TOML.")
    parser.add_argument("--activation-root", type=Path, default=DEFAULT_ACTIVATION_ROOT)
    parser.add_argument("--activation-manifest", type=Path, default=None)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--fit-rows", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--algorithm", choices=("parallel",), default="parallel")
    parser.add_argument("--norm-eps", type=float, default=1e-12)
    parser.add_argument("--logcosh-alpha", type=float, default=1.0)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True, help="Log fit metrics to Weights & Biases. Defaults to liusida/ICA-paper; pass --no-wandb to disable.")
    parser.add_argument("--wandb-token-file", type=Path, default=DEFAULT_WANDB_TOKEN_FILE)
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    fit_fastica_run(
        config=args.config,
        activation_root=args.activation_root,
        activation_manifest=args.activation_manifest,
        token_budget=args.token_budget,
        output_root=args.output_root,
        layers=args.layers,
        fit_rows=args.fit_rows,
        device_name=args.device,
        dtype_name=args.dtype,
        seed=int(args.seed),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        norm_eps=float(args.norm_eps),
        logcosh_alpha=float(args.logcosh_alpha),
        progress=bool(args.progress),
        wandb=bool(args.wandb),
        wandb_token_file=args.wandb_token_file,
        wandb_entity=str(args.wandb_entity),
        wandb_project=str(args.wandb_project),
        wandb_run_name=args.wandb_run_name,
        wandb_mode=str(args.wandb_mode),
        force=bool(args.force),
    )


def fit_fastica_run(
    *,
    config: Path,
    activation_root: Path = DEFAULT_ACTIVATION_ROOT,
    activation_manifest: Path | None = None,
    token_budget: int | None = None,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    layers: list[str] | None = None,
    fit_rows: int | None = None,
    device_name: str = "cuda",
    dtype_name: str = "float32",
    seed: int = 0,
    max_iter: int = 200,
    tol: float = 1e-4,
    norm_eps: float = 1e-12,
    logcosh_alpha: float = 1.0,
    progress: bool = True,
    wandb: bool = True,
    wandb_token_file: Path = DEFAULT_WANDB_TOKEN_FILE,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_run_name: str | None = None,
    wandb_mode: str = "online",
    force: bool = False,
) -> Path:
    started_at = time.time()

    activation_cfg = load_activation_config(config)
    token_budget = int(token_budget or activation_cfg["dataset"]["token_budget"])
    manifest_path = activation_manifest_path(
        explicit=activation_manifest,
        activation_root=activation_root,
        activation_cfg=activation_cfg,
        token_budget=token_budget,
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing activation manifest: {manifest_path}")

    manifest = load_json(manifest_path)
    model_short_name = str(activation_cfg["model"]["short_name"])
    layers = resolve_fit_layers(layers, manifest)
    device = resolve_device(device_name)
    dtype = torch_dtype(dtype_name)
    hidden_size = int(manifest["model"]["hidden_size"])
    n_components = hidden_size

    run_dir = output_root.resolve() / f"{model_short_name}_tok{token_budget}_c{hidden_size}_iter{max_iter}"
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(
        enabled=wandb,
        token_file=wandb_token_file,
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
        run_name=wandb_run_name or f"v9_fastica_{model_short_name}_tok{token_budget}_c{hidden_size}_iter{max_iter}",
        run_dir=run_dir,
        config={
            "model_short_name": model_short_name,
            "model": manifest.get("model", {}),
            "activation_manifest": str(manifest_path),
            "layers": layers,
            "preprocess": "row_normalize_center_whiten",
            "n_components": n_components,
            "component_rule": "c=d",
            "token_budget": token_budget,
            "fit_rows": fit_rows,
            "device": str(device),
            "dtype": dtype_name,
            "seed": int(seed),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "algorithm": "parallel",
            "nonlinearity": "logcosh",
            "logcosh_alpha": float(logcosh_alpha),
            "norm_eps": float(norm_eps),
            "output_dir": str(run_dir),
        },
    )

    summaries: dict[str, Any] = {}
    wandb_step = 0
    for layer in layers:
        artifact_path = run_dir / f"{layer}_fastica.pt"
        history_path = run_dir / f"{layer}_history.csv"
        if artifact_path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {artifact_path}; pass --force.")

        activations = load_layer_activations(
            capture_dir=manifest_path.parent,
            manifest=manifest,
            layer=layer,
            fit_rows=fit_rows,
            device=device,
            dtype=dtype,
        )
        if n_components > min(int(activations.shape[0]), int(activations.shape[1])):
            raise ValueError(
                f"c=d requires at least d rows. Got activations shape {tuple(activations.shape)} "
                f"and d={hidden_size}."
            )

        layer_index = activation_layers(manifest).index(layer)
        layer_seed = int(seed) + layer_index
        def log_metric_row(row: dict[str, float | int]) -> None:
            nonlocal wandb_step
            if wandb_run is None:
                return
            payload = {
                "global_step": wandb_step,
                "iteration": int(row["iteration"]),
                "layer": layer,
                "layer_index": layer_index,
                **{f"{layer}/{key}": value for key, value in row.items() if key not in {"iteration"}},
            }
            wandb_run.log(payload, step=wandb_step)
            wandb_step += 1

        artifact = fit_fastica_with_metrics(
            activations,
            n_components=n_components,
            seed=layer_seed,
            max_iter=int(max_iter),
            tol=float(tol),
            norm_eps=float(norm_eps),
            logcosh_alpha=float(logcosh_alpha),
            progress=bool(progress),
            metric_callback=log_metric_row,
        )
        metadata = {
            "layer": layer,
            "activation_manifest": str(manifest_path),
            "activation_shape": list(activations.shape),
            "artifact_format": "torch_save_tensors_and_metadata",
            "config": str(config.resolve()),
            "metric_history_csv": str(history_path),
            "fastica_torch": vendor_report(),
            "environment": environment_report(),
            "component_id_convention": (
                "Component id is the row index in saved FastICA tensors. No post-fit sorting, "
                "sign canonicalization, or renumbering is applied."
            ),
        }
        save_fastica_artifact(artifact_path, artifact, metadata)
        write_metric_history_csv(history_path, artifact["metric_history"])
        summaries[layer] = load_json(artifact_path.with_suffix(".json"))
        if wandb_run is not None:
            wandb_run.log(
                {
                    "global_step": wandb_step,
                    "layer": layer,
                    "layer_index": layer_index,
                    f"{layer}/final_lim": artifact["final_lim"],
                    f"{layer}/iterations": artifact["iterations"],
                    f"{layer}/converged": int(bool(artifact["converged"])),
                    **{f"{layer}/final_logcosh_{key}": value for key, value in artifact["final_logcosh_summary"].items()},
                    **{
                        f"{layer}/final_excess_kurtosis_{key}": value
                        for key, value in artifact["final_excess_kurtosis_summary"].items()
                    },
                },
                step=wandb_step,
            )
            wandb_step += 1
        del activations, artifact
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_manifest = {
        "artifact": f"v9_fastica_{model_short_name}_tok{token_budget}_c{hidden_size}_iter{max_iter}",
        "purpose": "Full-rank FastICA after row-normalization, centering, and whitening with logcosh/kurtosis history.",
        "model": manifest["model"],
        "activation_manifest": str(manifest_path),
        "output_dir": str(run_dir),
        "layers": layers,
        "settings": {
            "preprocess": "row_normalize_center_whiten",
            "n_components": n_components,
            "component_rule": "c=d",
            "token_budget": token_budget,
            "fit_rows": fit_rows,
            "device": str(device),
            "dtype": dtype_name,
            "seed": int(seed),
            "layer_seed_policy": "base_seed + layer_index_in_activation_manifest",
            "max_iter": int(max_iter),
            "tol": float(tol),
            "algorithm": "parallel",
            "nonlinearity": "logcosh",
            "logcosh_alpha": float(logcosh_alpha),
            "norm_eps": float(norm_eps),
        },
        "layer_summaries": summaries,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    if wandb_run is not None:
        wandb_run.summary.update({"elapsed_seconds": run_manifest["elapsed_seconds"], "output_dir": str(run_dir)})
        wandb_run.finish()
    print(f"wrote v9 FastICA artifacts: {run_dir}")
    return run_dir

if __name__ == "__main__":
    main()
