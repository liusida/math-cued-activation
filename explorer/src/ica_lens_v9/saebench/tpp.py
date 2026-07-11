from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .adapters import load_selected_sae
from .config import (
    ALL_METHODS,
    DEFAULT_FEATURE_INTERFACE_ROOT,
    DEFAULT_METHOD,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAEBENCH_ARTIFACTS,
    SAEBENCH_MODEL_NAMES,
    RunTarget,
    official_targets,
    preset_settings,
)
from .runner_utils import (
    activation_manifest_for_feature_interface,
    dry_run_payload,
    feature_dir_for_model,
    patch_qwen_sparse_or_tpp_loader,
    run_parent,
    setup_saebench_runtime,
    str_to_dtype,
    write_worker_result,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v9 ICA-SAE comparison with SAEBench TPP.")
    parser.add_argument("--models", nargs="*", default=None, choices=list(SAEBENCH_MODEL_NAMES))
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--methods", nargs="+", default=["ica_lens", "sae_baseline"], choices=[*ALL_METHODS, "all"])
    parser.add_argument("--preset", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT / "tpp")
    parser.add_argument("--feature-interface-root", type=Path, default=DEFAULT_FEATURE_INTERFACE_ROOT)
    parser.add_argument("--feature-interface-method", default=DEFAULT_METHOD)
    parser.add_argument("--saebench-artifacts-path", type=Path, default=DEFAULT_SAEBENCH_ARTIFACTS / "tpp")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--n-values", nargs="+", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--probe-epochs", type=int, default=None)
    parser.add_argument("--llm-batch-size", type=int, default=None)
    parser.add_argument("--sae-batch-size", type=int, default=None)
    parser.add_argument("--llm-dtype", default="float32")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--layer", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--method", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker:
        if args.model is None or args.layer is None or args.method is None:
            raise ValueError("--worker requires --model, --layer, --method")
        row = run_worker(args)
        print(json.dumps(row, indent=2, sort_keys=True, default=str))
        return

    targets = official_targets(models=args.models, layers=args.layers, methods=args.methods)
    if args.dry_run:
        print(json.dumps(dry_run_payload(targets=targets, output_root=args.output_root, saebench_artifacts_path=args.saebench_artifacts_path, preset=str(args.preset)), indent=2))
        return
    extra_args = [
        "--feature-interface-root",
        str(args.feature_interface_root.resolve()),
        "--feature-interface-method",
        str(args.feature_interface_method),
        "--llm-dtype",
        str(args.llm_dtype),
    ]
    if args.llm_batch_size is not None:
        extra_args += ["--llm-batch-size", str(args.llm_batch_size)]
    if args.sae_batch_size is not None:
        extra_args += ["--sae-batch-size", str(args.sae_batch_size)]
    if args.datasets is not None:
        extra_args += ["--datasets", *[str(item) for item in args.datasets]]
    if args.n_values is not None:
        extra_args += ["--n-values", *[str(item) for item in args.n_values]]
    if args.train_size is not None:
        extra_args += ["--train-size", str(args.train_size)]
    if args.test_size is not None:
        extra_args += ["--test-size", str(args.test_size)]
    if args.probe_epochs is not None:
        extra_args += ["--probe-epochs", str(args.probe_epochs)]
    run_parent(
        script_path=Path(__file__).resolve().parents[3] / "scripts" / "run_saebench_tpp.py",
        targets=targets,
        output_root=args.output_root,
        saebench_artifacts_path=args.saebench_artifacts_path,
        preset=str(args.preset),
        task="tpp",
        force_rerun=bool(args.force_rerun),
        save_activations=bool(args.save_activations),
        extra_args=extra_args,
    )


def run_worker(args: argparse.Namespace) -> dict[str, object]:
    model = str(args.model)
    layer = str(args.layer)
    method = str(args.method)
    settings = preset_settings("tpp", str(args.preset))  # type: ignore[arg-type]
    device = setup_saebench_runtime(model)
    from sae_bench.evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
    import sae_bench.evals.scr_and_tpp.main as tpp_main

    patch_qwen_sparse_or_tpp_loader(tpp_main, model_name=SAEBENCH_MODEL_NAMES[model])
    dtype = str_to_dtype(str(args.llm_dtype))
    feature_dir = feature_dir_for_model(model, feature_interface_root=args.feature_interface_root, method=str(args.feature_interface_method))
    activation_manifest = activation_manifest_for_feature_interface(feature_dir)

    eval_config = ScrAndTppEvalConfig(model_name=SAEBENCH_MODEL_NAMES[model], perform_scr=False)
    eval_config.dataset_names = list(args.datasets) if args.datasets is not None else list(settings.dataset_names)
    eval_config.n_values = list(args.n_values) if args.n_values is not None else list(settings.tpp_n_values)
    eval_config.train_set_size = int(args.train_size) if args.train_size is not None else int(settings.tpp_train_size)
    eval_config.test_set_size = int(args.test_size) if args.test_size is not None else int(settings.tpp_test_size)
    eval_config.context_length = int(settings.context_length)
    eval_config.probe_epochs = int(args.probe_epochs) if args.probe_epochs is not None else (4 if str(args.preset) == "smoke" else 20)
    eval_config.early_stopping_patience = eval_config.probe_epochs
    eval_config.llm_batch_size = int(args.llm_batch_size) if args.llm_batch_size is not None else 1
    eval_config.sae_batch_size = int(args.sae_batch_size) if args.sae_batch_size is not None else 125
    eval_config.llm_dtype = str(args.llm_dtype)
    eval_config.random_seed = 42
    eval_config.lower_vram_usage = False

    started = time.time()
    selected_saes, method_name, metadata = load_selected_sae(
        method=method,
        model=model,
        layer=layer,
        feature_interface_dir=feature_dir,
        output_root=args.output_root,
        activation_manifest_path=activation_manifest,
        device=device,
        dtype=dtype,
        force=bool(args.force_rerun),
    )
    saebench_output_path = args.output_root / "runs" / "tpp" / "saebench" / model / layer / method_name
    result = tpp_main.run_eval(
        eval_config,
        selected_saes=selected_saes,
        device=device,
        output_path=str(saebench_output_path),
        force_rerun=bool(args.force_rerun),
        clean_up_activations=not bool(args.save_activations or settings.save_activations),
        save_activations=bool(args.save_activations or settings.save_activations),
        artifacts_path=str(args.saebench_artifacts_path),
    )
    payload = _lookup_result(result, selected_saes[0][0], output_path=saebench_output_path)
    return write_worker_result(
        output_root=args.output_root,
        task="tpp",
        target=RunTarget(model=model, layer=layer, method=method),  # type: ignore[arg-type]
        method_name=method_name,
        elapsed_seconds=round(time.time() - started, 3),
        result=payload,
        metadata=metadata,
    )


def _lookup_result(result: dict[str, Any], name: str, *, output_path: Path) -> dict[str, Any]:
    if len(result) == 1:
        return next(iter(result.values()))
    if name in result:
        return result[name]
    if result:
        return next(iter(result.values()))
    candidates = sorted(output_path.glob(f"{name}_eval_results.json"))
    if not candidates:
        candidates = sorted(output_path.glob("*_eval_results.json"))
    if not candidates:
        raise RuntimeError(f"SAEBench returned no result and no cached result JSON was found in {output_path}.")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
