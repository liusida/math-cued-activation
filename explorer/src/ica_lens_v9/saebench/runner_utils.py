from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from ..io_utils import load_json
from ..paths import V9_ROOT
from .config import DEFAULT_FEATURE_INTERFACE_ROOT, DEFAULT_METHOD, DEFAULT_OUTPUT_ROOT, RunTarget, feature_interface_dir, layer_index, saebench_env


def run_parent(
    *,
    script_path: Path,
    targets: list[RunTarget],
    output_root: Path,
    saebench_artifacts_path: Path,
    preset: str,
    task: str,
    force_rerun: bool,
    save_activations: bool,
    extra_args: list[str] | None = None,
) -> None:
    extra_args = extra_args or []
    for index, target in enumerate(targets, start=1):
        _root, python = saebench_env(target.model)
        if not python.is_file():
            raise FileNotFoundError(
                f"Missing SAEBench interpreter: {python}. Run `bash scripts/setup_saebench_envs.sh` from v5 first."
            )
        command = [
            str(python),
            str(script_path),
            "--worker",
            "--model",
            target.model,
            "--layer",
            target.layer,
            "--method",
            target.method,
            "--preset",
            preset,
            "--output-root",
            str(output_root.resolve()),
            "--saebench-artifacts-path",
            str(saebench_artifacts_path.resolve()),
            *extra_args,
        ]
        if force_rerun:
            command.append("--force-rerun")
        if save_activations:
            command.append("--save-activations")
        print(f"[{index}/{len(targets)}] {task}: {target.model} {target.layer} {target.method}")
        subprocess.run(command, check=True, cwd=V9_ROOT)


def dry_run_payload(*, targets: list[RunTarget], output_root: Path, saebench_artifacts_path: Path, preset: str) -> dict[str, Any]:
    rows = []
    for target in targets:
        root, python = saebench_env(target.model)
        rows.append(
            {
                "model": target.model,
                "layer": target.layer,
                "method": target.method,
                "saebench_root": str(root),
                "saebench_python": str(python),
            }
        )
    return {
        "preset": preset,
        "evaluation_count": len(targets),
        "targets": rows,
        "output_root": str(output_root.resolve()),
        "saebench_artifacts_path": str(saebench_artifacts_path.resolve()),
    }


def activation_manifest_for_feature_interface(feature_dir: Path) -> Path:
    manifest = load_json(feature_dir / "manifest.json")
    return Path(str(manifest["source_activation_manifest"])).resolve()


def feature_dir_for_model(model: str, *, feature_interface_root: Path = DEFAULT_FEATURE_INTERFACE_ROOT, method: str = DEFAULT_METHOD) -> Path:
    return feature_interface_dir(model, feature_interface_root=feature_interface_root, method=method)


def setup_saebench_runtime(model: str) -> str:
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    root, _python = saebench_env(model)
    for path in (V9_ROOT / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from sae_bench.sae_bench_utils.general_utils import setup_environment

    return setup_environment()


def patch_qwen_sparse_or_tpp_loader(module: Any, *, model_name: str) -> None:
    if model_name != "Qwen/Qwen3.5-2B-Base":
        return
    from sae_bench.sae_bench_utils import activation_collection
    import sae_bench.evals.sparse_probing.probe_training as probe_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class Qwen35HookedTransformerShim:
        @staticmethod
        def from_pretrained_no_processing(model_name: str, device: str, dtype: Any) -> Any:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
            hf_model.to(device)
            hf_model.eval()
            return activation_collection.HFCausalLMActivationModel(hf_model, tokenizer)

    module.HookedTransformer = Qwen35HookedTransformerShim
    _patch_probe_test_dtype(probe_training)


def _patch_probe_test_dtype(probe_training: Any) -> None:
    if getattr(probe_training.test_probe_gpu, "_ica_lens_dtype_patch", False):
        return
    original_test_probe_gpu = probe_training.test_probe_gpu

    def test_probe_gpu_dtype_safe(inputs: Any, labels: Any, batch_size: int, probe: Any) -> float:
        probe_dtype = probe.net.weight.dtype
        if getattr(inputs, "dtype", None) != probe_dtype:
            inputs = inputs.to(dtype=probe_dtype)
        return original_test_probe_gpu(inputs, labels, batch_size, probe)

    test_probe_gpu_dtype_safe._ica_lens_dtype_patch = True  # type: ignore[attr-defined]
    probe_training.test_probe_gpu = test_probe_gpu_dtype_safe


def str_to_dtype(value: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }[value]


def write_worker_result(
    *,
    output_root: Path,
    task: str,
    target: RunTarget,
    method_name: str,
    elapsed_seconds: float,
    result: dict[str, Any],
    metadata: dict[str, object],
) -> dict[str, object]:
    row = {
        "task": task,
        "model_name": target.model,
        "layer": target.layer,
        "method": method_name,
        "elapsed_seconds": elapsed_seconds,
        **metadata,
    }
    sae_cfg = result.get("sae_cfg_dict", {})
    if isinstance(sae_cfg, dict) and "d_sae" in sae_cfg:
        row.setdefault("n_saebench_features", sae_cfg.get("d_sae"))
    row.update(flatten_metrics(result.get("eval_result_metrics", {})))

    run_dir = output_root / "runs" / task
    raw_dir = run_dir / "raw" / target.model / target.layer / method_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    row_path = raw_dir / "row.json"
    row_path.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    collect_rows(output_root=output_root, task=task)
    return row


def collect_rows(*, output_root: Path, task: str) -> list[dict[str, object]]:
    run_dir = output_root / "runs" / task
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(run_dir.glob("raw/*/*/*/row.json"))]
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_dir / f"{task}.json", {"task": task, "rows": rows, "collected_at_unix": time.time()})
    _write_csv(summary_dir / f"{task}_long.csv", rows)
    _write_csv(summary_dir / f"{task}_summary.csv", summarize_rows(rows))
    return rows


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ["task", "model_name", "layer", "method", "n_saebench_features", "elapsed_seconds"]
    metric_keys = sorted(k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and k not in keys)
    return [{key: row.get(key) for key in keys + metric_keys} for row in rows]


def flatten_metrics(value: Any, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            out.update(flatten_metrics(child, name))
    elif isinstance(value, (int, float, str, bool)) or value is None:
        out[prefix] = value
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
