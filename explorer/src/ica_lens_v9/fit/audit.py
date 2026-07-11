from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from ..paths import V9_ROOT


DEFAULT_ICA_ROOT = V9_ROOT / "artifacts" / "ica"
DEFAULT_OUTPUT_ROOT = V9_ROOT / "results" / "fit_audit"


def audit_fastica_runs(
    *,
    ica_root: Path = DEFAULT_ICA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_names: list[str] | None = None,
) -> Path:
    ica_root = ica_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dirs = [ica_root / name for name in run_names] if run_names is not None else sorted(p for p in ica_root.iterdir() if p.is_dir())

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        if not (run_dir / "manifest.json").is_file():
            print(f"skipping incomplete run without manifest: {run_dir}")
            continue
        rows.extend(_audit_run(run_dir))
    _add_relative_flags(rows)

    output_csv = output_root / "fastica_fit_audit.csv"
    _write_rows(output_csv, rows)
    suspicious = [row for row in rows if row["status"] != "ok"]
    print(f"wrote FastICA fit audit: {output_csv}")
    print(f"audited {len(rows)} layers; suspicious={len(suspicious)}")
    for row in suspicious:
        print(
            f"{row['status']}: {row['run']}/{row['layer']} "
            f"final_lim={row['lim_final']:.4g} best_lim={row['lim_best']:.4g} "
            f"kurt_mean={row['kurtosis_final_mean']:.4g} reason={row['reason']}"
        )
    return output_csv


def _audit_run(run_dir: Path) -> list[dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = manifest.get("settings", {})
    max_iter = int(settings.get("max_iter", 0))
    layers = [str(layer) for layer in manifest.get("layers", [])]
    rows = []
    for layer in layers:
        history_path = run_dir / f"{layer}_history.csv"
        metadata_path = run_dir / f"{layer}_fastica.json"
        artifact_path = run_dir / f"{layer}_fastica.pt"
        missing = [str(path.name) for path in (history_path, metadata_path, artifact_path) if not path.exists()]
        if missing:
            rows.append(
                {
                    "run": run_dir.name,
                    "model": manifest.get("model", {}).get("short_name", ""),
                    "layer": layer,
                    "status": "missing",
                    "reason": "missing " + ", ".join(missing),
                    "iterations": 0,
                    "max_iter": max_iter,
                    "converged": False,
                    "lim_final": float("nan"),
                    "lim_best": float("nan"),
                    "lim_tail_min": float("nan"),
                    "kurtosis_final_mean": float("nan"),
                    "kurtosis_best_mean": float("nan"),
                    "kurtosis_gain_10_to_final": float("nan"),
                    "logcosh_final_mean": float("nan"),
                }
            )
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        history = _read_history(history_path)
        lim_values = [float(row["lim_max"]) for row in history]
        kurtosis_values = [float(row["excess_kurtosis_mean"]) for row in history]
        logcosh_values = [float(row["logcosh_mean"]) for row in history]
        tail = lim_values[-20:] if len(lim_values) >= 20 else lim_values
        final_kurtosis = _metadata_summary_value(metadata, "final_excess_kurtosis_summary", "mean")
        if final_kurtosis is None:
            final_kurtosis = kurtosis_values[-1]
        final_logcosh = _metadata_summary_value(metadata, "final_logcosh_summary", "mean")
        if final_logcosh is None:
            final_logcosh = logcosh_values[-1]
        n_components = int(metadata.get("n_components", len(history[0]) if history else 0))
        reason = _basic_reason(
            history_rows=len(history),
            max_iter=max_iter,
            final_lim=lim_values[-1],
            best_lim=min(lim_values),
            final_kurtosis=final_kurtosis,
            best_kurtosis=max(kurtosis_values),
        )
        rows.append(
            {
                "run": run_dir.name,
                "model": manifest.get("model", {}).get("short_name", ""),
                "layer": layer,
                "status": "ok" if reason == "" else "check",
                "reason": reason,
                "iterations": int(metadata.get("iterations", len(history))),
                "max_iter": max_iter,
                "converged": bool(metadata.get("converged", False)),
                "lim_final": lim_values[-1],
                "lim_best": min(lim_values),
                "lim_tail_min": min(tail),
                "kurtosis_final_mean": final_kurtosis,
                "kurtosis_final_sum": final_kurtosis * n_components,
                "kurtosis_best_mean": max(kurtosis_values),
                "kurtosis_best_sum": max(kurtosis_values) * n_components,
                "kurtosis_gain_10_to_final": final_kurtosis - kurtosis_values[min(9, len(kurtosis_values) - 1)],
                "logcosh_final_mean": final_logcosh,
                "logcosh_final_sum": final_logcosh * n_components,
                "logcosh_best_mean": max(logcosh_values),
                "logcosh_best_sum": max(logcosh_values) * n_components,
            }
        )
    return rows


def _basic_reason(
    *,
    history_rows: int,
    max_iter: int,
    final_lim: float,
    best_lim: float,
    final_kurtosis: float,
    best_kurtosis: float,
) -> str:
    reasons = []
    if history_rows == 0:
        reasons.append("empty history")
    if max_iter > 0 and history_rows < max_iter and final_lim >= 1e-4:
        reasons.append("short history without tol convergence")
    if final_kurtosis < 3.0:
        reasons.append("low final excess kurtosis")
    if best_kurtosis > 0 and final_kurtosis < 0.75 * best_kurtosis:
        reasons.append("lost >25% of best history kurtosis")
    if best_lim > 0 and final_lim > 20.0 * best_lim and final_lim > 0.02:
        reasons.append("final lim much worse than best lim")
    return "; ".join(reasons)


def _add_relative_flags(rows: list[dict[str, Any]]) -> None:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_run.setdefault(str(row["run"]), []).append(row)
    for run_rows in by_run.values():
        values = [float(row["kurtosis_final_mean"]) for row in run_rows if row["status"] != "missing"]
        if not values:
            continue
        median_kurtosis = statistics.median(values)
        for row in run_rows:
            if row["status"] == "missing":
                continue
            row["run_median_kurtosis_final_mean"] = median_kurtosis
            if median_kurtosis > 0 and float(row["kurtosis_final_mean"]) < 0.5 * median_kurtosis:
                row["status"] = "check"
                row["reason"] = _append_reason(str(row["reason"]), "low kurtosis relative to same run")


def _append_reason(old: str, new: str) -> str:
    return new if old == "" else old + "; " + new


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metadata_summary_value(metadata: dict[str, Any], summary_key: str, stat_key: str) -> float | None:
    summary = metadata.get(summary_key)
    if not isinstance(summary, dict) or stat_key not in summary:
        return None
    return float(summary[stat_key])


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No audit rows to write.")
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "status",
        "reason",
        "run",
        "model",
        "layer",
        "iterations",
        "max_iter",
        "converged",
        "lim_final",
        "lim_best",
        "lim_tail_min",
        "kurtosis_final_mean",
        "kurtosis_final_sum",
        "kurtosis_best_mean",
        "kurtosis_best_sum",
        "kurtosis_gain_10_to_final",
        "run_median_kurtosis_final_mean",
        "logcosh_final_mean",
        "logcosh_final_sum",
        "logcosh_best_mean",
        "logcosh_best_sum",
    ]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
