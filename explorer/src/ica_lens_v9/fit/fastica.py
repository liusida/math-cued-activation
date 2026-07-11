from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from tqdm.auto import tqdm

from ..paths import V5_SRC, VENDOR_FASTICA_SRC
from ..torch_utils import summary as tensor_summary


for path in (VENDOR_FASTICA_SRC, V5_SRC):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fastica_torch import FastICA  # noqa: E402
import fastica_torch.fastica as fastica_module  # noqa: E402


def fit_fastica_with_metrics(
    activations: torch.Tensor,
    *,
    n_components: int,
    seed: int,
    max_iter: int,
    tol: float,
    norm_eps: float,
    logcosh_alpha: float,
    progress: bool,
    metric_callback: Callable[[dict[str, float | int]], None] | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    metric_history: list[dict[str, float | int]] = []

    rows = int(activations.shape[0])
    hidden_size = int(activations.shape[1])
    x = _normalize_rows_inplace(activations, norm_eps=norm_eps)
    ica = FastICA(
        n_components=n_components,
        algorithm="parallel",
        # Avoid vendor unit-variance postprocessing because it materializes
        # the full sources matrix. We apply the same scaling in batches below.
        whiten="arbitrary-variance",
        whiten_solver="eigh",
        fun="logcosh",
        fun_args={"alpha": float(logcosh_alpha)},
        max_iter=max_iter,
        tol=tol,
        random_state=seed,
        float64_covariance=True,
        progress=progress,
    )

    with _recording_parallel_fastica(
        metric_history,
        started_at=started_at,
        logcosh_alpha=logcosh_alpha,
        metric_callback=metric_callback,
    ):
        ica.fit(x)

    raw_components = ica.components_.detach().clone()
    raw_unmixing = ica._unmixing.detach().clone()
    mean_device = ica.mean_.detach().clone().reshape(1, -1)
    source_std = _batched_source_std(x, mean=mean_device, components=raw_components)
    scaled_components = raw_components / source_std.unsqueeze(1).clamp_min(norm_eps)
    scaled_unmixing = raw_unmixing / source_std.unsqueeze(1).clamp_min(norm_eps)
    final_logcosh, final_excess_kurtosis = _batched_source_metrics(
        x,
        mean=mean_device,
        components=scaled_components,
        logcosh_alpha=logcosh_alpha,
    )

    components = scaled_components.cpu()
    mean = mean_device.cpu()
    whitening = ica.whitening_.detach().clone().cpu()
    unmixing = scaled_unmixing.cpu()
    directions = unmixing / torch.linalg.vector_norm(unmixing, dim=1, keepdim=True).clamp_min(norm_eps)
    final_logcosh = final_logcosh.cpu()
    final_excess_kurtosis = final_excess_kurtosis.cpu()
    lim_history = list(getattr(ica, "lim_history_", []))
    final_lim = float(lim_history[-1]) if lim_history else None
    if final_lim is None and metric_history:
        final_lim = float(metric_history[-1]["lim_max"])

    return {
        "method": "fastica",
        "algorithm": "parallel",
        "preprocess": "row_normalize_center_whiten",
        "mean": mean,
        "whitening": whitening,
        "components": components,
        "unmixing": unmixing,
        "directions": directions,
        "final_logcosh_per_component": final_logcosh,
        "final_excess_kurtosis_per_component": final_excess_kurtosis,
        "rows": rows,
        "hidden_size": hidden_size,
        "n_components": int(n_components),
        "iterations": int(ica.n_iter_),
        "converged": bool(int(ica.n_iter_) < max_iter),
        "lim_history": lim_history,
        "metric_history": metric_history,
        "final_lim": final_lim,
        "final_logcosh_summary": tensor_summary(final_logcosh),
        "final_excess_kurtosis_summary": tensor_summary(final_excess_kurtosis),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "fun": "logcosh",
        "logcosh_alpha": float(logcosh_alpha),
        "seed": int(seed),
        "norm_eps": float(norm_eps),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


@contextmanager
def _recording_parallel_fastica(
    metric_history: list[dict[str, float | int]],
    *,
    started_at: float,
    logcosh_alpha: float,
    metric_callback: Callable[[dict[str, float | int]], None] | None,
) -> Iterator[None]:
    original_ica_par = fastica_module._ica_par

    def wrapped_ica_par(*args: Any, **kwargs: Any) -> Any:
        return _ica_par_with_metrics(
            *args,
            metric_history=metric_history,
            started_at=started_at,
            logcosh_alpha=logcosh_alpha,
            metric_callback=metric_callback,
            **kwargs,
        )

    fastica_module._ica_par = wrapped_ica_par
    try:
        yield
    finally:
        fastica_module._ica_par = original_ica_par


def _ica_par_with_metrics(
    X: torch.Tensor,
    tol: float,
    g: Callable[..., Any],
    fun_args: dict[str, Any],
    max_iter: int,
    w_init: torch.Tensor,
    progress: bool = False,
    lim_history: list[float] | None = None,
    *,
    metric_history: list[dict[str, float | int]],
    started_at: float,
    logcosh_alpha: float,
    metric_callback: Callable[[dict[str, float | int]], None] | None,
) -> tuple[torch.Tensor, int]:
    W = fastica_module._sym_decorrelation(w_init)
    ii = 0
    iterator = range(max_iter)
    if progress:
        iterator = tqdm(iterator, total=max_iter, desc="FastICA parallel", unit="iter", dynamic_ncols=True)

    for ii in iterator:
        wtx = W @ X
        gwtx, g_wtx = g(wtx, fun_args)
        term1 = (gwtx @ X.T) / X.shape[1]
        term2 = g_wtx.unsqueeze(1) * W
        W1 = fastica_module._sym_decorrelation(term1 - term2)

        dot_products = torch.sum(W1 * W, dim=1)
        lim_vec = torch.abs(torch.abs(dot_products) - 1)
        lim = torch.max(lim_vec)
        if lim_history is not None:
            lim_history.append(float(lim.detach().cpu()))
        row = {
            "iteration": int(ii + 1),
            "elapsed_seconds": round(time.time() - started_at, 6),
            **{f"lim_{key}": value for key, value in tensor_summary(lim_vec).items()},
            **{f"logcosh_{key}": value for key, value in tensor_summary(_per_component_logcosh(wtx, alpha=logcosh_alpha)).items()},
            **{
                f"excess_kurtosis_{key}": value
                for key, value in tensor_summary(_per_component_excess_kurtosis(wtx)).items()
            },
        }
        metric_history.append(row)
        if metric_callback is not None:
            metric_callback(row)
        if progress:
            iterator.set_postfix(
                lim=f"{lim.item():.2e}",
                logcosh=f"{float(metric_history[-1]['logcosh_mean']):.3g}",
                kurt=f"{float(metric_history[-1]['excess_kurtosis_mean']):.3g}",
            )
        if not torch.isfinite(lim) or lim.item() > 1e20:
            raise RuntimeError(f"FastICA diverged at iter {ii}: lim={lim.item():.3e}")

        W = W1
        if lim < tol:
            break

    return W, ii + 1


def _per_component_logcosh(values: torch.Tensor, *, alpha: float) -> torch.Tensor:
    scaled = values * alpha
    log_two = math.log(2.0)
    logcosh = scaled + torch.nn.functional.softplus(-2.0 * scaled) - log_two
    return (logcosh / alpha).mean(dim=1)


def _per_component_excess_kurtosis(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=1, keepdim=True)
    second = (centered**2).mean(dim=1).clamp_min(torch.finfo(values.dtype).tiny)
    fourth = (centered**4).mean(dim=1)
    return fourth / (second**2) - 3.0


def _normalize_rows(values: torch.Tensor, *, norm_eps: float) -> torch.Tensor:
    return values / torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(norm_eps)


def _normalize_rows_inplace(values: torch.Tensor, *, norm_eps: float) -> torch.Tensor:
    values.div_(torch.linalg.vector_norm(values, dim=1, keepdim=True).clamp_min(norm_eps))
    return values


def _batched_source_std(
    x: torch.Tensor,
    *,
    mean: torch.Tensor,
    components: torch.Tensor,
    batch_size: int = 8192,
) -> torch.Tensor:
    n_components = int(components.shape[0])
    total = 0
    sum_ = torch.zeros(n_components, dtype=torch.float64, device="cpu")
    sumsq = torch.zeros(n_components, dtype=torch.float64, device="cpu")
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), batch_size):
            batch = x[start : start + batch_size]
            sources = (batch - mean) @ components.T
            total += int(sources.shape[0])
            sum_ += sources.sum(dim=0).detach().cpu().to(torch.float64)
            sumsq += (sources * sources).sum(dim=0).detach().cpu().to(torch.float64)
    mean_source = sum_ / float(total)
    var = (sumsq / float(total) - mean_source * mean_source).clamp_min(0.0)
    return torch.sqrt(var).to(device=components.device, dtype=components.dtype)


def _batched_source_metrics(
    x: torch.Tensor,
    *,
    mean: torch.Tensor,
    components: torch.Tensor,
    logcosh_alpha: float,
    batch_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_components = int(components.shape[0])
    total = 0
    sum1 = torch.zeros(n_components, dtype=torch.float64, device="cpu")
    sum2 = torch.zeros(n_components, dtype=torch.float64, device="cpu")
    sum3 = torch.zeros(n_components, dtype=torch.float64, device="cpu")
    sum4 = torch.zeros(n_components, dtype=torch.float64, device="cpu")
    logcosh_sum = torch.zeros(n_components, dtype=torch.float64, device="cpu")
    log_two = math.log(2.0)
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), batch_size):
            batch = x[start : start + batch_size]
            sources = (batch - mean) @ components.T
            total += int(sources.shape[0])
            sources64 = sources.detach().cpu().to(torch.float64)
            sum1 += sources64.sum(dim=0)
            sum2 += (sources64**2).sum(dim=0)
            sum3 += (sources64**3).sum(dim=0)
            sum4 += (sources64**4).sum(dim=0)
            scaled = sources64 * float(logcosh_alpha)
            logcosh_sum += ((scaled + torch.nn.functional.softplus(-2.0 * scaled) - log_two) / float(logcosh_alpha)).sum(dim=0)
    n = float(total)
    m1 = sum1 / n
    m2 = sum2 / n
    m3 = sum3 / n
    m4 = sum4 / n
    central2 = (m2 - m1 * m1).clamp_min(torch.finfo(torch.float64).tiny)
    central4 = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * (m1**4)
    excess_kurtosis = central4 / (central2 * central2) - 3.0
    return logcosh_sum / n, excess_kurtosis
