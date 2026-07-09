#!/usr/bin/env python3
"""Minimal repro for slow same-dtype CUDA copy from safetensors PySafeSlice."""

from __future__ import annotations

import argparse
import gc
import tempfile
import time
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
import safetensors
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--cols", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--path", type=Path, default=None)
    return parser.parse_args()


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def time_case(name: str, repeat: int, fn) -> list[float]:
    times = []
    for _ in range(repeat):
        clear_cuda()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        del out
    print(f"{name:24} avg={sum(times)/len(times):8.4f}s min={min(times):8.4f}s max={max(times):8.4f}s")
    return times


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    other_dtype = torch.bfloat16 if dtype == torch.float16 else torch.float16
    shape = (args.rows, args.cols)
    path = args.path
    temp_dir = None
    if path is None:
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "pysafeslice_repro.safetensors"

    print("Environment")
    print(f"  torch:        {torch.__version__}")
    print(f"  transformers: {transformers.__version__}")
    print(f"  safetensors:  {safetensors.__version__}")
    print(f"  cuda:         {torch.version.cuda}")
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print()
    print("Test Tensor")
    print(f"shape: {shape}")
    print(f"dtype: {dtype}")
    print(f"file: {path}")

    tensor = torch.zeros(shape, dtype=dtype)
    save_file({"x": tensor}, path)
    del tensor
    print(f"file size: {path.stat().st_size / 1024**2:.1f} MiB")

    with safe_open(path, framework="pt", device="cpu", backend="mmap") as handle:
        sl = handle.get_slice("x")
        x = sl[...]
        print(f"slice type: {type(sl)}")
        print(f"materialized: dtype={x.dtype}, device={x.device}, contiguous={x.is_contiguous()}")
        del x
        print()

        direct_same = time_case("direct same dtype", args.repeat, lambda: sl[...].to("cuda", dtype=dtype))
        clone_same = time_case("clone then same", args.repeat, lambda: sl[...].clone().to("cuda", dtype=dtype))
        cpu_copy_same = time_case("cpu copy then same", args.repeat, lambda: sl[...].to("cpu", copy=True).to("cuda", dtype=dtype))
        direct_other = time_case("direct other dtype", args.repeat, lambda: sl[...].to("cuda", dtype=other_dtype))

        direct_avg = mean(direct_same)
        best_workaround = min(mean(clone_same), mean(cpu_copy_same), mean(direct_other))
        ratio = direct_avg / best_workaround if best_workaround > 0 else float("inf")
        print()
        print("Judgement")
        print(f"  direct_same / best_workaround: {ratio:.2f}x")
        if ratio >= 2.0:
            print("  RESULT: reproduced suspicious slowdown.")
            print("  Direct same-dtype CUDA copy from PySafeSlice is much slower than clone/copy/cast workaround.")
        elif ratio >= 1.3:
            print("  RESULT: mild slowdown.")
            print("  Direct same-dtype CUDA copy is slower, but not dramatically so on this setup.")
        else:
            print("  RESULT: no significant slowdown reproduced.")

    if temp_dir is not None:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
