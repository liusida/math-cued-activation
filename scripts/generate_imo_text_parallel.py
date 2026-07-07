#!/usr/bin/env python3
"""Launch sharded IMO-AnswerBench generation workers across available GPUs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKER_SCRIPT = PROJECT_ROOT / "scripts" / "generate_imo_text.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Manage parallel scripts/generate_imo_text.py workers. "
            "Unknown args are forwarded to each worker."
        )
    )
    parser.add_argument(
        "--gpus",
        help='Comma-separated GPU ids to use, e.g. "0,1,2,3". Default: auto-detect all CUDA GPUs.',
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Use the first N detected GPUs. Ignored when --gpus is set.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs/imo-text-parallel"))
    parser.add_argument("--worker-script", type=Path, default=DEFAULT_WORKER_SCRIPT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print worker commands without launching them.",
    )
    args, worker_args = parser.parse_known_args()
    if "--shard-count" in worker_args or "--shard-index" in worker_args:
        raise SystemExit("Do not pass --shard-count/--shard-index; the manager sets them.")
    return args, worker_args


def detect_gpus() -> list[str]:
    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return [str(index) for index in range(count)]
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def selected_gpus(args: argparse.Namespace) -> list[str]:
    if args.gpus:
        gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    else:
        gpus = detect_gpus()
        if args.num_gpus is not None:
            gpus = gpus[: args.num_gpus]
    if not gpus:
        raise SystemExit("No GPUs found. Pass --gpus explicitly if auto-detection failed.")
    return gpus


def worker_command(
    worker_script: Path,
    worker_args: list[str],
    shard_count: int,
    shard_index: int,
) -> list[str]:
    return [
        sys.executable,
        str(worker_script),
        *worker_args,
        "--shard-count",
        str(shard_count),
        "--shard-index",
        str(shard_index),
    ]


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    live = [process for process in processes if process.poll() is None]
    for process in live:
        process.terminate()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and any(process.poll() is None for process in live):
        time.sleep(0.5)
    for process in live:
        if process.poll() is None:
            process.kill()


def main() -> None:
    args, worker_args = parse_args()
    gpus = selected_gpus(args)
    worker_script = args.worker_script.expanduser().resolve()
    if not worker_script.exists():
        raise SystemExit(f"Worker script not found: {worker_script}")

    log_dir = args.log_dir.expanduser()
    shard_count = len(gpus)
    commands = [
        (
            gpu,
            worker_command(
                worker_script=worker_script,
                worker_args=worker_args,
                shard_count=shard_count,
                shard_index=shard_index,
            ),
        )
        for shard_index, gpu in enumerate(gpus)
    ]

    print(f"Launching {len(commands)} generation worker(s): GPUs {', '.join(gpus)}", flush=True)
    for shard_index, (gpu, command) in enumerate(commands):
        print(f"[shard {shard_index}/{shard_count} gpu {gpu}] {' '.join(command)}", flush=True)
    if args.dry_run:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen] = []
    log_files = []

    def stop_children(signum, frame) -> None:
        del frame
        print(f"\nReceived signal {signum}; stopping workers...", flush=True)
        terminate_processes(processes)
        raise SystemExit(130)

    old_sigint = signal.signal(signal.SIGINT, stop_children)
    old_sigterm = signal.signal(signal.SIGTERM, stop_children)
    try:
        for shard_index, (gpu, command) in enumerate(commands):
            log_path = log_dir / f"shard_{shard_index:02d}_gpu_{gpu}.log"
            log_file = log_path.open("w")
            log_files.append(log_file)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            print(f"Starting shard {shard_index}/{shard_count} on GPU {gpu}; log: {log_path}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(process)

        failures: list[tuple[int, int, str]] = []
        for shard_index, process in enumerate(processes):
            return_code = process.wait()
            gpu = gpus[shard_index]
            log_path = log_dir / f"shard_{shard_index:02d}_gpu_{gpu}.log"
            if return_code == 0:
                print(f"Finished shard {shard_index}/{shard_count} on GPU {gpu}", flush=True)
            else:
                failures.append((shard_index, return_code, str(log_path)))
                print(
                    f"Failed shard {shard_index}/{shard_count} on GPU {gpu}: "
                    f"exit {return_code}; log: {log_path}",
                    flush=True,
                )

        if failures:
            raise SystemExit(f"{len(failures)} worker(s) failed.")
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        for log_file in log_files:
            log_file.close()


if __name__ == "__main__":
    main()
