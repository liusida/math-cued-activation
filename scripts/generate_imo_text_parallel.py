#!/usr/bin/env python3
"""Dynamically schedule IMO-AnswerBench generation workers across GPUs."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run scripts/generate_imo_text.py with dynamic GPU scheduling. "
            "Unknown args are forwarded to each generation worker."
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
    parser.add_argument(
        "--batch-size-per-gpu",
        type=int,
        default=1,
        help="Number of problems each GPU worker generates in one batched model.generate call.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Regenerate problems even when their local replay JSON already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scheduling information without launching workers.",
    )
    args, worker_args = parser.parse_known_args()
    if "--shard-count" in worker_args or "--shard-index" in worker_args:
        raise SystemExit("Do not pass --shard-count/--shard-index; dynamic scheduling replaces sharding.")
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


def load_selected_rows(worker_args: list[str]) -> tuple[argparse.Namespace, list[dict]]:
    from generate_imo_text import parse_args as parse_worker_args
    from run_vibethinker import load_imo_answerbench_rows

    args = parse_worker_args(worker_args)
    rows = load_imo_answerbench_rows(
        args.sample_size,
        args.seed,
        args.problem_id,
        args.start_index,
        args.shuffle,
    )
    if not rows:
        raise SystemExit("No IMO-AnswerBench rows matched the request.")
    return args, rows


def generation_json_path(worker_args: argparse.Namespace, row: dict) -> Path:
    from run_vibethinker import (
        IMO_ANSWERBENCH_ID,
        generated_text_output_dir,
        resolve_model_id,
        safe_filename,
    )

    output_dir = generated_text_output_dir(
        root=worker_args.generated_text_dir,
        dataset_id=IMO_ANSWERBENCH_ID,
        model_id=resolve_model_id(worker_args.model),
    )
    return output_dir / f"{safe_filename(str(row['Problem ID']))}.json"


def pending_rows(worker_args: argparse.Namespace, rows: list[dict], rerun_existing: bool) -> list[dict]:
    if rerun_existing:
        return rows
    pending = [row for row in rows if not generation_json_path(worker_args, row).exists()]
    skipped = len(rows) - len(pending)
    if skipped:
        print(f"Skipping {skipped} already generated problem(s).", flush=True)
    return pending


def gpu_worker(
    gpu: str,
    worker_index: int,
    worker_args: list[str],
    task_queue,
    result_queue,
    log_dir: str,
    batch_size: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    log_path = Path(log_dir) / f"gpu_{gpu}_worker_{worker_index:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", buffering=1) as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            try:
                from generate_imo_text import parse_args as parse_worker_args
                from generate_imo_text import run_rows
                from run_vibethinker import load_model_and_tokenizer

                args = parse_worker_args(worker_args)
                args.shard_count = 1
                args.shard_index = 0
                args.batch_size = batch_size

                print(f"GPU worker {worker_index} starting on physical GPU {gpu}", flush=True)
                model, tokenizer = load_model_and_tokenizer(args)
                print(f"GPU worker {worker_index} model loaded", flush=True)

                while True:
                    first_task = task_queue.get()
                    if first_task is None:
                        print(f"GPU worker {worker_index} received stop signal", flush=True)
                        return

                    tasks = [first_task]
                    while len(tasks) < batch_size:
                        try:
                            task = task_queue.get_nowait()
                        except queue.Empty:
                            break
                        if task is None:
                            task_queue.put(None)
                            break
                        tasks.append(task)

                    task_numbers = [task[0] for task in tasks]
                    total_tasks = tasks[0][1]
                    rows = [task[2] for task in tasks]
                    problem_ids = [row["Problem ID"] for row in rows]
                    started = time.monotonic()
                    print(
                        f"GPU worker {worker_index} starting tasks "
                        f"{task_numbers[0]}-{task_numbers[-1]}/{total_tasks}: {', '.join(problem_ids)}",
                        flush=True,
                    )
                    try:
                        correct = run_rows(
                            args,
                            rows,
                            selected_count=total_tasks,
                            worker_label=f"[gpu {gpu} tasks {task_numbers[0]}-{task_numbers[-1]}/{total_tasks}]",
                            model=model,
                            tokenizer=tokenizer,
                        )
                    except Exception:
                        traceback.print_exc()
                        seconds = time.monotonic() - started
                        for task_number, problem_id in zip(task_numbers, problem_ids):
                            result_queue.put(
                                {
                                    "ok": False,
                                    "gpu": gpu,
                                    "worker_index": worker_index,
                                    "task_number": task_number,
                                    "problem_id": problem_id,
                                    "seconds": seconds,
                                }
                            )
                    else:
                        seconds = time.monotonic() - started
                        for task_number, problem_id in zip(task_numbers, problem_ids):
                            result_queue.put(
                                {
                                    "ok": True,
                                    "gpu": gpu,
                                    "worker_index": worker_index,
                                    "task_number": task_number,
                                    "problem_id": problem_id,
                                    "correct": int(correct),
                                    "seconds": seconds,
                                    "batch_size": len(tasks),
                                }
                            )
            except BaseException:
                traceback.print_exc()
                result_queue.put(
                    {
                        "ok": False,
                        "gpu": gpu,
                        "worker_index": worker_index,
                        "task_number": None,
                        "problem_id": "__worker_crash__",
                        "seconds": 0.0,
                    }
                )
                raise


def terminate_processes(processes: list[mp.Process]) -> None:
    live = [process for process in processes if process.is_alive()]
    for process in live:
        process.terminate()
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and any(process.is_alive() for process in live):
        time.sleep(0.5)
    for process in live:
        if process.is_alive():
            process.kill()


def main() -> None:
    args, worker_args_list = parse_args()
    if args.batch_size_per_gpu < 1:
        raise SystemExit("--batch-size-per-gpu must be at least 1.")
    gpus = selected_gpus(args)
    worker_args, rows = load_selected_rows(worker_args_list)
    rows = pending_rows(worker_args, rows, args.rerun_existing)
    if not rows:
        print("All selected problems already have local generation JSON files.", flush=True)
        return

    total_tasks = len(rows)
    log_dir = args.log_dir.expanduser()
    print(
        f"Dynamic scheduling {total_tasks} problem(s) across GPUs {', '.join(gpus)} "
        f"with batch size {args.batch_size_per_gpu}/GPU",
        flush=True,
    )
    print(f"Logs: {log_dir}", flush=True)
    if args.dry_run:
        for task_number, row in enumerate(rows, start=1):
            print(f"task {task_number}/{total_tasks}: {row['Problem ID']}", flush=True)
        return

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for task_number, row in enumerate(rows, start=1):
        task_queue.put((task_number, total_tasks, row))
    for _ in gpus:
        task_queue.put(None)

    processes: list[mp.Process] = []

    def stop_children(signum, frame) -> None:
        del frame
        print(f"\nReceived signal {signum}; stopping workers...", flush=True)
        terminate_processes(processes)
        raise SystemExit(130)

    old_sigint = signal.signal(signal.SIGINT, stop_children)
    old_sigterm = signal.signal(signal.SIGTERM, stop_children)
    failures = []
    completed = 0
    started_at = time.monotonic()
    try:
        for worker_index, gpu in enumerate(gpus):
            process = ctx.Process(
                target=gpu_worker,
                args=(
                    gpu,
                    worker_index,
                    worker_args_list,
                    task_queue,
                    result_queue,
                    str(log_dir),
                    args.batch_size_per_gpu,
                ),
                name=f"imo-text-gpu-{gpu}",
            )
            process.start()
            processes.append(process)
            print(
                f"Started worker {worker_index} on GPU {gpu}; "
                f"log: {log_dir / f'gpu_{gpu}_worker_{worker_index:02d}.log'}",
                flush=True,
            )

        while completed < total_tasks:
            try:
                result = result_queue.get(timeout=5)
            except queue.Empty:
                dead = [process for process in processes if not process.is_alive() and process.exitcode not in (0, None)]
                if dead:
                    for process in dead:
                        failures.append({"problem_id": "__worker_exit__", "exitcode": process.exitcode})
                    break
                continue

            completed += 1
            elapsed = time.monotonic() - started_at
            rate = completed / elapsed if elapsed > 0 else 0.0
            status = "ok" if result["ok"] else "failed"
            if not result["ok"]:
                failures.append(result)
            print(
                f"[{completed}/{total_tasks}] {status} gpu={result['gpu']} "
                f"problem={result['problem_id']} time={result['seconds']:.1f}s "
                f"rate={rate:.3f} problems/s",
                flush=True,
            )

        for process in processes:
            process.join()

        crashed = [process for process in processes if process.exitcode not in (0, None)]
        if crashed:
            for process in crashed:
                failures.append({"problem_id": "__worker_crash__", "exitcode": process.exitcode})

        if failures:
            raise SystemExit(f"{len(failures)} task/worker failure(s); inspect logs in {log_dir}.")

        elapsed = time.monotonic() - started_at
        print(
            f"Finished {completed}/{total_tasks} problem(s) in {elapsed / 60:.1f} min "
            f"({completed / elapsed:.3f} problems/s).",
            flush=True,
        )
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if any(process.is_alive() for process in processes):
            terminate_processes(processes)


if __name__ == "__main__":
    main()
