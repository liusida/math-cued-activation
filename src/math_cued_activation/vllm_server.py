from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

from .config import PipelineConfig


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _is_vllm_process(pid: int) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return "vllm" in command and "serve" in command


def start_vllm(config: PipelineConfig) -> None:
    cfg = config.vllm
    old_pid = _read_pid(cfg.pid_file)
    if old_pid is not None and _alive(old_pid) and _is_vllm_process(old_pid):
        raise RuntimeError(f"vLLM is already running with PID {old_pid}")

    cfg.pid_file.parent.mkdir(parents=True, exist_ok=True)
    cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "vllm", "serve", config.model.id,
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--dtype", cfg.dtype,
        "--max-model-len", str(cfg.max_model_len),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
    ]
    if cfg.enforce_eager:
        command.append("--enforce-eager")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    if cfg.disable_flashinfer_sampler:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

    with cfg.log_file.open("ab") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    cfg.pid_file.write_text(f"{process.pid}\n")
    print(f"started vLLM PID {process.pid}; log: {cfg.log_file}")

    models_url = f"http://{cfg.host}:{cfg.port}/v1/models"
    deadline = time.monotonic() + cfg.startup_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            cfg.pid_file.unlink(missing_ok=True)
            raise RuntimeError(f"vLLM exited with code {process.returncode}; inspect {cfg.log_file}")
        try:
            with urllib.request.urlopen(models_url, timeout=2) as response:
                payload = json.load(response)
            if response.status == 200 and payload.get("data"):
                print(f"vLLM ready at {models_url}")
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(2)

    os.killpg(process.pid, signal.SIGTERM)
    cfg.pid_file.unlink(missing_ok=True)
    raise TimeoutError(f"vLLM did not become ready within {cfg.startup_timeout:g}s; inspect {cfg.log_file}")


def stop_vllm(config: PipelineConfig) -> None:
    cfg = config.vllm
    pid = _read_pid(cfg.pid_file)
    if pid is None:
        print(f"vLLM is not recorded as running: {cfg.pid_file}")
        return
    if not _alive(pid):
        cfg.pid_file.unlink(missing_ok=True)
        print(f"removed stale vLLM PID file for {pid}")
        return
    if not _is_vllm_process(pid):
        raise RuntimeError(f"refusing to stop PID {pid}: it is not a vLLM serve process; remove stale file {cfg.pid_file}")

    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + cfg.shutdown_timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            cfg.pid_file.unlink(missing_ok=True)
            print(f"stopped vLLM PID {pid}")
            return
        time.sleep(0.5)
    os.killpg(pid, signal.SIGKILL)
    cfg.pid_file.unlink(missing_ok=True)
    print(f"force-stopped vLLM PID {pid} after {cfg.shutdown_timeout:g}s")
