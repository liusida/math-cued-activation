from __future__ import annotations

import contextlib
import errno
import io
import json
import os
import runpy
import socket
import struct
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


HEADER = struct.Struct("!I")
DEFAULT_PORT = "9880"
MODEL_CACHE: dict[tuple[str, str], Any] = {}
TOKENIZER_CACHE: dict[tuple[str, str], Any] = {}
PATCHED = False
CURRENT_CANCEL_EVENT: threading.Event | None = None
CURRENT_CLIENT_CONNECTED = True
DAEMON_PATH = Path(__file__).resolve()
PROJECT_ROOT = DAEMON_PATH.parents[1]
RELOAD_MODULE_ROOTS = (PROJECT_ROOT,)
SKIP_RELOAD_DIRS = {".venv", "__pycache__", ".git"}


def log(message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", file=sys.__stderr__, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        args = ["serve"]
    if args[0] == "serve":
        return serve()
    if args[0] == "run" and len(args) >= 2:
        script = args[1]
        return client(script=script, argv=[str(resolve_script(script)), *args[2:]])
    print("usage: daemon.py serve | daemon.py run SCRIPT [ARGS...]", file=sys.stderr)
    return 2


def serve() -> int:
    host = "127.0.0.1"
    port = int(os.environ.get("VIBETHINKER_DAEMON_PORT", DEFAULT_PORT))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"port {port} is already in use; set VIBETHINKER_DAEMON_PORT to use another port",
                file=sys.stderr,
            )
            return 1
        raise
    sock.listen(4)
    log(f"vibethinker daemon listening on {host}:{port}")
    log(f"project root: {PROJECT_ROOT}")
    try:
        while True:
            conn, _addr = sock.accept()
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    msg = recv_msg(conn)
                    if msg.get("cmd") != "run":
                        log("rejected request: expected cmd=run")
                        send_msg(conn, {"type": "done", "ok": False, "error": "expected cmd=run"})
                        continue
                    run_guest_with_monitor(script=msg["script"], argv=msg["argv"], conn=conn)
                except socket.timeout:
                    log("request timed out before script execution")
                except ConnectionError:
                    log("client disconnected before script execution")
                except BaseException:
                    log("request crashed before script execution")
                    traceback.print_exc(file=sys.__stderr__)
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        sock.close()
        MODEL_CACHE.clear()
        TOKENIZER_CACHE.clear()
    return 0


def client(*, script: str, argv: list[str]) -> int:
    host = "127.0.0.1"
    port = int(os.environ.get("VIBETHINKER_DAEMON_PORT", DEFAULT_PORT))
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
    except ConnectionRefusedError:
        print("no vibethinker daemon; start it with: uv run python scripts/daemon.py serve", file=sys.stderr)
        return 1
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        send_msg(sock, {"cmd": "run", "script": str(resolve_script(script)), "argv": argv})
        try:
            while True:
                reply = recv_msg(sock)
                kind = reply.get("type")
                if kind == "chunk":
                    stream = sys.stderr if reply.get("stream") == "stderr" else sys.stdout
                    stream.write(reply.get("text", ""))
                    stream.flush()
                elif kind == "done":
                    if not reply.get("ok"):
                        print(reply.get("error", "script failed"), file=sys.stderr)
                    return 0 if reply.get("ok") else 1
                else:
                    print(f"unknown daemon reply: {reply!r}", file=sys.stderr)
                    return 1
        except KeyboardInterrupt:
            try:
                send_msg(sock, {"cmd": "cancel"})
                print("cancel requested; daemon will stop generation at the next token", file=sys.stderr)
            except OSError:
                pass
            return 130
    finally:
        sock.close()


def run_guest_with_monitor(*, script: str, argv: list[str], conn: socket.socket) -> None:
    cancel_event = threading.Event()
    worker = threading.Thread(
        target=run_guest,
        kwargs={"script": script, "argv": argv, "conn": conn, "cancel_event": cancel_event},
        daemon=True,
    )
    worker.start()
    conn.settimeout(0.25)
    try:
        while worker.is_alive():
            try:
                msg = recv_msg(conn)
            except socket.timeout:
                continue
            except ConnectionError:
                if not cancel_event.is_set():
                    log("client disconnected; cancelling active run")
                    cancel_event.set()
                break
            if msg.get("cmd") == "cancel":
                if not cancel_event.is_set():
                    log("client requested cancellation")
                    cancel_event.set()
                break
            log(f"ignored client message while script is running: {msg!r}")
    finally:
        conn.settimeout(None)
        worker.join(timeout=2.0)
        if worker.is_alive():
            log("active run is still winding down after cancellation; restart daemon if you need an immediate clean slate")
            while worker.is_alive():
                worker.join(timeout=5.0)
                log("waiting for cancelled run to finish cleanup")


def run_guest(*, script: str, argv: list[str], conn: socket.socket, cancel_event: threading.Event) -> None:
    global CURRENT_CANCEL_EVENT, CURRENT_CLIENT_CONNECTED
    path = resolve_script(script)
    if not path.is_file():
        log(f"script not found: {path}")
        safe_send_msg(conn, {"type": "done", "ok": False, "error": f"not found: {path}"})
        return

    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    old_cancel_event = CURRENT_CANCEL_EVENT
    old_client_connected = CURRENT_CLIENT_CONNECTED
    started = time.monotonic()
    display_argv = " ".join(argv[1:])
    try:
        display_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        display_path = path
    log(f"start {display_path} {display_argv}".rstrip())
    try:
        CURRENT_CANCEL_EVENT = cancel_event
        CURRENT_CLIENT_CONNECTED = True
        patch_transformers_loaders()
        reload_project_modules()
        log("project modules refreshed")
        sys.argv = argv
        os.chdir(PROJECT_ROOT)
        stdout = ConnStream(conn, "stdout")
        stderr = ConnStream(conn, "stderr")
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            runpy.run_path(str(path), init_globals={"__name__": "__main__", "__file__": str(path)}, run_name="__main__")
        elapsed = time.monotonic() - started
        if cancel_event.is_set():
            log(f"cancelled {path.name} in {elapsed:.1f}s")
            safe_send_msg(conn, {"type": "done", "ok": False, "error": "cancelled"})
        else:
            log(f"done {path.name} in {elapsed:.1f}s")
            safe_send_msg(conn, {"type": "done", "ok": True})
    except ClientDisconnected:
        cancel_event.set()
        elapsed = time.monotonic() - started
        log(f"client disconnected from {path.name} after {elapsed:.1f}s")
    except BaseException:
        elapsed = time.monotonic() - started
        log(f"failed {path.name} after {elapsed:.1f}s")
        try:
            safe_send_msg(conn, {"type": "chunk", "stream": "stderr", "text": traceback.format_exc()})
        finally:
            safe_send_msg(conn, {"type": "done", "ok": False, "error": "script failed"})
    finally:
        CURRENT_CANCEL_EVENT = old_cancel_event
        CURRENT_CLIENT_CONNECTED = old_client_connected
        sys.argv = old_argv
        os.chdir(old_cwd)


def resolve_script(script: str) -> Path:
    path = Path(script).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def reload_project_modules() -> None:
    removed = 0
    for name in list(sys.modules):
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        try:
            path = Path(module_file).resolve()
        except OSError:
            continue
        if path == DAEMON_PATH or not any(_is_relative_to(path, root) for root in RELOAD_MODULE_ROOTS):
            continue
        if any(part in SKIP_RELOAD_DIRS for part in path.parts):
            continue
        sys.modules.pop(name, None)
        removed += 1
    if removed:
        log(f"reloaded {removed} project module(s)")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def patch_transformers_loaders() -> None:
    global PATCHED
    if PATCHED:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    original_model_loader = AutoModelForCausalLM.from_pretrained.__func__
    original_tokenizer_loader = AutoTokenizer.from_pretrained.__func__

    def cached_model_loader(cls: Any, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        key = cache_key(pretrained_model_name_or_path, kwargs)
        if key in MODEL_CACHE:
            log(f"reuse model: {key[0]}")
            return MODEL_CACHE[key]
        cancel_event = CURRENT_CANCEL_EVENT
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled before model load")
        log(f"load model: {key[0]}")
        started = time.monotonic()
        model = original_model_loader(cls, pretrained_model_name_or_path, *args, **kwargs)
        if cancel_event is not None and cancel_event.is_set():
            log(f"model load finished after cancellation: {key[0]}; not caching")
            raise RuntimeError("cancelled during model load")
        patch_model_generate(model)
        MODEL_CACHE[key] = model
        log(f"model loaded: {key[0]} in {time.monotonic() - started:.1f}s")
        return model

    def cached_tokenizer_loader(cls: Any, pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        key = cache_key(pretrained_model_name_or_path, kwargs)
        if key in TOKENIZER_CACHE:
            log(f"reuse tokenizer: {key[0]}")
            return TOKENIZER_CACHE[key]
        cancel_event = CURRENT_CANCEL_EVENT
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("cancelled before tokenizer load")
        log(f"load tokenizer: {key[0]}")
        started = time.monotonic()
        tokenizer = original_tokenizer_loader(cls, pretrained_model_name_or_path, *args, **kwargs)
        if cancel_event is not None and cancel_event.is_set():
            log(f"tokenizer load finished after cancellation: {key[0]}; not caching")
            raise RuntimeError("cancelled during tokenizer load")
        TOKENIZER_CACHE[key] = tokenizer
        log(f"tokenizer loaded: {key[0]} in {time.monotonic() - started:.1f}s")
        return tokenizer

    AutoModelForCausalLM.from_pretrained = classmethod(cached_model_loader)
    AutoTokenizer.from_pretrained = classmethod(cached_tokenizer_loader)
    PATCHED = True
    log("transformers loaders patched")


def patch_model_generate(model: Any) -> None:
    if getattr(model, "_vibethinker_daemon_generate_patched", False):
        return

    from transformers import StoppingCriteria, StoppingCriteriaList

    original_generate = model.generate

    class CancelStoppingCriteria(StoppingCriteria):
        def __init__(self, event: threading.Event) -> None:
            self.event = event
            self.logged = False

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            del input_ids, scores, kwargs
            if self.event.is_set() and not self.logged:
                log("generation cancellation observed; stopping at current token")
                self.logged = True
            return self.event.is_set()

    def generate_with_cancel(*args: Any, **kwargs: Any) -> Any:
        cancel_event = CURRENT_CANCEL_EVENT
        if cancel_event is not None:
            stopping_criteria = kwargs.get("stopping_criteria")
            if stopping_criteria is None:
                stopping_criteria = StoppingCriteriaList()
            elif not isinstance(stopping_criteria, StoppingCriteriaList):
                stopping_criteria = StoppingCriteriaList(stopping_criteria)
            stopping_criteria.append(CancelStoppingCriteria(cancel_event))
            kwargs["stopping_criteria"] = stopping_criteria
        return original_generate(*args, **kwargs)

    model.generate = generate_with_cancel
    model._vibethinker_daemon_generate_patched = True


def cache_key(model_id: Any, kwargs: dict[str, Any]) -> tuple[str, str]:
    stable = {
        key: str(value)
        for key, value in kwargs.items()
        if key in {"dtype", "torch_dtype", "local_files_only", "trust_remote_code", "revision"}
    }
    return str(model_id), json.dumps(stable, sort_keys=True)


class ConnStream(io.TextIOBase):
    encoding = "utf-8"

    def __init__(self, conn: socket.socket, stream: str) -> None:
        super().__init__()
        self.conn = conn
        self.stream = stream

    def write(self, text: str) -> int:
        if text:
            send_chunk_or_cancel(self.conn, self.stream, text)
        return len(text)

    def flush(self) -> None:
        return None


def recvn(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("connection closed")
        data.extend(chunk)
    return bytes(data)


def send_msg(sock: socket.socket, obj: dict[str, Any]) -> None:
    raw = json.dumps(obj).encode("utf-8")
    sock.sendall(HEADER.pack(len(raw)) + raw)


class ClientDisconnected(Exception):
    pass


def send_chunk_or_cancel(sock: socket.socket, stream: str, text: str) -> None:
    global CURRENT_CLIENT_CONNECTED
    if not CURRENT_CLIENT_CONNECTED:
        return
    try:
        send_msg(sock, {"type": "chunk", "stream": stream, "text": text})
    except OSError:
        CURRENT_CLIENT_CONNECTED = False
        if CURRENT_CANCEL_EVENT is not None and not CURRENT_CANCEL_EVENT.is_set():
            log("client disconnected while receiving output; cancelling active run")
            CURRENT_CANCEL_EVENT.set()


def safe_send_msg(sock: socket.socket, obj: dict[str, Any]) -> None:
    global CURRENT_CLIENT_CONNECTED
    if not CURRENT_CLIENT_CONNECTED:
        raise ClientDisconnected()
    try:
        send_msg(sock, obj)
    except OSError as exc:
        CURRENT_CLIENT_CONNECTED = False
        raise ClientDisconnected() from exc


def recv_msg(sock: socket.socket) -> dict[str, Any]:
    (n_bytes,) = HEADER.unpack(recvn(sock, HEADER.size))
    return json.loads(recvn(sock, n_bytes).decode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
