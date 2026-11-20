from __future__ import annotations

import inspect
import json
import os
import select
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .tool_base import PROJECT_ROOT


def python_for_environment(environment):
    if not environment:
        return None
    environment = Path(os.path.expandvars(os.path.expanduser(str(environment))))
    if environment.is_absolute() or len(environment.parts) > 1:
        candidates = [environment / "bin" / "python", environment]
    else:
        roots = [
            os.getenv("AGENT4AIGC_CONDA_ENVS_ROOT", "/data/chenzhipeng/miniconda3/envs"),
            str(Path(sys.prefix).parent),
            str(Path.home() / "miniconda3" / "envs"),
        ]
        candidates = [Path(root) / environment / "bin" / "python" for root in dict.fromkeys(roots)]
    return next((str(path) for path in candidates if path.is_file() and os.access(path, os.X_OK)), None)


def tool_environment(python, device):
    env = os.environ.copy()
    if str(device).startswith("npu") and not env.get("ASCEND_HOME_PATH"):
        setup = Path(env.get("AGENT4AIGC_CANN_SETUP", "/usr/local/Ascend/ascend-toolkit/set_env.sh"))
        if not setup.is_file():
            raise FileNotFoundError(f"CANN environment script does not exist: {setup}")
        completed = subprocess.run(
            ["bash", "-c", 'source "$1" >/dev/null && env -0', "bash", str(setup)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        if completed.returncode:
            raise RuntimeError("Cannot activate CANN: " + completed.stderr.decode("utf-8", "replace")[-1500:])
        env = dict(item.decode("utf-8", "surrogateescape").split("=", 1)
                   for item in completed.stdout.split(b"\0") if b"=" in item)
    prefix = Path(python).resolve().parent.parent
    env["CONDA_PREFIX"] = str(prefix)
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


class SubprocessTool:
    def __init__(self, name, info, device="npu:6"):
        self.name, self.info, self.device = name, dict(info), device
        self.python = python_for_environment(info.get("conda_env"))
        if not self.python:
            raise FileNotFoundError(f"Conda environment not found for {name}: {info.get('conda_env')}")
        self._process = None
        self._read_fd = None
        self._buffer = b""
        self._lock = threading.RLock()
        self.log_path = None
        self._log = None

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.name} RPC timed out; see {self.log_path}")
            ready, _, _ = select.select([self._read_fd], [], [], min(remaining, 1.0))
            if ready:
                data = os.read(self._read_fd, 65536)
                if not data:
                    raise RuntimeError(f"{self.name} worker exited; see {self.log_path}")
                self._buffer += data
                if len(self._buffer) > 4 * 1024 * 1024:
                    raise RuntimeError("Unexpectedly large tool RPC response")
            elif self._process.poll() is not None:
                raise RuntimeError(f"{self.name} worker exited ({self._process.returncode}); see {self.log_path}")
        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    def _request(self, action, **kwargs):
        request_id = uuid.uuid4().hex
        request = {"action": action, "request_id": request_id, **kwargs}
        timeout_name = "LOAD" if action == "load" else "INFER"
        timeout = float(os.getenv(f"AGENT4AIGC_TOOL_{timeout_name}_TIMEOUT", "3600"))
        try:
            self._process.stdin.write(json.dumps(request, ensure_ascii=False, allow_nan=False) + "\n")
            self._process.stdin.flush()
            response = self._receive(timeout)
            if response.get("request_id") != request_id:
                raise RuntimeError("Mismatched tool RPC request_id")
            if response.get("status") != "ok":
                raise RuntimeError(f"{self.name}: {response.get('error', 'unknown error')}; see {self.log_path}")
            return response.get("result")
        except BaseException:
            self.close()
            raise

    def server_load_model(self):
        with self._lock:
            if self._process is not None:
                if self._process.poll() is None:
                    return
                self.close()
            read_fd, write_fd = os.pipe()
            self._read_fd = read_fd
            logs = PROJECT_ROOT / "outputs" / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            self.log_path = logs / f"{self.name}_{uuid.uuid4().hex}.log"
            self._log = self.log_path.open("a", encoding="utf-8")
            try:
                env = tool_environment(self.python, self.device)
                self._process = subprocess.Popen(
                    [self.python, "-m", "ToolSets.toolset_utils.runtime", "--worker", str(write_fd)],
                    stdin=subprocess.PIPE, stdout=self._log, stderr=self._log, text=True,
                    cwd=PROJECT_ROOT, env=env, pass_fds=(write_fd,),
                )
            except BaseException:
                os.close(read_fd)
                self._read_fd = None
                self._log.close()
                raise
            finally:
                os.close(write_fd)
            self._request("load", name=self.name, info=self.info, device=self.device)

    def server_inference(self, **kwargs):
        with self._lock:
            self.server_load_model()
            return self._request("inference", args=kwargs)

    def server_unload_model(self):
        self.close()

    def close(self):
        with self._lock:
            process, self._process = self._process, None
            if process is not None:
                try:
                    if process.poll() is None and process.stdin is not None:
                        process.stdin.write('{"action":"shutdown"}\n')
                        process.stdin.flush()
                        process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                finally:
                    if process.stdin:
                        process.stdin.close()
            if self._read_fd is not None:
                os.close(self._read_fd)
                self._read_fd = None
            if self._log is not None:
                self._log.close()
                self._log = None
            self._buffer = b""


def _worker(write_fd):
    import asyncio
    import importlib
    import traceback
    output = os.fdopen(write_fd, "w", buffering=1, encoding="utf-8")
    instance = None

    def call(method, **kwargs):
        result = method(**kwargs)
        return asyncio.run(result) if inspect.isawaitable(result) else result

    try:
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                action = request["action"]
                if action == "shutdown":
                    break
                if action == "load":
                    info = request["info"]
                    cls = getattr(importlib.import_module(info["module"]), info["class_name"])
                    instance = cls(device=request["device"], tool_info=info, model_path=info.get("model_path"))
                    result = call(instance.server_load_model)
                elif action == "inference":
                    if instance is None:
                        raise RuntimeError("Tool model is not loaded")
                    result = call(instance.server_inference, **request["args"])
                else:
                    raise ValueError(f"Unknown RPC action: {action}")
                response = {"request_id": request.get("request_id"), "status": "ok", "result": result}
            except BaseException as exc:
                traceback.print_exc(file=sys.stderr)
                response = {"request_id": request.get("request_id"), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            output.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
    finally:
        if instance is not None:
            try:
                call(instance.server_unload_model)
            except BaseException:
                traceback.print_exc(file=sys.stderr)
        output.close()


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("This module is an internal tool worker")
    _worker(int(sys.argv[2]))

