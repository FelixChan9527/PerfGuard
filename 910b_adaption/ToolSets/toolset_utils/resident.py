from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
import uuid

from .runtime import python_for_environment, tool_environment
from .tool_base import PROJECT_ROOT

SERVICE_DIR = PROJECT_ROOT / 'deployment/tools/state'


def identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        stat = (proc / 'stat').read_text()
        fields = stat[stat.rfind(')') + 2:].split()
        return {'uid': proc.stat().st_uid, 'start_time': fields[19], 'state': fields[0],
                'argv': (proc / 'cmdline').read_bytes().split(b'\0')}
    except (OSError, IndexError, ValueError):
        return None


def owned_alive(state):
    process = identity(state['pid']) if state else None
    return bool(process and process['uid'] == os.getuid() and process['state'] != 'Z'
                and process['start_time'] == state['start_time']
                and b'ToolSets.toolset_utils.resident' in process['argv']
                and os.fsencode(state['spec_path']) in process['argv'])


def paths(name):
    if not re.fullmatch(r'[A-Za-z0-9_]+', name):
        raise ValueError('Invalid service tool name')
    SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    return SERVICE_DIR / (name + '.json'), SERVICE_DIR / (name + '.lock')


def read_state(name):
    path, _ = paths(name)
    return json.loads(path.read_text()) if path.is_file() else None


def fingerprint(name, info, device):
    data = json.dumps({'name': name, 'info': info, 'device': device}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def endpoint(info):
    value = info['endpoint'].rstrip('/')
    parsed = urlsplit(value)
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or not parsed.port
            or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError('Tool endpoint must be a loopback HTTP URL without a path or credentials')
    return value


def http_json(url, payload=None, timeout=3):
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() if payload is not None else None
    request = Request(url, data=data, headers={'Content-Type': 'application/json'})
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.loads(response.read(4 * 1024 * 1024))


def status(name):
    state = read_state(name)
    if not state:
        return {'name': name, 'status': 'not_started', 'owned_process_alive': False}
    result = {**state, 'owned_process_alive': owned_alive(state)}
    if not result['owned_process_alive']:
        return {**result, 'status': 'stopped'}
    try:
        health = http_json(state['endpoint'] + '/health')
        if health.get('pid') != state['pid'] or health.get('fingerprint') != state['fingerprint']:
            return {**result, 'status': 'identity_mismatch'}
        return {**result, 'status': 'ready', 'health': health}
    except (URLError, TimeoutError, OSError, ValueError) as error:
        return {**result, 'status': 'starting', 'health_error': str(error)}


def start(name, info, device):
    state_path, lock_path = paths(name)
    expected = fingerprint(name, info, device)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_state(name)
        if owned_alive(state):
            if state['fingerprint'] != expected:
                raise ValueError(f'{name} is already running with different settings; explicitly stop it before changing devices or weights')
            return state
        url = endpoint(info)
        import socket
        with socket.socket() as probe:
            try:
                probe.bind(('127.0.0.1', urlsplit(url).port))
            except OSError as error:
                raise RuntimeError(f'{url} is occupied; existing processes were left untouched') from error
        python = python_for_environment(info.get('conda_env'))
        if python is None:
            raise FileNotFoundError(f'Conda environment missing: {info.get("conda_env")}')
        spec_path = SERVICE_DIR / (name + '.spec.json')
        spec_path.write_text(json.dumps({'name': name, 'info': info, 'device': device,
                                         'fingerprint': expected}, ensure_ascii=False, indent=2) + '\n')
        logs = PROJECT_ROOT / 'deployment/tools/logs'
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f'{name}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        with log_path.open('ab') as log:
            process = subprocess.Popen([python, '-m', 'ToolSets.toolset_utils.resident', '--serve', str(spec_path)],
                                       env=tool_environment(python, device), cwd=PROJECT_ROOT,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        process_id = identity(process.pid)
        if process_id is None:
            raise RuntimeError(f'{name} exited; inspect {log_path}')
        state = {'name': name, 'pid': process.pid, 'start_time': process_id['start_time'],
                 'spec_path': str(spec_path), 'fingerprint': expected, 'endpoint': url,
                 'device': device, 'python': python, 'log_path': str(log_path),
                 'started_at_unix': time.time()}
        temporary = state_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(state_path)
        return state


def stop(name):
    _, lock_path = paths(name)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read_state(name)
        if not owned_alive(state):
            return {'name': name, 'status': 'already_stopped'}
        if os.getpgid(state['pid']) != state['pid']:
            raise RuntimeError('Refusing to stop a process outside its owned service group')
        os.killpg(state['pid'], signal.SIGTERM)
        return {'name': name, 'status': 'shutdown_requested', 'pid': state['pid']}


class ResidentTool:
    def __init__(self, name, info, device):
        self.name, self.info, self.device = name, dict(info), device
        self.state = None
        self.log_path = None
        self._lock = threading.RLock()

    @property
    def pid(self):
        return self.state['pid'] if self.state else None

    def server_load_model(self):
        with self._lock:
            self.state = start(self.name, self.info, self.device)
            self.log_path = Path(self.state['log_path'])
            deadline = time.monotonic() + float(os.getenv('AGENT4AIGC_TOOL_LOAD_TIMEOUT', '3600'))
            while time.monotonic() < deadline:
                current = status(self.name)
                if current['status'] == 'ready':
                    return
                if not current['owned_process_alive'] or current['status'] == 'identity_mismatch':
                    raise RuntimeError(f'{self.name} service failed; inspect {self.log_path}')
                time.sleep(0.5)
            raise TimeoutError(f'{self.name} model startup timed out; service remains running: {self.log_path}')

    def server_inference(self, **kwargs):
        with self._lock:
            self.server_load_model()
            request_id = uuid.uuid4().hex
            progress_stop = threading.Event()
            progress_state = {'last': None}

            def render(progress):
                current, total = int(progress['current']), int(progress['total'])
                percent = min(100.0, max(0.0, current * 100.0 / total))
                width = 24
                filled = round(width * percent / 100)
                label = 'Step1X-Edit' if self.name == 'Step1X_Edit_ImageEditTool' else self.name
                line = (f'\r[{label}] {progress["stage"]:<10} '
                        f'[{"=" * filled}{"." * (width - filled)}] '
                        f'{current:>3}/{total:<3} {percent:5.1f}%')
                sys.stderr.write(line)
                sys.stderr.flush()
                progress_state['last'] = progress

            def poll_progress():
                while not progress_stop.is_set():
                    try:
                        health = http_json(self.state['endpoint'] + '/health', timeout=1)
                        progress = health.get('progress')
                        if progress and progress.get('request_id') == request_id:
                            previous = progress_state['last']
                            if not previous or (progress.get('current'), progress.get('stage')) != (
                                    previous.get('current'), previous.get('stage')):
                                render(progress)
                    except (URLError, TimeoutError, OSError, ValueError):
                        pass
                    progress_stop.wait(0.35)

            watcher = threading.Thread(target=poll_progress, name=f'{self.name}-progress', daemon=True)
            watcher.start()
            succeeded = False
            try:
                result = http_json(self.state['endpoint'] + '/rpc',
                                   {'action': 'inference', 'request_id': request_id, 'args': kwargs,
                                    'fingerprint': self.state['fingerprint']},
                                   timeout=float(os.getenv('AGENT4AIGC_TOOL_INFER_TIMEOUT', '3600')))
                if result.get('request_id') != request_id:
                    raise RuntimeError('Mismatched resident tool request_id')
                if result.get('status') != 'ok':
                    raise RuntimeError(f'{self.name}: {result.get("error")}; see {self.log_path}')
                succeeded = True
                return result['result']
            finally:
                progress_stop.set()
                watcher.join(timeout=2)
                last = progress_state['last']
                if last:
                    if succeeded:
                        render({**last, 'current': last['total']})
                    sys.stderr.write('\n')
                    sys.stderr.flush()

    def server_unload_model(self):

        pass

    close = server_unload_model


def serve(spec_path):
    specification = json.loads(Path(spec_path).read_text())
    name, info, device = (specification[key] for key in ('name', 'info', 'device'))
    url = endpoint(info)
    call_lock = threading.Lock()
    stats = {'inference_calls': 0, 'model_loads': 0, 'progress': None}

    def call(method, **kwargs):
        value = method(**kwargs)
        return asyncio.run(value) if inspect.isawaitable(value) else value

    cls = getattr(importlib.import_module(info['module']), info['class_name'])
    instance = cls(device=device, tool_info=info, model_path=info.get('model_path'))
    call(instance.server_load_model)
    stats['model_loads'] = 1

    class Handler(BaseHTTPRequestHandler):
        def log_request(self, code='-', size='-'):
            if self.path != '/health' or str(code) != '200':
                super().log_request(code, size)

        def reply(self, value, status_code=200):
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != '/health':
                return self.reply({'error': 'Unknown route'}, 404)
            self.reply({'name': name, 'pid': os.getpid(), 'device': device, 'status': 'ready',
                        'fingerprint': specification['fingerprint'], **stats})

        def do_POST(self):
            request = {}
            try:
                if self.path != '/rpc':
                    return self.reply({'error': 'Unknown route'}, 404)
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError('Invalid request length')
                request = json.loads(self.rfile.read(size))
                if request.get('fingerprint') != specification['fingerprint']:
                    raise ValueError('Model service fingerprint mismatch')
                if request.get('action') != 'inference':
                    raise ValueError('Only inference is available; manage lifecycle with deployment/tools/serve.py')
                with call_lock:
                    activate = getattr(instance, '_activate_device', None)
                    if activate:
                        activate()
                    stats['progress'] = None

                    def update_progress(stage, current, total):
                        try:
                            current, total = int(current), int(total)
                        except (TypeError, ValueError):
                            return
                        if total <= 0:
                            return
                        current = min(total, max(0, current))
                        stage = re.sub(r'[^A-Za-z0-9 _.-]', '', str(stage))[:32] or 'inference'
                        stats['progress'] = {
                            'request_id': request.get('request_id'), 'stage': stage,
                            'current': current, 'total': total,
                            'percent': round(current * 100.0 / total, 1),
                        }

                    instance._progress_callback = update_progress
                    try:
                        result = call(instance.server_inference, **request['args'])
                        stats['inference_calls'] += 1
                    finally:
                        instance._progress_callback = None
                        stats['progress'] = None
                self.reply({'request_id': request.get('request_id'), 'status': 'ok', 'result': result})
            except Exception as error:
                traceback.print_exc()
                self.reply({'request_id': request.get('request_id'), 'status': 'error',
                            'error': f'{type(error).__name__}: {error}'})

    server = ThreadingHTTPServer(('127.0.0.1', urlsplit(url).port), Handler)
    server.daemon_threads = False
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    print(json.dumps({'status': 'ready', 'name': name, 'pid': os.getpid(), 'endpoint': url}), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        call(instance.server_unload_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', required=True)
    serve(parser.parse_args().serve)
