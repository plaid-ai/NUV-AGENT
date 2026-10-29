"""Loopback-only demo display and allowlisted input selection. No cloud/auth dependencies."""
from __future__ import annotations

import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


def preview_branch() -> str:
    return (
        'tee name=local_demo_split '
        'local_demo_split. ! queue leaky=downstream max-size-buffers=1 '
        'max-size-bytes=0 max-size-time=0 ! videorate drop-only=true ! '
        'video/x-raw,framerate=5/1 ! videoconvert ! jpegenc quality=85 ! '
        'appsink name=local_demo_sink emit-signals=true max-buffers=1 drop=true sync=false '
        'local_demo_split. ! queue leaky=downstream max-size-buffers=1 '
        'max-size-bytes=0 max-size-time=0 ! '
    )


class LocalDemoServer:
    def __init__(self, assets: str | Path, port: int = 8080, profiles=None, restart=None):
        self.profiles = profiles
        self.restart = restart
        self.assets = Path(assets).resolve()
        self._lock = threading.Lock()
        self._jpeg = b''
        self._updated = 0.0
        self._sequence = 0
        self._server = None
        self._thread = None
        self.port = port

    def publish(self, jpeg: bytes) -> None:
        if not jpeg or len(jpeg) > 8 * 1024 * 1024:
            return
        with self._lock:
            self._jpeg = jpeg
            self._updated = time.monotonic()
            self._sequence += 1

    def snapshot(self):
        with self._lock:
            if not self._jpeg:
                return None, 0, 0.0
            age = time.monotonic() - self._updated
            if age > 3:
                return None, 0, age
            return self._jpeg, self._sequence, age

    def start(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(3)

            def log_message(self, *_args):
                pass

            def send_bytes(self, code, data, mime='text/plain', sequence=None):
                self.send_response(code)
                self.send_header('Content-Type', mime)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; font-src 'self'; connect-src 'self'; frame-ancestors 'none'")
                if sequence is not None:
                    self.send_header('X-Frame-Sequence', str(sequence))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                allowed = {f'localhost:{owner.port}', f'127.0.0.1:{owner.port}'}
                if (self.headers.get('Host') not in allowed
                    or self.headers.get('Origin') not in {f'http://{host}' for host in allowed}
                    or self.headers.get('Content-Type') != 'application/json'):
                    self.send_bytes(403, b'Local control only')
                    return
                if self.path != '/api/profile' or owner.profiles is None or owner.restart is None:
                    self.send_bytes(404, b'Not found')
                    return
                try:
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= 1024:
                        raise ValueError('Invalid request size')
                    body = json.loads(self.rfile.read(size))
                    result = owner.profiles.select(body['id'], owner.restart)
                    self.send_bytes(202, json.dumps(result).encode(), 'application/json')
                except (ValueError, KeyError, TypeError):
                    self.send_bytes(400, b'Invalid demo profile')
                except RuntimeError:
                    self.send_bytes(409, b'Demo switch unavailable')
                except OSError:
                    self.send_bytes(503, b'Demo selection storage unavailable')

            def do_GET(self):
                # Reject DNS rebinding and non-local origins even though we bind loopback.
                allowed = {f'localhost:{owner.port}', f'127.0.0.1:{owner.port}'}
                if self.headers.get('Host') not in allowed:
                    self.send_bytes(403, b'Local display only')
                    return
                origin = self.headers.get('Origin')
                if origin and origin not in {f'http://{host}' for host in allowed}:
                    self.send_bytes(403, b'Local display only')
                    return
                path = unquote(urlsplit(self.path).path)
                if path == '/api/profiles' and owner.profiles is not None:
                    self.send_bytes(200, json.dumps(owner.profiles.snapshot()).encode(), 'application/json')
                    return
                if path == '/api/frame.jpg':
                    jpeg, sequence, _age = owner.snapshot()
                    if jpeg is None:
                        self.send_bytes(503, b'Demo frame unavailable')
                    else:
                        self.send_bytes(200, jpeg, 'image/jpeg', sequence)
                    return
                target = (owner.assets / ('index.html' if path == '/' else path.lstrip('/'))).resolve()
                if not target.is_relative_to(owner.assets) or not target.is_file():
                    self.send_bytes(404, b'Not found')
                    return
                self.send_bytes(200, target.read_bytes(), mimetypes.guess_type(target)[0] or 'application/octet-stream')

        self._server = ThreadingHTTPServer(('127.0.0.1', self.port), Handler)
        self.port = self._server.server_port
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True, name='local-demo-http')
        self._thread.start()

    def close(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=4)
            self._thread = None
