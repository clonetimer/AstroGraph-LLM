"""Bounded HTTP worker must not proxy loopback model requests."""
from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest

WORKER = Path(__file__).resolve().parents[1] / 'src' / 'astrograph' / 'integrations' / 'http_worker.py'


@contextmanager
def local_endpoint(label):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            calls.append(body)
            raw = json.dumps({'destination': label}).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield 'http://127.0.0.1:%d/v1/chat/completions' % server.server_port, calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def call_worker(url, proxy_base):
    environ = dict(os.environ)
    environ.update({
        'HTTP_PROXY': proxy_base, 'http_proxy': proxy_base,
        'HTTPS_PROXY': proxy_base, 'https_proxy': proxy_base,
        'NO_PROXY': '', 'no_proxy': '',
    })
    config = {
        'url': url, 'payload': {'example': 'diagnosis source'},
        'headers': {}, 'timeout_s': 10, 'max_response_bytes': 8192,
    }
    process = subprocess.run(
        [sys.executable, str(WORKER)],
        input=json.dumps(config).encode('utf-8'),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=environ, timeout=15, check=True,
    )
    envelope = json.loads(process.stdout.decode('utf-8'))
    if not envelope.get('ok'):
        raise AssertionError('worker rejected request: %s' % envelope.get('error'))
    return envelope['document']


class LoopbackProxyIsolationTests(unittest.TestCase):
    def test_loopback_worker_ignores_environment_proxy_even_without_no_proxy(self):
        with local_endpoint('PROXY') as (proxy_url, proxy_calls):
            with local_endpoint('MODEL') as (model_url, model_calls):
                result = call_worker(model_url, proxy_url.split('/v1/', 1)[0])
                self.assertEqual(result, {'destination': 'MODEL'})
                self.assertEqual(len(model_calls), 1)
                self.assertEqual(len(proxy_calls), 0)

    def test_non_loopback_endpoint_can_use_configured_proxy(self):
        with local_endpoint('PROXY') as (proxy_url, proxy_calls):
            result = call_worker(
                'http://nonexistent-provider.invalid/v1/chat/completions',
                proxy_url.split('/v1/', 1)[0],
            )
            self.assertEqual(result, {'destination': 'PROXY'})
            self.assertEqual(len(proxy_calls), 1)


if __name__ == '__main__':
    unittest.main()
