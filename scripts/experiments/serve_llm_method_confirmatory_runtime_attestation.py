#!/usr/bin/env python3
"""Serve freshly recomputed R16 model runtime attestation using the Python standard library."""
from __future__ import annotations
import argparse, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"src"))
from astrograph.evaluation.confirmatory_runtime_identity import build_model_runtime_attestation, load_model_runtime_spec

def main()->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--spec",type=Path,required=True); p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8765); p.add_argument("--allow-unauthenticated-remote",action="store_true"); a=p.parse_args()
    try: loopback=ip_address(a.host).is_loopback
    except ValueError: loopback=a.host=="localhost"
    if not loopback and not a.allow_unauthenticated_remote: p.error("non-loopback binding requires --allow-unauthenticated-remote; prefer an authenticated tunnel/proxy")
    spec_path=a.spec.resolve(); load_model_runtime_spec(spec_path)
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in {"/attestation","/attestation/"}: self.send_response(404); self.end_headers(); return
            try:
                payload=build_model_runtime_attestation(load_model_runtime_spec(spec_path)); raw=(json.dumps(payload,ensure_ascii=False,separators=(",",":"))+"\n").encode("utf-8")
                self.send_response(200); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(raw))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(raw)
            except Exception as exc:
                raw=json.dumps({"error":f"{type(exc).__name__}:{exc}"}).encode(); self.send_response(500); self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(raw))); self.end_headers(); self.wfile.write(raw)
        def log_message(self,fmt,*args): sys.stderr.write("[attestation] "+(fmt%args)+"\n")
    server=ThreadingHTTPServer((a.host,a.port),H); print(f"serving confirmatory runtime attestation on http://{a.host}:{a.port}/attestation",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
    return 0
if __name__=="__main__": raise SystemExit(main())
