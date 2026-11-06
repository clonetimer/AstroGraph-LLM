"""Private bounded worker. Response capture is explicit; headers never leave stdin."""
import base64
import json
import sys
import ssl

import urllib.request
import urllib.error
from urllib.parse import urlsplit


def verified_tls_context():
    """Add the shipped Mozilla roots without disabling OS trust or verification."""
    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        pass  # Full-source installations can rely on OS trust; portable profiles bundle roots.
    else:
        context.load_verify_locations(cafile=certifi.where())
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Prevent accidental forwarding of API credentials to redirected hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("HTTP_REDIRECT_NOT_ALLOWED")


def main():
    captured={}
    config={}
    try:
        config=json.loads(sys.stdin.buffer.read(16_000_001))
        limit=int(config["max_response_bytes"])
        # Must exactly match llm.audit.wire_bytes; verified against a real loopback socket.
        body=json.dumps(config["payload"],ensure_ascii=False,sort_keys=True,
                        separators=(",",":"),allow_nan=False).encode("utf-8")
        req=urllib.request.Request(config["url"],data=body,
            headers={"Content-Type":"application/json",**config["headers"]},method="POST")
        # urllib's default redirect handler can forward Authorization headers.
        # Provider endpoint changes must be explicit configuration changes.
        # A loopback provider must NEVER send diagnostic material or credentials
        # to an operating-system or environment-configured HTTP proxy. This is
        # stronger than NO_PROXY, whose handling differs across Windows/Python.
        hostname = urlsplit(config["url"]).hostname
        loopback = hostname is not None and hostname.lower() in {"127.0.0.1", "localhost", "::1"}
        proxy_handler = urllib.request.ProxyHandler({}) if loopback else urllib.request.ProxyHandler()
        with urllib.request.build_opener(
            proxy_handler, _NoRedirect(), urllib.request.HTTPSHandler(context=verified_tls_context())
        ).open(
            req, timeout=config["timeout_s"]
        ) as response:
            length=response.headers.get("Content-Length")
            if length and int(length)>limit:raise ValueError("HTTP_RESPONSE_TOO_LARGE")
            raw=response.read(limit+1)
            status=response.status
        if len(raw)>limit:raise ValueError("HTTP_RESPONSE_TOO_LARGE")
        if config.get("capture_response"):
            captured={"raw_body_b64":base64.b64encode(raw).decode("ascii"),"http_status":status}
        out={"ok":True,"document":json.loads(raw)}
    except urllib.error.HTTPError as exc:
        # Sanitized error to caller; bounded raw error body only to explicit task audit.
        if config.get("capture_response"):
            raw=exc.read(int(config["max_response_bytes"])+1)
            if len(raw)<=int(config["max_response_bytes"]):
                captured={"raw_body_b64":base64.b64encode(raw).decode("ascii"),"http_status":exc.code}
        exc.close()
        out={"ok":False,"error":"HTTP_STATUS_"+str(exc.code)}
    except Exception as exc:
        code=(str(exc) if str(exc) in {"HTTP_RESPONSE_TOO_LARGE", "HTTP_REDIRECT_NOT_ALLOWED"}
              else type(exc).__name__)
        out={"ok":False,"error":code}
    out.update(captured)
    # This is a machine protocol; a Windows/GBK console encoding must not alter it.
    sys.stdout.buffer.write(json.dumps(out,ensure_ascii=False).encode("utf-8"))

if __name__=="__main__":main()
