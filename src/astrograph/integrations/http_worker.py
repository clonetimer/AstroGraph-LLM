"""Private bounded worker. Response capture is explicit; headers never leave stdin."""
import base64
import json
import sys
import urllib.request
import urllib.error


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
        with urllib.request.urlopen(req,timeout=config["timeout_s"]) as response:
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
        code=str(exc) if str(exc)=="HTTP_RESPONSE_TOO_LARGE" else type(exc).__name__
        out={"ok":False,"error":code}
    out.update(captured)
    sys.stdout.write(json.dumps(out,ensure_ascii=False))

if __name__=="__main__":main()
