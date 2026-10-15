"""Worker for host-approved one-file Python plugins; not an untrusted-code sandbox."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import types


def main() -> int:
    request_path, output_path = map(Path, sys.argv[1:3])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        path = Path(request["module_file"]).resolve(strict=True)
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != request["artifact_sha256"]:
            raise ValueError("artifact changed")
        module = types.ModuleType("astrograph_approved_plugin")
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        # Execute the exact bytes that were hashed, not a second pathname read.
        exec(compile(source, str(path), "exec"), module.__dict__)
        result = getattr(module, request["function_name"])(request["payload"])
        data = json.dumps(result, sort_keys=True, allow_nan=False, ensure_ascii=False).encode()
        if len(data) > request["max_output_bytes"]:
            raise ValueError("output too large")
        output = {"status":"SUCCEEDED", "output":result}
    except Exception:
        output = {"status":"FAILED", "error_code":"TOOL_WORKER_FAILED"}
    output_path.write_text(json.dumps(output, allow_nan=False, ensure_ascii=False),encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
