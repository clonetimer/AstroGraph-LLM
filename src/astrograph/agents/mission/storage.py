"""Task artifacts, immutable completed results and fail-closed interrupted runs."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def _replace_atomic(temp: str | Path, target: str | Path) -> None:
    """Commit an atomic replacement, tolerating only transient Windows locks."""
    for attempt in range(5):
        try:
            os.replace(temp, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


class Workspace:
    def __init__(self, output_root, read_roots, task_id, *, max_bytes=64_000_000):
        self.root=Path(output_root).resolve()/task_id
        self.read_roots=tuple(Path(p).resolve() for p in read_roots)
        self.max_bytes=max_bytes

    def read_path(self, value: str, *, directory=False) -> Path:
        candidate=Path(value).expanduser()
        # Authorize the lexical target before requiring it to exist.  Besides
        # preserving PermissionError as the public boundary, this avoids using
        # filesystem existence as an oracle for paths outside approved roots.
        p=candidate.resolve(strict=False)
        roots=(*self.read_roots,self.root)
        if not any(p.is_relative_to(root) for root in roots):
            raise PermissionError("artifact is outside host-approved read roots")
        p=candidate.resolve(strict=True)
        # Recheck after resolving existing symlinks/junctions so an approved
        # lexical path cannot escape its host-approved root.
        if not any(p.is_relative_to(root) for root in roots):
            raise PermissionError("artifact is outside host-approved read roots")
        if directory:
            if not p.is_dir(): raise ValueError("directory artifact required")
        else:
            if not p.is_file() or p.stat().st_size>self.max_bytes:
                raise ValueError("artifact is not a regular file or exceeds host size budget")
        return p

    def write_bytes(self, name: str, value: bytes) -> Path:
        p=self.root/name
        if p.resolve().parent!=self.root.resolve():raise ValueError("artifact name must be a basename")
        if len(value)>self.max_bytes:raise ValueError("audit artifact exceeds host size budget")
        fd,temp=tempfile.mkstemp(prefix=".pending-",dir=self.root)
        try:
            with os.fdopen(fd,"wb") as f:
                f.write(value);f.flush();os.fsync(f.fileno())
            os.link(temp,p)
        finally:
            if os.path.exists(temp):os.unlink(temp)
        return p

    def write_json(self, name: str, value: Any) -> Path:
        p=self.root/name
        if p.resolve().parent!=self.root.resolve():raise ValueError("artifact name must be a basename")
        fd,temp=tempfile.mkstemp(prefix=".pending-",dir=self.root)
        try:
            with os.fdopen(fd,"wb") as f:
                f.write(canonical(value));f.flush();os.fsync(f.fileno())
            os.link(temp,p)  # fails if a different writer already committed
        finally:
            if os.path.exists(temp):os.unlink(temp)
        return p

    def replace_json(self, name: str, value: Any):
        p=self.root/name
        if p.resolve().parent!=self.root.resolve():raise ValueError("basename required")
        fd,temp=tempfile.mkstemp(prefix=".pending-",dir=self.root)
        try:
            with os.fdopen(fd,"wb") as f:
                f.write(canonical(value));f.flush();os.fsync(f.fileno())
            _replace_atomic(temp,p)
        finally:
            if os.path.exists(temp):os.unlink(temp)
        return p

    def event(self, event: str, **details):
        p=self.root/"events.jsonl"
        previous="0"*64
        if p.exists():
            lines=p.read_text(encoding="utf-8").splitlines()
            if lines: previous=json.loads(lines[-1])["event_sha256"]
        body={"at":datetime.now(timezone.utc).isoformat(),"event":event,"previous_sha256":previous,**details}
        body["event_sha256"]=sha(body)
        with p.open("ab") as f:f.write(canonical(body)+b"\n")

    def begin(self, identity: dict, *, resume_read_only=False):
        if self.root.exists():
            manifest=self.root/"request_identity.json"
            if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8"))!=identity:
                raise ValueError("task ID already used with a different request, input, or policy")
            result=self.root/"result.json"
            receipt=self.root/"result_receipt.json"
            if result.is_file() and receipt.is_file():
                if file_sha(result)!=json.loads(receipt.read_text(encoding="utf-8"))["sha256"]:
                    raise ValueError("persisted result integrity mismatch")
                # Recheck every committed step artifact; do not trust a successful status alone.
                self.verify_events()
                for item in json.loads((self.root/"artifacts.json").read_text(encoding="utf-8")):
                    path=self.root/item["name"]
                    if not path.resolve().is_relative_to(self.root.resolve()):raise ValueError("artifact escaped task root")
                    if file_sha(path)!=item["sha256"]:raise ValueError("task artifact changed after completion")
                return json.loads(result.read_text(encoding="utf-8"))
            if resume_read_only and not result.exists() and not (self.root/"artifacts.json").exists():
                self.verify_events()
                safe={"model.infer","model.capabilities","graph.reason","document.convert","document.clean","rag.retrieve"}
                events=[json.loads(line) for line in (self.root/"events.jsonl").read_text(encoding="utf-8").splitlines()]
                started=[e for e in events if e["event"]=="TOOL_STARTED"]
                if not started or any(e["tool_id"] not in safe for e in started):
                    raise RuntimeError("INTERRUPTED_RUN_REQUIRES_REVIEW: unsafe or unknown prefix")
                cache=[]
                for event in events:
                    if event["event"]!="TOOL_FINISHED":continue
                    path=self.root/f"step-{event['step']:03d}.json"
                    value=json.loads(path.read_text(encoding="utf-8"))
                    if sha(value["result"])!=event["output_sha256"]:
                        raise ValueError("checkpoint artifact integrity mismatch")
                    if value["tool_id"]=="model.infer" and value["result"].get("mode")!="BOUND_PRECOMPUTED_MODEL_OUTPUT_VALIDATED_NOT_RERUN":
                        raise RuntimeError("INTERRUPTED_RUN_REQUIRES_REVIEW: native model worker prefix")
                    if value["tool_id"]=="document.convert" and any(d.get("parser")!="UTF8_NATIVE_TEXT" for d in value["result"].get("documents",[])):
                        raise RuntimeError("INTERRUPTED_RUN_REQUIRES_REVIEW: external parser prefix")
                    cache.append(value)
                # Orphan step artifacts indicate a crash before a receipt; retain for review.
                committed={e['step'] for e in events if e['event']=='TOOL_FINISHED'}
                if any(int(p.stem.split('-')[1]) not in committed for p in self.root.glob('step-*.json')):
                    raise RuntimeError("INTERRUPTED_RUN_REQUIRES_REVIEW: uncommitted step artifact")
                self.event("READ_ONLY_PREFIX_RESUMED",reused_steps=len(cache))
                return {"__resume_read_only__":True,"steps":cache,"max_step":max(e["step"] for e in started)}
            raise RuntimeError("INTERRUPTED_RUN_REQUIRES_REVIEW: no automatic replay of uncertain side effects")
        self.root.mkdir(parents=True,exist_ok=False)
        self.write_json("request_identity.json",identity)
        self.event("TASK_STARTED",identity_sha256=sha(identity))
        return None

    def finish(self,result: dict):
        self.event("TASK_FINISHED",status=result["status"])
        artifacts=[{"name":p.relative_to(self.root).as_posix(),"sha256":file_sha(p)} for p in sorted(self.root.rglob("*"))
                   if p.is_file() and p.name not in {"result.json","result_receipt.json","artifacts.json"}]
        self.write_json("artifacts.json",artifacts)
        p=self.write_json("result.json",result)
        self.write_json("result_receipt.json",{"sha256":file_sha(p)})

    def verify_events(self):
        previous="0"*64
        for line in (self.root/"events.jsonl").read_text(encoding="utf-8").splitlines():
            body=json.loads(line);digest=body.pop("event_sha256")
            if body["previous_sha256"]!=previous or sha(body)!=digest:raise ValueError("event chain integrity mismatch")
            previous=digest
        return previous
