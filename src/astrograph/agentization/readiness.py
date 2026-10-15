from __future__ import annotations
import re, zipfile
from pathlib import Path
from typing import Any, Iterable
from .artifacts import sha256_file

EXACT_MODEL_NAMES={"p2_innovation.pt","adcs_p2_innovation.pt"}
MODEL_RE=re.compile(r"(^|/)([^/]*p2[^/]*innovation[^/]*\.(pt|pth|ckpt))$",re.I)
TEMP_RE=re.compile(r"(temporal.*(3.*8|groups[_-]?3[_-]?8)|groups[_-]?3[_-]?8.*temporal|frozen_temporal_policy_receipt)",re.I)

def audit_roots(roots: Iterable[Path]) -> dict[str,Any]:
    candidates=[]
    for root in roots:
        root=Path(root)
        if not root.exists(): continue
        if root.is_file() and root.suffix.lower()==".zip":
            try:
                with zipfile.ZipFile(root) as zf:
                    for info in zf.infolist():
                        name=info.filename
                        kind="MODEL_CANDIDATE" if MODEL_RE.search(name) else ("TEMPORAL_CANDIDATE" if TEMP_RE.search(name) else None)
                        if kind: candidates.append({"container":str(root),"member":name,"bytes":info.file_size,"kind":kind,"attestation":"UNVERIFIED_CANDIDATE_ONLY"})
            except (zipfile.BadZipFile,PermissionError,OSError): pass
        elif root.is_dir():
            for p in root.rglob('*'):
                if not p.is_file(): continue
                rel=p.relative_to(root).as_posix()
                kind="MODEL_CANDIDATE" if MODEL_RE.search(rel) else ("TEMPORAL_CANDIDATE" if TEMP_RE.search(rel) else None)
                if kind: candidates.append({"container":str(root),"path":str(p),"bytes":p.stat().st_size,"sha256":sha256_file(p),"kind":kind,"attestation":"UNVERIFIED_CANDIDATE_ONLY"})
    exact=[c for c in candidates if Path(c.get("member",c.get("path",""))).name.lower() in EXACT_MODEL_NAMES]
    temporal=[c for c in candidates if c["kind"]=="TEMPORAL_CANDIDATE"]
    return {"schema_version":"astrograph-authoritative-p2-chain-audit.v1","candidate_count":len(candidates),"candidates":candidates,"authoritative_model_status":"NOT_ESTABLISHED" if not exact else "CANDIDATE_REQUIRES_PACKAGE_AND_RECEIPT_ATTESTATION","temporal_v2_groups_3_8_status":"NOT_ESTABLISHED" if not temporal else "CANDIDATE_REQUIRES_TRACE_AND_RECEIPT_ATTESTATION","overall_status":"BLOCKED","claim_boundary":"Filename matches do not establish authoritative scientific identity."}
