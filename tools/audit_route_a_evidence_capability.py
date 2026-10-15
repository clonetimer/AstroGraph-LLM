
#!/usr/bin/env python3
"""Audit a Route A evidence capability manifest without model predictions."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from astrograph.knowledge.evidence_capability import (
    EvidenceCapabilityManifest, EvidenceCapabilityRecord, EvidenceMaturity,
    EvidencePolarity, EvidenceProvenance, OperationalContext, compute_capability_vector,
)

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--subsystem", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--feature", action="append", default=[])
    ap.add_argument("--generator-domain")
    args=ap.parse_args()
    raw=json.loads(args.manifest.read_text())
    provenance=tuple(EvidenceProvenance(**x) for x in raw["provenance"])
    records=[]
    for x in raw["records"]:
        x=dict(x); x["polarity"]=EvidencePolarity(x["polarity"]); x["maturity"]=EvidenceMaturity(x["maturity"])
        for key in ["provenance_ids","required_telemetry_features","applicable_modes","applicable_generator_domains"]:
            x[key]=tuple(x.get(key,()))
        records.append(EvidenceCapabilityRecord(**x))
    m=EvidenceCapabilityManifest(raw["schema"],tuple(raw["fault_universe"]),provenance,tuple(records),raw.get("manifest_sha256"))
    context=OperationalContext(args.subsystem,args.mode,tuple(args.feature),args.generator_domain)
    print(json.dumps(compute_capability_vector(m,context).as_dict(),sort_keys=True,indent=2))
    return 0
if __name__ == "__main__": raise SystemExit(main())
