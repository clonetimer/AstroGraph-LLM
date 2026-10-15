from __future__ import annotations
from typing import Any
from .protocol import CapabilityProtocolError, sha256_json

ALLOWED_STATES=("PROPOSED","SOURCE_GROUNDED","COMPILED","UNIT_VALIDATED","DEVELOPMENT_VALIDATED","REVIEWED","FROZEN")
AUTO_MAX_STATE="UNIT_VALIDATED"

def propose_card(args: dict[str,Any], source_receipt: dict[str,Any]) -> dict[str,Any]:
    required=("fault_id","evidence_type","telemetry_fields","predicate","applicability_envelope")
    missing=[k for k in required if not args.get(k)]
    if missing: raise CapabilityProtocolError("INCOMPLETE_EVIDENCE_CARD",f"missing fields: {missing}")
    card={"schema_version":"astrograph-evidence-card.v1","status":"PROPOSED","fault_id":str(args["fault_id"]),"evidence_type":str(args["evidence_type"]),"telemetry_fields":list(args["telemetry_fields"]),"predicate":args["predicate"],"applicability_envelope":args["applicability_envelope"],"counter_evidence":args.get("counter_evidence",[]),"exclusion_conditions":args.get("exclusion_conditions",[]),"provenance_independence_group":args.get("provenance_independence_group","UNRESOLVED"),"source_receipt":source_receipt,"human_review":"REQUIRED_BEFORE_REVIEWED_OR_FROZEN"}
    card["card_sha256"]=sha256_json(card)
    return card

def validate_card(card: dict[str,Any], requested_state: str | None=None) -> dict[str,Any]:
    state=str(card.get("status",""))
    if state not in ALLOWED_STATES: raise CapabilityProtocolError("INVALID_EVIDENCE_STATE",f"invalid evidence state {state!r}")
    if requested_state in {"DEVELOPMENT_VALIDATED","REVIEWED","FROZEN"}:
        raise CapabilityProtocolError("HUMAN_SCIENTIFIC_APPROVAL_REQUIRED",f"agent cannot promote evidence to {requested_state}")
    if not card.get("source_receipt",{}).get("verified"):
        raise CapabilityProtocolError("SOURCE_NOT_PINNED","evidence card source must be SHA-256 verified")
    return {"valid":True,"current_state":state,"automatic_promotion_ceiling":AUTO_MAX_STATE,"card_sha256":sha256_json({k:v for k,v in card.items() if k!="card_sha256"})}
