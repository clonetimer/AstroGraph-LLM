
"""Evidence-capability lattice for incomplete-knowledge diagnosis.

Capability is a vector over ontology mapping, source pinning, executable positive
rules, contextual applicability, development validation, freeze state,
provenance independence, and conflict.  Episode rankings never define coverage.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib, json, re
from typing import Any, Iterable, Mapping
_SHA256=re.compile(r"^[0-9a-f]{64}$")
_BANNED_RUNTIME_KEYS={"ground_truth","true_label","fault_label","y_true","target","correctness","accuracy","test_outcome"}
class EvidencePolarity(str,Enum):
    REQUIRED="REQUIRED"; SUPPORTING="SUPPORTING"; COUNTER="COUNTER"; EXCLUSION="EXCLUSION"
class EvidenceMaturity(str,Enum):
    DRAFT="DRAFT"; SOURCE_PINNED="SOURCE_PINNED"; EXECUTABLE="EXECUTABLE"; UNIT_VALIDATED="UNIT_VALIDATED"; DEVELOPMENT_VALIDATED="DEVELOPMENT_VALIDATED"; FROZEN="FROZEN"
class ProvenanceRole(str,Enum):
    DOMAIN_EVIDENCE="DOMAIN_EVIDENCE"; TRANSFORMATION="TRANSFORMATION"; VALIDATION="VALIDATION"; REVIEW="REVIEW"
class SourceKind(str,Enum):
    STANDARD="standard"; FMEA="fmea"; PEER_REVIEWED="peer_reviewed"; SIMULATOR_SPEC="simulator_spec"; HIL_RECORD="hil_record"; FLIGHT_RECORD="flight_record"; EXPERT_REVIEW="expert_review"; AGENT_GENERATED="agent_generated"; DERIVED_CODE="derived_code"; OTHER="other"
_MATURITY_ORDER={m:i for i,m in enumerate(EvidenceMaturity)}
_POSITIVE={EvidencePolarity.REQUIRED,EvidencePolarity.SUPPORTING}
def _canonical_bytes(v:Any)->bytes: return (json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
def content_sha256(v:Any)->str: return hashlib.sha256(_canonical_bytes(v)).hexdigest()
def _assert_sha(v:str|None,n:str,required:bool=False)->None:
    if v is None:
        if required: raise ValueError(f"{n} is required")
        return
    if not _SHA256.fullmatch(v): raise ValueError(f"{n} must be lowercase SHA-256")
def _walk_keys(v:Any)->Iterable[str]:
    if isinstance(v,Mapping):
        for k,c in v.items(): yield str(k).strip().lower(); yield from _walk_keys(c)
    elif isinstance(v,(list,tuple)):
        for c in v: yield from _walk_keys(c)
@dataclass(frozen=True)
class OperationalContext:
    subsystem:str; operating_mode:str; telemetry_features:tuple[str,...]; generator_domain:str|None=None; environment_tags:tuple[str,...]=()
    def __post_init__(self):
        if not self.subsystem.strip() or not self.operating_mode.strip(): raise ValueError("subsystem and operating_mode required")
        if len(set(self.telemetry_features))!=len(self.telemetry_features): raise ValueError("telemetry_features must be unique")
@dataclass(frozen=True)
class EvidenceProvenance:
    provenance_id:str; source_kind:SourceKind|str; source_artifact_sha256:str; source_locator:str; independence_group:str
    role:ProvenanceRole|str=ProvenanceRole.DOMAIN_EVIDENCE
    transformation_receipt_sha256:str|None=None; human_reviewers:tuple[str,...]=()
    def __post_init__(self):
        if not self.provenance_id.strip() or not self.independence_group.strip(): raise ValueError("provenance_id and independence_group required")
        kind=SourceKind(self.source_kind); role=ProvenanceRole(self.role)
        object.__setattr__(self,"source_kind",kind); object.__setattr__(self,"role",role)
        _assert_sha(self.source_artifact_sha256,"source_artifact_sha256",True); _assert_sha(self.transformation_receipt_sha256,"transformation_receipt_sha256")
        if role is ProvenanceRole.DOMAIN_EVIDENCE and kind in {SourceKind.AGENT_GENERATED,SourceKind.DERIVED_CODE}:
            raise ValueError("agent/code transformations cannot be scientific domain evidence")
@dataclass(frozen=True)
class EvidenceCapabilityRecord:
    evidence_id:str; fault_id:str; polarity:EvidencePolarity; maturity:EvidenceMaturity; predicate_sha256:str|None
    provenance_ids:tuple[str,...]; required_telemetry_features:tuple[str,...]; applicable_modes:tuple[str,...]
    applicable_generator_domains:tuple[str,...]=(); validation_receipt_sha256:str|None=None; freeze_receipt_sha256:str|None=None
    runtime_contract:Mapping[str,Any]=field(default_factory=dict)
    def __post_init__(self):
        if not self.evidence_id.strip() or not self.fault_id.strip(): raise ValueError("evidence_id and fault_id required")
        if len(set(self.provenance_ids))!=len(self.provenance_ids): raise ValueError("provenance_ids must be unique")
        if len(set(self.required_telemetry_features))!=len(self.required_telemetry_features): raise ValueError("required telemetry must be unique")
        _assert_sha(self.predicate_sha256,"predicate_sha256"); _assert_sha(self.validation_receipt_sha256,"validation_receipt_sha256"); _assert_sha(self.freeze_receipt_sha256,"freeze_receipt_sha256")
        leaked=sorted(set(_walk_keys(self.runtime_contract))&_BANNED_RUNTIME_KEYS)
        if leaked: raise ValueError(f"runtime contract contains truth/outcome keys: {leaked}")
        rank=_MATURITY_ORDER[self.maturity]
        if rank>=_MATURITY_ORDER[EvidenceMaturity.EXECUTABLE] and self.predicate_sha256 is None: raise ValueError("executable evidence requires predicate")
        if rank>=_MATURITY_ORDER[EvidenceMaturity.SOURCE_PINNED] and not self.provenance_ids: raise ValueError("source-pinned evidence requires provenance")
        if rank>=_MATURITY_ORDER[EvidenceMaturity.DEVELOPMENT_VALIDATED] and self.validation_receipt_sha256 is None: raise ValueError("development validated evidence requires receipt")
        if self.maturity is EvidenceMaturity.FROZEN and self.freeze_receipt_sha256 is None: raise ValueError("frozen evidence requires freeze receipt")
    @property
    def positive_vote_capable(self)->bool: return self.polarity in _POSITIVE and _MATURITY_ORDER[self.maturity]>=_MATURITY_ORDER[EvidenceMaturity.EXECUTABLE]
@dataclass(frozen=True)
class EvidenceCapabilityManifest:
    schema:str; fault_universe:tuple[str,...]; provenance:tuple[EvidenceProvenance,...]; records:tuple[EvidenceCapabilityRecord,...]; manifest_sha256:str|None=None
    def __post_init__(self):
        if self.schema!="astrograph-evidence-capability-manifest.v1": raise ValueError("unsupported capability schema")
        if not self.fault_universe or len(set(self.fault_universe))!=len(self.fault_universe): raise ValueError("fault_universe must be non-empty and unique")
        pids=[p.provenance_id for p in self.provenance]; eids=[e.evidence_id for e in self.records]
        if len(set(pids))!=len(pids): raise ValueError("duplicate provenance_id")
        if len(set(eids))!=len(eids): raise ValueError("duplicate evidence_id")
        pmap={p.provenance_id:p for p in self.provenance}; universe=set(self.fault_universe)
        for rec in self.records:
            if rec.fault_id not in universe: raise ValueError(f"fault outside universe: {rec.fault_id}")
            missing=set(rec.provenance_ids)-set(pmap)
            if missing: raise ValueError(f"missing provenance for {rec.evidence_id}: {sorted(missing)}")
            if rec.positive_vote_capable:
                domain=[pmap[x] for x in rec.provenance_ids if pmap[x].role is ProvenanceRole.DOMAIN_EVIDENCE]
                if not domain: raise ValueError(f"positive executable evidence lacks domain-evidence provenance: {rec.evidence_id}")
            if rec.maturity is EvidenceMaturity.FROZEN:
                reviewed=any(pmap[x].human_reviewers or pmap[x].role is ProvenanceRole.REVIEW for x in rec.provenance_ids)
                if not reviewed: raise ValueError(f"frozen evidence lacks human/domain review provenance: {rec.evidence_id}")
        expected=self.compute_sha256()
        if self.manifest_sha256 is not None and self.manifest_sha256!=expected: raise ValueError("manifest_sha256 mismatch")
    def canonical_payload(self)->dict[str,Any]:
        def norm(x:Any)->Any:
            if isinstance(x,Enum): return x.value
            if isinstance(x,Mapping): return {k:norm(v) for k,v in x.items()}
            if isinstance(x,(list,tuple)): return [norm(v) for v in x]
            return x
        return {"schema":self.schema,"fault_universe":sorted(self.fault_universe),"provenance":sorted((norm(asdict(p)) for p in self.provenance),key=lambda x:x["provenance_id"]),"records":sorted((norm(asdict(r)) for r in self.records),key=lambda x:x["evidence_id"])}
    def compute_sha256(self)->str: return content_sha256(self.canonical_payload())
@dataclass(frozen=True)
class CapabilityVector:
    universe_size:int; ontology_mapped_fraction:float; source_pinned_fraction:float; executable_positive_fraction:float; telemetry_applicable_fraction:float
    development_validated_fraction:float; frozen_fraction:float; provenance_independent_fraction:float; conflict_fraction:float
    executable_positive_fault_ids:tuple[str,...]; telemetry_applicable_fault_ids:tuple[str,...]; development_validated_fault_ids:tuple[str,...]; frozen_fault_ids:tuple[str,...]
    provenance_independent_fault_ids:tuple[str,...]; conflicted_fault_ids:tuple[str,...]; missing_executable_fault_ids:tuple[str,...]
    def as_dict(self)->dict[str,Any]: return asdict(self)
def compute_capability_vector(manifest:EvidenceCapabilityManifest,context:OperationalContext,*,minimum_independent_groups:int=2)->CapabilityVector:
    if minimum_independent_groups<1: raise ValueError("minimum_independent_groups must be >=1")
    universe=set(manifest.fault_universe); by={f:[] for f in universe}
    for r in manifest.records: by[r.fault_id].append(r)
    pmap={p.provenance_id:p for p in manifest.provenance}
    ontology={f for f,rs in by.items() if rs}
    pinned={f for f,rs in by.items() if any(_MATURITY_ORDER[r.maturity]>=_MATURITY_ORDER[EvidenceMaturity.SOURCE_PINNED] for r in rs)}
    executable={f for f,rs in by.items() if any(r.positive_vote_capable for r in rs)}
    available=set(context.telemetry_features); applicable=set(); dev=set(); frozen=set(); independent=set(); conflicted=set()
    for f,rs in by.items():
        positive=[r for r in rs if r.positive_vote_capable]
        active=[r for r in positive if set(r.required_telemetry_features).issubset(available) and (not r.applicable_modes or context.operating_mode in r.applicable_modes) and (not r.applicable_generator_domains or context.generator_domain in r.applicable_generator_domains)]
        if active: applicable.add(f)
        if any(_MATURITY_ORDER[r.maturity]>=_MATURITY_ORDER[EvidenceMaturity.DEVELOPMENT_VALIDATED] for r in active): dev.add(f)
        if any(r.maturity is EvidenceMaturity.FROZEN for r in active): frozen.add(f)
        groups={pmap[pid].independence_group for r in active for pid in r.provenance_ids if pmap[pid].role is ProvenanceRole.DOMAIN_EVIDENCE}
        if len(groups)>=minimum_independent_groups: independent.add(f)
        pol={r.polarity for r in rs if _MATURITY_ORDER[r.maturity]>=_MATURITY_ORDER[EvidenceMaturity.EXECUTABLE]}
        if pol&_POSITIVE and pol&{EvidencePolarity.COUNTER,EvidencePolarity.EXCLUSION}: conflicted.add(f)
    n=len(universe); frac=lambda s:len(s)/n
    return CapabilityVector(n,frac(ontology),frac(pinned),frac(executable),frac(applicable),frac(dev),frac(frozen),frac(independent),frac(conflicted),tuple(sorted(executable)),tuple(sorted(applicable)),tuple(sorted(dev)),tuple(sorted(frozen)),tuple(sorted(independent)),tuple(sorted(conflicted)),tuple(sorted(universe-executable)))
