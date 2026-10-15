from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from astrograph.agents.base import ControlledAgent
from astrograph.knowledge.schema_v3 import (
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
    RELATION_ALLOWLIST,
)


T = TypeVar("T", bound=BaseModel)


class KnowledgeExtractionBackend(Protocol):
    model_id: str
    model_version: str

    def generate_structured(
        self,
        *,
        agent_id: str,
        output_model: type[T],
        payload: dict[str, Any],
    ) -> T: ...


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_model(value: BaseModel) -> str:
    return _sha256_bytes(value.model_dump_json(exclude_none=False).encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def knowledge_source_profile_identity_payload(
    profile: "KnowledgeSourceProfile",
) -> dict[str, Any]:
    """Return the relocatable content identity while retaining path in the profile."""

    payload = profile.model_dump(mode="json")
    payload.pop("source_path", None)
    return payload


def knowledge_source_profile_sha256(profile: "KnowledgeSourceProfile") -> str:
    return _sha256_bytes(
        json.dumps(
            knowledge_source_profile_identity_payload(profile),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def knowledge_source_profile_set_sha256(
    profiles: list["KnowledgeSourceProfile"],
) -> str:
    payload = [knowledge_source_profile_identity_payload(profile) for profile in profiles]
    return _sha256_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _profile_set_sha256(profiles: list["KnowledgeSourceProfile"]) -> str:
    return knowledge_source_profile_set_sha256(profiles)


class SourceFragmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    location: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=8_000)


class KnowledgeSourceFragment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fragment_id: str = Field(pattern=r"^fragment:[^\s]+$")
    source_id: str = Field(min_length=1)
    location: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=8_000)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class KnowledgeSourceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-knowledge-source-profile.v1"
    source_id: str = Field(pattern=r"^source:[^\s]+$")
    source_path: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    independence_group: str = Field(min_length=1)
    source_role: Literal[
        "DOMAIN_DOCUMENT", "ENGINEERING_STANDARD", "GENERATOR_SOURCE",
        "DATASET_INDEX", "ASTROGRAPH_ASSET", "EXPERIMENT_EVIDENCE",
    ] = "DOMAIN_DOCUMENT"
    license_id: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    extraction_method: str = Field(min_length=1)
    extracted_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fragments: list[KnowledgeSourceFragment] = Field(min_length=1)


class KnowledgeNodeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(pattern=r"^candidate:[^\s]+$")
    node_type: KnowledgeNodeTypeV3
    canonical_name: str = Field(min_length=1)
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    canonical_id_hint: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    source_fragment_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguous: bool = False

    @field_validator("source_fragment_ids")
    @classmethod
    def unique_fragments(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class KnowledgeRelationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edge_id: str = Field(pattern=r"^candidate-edge:[^\s]+$")
    source_id: str = Field(min_length=1)
    relation_type: KnowledgeRelationTypeV3
    target_id: str = Field(min_length=1)
    source_fragment_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    ambiguous: bool = False
    rationale: str = Field(min_length=1)
    predicate_definition: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_fragment_ids")
    @classmethod
    def unique_fragments(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))


class KnowledgeExtractionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-knowledge-extraction-proposal.v1"
    source_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_schema_version: Literal["astrograph-knowledge-schema.v3"] = "astrograph-knowledge-schema.v3"
    nodes: list[KnowledgeNodeCandidate] = Field(default_factory=list)
    relations: list[KnowledgeRelationCandidate] = Field(default_factory=list)
    unresolved_claims: list[str] = Field(default_factory=list)
    proposer_mode: Literal["DETERMINISTIC_DRAFT", "LLM"] = "DETERMINISTIC_DRAFT"
    proposer_id: str = Field(min_length=1)
    proposer_version: str = Field(min_length=1)
    executable_content_present: bool = False


class KnowledgeExtractionFragmentProposal(KnowledgeExtractionProposal):
    """Bound one-fragment response so guided decoding cannot expand without limit."""

    nodes: list[KnowledgeNodeCandidate] = Field(default_factory=list, max_length=6)
    relations: list[KnowledgeRelationCandidate] = Field(default_factory=list, max_length=6)
    unresolved_claims: list[str] = Field(default_factory=list, max_length=6)


class KnowledgeCandidateFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: Literal["BLOCKER", "WARNING"]
    code: str = Field(min_length=1)
    subject_id: str | None = None
    detail: str = Field(min_length=1)


class KnowledgeCandidateValidationReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-knowledge-candidate-validation-receipt.v1"
    status: Literal["BLOCKED", "READY_FOR_HUMAN_REVIEW"]
    proposal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_node_count: int = Field(ge=0)
    reviewed_relation_count: int = Field(ge=0)
    findings: list[KnowledgeCandidateFinding] = Field(default_factory=list)
    publication_allowed: bool = False
    requires_human_approval: bool = True
    active_graph_modified: bool = False
    claim_boundary: str = (
        "READY means only that source, identity, confidence and V3 relation-shape contracts passed. "
        "It does not establish factual or causal truth and cannot publish or modify the active graph."
    )


class KnowledgeCurationTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int = Field(ge=1)
    agent_id: str = Field(min_length=1)
    outcome: str = Field(min_length=1)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ControlledKnowledgeCurationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-controlled-knowledge-curation.v1"
    source_profiles: list[KnowledgeSourceProfile] = Field(min_length=1)
    proposal: KnowledgeExtractionProposal
    validation: KnowledgeCandidateValidationReceipt
    trace: list[KnowledgeCurationTraceStep] = Field(min_length=3, max_length=3)
    files_written: list[str] = Field(default_factory=list, max_length=0)


class KnowledgeSourceProfilerAgent(ControlledAgent):
    agent_id = "knowledge-source-profiler-agent"
    agent_version = "0.1.0"

    _MEDIA_TYPES = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".csv": "text/csv",
        ".yaml": "application/yaml",
        ".yml": "application/yaml",
    }

    def profile(
        self,
        *,
        source_id: str,
        source_path: str | Path,
        independence_group: str,
        license_id: str,
        source_role: Literal[
            "DOMAIN_DOCUMENT", "ENGINEERING_STANDARD", "GENERATOR_SOURCE",
            "DATASET_INDEX", "ASTROGRAPH_ASSET", "EXPERIMENT_EVIDENCE",
        ] = "DOMAIN_DOCUMENT",
        extraction_method: str,
        fragments: list[SourceFragmentInput | dict[str, str]],
    ) -> KnowledgeSourceProfile:
        path = Path(source_path).resolve()
        if not path.is_file():
            raise ValueError("knowledge source must be an existing file")
        if path.suffix.lower() not in self._MEDIA_TYPES:
            raise ValueError(f"unsupported knowledge source suffix: {path.suffix.lower()}")
        normalized_id = source_id if source_id.startswith("source:") else f"source:{source_id}"
        inputs = [SourceFragmentInput.model_validate(item) for item in fragments]
        if not inputs:
            raise ValueError("at least one source fragment is required")
        profiled: list[KnowledgeSourceFragment] = []
        for item in inputs:
            text_sha = _sha256_bytes(item.text.encode("utf-8"))
            identity_sha = _sha256_bytes(
                json.dumps(
                    {"location": item.location, "text_sha256": text_sha},
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            fragment_id = (
                f"fragment:{normalized_id.removeprefix('source:')}:{identity_sha[:16]}"
            )
            profiled.append(
                KnowledgeSourceFragment(
                    fragment_id=fragment_id,
                    source_id=normalized_id,
                    location=item.location,
                    text=item.text,
                    text_sha256=text_sha,
                )
            )
        if len({item.fragment_id for item in profiled}) != len(profiled):
            raise ValueError("source fragments must contain distinct location/text identities")
        extracted_sha = _sha256_bytes(
            "\n".join(item.text for item in profiled).encode("utf-8")
        )
        return KnowledgeSourceProfile(
            source_id=normalized_id,
            source_path=str(path),
            source_sha256=_sha256_file(path),
            independence_group=independence_group,
            source_role=source_role,
            license_id=license_id,
            media_type=self._MEDIA_TYPES[path.suffix.lower()],
            extraction_method=extraction_method,
            extracted_text_sha256=extracted_sha,
            fragments=profiled,
        )


class KnowledgeCurationAgent(ControlledAgent):
    """Proposes offline knowledge candidates; it has no publication capability."""

    agent_id = "knowledge-curation-agent"
    agent_version = "0.5.0"

    def __init__(
        self,
        *args: Any,
        backend: KnowledgeExtractionBackend | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.backend = backend

    def propose(
        self,
        *,
        source_profiles: list[KnowledgeSourceProfile],
        approved_existing_nodes: dict[str, KnowledgeNodeTypeV3 | str],
        draft_proposal: dict[str, Any] | None = None,
        allowed_node_types: list[KnowledgeNodeTypeV3 | str] | None = None,
        allowed_relation_shapes: list[str] | None = None,
        additional_rules: list[str] | None = None,
        governance_context: dict[str, Any] | None = None,
        prompt_fragment_ids: list[str] | None = None,
        response_model: type[KnowledgeExtractionProposal] = KnowledgeExtractionProposal,
    ) -> KnowledgeExtractionProposal:
        profile_sha = _profile_set_sha256(source_profiles)
        if self.backend is None:
            if draft_proposal is None:
                raise ValueError("deterministic curation mode requires draft_proposal")
            proposal = KnowledgeExtractionProposal.model_validate(draft_proposal)
            return proposal.model_copy(
                update={
                    "proposer_mode": "DETERMINISTIC_DRAFT",
                    "proposer_id": self.agent_id,
                    "proposer_version": self.agent_version,
                }
            )
        scoped_node_types = (
            [KnowledgeNodeTypeV3(item).value for item in allowed_node_types]
            if allowed_node_types is not None
            else [item.value for item in KnowledgeNodeTypeV3]
        )
        scoped_relation_shapes = (
            sorted(set(allowed_relation_shapes))
            if allowed_relation_shapes is not None
            else sorted(
                f"{source.value}->{relation.value}->{target.value}"
                for source, relation, target in RELATION_ALLOWLIST
            )
        )
        prompt_profiles = [item.model_dump(mode="json") for item in source_profiles]
        if prompt_fragment_ids is not None:
            requested_fragments = set(prompt_fragment_ids)
            known_fragments = {
                fragment.fragment_id
                for profile in source_profiles
                for fragment in profile.fragments
            }
            if not requested_fragments or not requested_fragments.issubset(known_fragments):
                raise ValueError("prompt_fragment_ids must be a nonempty subset of profiled fragments")
            for profile in prompt_profiles:
                profile["fragments"] = [
                    fragment
                    for fragment in profile["fragments"]
                    if fragment["fragment_id"] in requested_fragments
                ]
            prompt_profiles = [profile for profile in prompt_profiles if profile["fragments"]]
        output_schema = response_model.model_json_schema()
        payload = {
            "source_profiles": prompt_profiles,
            "source_profile_sha256": profile_sha,
            "target_schema_version": "astrograph-knowledge-schema.v3",
            "approved_existing_nodes": {
                key: KnowledgeNodeTypeV3(value).value
                for key, value in sorted(approved_existing_nodes.items())
            },
            "governance_context": governance_context or {},
            "allowed_node_types": scoped_node_types,
            "allowed_relation_shapes": scoped_relation_shapes,
            "output_protocol": {
                "schema_name": response_model.__name__,
                "schema_sha256": _sha256_bytes(
                    json.dumps(
                        output_schema,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
                "delivery": "enforced by response_format.json_schema",
            },
            "extraction_passes": [
                {
                    "pass_id": "ENTITY",
                    "objective": "Extract typed concepts and stable aliases grounded in supplied fragments.",
                    "allowed_output": "nodes",
                },
                {
                    "pass_id": "RELATION",
                    "objective": "Extract only allowlisted relation shapes between approved or proposed nodes.",
                    "allowed_output": "relations",
                },
                {
                    "pass_id": "ATTRIBUTE",
                    "objective": "Define each proposed concept/relation and retain schema-safe attributes.",
                    "allowed_output": "node descriptions, aliases and properties; relation predicate definitions",
                },
            ],
            "rules": [
                "Return draft candidates only; never emit executable code, Cypher, commands or publication requests.",
                "Cite only supplied fragment IDs and use only approved existing nodes or candidate-prefixed new nodes.",
                "Do not infer causality from temporal order or co-occurrence.",
                "Mark uncertain claims ambiguous or unresolved.",
                *(additional_rules or []),
            ],
        }
        generated = self.backend.generate_structured(
            agent_id=self.agent_id,
            output_model=response_model,
            payload=payload,
        )
        proposal = KnowledgeExtractionProposal.model_validate(generated)
        return proposal.model_copy(
            update={
                "proposer_mode": "LLM",
                "proposer_id": self.backend.model_id,
                "proposer_version": self.backend.model_version,
            }
        )


class KnowledgeCandidateCriticAgent(ControlledAgent):
    agent_id = "knowledge-candidate-critic-agent"
    agent_version = "0.1.0"

    _CAUSAL_OR_MECHANISM_RELATIONS = {
        KnowledgeRelationTypeV3.PROPAGATES_TO,
        KnowledgeRelationTypeV3.MANIFESTS_AS,
        KnowledgeRelationTypeV3.SUSCEPTIBLE_TO,
    }

    def review(
        self,
        *,
        source_profiles: list[KnowledgeSourceProfile],
        proposal: KnowledgeExtractionProposal,
        approved_existing_nodes: dict[str, KnowledgeNodeTypeV3 | str],
        approved_extraction_methods: list[str],
        minimum_confidence: float = 0.75,
        minimum_independent_causal_sources: int = 2,
    ) -> KnowledgeCandidateValidationReceipt:
        findings: list[KnowledgeCandidateFinding] = []

        def add(code: str, detail: str, subject_id: str | None = None, severity: str = "BLOCKER") -> None:
            findings.append(
                KnowledgeCandidateFinding(
                    severity=severity,
                    code=code,
                    subject_id=subject_id,
                    detail=detail,
                )
            )

        expected_profile_sha = _profile_set_sha256(source_profiles)
        if proposal.source_profile_sha256 != expected_profile_sha:
            add("SOURCE_PROFILE_HASH_MISMATCH", "proposal does not target the reviewed source profile set")
        approved_methods = set(approved_extraction_methods)
        fragments: dict[str, KnowledgeSourceFragment] = {}
        fragment_independence_groups: dict[str, str] = {}
        for profile in source_profiles:
            path = Path(profile.source_path)
            if not path.is_file() or _sha256_file(path) != profile.source_sha256:
                add("SOURCE_FILE_HASH_DRIFT", "source file is missing or changed after profiling", profile.source_id)
            if profile.extraction_method not in approved_methods:
                add(
                    "UNAPPROVED_EXTRACTION_METHOD",
                    f"extraction method is not approved: {profile.extraction_method}",
                    profile.source_id,
                )
            actual_extracted_sha = _sha256_bytes(
                "\n".join(item.text for item in profile.fragments).encode("utf-8")
            )
            if actual_extracted_sha != profile.extracted_text_sha256:
                add("EXTRACTED_TEXT_HASH_DRIFT", "profiled extracted text hash no longer matches", profile.source_id)
            for fragment in profile.fragments:
                actual_fragment_sha = _sha256_bytes(fragment.text.encode("utf-8"))
                if fragment.source_id != profile.source_id or actual_fragment_sha != fragment.text_sha256:
                    add("FRAGMENT_PROVENANCE_INVALID", "fragment source or text hash is invalid", fragment.fragment_id)
                if fragment.fragment_id in fragments:
                    add("DUPLICATE_FRAGMENT_ID", "fragment ID occurs more than once", fragment.fragment_id)
                fragments[fragment.fragment_id] = fragment
                fragment_independence_groups[fragment.fragment_id] = profile.independence_group

        if proposal.executable_content_present:
            add("EXECUTABLE_CONTENT_FORBIDDEN", "knowledge proposals cannot contain executable content")
        if not proposal.nodes and not proposal.relations:
            add("EMPTY_KNOWLEDGE_PROPOSAL", "at least one candidate node or relation is required")

        candidate_types: dict[str, KnowledgeNodeTypeV3] = {}
        seen_node_ids: set[str] = set()
        for node in proposal.nodes:
            if node.node_id in seen_node_ids:
                add("DUPLICATE_CANDIDATE_NODE_ID", "candidate node ID occurs more than once", node.node_id)
            seen_node_ids.add(node.node_id)
            candidate_types[node.node_id] = node.node_type
            self._review_grounding(
                node.source_fragment_ids,
                fragments,
                node.node_id,
                node.confidence,
                node.ambiguous,
                minimum_confidence,
                add,
            )

        existing_types = {
            key: KnowledgeNodeTypeV3(value) for key, value in approved_existing_nodes.items()
        }
        endpoint_types = {**existing_types, **candidate_types}
        seen_edge_ids: set[str] = set()
        for relation in proposal.relations:
            if relation.edge_id in seen_edge_ids:
                add("DUPLICATE_CANDIDATE_EDGE_ID", "candidate edge ID occurs more than once", relation.edge_id)
            seen_edge_ids.add(relation.edge_id)
            self._review_grounding(
                relation.source_fragment_ids,
                fragments,
                relation.edge_id,
                relation.confidence,
                relation.ambiguous,
                minimum_confidence,
                add,
            )
            source_type = endpoint_types.get(relation.source_id)
            target_type = endpoint_types.get(relation.target_id)
            if source_type is None:
                add("UNAPPROVED_OR_UNKNOWN_SOURCE_NODE", "relation source is not approved or proposed", relation.edge_id)
            if target_type is None:
                add("UNAPPROVED_OR_UNKNOWN_TARGET_NODE", "relation target is not approved or proposed", relation.edge_id)
            if source_type is not None and target_type is not None:
                triple = (source_type, relation.relation_type, target_type)
                if triple not in RELATION_ALLOWLIST:
                    add(
                        "V3_RELATION_SHAPE_NOT_ALLOWED",
                        f"{source_type.value}->{relation.relation_type.value}->{target_type.value}",
                        relation.edge_id,
                    )
            if relation.relation_type in self._CAUSAL_OR_MECHANISM_RELATIONS:
                independent_sources = {
                    fragment_independence_groups[fragment_id]
                    for fragment_id in relation.source_fragment_ids
                    if fragment_id in fragment_independence_groups
                }
                if len(independent_sources) < minimum_independent_causal_sources:
                    add(
                        "INSUFFICIENT_INDEPENDENT_CAUSAL_SOURCES",
                        f"requires {minimum_independent_causal_sources} independent sources; got {len(independent_sources)}",
                        relation.edge_id,
                    )

        for unresolved in proposal.unresolved_claims:
            add("UNRESOLVED_KNOWLEDGE_CLAIM", unresolved, severity="WARNING")

        blocked = any(item.severity == "BLOCKER" for item in findings)
        return KnowledgeCandidateValidationReceipt(
            status="BLOCKED" if blocked else "READY_FOR_HUMAN_REVIEW",
            proposal_sha256=_sha256_model(proposal),
            source_profile_sha256=expected_profile_sha,
            reviewed_node_count=len(proposal.nodes),
            reviewed_relation_count=len(proposal.relations),
            findings=findings,
        )

    @staticmethod
    def _review_grounding(
        fragment_ids: list[str],
        fragments: dict[str, KnowledgeSourceFragment],
        subject_id: str,
        confidence: float,
        ambiguous: bool,
        minimum_confidence: float,
        add: Any,
    ) -> None:
        unknown = sorted(set(fragment_ids) - set(fragments))
        if unknown:
            add("UNKNOWN_SOURCE_FRAGMENT", f"candidate cites unknown fragments: {unknown}", subject_id)
        if confidence < minimum_confidence:
            add("LOW_CONFIDENCE", f"confidence is below {minimum_confidence:g}", subject_id)
        if ambiguous:
            add("AMBIGUOUS_CANDIDATE", "candidate is explicitly marked ambiguous", subject_id)


class VllmKnowledgeExtractionBackend:
    """Strict OpenAI-compatible structured backend for a local vLLM service."""

    model_version = "openai-compatible-json-schema-v1"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        model_id: str = "auto",
        timeout_s: float = 120.0,
        max_tokens: int = 8192,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("vLLM base_url must be an absolute HTTP(S) URL")
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.response_trace: list[dict[str, Any]] = []

    def _json_request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib_request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - defensive error reporting
                detail = ""
            raise RuntimeError(
                f"vLLM structured extraction unavailable: HTTP {exc.code}: {detail[:1000]}"
            ) from exc
        except (urllib_error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"vLLM structured extraction unavailable: {exc}") from exc

    def _resolve_model(self) -> str:
        if self.model_id != "auto":
            return self.model_id
        payload = self._json_request("/v1/models")
        identifiers = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
        if not identifiers:
            raise RuntimeError("vLLM /v1/models returned no model identifiers")
        self.model_id = identifiers[0]
        return self.model_id

    def discover_model(self) -> str:
        """Verify the endpoint and return its selected model identifier."""
        return self._resolve_model()

    def generate_structured(
        self,
        *,
        agent_id: str,
        output_model: type[T],
        payload: dict[str, Any],
    ) -> T:
        schema = output_model.model_json_schema()
        response = self._json_request("/v1/chat/completions", {
            "model": self._resolve_model(),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an offline knowledge extraction worker. Follow the supplied schema and "
                        "provenance rules. Return only the requested JSON object; never publish knowledge."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        })
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("vLLM response does not contain structured message content") from exc
        self.response_trace.append({
            "trace_index": len(self.response_trace),
            "agent_id": agent_id,
            "model_id": self.model_id,
            "output_model": output_model.__name__,
            "prompt_fragment_ids": list(
                payload.get("governance_context", {}).get("prompt_fragment_ids", [])
            ),
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage"),
            "response_id": response.get("id"),
            "raw_content": content,
        })
        if isinstance(content, dict):
            return output_model.model_validate(content)
        text = str(content).strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[7:-3].strip()
        return output_model.model_validate_json(text)

class ControlledKnowledgeCurationWorkflow:
    workflow_id = "controlled-knowledge-curation-workflow"
    workflow_version = "0.1.0"

    def __init__(
        self,
        *,
        profiler: KnowledgeSourceProfilerAgent | None = None,
        curator: KnowledgeCurationAgent | None = None,
        critic: KnowledgeCandidateCriticAgent | None = None,
    ) -> None:
        self.profiler = profiler or KnowledgeSourceProfilerAgent()
        self.curator = curator or KnowledgeCurationAgent()
        self.critic = critic or KnowledgeCandidateCriticAgent()

    def run(
        self,
        *,
        sources: list[dict[str, Any]],
        approved_existing_nodes: dict[str, KnowledgeNodeTypeV3 | str],
        approved_extraction_methods: list[str],
        draft_proposal: dict[str, Any] | None = None,
        minimum_confidence: float = 0.75,
        minimum_independent_causal_sources: int = 2,
    ) -> ControlledKnowledgeCurationResult:
        profiles = [self.profiler.profile(**source) for source in sources]
        proposal = self.curator.propose(
            source_profiles=profiles,
            approved_existing_nodes=approved_existing_nodes,
            draft_proposal=draft_proposal,
        )
        validation = self.critic.review(
            source_profiles=profiles,
            proposal=proposal,
            approved_existing_nodes=approved_existing_nodes,
            approved_extraction_methods=approved_extraction_methods,
            minimum_confidence=minimum_confidence,
            minimum_independent_causal_sources=minimum_independent_causal_sources,
        )
        profile_sha = _profile_set_sha256(profiles)
        return ControlledKnowledgeCurationResult(
            source_profiles=profiles,
            proposal=proposal,
            validation=validation,
            trace=[
                KnowledgeCurationTraceStep(
                    sequence=1,
                    agent_id=self.profiler.agent_id,
                    outcome="PROFILED",
                    artifact_sha256=profile_sha,
                ),
                KnowledgeCurationTraceStep(
                    sequence=2,
                    agent_id=self.curator.agent_id,
                    outcome="PROPOSED",
                    artifact_sha256=_sha256_model(proposal),
                ),
                KnowledgeCurationTraceStep(
                    sequence=3,
                    agent_id=self.critic.agent_id,
                    outcome=validation.status,
                    artifact_sha256=_sha256_model(validation),
                ),
            ],
        )


__all__ = [
    "ControlledKnowledgeCurationResult",
    "ControlledKnowledgeCurationWorkflow",
    "KnowledgeCandidateCriticAgent",
    "KnowledgeCandidateFinding",
    "KnowledgeCandidateValidationReceipt",
    "KnowledgeCurationAgent",
    "KnowledgeCurationTraceStep",
    "KnowledgeExtractionBackend",
    "KnowledgeExtractionFragmentProposal",
    "KnowledgeExtractionProposal",
    "KnowledgeNodeCandidate",
    "KnowledgeRelationCandidate",
    "KnowledgeSourceFragment",
    "KnowledgeSourceProfile",
    "KnowledgeSourceProfilerAgent",
    "knowledge_source_profile_identity_payload",
    "knowledge_source_profile_sha256",
    "knowledge_source_profile_set_sha256",
    "SourceFragmentInput",
    "VllmKnowledgeExtractionBackend",
]
