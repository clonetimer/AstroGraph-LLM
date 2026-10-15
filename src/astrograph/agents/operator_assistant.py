from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from astrograph.agents.base import ControlledAgent
from astrograph.audit.store import sha256_json


T = TypeVar("T", bound=BaseModel)


class OperatorExplanationBackend(Protocol):
    model_id: str
    model_version: str

    def generate_structured(
        self,
        *,
        agent_id: str,
        output_model: type[T],
        payload: dict[str, Any],
    ) -> T: ...


class LockedExplanationFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)


class RetrievedExplanationSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snippet: str = Field(min_length=1)
    knowledge_version: str = Field(min_length=1)


class LockedExplanationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-locked-explanation-payload.v1"
    diagnosis_id: str = Field(min_length=1)
    diagnosis_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    facts: list[LockedExplanationFact] = Field(min_length=1)
    retrieved_sources: list[RetrievedExplanationSource] = Field(default_factory=list)
    allowed_caveat_codes: list[str] = Field(default_factory=list)

    @property
    def envelope_hash(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


class ExplanationSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-explanation-selection.v1"
    locked_envelope_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    ordered_fact_ids: list[str] = Field(min_length=1)
    context_source_ids: list[str] = Field(default_factory=list)
    caveat_codes: list[str] = Field(default_factory=list)
    emphasis: str = "EVIDENCE_FIRST"


class CitationBoundExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-citation-bound-explanation.v1"
    diagnosis_id: str
    diagnosis_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locked_envelope_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rendered_markdown: str
    selected_fact_ids: list[str]
    cited_source_ids: list[str]
    cited_source_sha256: dict[str, str]
    model_id: str
    model_version: str
    claim_boundary: str = (
        "The model selects locked facts and reviewed retrieval sources only. Text is rendered "
        "deterministically; this agent cannot change diagnosis IDs, scores, evidence, or fusion results."
    )


class OperatorAssistantAgent(ControlledAgent):
    """Citation-bound selector over locked diagnosis facts and retrieved sources."""

    agent_id = "operator-assistant-agent"
    agent_version = "0.5.0"

    def __init__(
        self,
        *args: Any,
        backend: OperatorExplanationBackend | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.backend = backend

    def explain(
        self,
        payload: LockedExplanationPayload,
        *,
        draft_selection: dict[str, Any] | None = None,
    ) -> CitationBoundExplanation:
        if self.backend is None:
            if draft_selection is None:
                raise ValueError("deterministic explanation mode requires draft_selection")
            selection = ExplanationSelection.model_validate(draft_selection)
            model_id = self.agent_id
            model_version = self.agent_version
        else:
            selection = self.backend.generate_structured(
                agent_id=self.agent_id,
                output_model=ExplanationSelection,
                payload={
                    "locked_envelope_hash": payload.envelope_hash,
                    "facts": [item.model_dump(mode="json") for item in payload.facts],
                    "retrieved_sources": [
                        item.model_dump(mode="json") for item in payload.retrieved_sources
                    ],
                    "allowed_caveat_codes": payload.allowed_caveat_codes,
                    "output_protocol": ExplanationSelection.model_json_schema(),
                    "rules": [
                        "Select IDs only; do not rewrite facts or retrieved text.",
                        "Do not add diagnosis, score, evidence, citation, query, or remediation fields.",
                    ],
                },
            )
            selection = ExplanationSelection.model_validate(selection)
            model_id = self.backend.model_id
            model_version = self.backend.model_version

        if selection.locked_envelope_hash != payload.envelope_hash:
            raise ValueError("explanation selection does not match the locked payload")
        if len(selection.ordered_fact_ids) != len(set(selection.ordered_fact_ids)):
            raise ValueError("explanation selection contains duplicate fact IDs")
        if len(selection.context_source_ids) != len(set(selection.context_source_ids)):
            raise ValueError("explanation selection contains duplicate source IDs")
        facts = {item.fact_id: item for item in payload.facts}
        sources = {item.source_id: item for item in payload.retrieved_sources}
        unknown_facts = set(selection.ordered_fact_ids) - set(facts)
        unknown_sources = set(selection.context_source_ids) - set(sources)
        unknown_caveats = set(selection.caveat_codes) - set(payload.allowed_caveat_codes)
        if unknown_facts:
            raise ValueError(f"explanation selected unknown locked facts: {sorted(unknown_facts)}")
        if unknown_sources:
            raise ValueError(f"explanation selected unknown retrieval sources: {sorted(unknown_sources)}")
        if unknown_caveats:
            raise ValueError(f"explanation selected unknown caveat codes: {sorted(unknown_caveats)}")

        cited = set(selection.context_source_ids)
        lines = [f"# 诊断解释：{payload.diagnosis_id}", "", "## 已锁定事实", ""]
        for fact_id in selection.ordered_fact_ids:
            fact = facts[fact_id]
            unknown_fact_sources = set(fact.source_ids) - set(sources)
            if unknown_fact_sources:
                raise ValueError(
                    f"locked fact {fact_id} references unknown sources: {sorted(unknown_fact_sources)}"
                )
            cited.update(fact.source_ids)
            suffix = "" if not fact.source_ids else " " + " ".join(f"[{item}]" for item in fact.source_ids)
            lines.append(f"- {fact.text}{suffix}")
        if selection.context_source_ids:
            lines.extend(["", "## 检索上下文", ""])
            for source_id in selection.context_source_ids:
                source = sources[source_id]
                lines.append(f"- [{source_id}] {source.snippet}")
        if selection.caveat_codes:
            lines.extend(["", "## 限制", ""])
            lines.extend(f"- {item}" for item in selection.caveat_codes)
        lines.extend(["", "## 引用", ""])
        for source_id in sorted(cited):
            source = sources[source_id]
            lines.append(
                f"- [{source_id}] {source.title}; {source.uri}; sha256={source.sha256}; "
                f"knowledge_version={source.knowledge_version}"
            )
        return CitationBoundExplanation(
            diagnosis_id=payload.diagnosis_id,
            diagnosis_payload_sha256=payload.diagnosis_payload_sha256,
            locked_envelope_hash=payload.envelope_hash,
            rendered_markdown="\n".join(lines),
            selected_fact_ids=selection.ordered_fact_ids,
            cited_source_ids=sorted(cited),
            cited_source_sha256={item: sources[item].sha256 for item in sorted(cited)},
            model_id=model_id,
            model_version=model_version,
        )


__all__ = [
    "CitationBoundExplanation",
    "ExplanationSelection",
    "LockedExplanationFact",
    "LockedExplanationPayload",
    "OperatorAssistantAgent",
    "OperatorExplanationBackend",
    "RetrievedExplanationSource",
]
