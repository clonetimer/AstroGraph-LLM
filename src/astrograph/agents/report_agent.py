from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astrograph.agents.base import AgentResult, ControlledAgent
from astrograph.audit.store import sha256_json
from astrograph.diagnosis.report import render_markdown
from astrograph.schemas import AnomalyEvidence, DiagnosisResult


class ReportPayload(BaseModel):
    """Locked input for report generation.

    ReportAgent may transform this payload into readable text, but must not
    modify diagnosis status, fault ids, scores, evidence, counter-evidence, or
    version fields.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    request_id: str
    anomalies: list[AnomalyEvidence]
    diagnosis: DiagnosisResult
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def payload_hash(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


class ReportAgent(ControlledAgent):
    agent_id = "report-agent"
    agent_version = "0.4.4"

    def render(self, payload: ReportPayload) -> AgentResult:
        report = render_markdown(
            payload.case_id,
            payload.request_id,
            list(payload.anomalies),
            payload.diagnosis,
        )
        return AgentResult(
            agent_id=self.agent_id,
            status="SUCCEEDED",
            output={
                "report_markdown": report,
                "locked_payload_hash": payload.payload_hash,
                "report_agent_version": self.agent_version,
                "claim_boundary": "Report Agent renders locked deterministic diagnosis payload only.",
            },
        )
