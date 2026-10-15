"""R36 bounded-window online-compatible diagnosis and operator packet loop.

This API consumes complete semantic frames/windows.  It does not implement
stream watermarks, backpressure, late-event handling or incremental state.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from astrograph.diagnosis.generic import (
    AssetTopologyContext,
    DataQualityContext,
    GenericDiagnosisPipeline,
    GenericDiagnosisResult,
    RootCauseEvidenceContext,
)
from astrograph.physics.contracts import DynamicsContract
from astrograph.semantics import SemanticTelemetryFrame
from astrograph.state_estimation import (
    StateEstimateResult,
    StateEstimator,
)


class _Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


def _sha(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


class LiveDiagnosticWindow(_Strict):
    case_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)

    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def ordered(self) -> "LiveDiagnosticWindow":
        if self.end_s <= self.start_s:
            raise ValueError(
                "live diagnostic window end must "
                "be greater than start"
            )
        return self


class OperatorDiagnosticPacket(_Strict):

    schema_version: Literal[
        "astrograph-live-diagnostic-operator-packet.v1"
    ] = "astrograph-live-diagnostic-operator-packet.v1"

    case_id: str
    window_id: str

    start_s: float
    end_s: float

    semantic_data_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    state_estimate_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    diagnosis_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    state_estimate: StateEstimateResult
    diagnosis: GenericDiagnosisResult

    labels_used: Literal[False] = False
    truth_used: Literal[False] = False

    control_action_authorized: Literal[False] = False

    claim_boundary: str = (
        "Bounded-window online-compatible technical diagnosis packet for "
        "operator review. It is not a streaming-runtime receipt. "
        "It cannot command spacecraft control, approve release, "
        "or convert affected-component evidence into causal "
        "root-cause proof."
    )

    @computed_field
    @property
    def packet_sha256(self) -> str:
        return _sha(
            self.model_dump(
                mode="json",
                exclude={"packet_sha256"},
            )
        )


class AppendOnlyOperatorPacketStore:

    def __init__(
        self,
        path: str | Path,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def packets(
        self,
    ) -> list[OperatorDiagnosticPacket]:

        if not self.path.is_file():
            return []

        return [
            OperatorDiagnosticPacket.model_validate_json(
                line
            )
            for line in self.path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

    def append(
        self,
        packet: OperatorDiagnosticPacket,
    ) -> None:

        existing = self.packets()

        for previous in existing:

            if (
                previous.packet_sha256
                == packet.packet_sha256
            ):
                return

            if previous.case_id == packet.case_id:

                overlap = (
                    max(
                        previous.start_s,
                        packet.start_s,
                    )
                    <
                    min(
                        previous.end_s,
                        packet.end_s,
                    )
                )

                if overlap:
                    raise ValueError(
                        "OVERLAPPING_LIVE_DIAGNOSTIC_WINDOW"
                    )

                if (
                    previous.window_id
                    == packet.window_id
                ):
                    raise ValueError(
                        "LIVE_DIAGNOSTIC_WINDOW_ID_REUSED"
                    )

        # Important:
        # packet_sha256 is a computed field.
        # Do NOT persist it into a strict model JSON document.
        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:

            handle.write(
                json.dumps(
                    packet.model_dump(
                        mode="json",
                        exclude={"packet_sha256"},
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


class LiveDiagnosticLoop:

    def __init__(
        self,
        *,
        estimator: StateEstimator,
        dynamics_contract: DynamicsContract,
        pipeline: GenericDiagnosisPipeline | None = None,
        packet_store:
            AppendOnlyOperatorPacketStore | None = None,
    ) -> None:

        self.estimator = estimator
        self.dynamics_contract = dynamics_contract
        self.pipeline = (
            pipeline
            or GenericDiagnosisPipeline()
        )
        self.packet_store = packet_store

    def run(
        self,
        *,
        frame: SemanticTelemetryFrame,
        topology: AssetTopologyContext,
        data_quality: DataQualityContext,
        window: LiveDiagnosticWindow,
        root_cause_evidence:
            RootCauseEvidenceContext | None = None,
    ) -> OperatorDiagnosticPacket:

        if frame.labels_used or frame.truth_used:
            raise ValueError(
                "live diagnostic loop accepts "
                "truth-free telemetry only"
            )

        if (
            topology.component_archetype_id
            !=
            self.dynamics_contract.component_archetype_id
        ):
            raise ValueError(
                "topology archetype does not match "
                "dynamics contract"
            )

        if not frame.timestamps_s:
            raise ValueError(
                "empty telemetry frame"
            )

        if (
            frame.timestamps_s[0] < window.start_s
            or
            frame.timestamps_s[-1] > window.end_s
        ):
            raise ValueError(
                "telemetry timestamps escape declared "
                "live diagnostic window"
            )

        frame_archetypes = {
            signal.component_archetype_id
            for signal in frame.signals
        }

        if frame_archetypes != {
            self.dynamics_contract.component_archetype_id
        }:
            raise ValueError(
                "telemetry frame archetype does not match "
                "dynamics contract"
            )

        state_estimate = self.estimator.estimate(
            frame,
            self.dynamics_contract,
        )

        diagnosis = self.pipeline.diagnose(
            state_estimate=state_estimate,
            topology=topology,
            data_quality=data_quality,
            root_cause_evidence=root_cause_evidence,
        )

        packet = OperatorDiagnosticPacket(
            case_id=window.case_id,
            window_id=window.window_id,
            start_s=window.start_s,
            end_s=window.end_s,

            semantic_data_sha256=(
                frame.semantic_data_sha256
            ),

            state_estimate_sha256=_sha(
                state_estimate.model_dump(
                    mode="json"
                )
            ),

            diagnosis_sha256=_sha(
                diagnosis.model_dump(
                    mode="json"
                )
            ),

            state_estimate=state_estimate,
            diagnosis=diagnosis,
        )

        if self.packet_store is not None:
            self.packet_store.append(packet)

        return packet


__all__ = [
    "LiveDiagnosticWindow",
    "OperatorDiagnosticPacket",
    "AppendOnlyOperatorPacketStore",
    "LiveDiagnosticLoop",
]
