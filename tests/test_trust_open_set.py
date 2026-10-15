from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astrograph.diagnosis.sat_sim_catalog import _battery_capacity_loss_observed
from astrograph.diagnosis.sat_sim_open_set import (
    OpenSetDiagnosisStatus,
    _select_knownness_threshold,
)
from astrograph.errors import UnknownFaultError
from astrograph.knowledge.schema import (
    AssertionStatus,
    DEFAULT_KNOWLEDGE_SCHEMA,
    KnowledgeNodeType,
    KnowledgeRelationType,
    validate_node_type,
    validate_relation_type,
)


def test_separated_membership_threshold_is_inside_gap_not_on_known_endpoint() -> None:
    probabilities = np.asarray([0.99984, 0.99994, 0.000058, 0.000059], dtype=float)
    flags = np.asarray([1, 1, 0, 0], dtype=int)
    threshold, evidence = _select_knownness_threshold(probabilities, flags)
    assert 0.000059 < threshold < 0.99984
    assert threshold < 0.99
    assert evidence["threshold_strategy"] == "LOGIT_MIDPOINT_OF_CALIBRATION_SEPARATION_GAP"


def test_overlapping_membership_threshold_uses_recall_fallback() -> None:
    probabilities = np.asarray([0.9, 0.7, 0.8, 0.4], dtype=float)
    flags = np.asarray([1, 1, 0, 0], dtype=int)
    threshold, evidence = _select_knownness_threshold(probabilities, flags)
    assert threshold in set(probabilities.tolist())
    assert evidence["threshold_strategy"] == "CONSTRAINED_BALANCED_RECALL_OVERLAP_FALLBACK"


def test_knowledge_schema_is_explicit_and_rejects_free_form_types() -> None:
    assert validate_node_type(KnowledgeNodeType.FAULT_MODE) == KnowledgeNodeType.FAULT_MODE
    assert validate_relation_type(KnowledgeRelationType.SUPPORTS) == KnowledgeRelationType.SUPPORTS
    with pytest.raises(ValueError):
        validate_node_type("WhateverNode")
    with pytest.raises(ValueError):
        validate_relation_type("CAUSES_MAGICALLY")


def test_assertion_status_distinguishes_found_inferred_and_ambiguous() -> None:
    assert {item.value for item in AssertionStatus} >= {"EXTRACTED", "DERIVED", "INFERRED", "AMBIGUOUS"}


def test_structured_unknown_fault_error_exposes_correct_failure_action() -> None:
    error = UnknownFaultError("Catalog membership rejected.", details={"case_id": "opaque"})
    payload = error.to_dict()
    assert payload["error_code"] == "ASTROGRAPH_UNKNOWN_FAULT"
    assert payload["recoverable"] is True
    assert payload["requires_human_review"] is True
    assert payload["recommended_action"]
    assert "confidence" not in payload["claim_boundary"].lower()


def test_battery_capacity_resolver_requires_physical_ratio_transition() -> None:
    normal = pd.DataFrame({"eps.battery_capacity_ratio": [1.0, 1.0, 1.0]})
    loss = pd.DataFrame({"eps.battery_capacity_ratio": [1.0, 0.72, 0.72]})
    assert _battery_capacity_loss_observed(normal) is False
    assert _battery_capacity_loss_observed(loss) is True

from astrograph.integrations.rw_jam_promotion import (
    KnowledgePromotionEvidence,
    Neo4jCausalReasoningService,
    Neo4jEvidencePathService,
)
from astrograph.schemas import KnowledgeEdge


class _OpenSetNeo4jRepo:
    backend_id = "neo4j"
    knowledge_version = "kg-schema-v2"

    def health_check(self):
        return {"status": "PASS", "release_count": 1}

    def retrieve(self, anomaly_ids):
        return [KnowledgeEdge(
            fault_id="F-1", fault_name="fault", anomaly_id=anomaly_ids[0], anomaly_name="anomaly",
            weight=0.6, edge_role="direct", evidence_level="E1", evidence_source="basilisk",
            source_ref="RUN-1", review_status="approved", provisional=False, knowledge_version=self.knowledge_version,
        )]


def test_neo4j_service_reports_evidence_paths_not_causal_effects() -> None:
    assert Neo4jCausalReasoningService is Neo4jEvidencePathService
    result = Neo4jEvidencePathService(_OpenSetNeo4jRepo()).reason(["A-1"])
    dumped = result.model_dump(mode="json")
    assert "evidence_paths" in dumped
    assert "causal_paths" not in dumped
    path = dumped["evidence_paths"][0]
    assert path["aggregate_support_score"] == 0.6
    assert path["score_semantics"] == "ASSOCIATIVE_EDGE_WEIGHT_SUM_NOT_CAUSAL_EFFECT"


def test_legacy_causal_paths_input_is_read_only_compatibility_alias() -> None:
    value = KnowledgePromotionEvidence.model_validate({
        "backend": "neo4j", "knowledge_version": "kg-v2", "health_status": "PASS",
        "causal_paths": [{"fault_id": "F-1"}],
    })
    assert value.evidence_paths == [{"fault_id": "F-1"}]
    assert value.model_dump(mode="json").get("causal_paths") is None
