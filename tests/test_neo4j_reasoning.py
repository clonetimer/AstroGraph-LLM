from astrograph.integrations import Neo4jCausalReasoningService
from astrograph.schemas import KnowledgeEdge


class FakeNeo4jRepository:
    backend_id = "neo4j"
    knowledge_version = "kg-v1"

    def health_check(self):
        return {"status": "PASS", "release_count": 1, "edge_count": 2}

    def retrieve(self, anomaly_ids):
        return [KnowledgeEdge(
            fault_id="ADCS_RW_JAM", fault_name="RW jam", anomaly_id=anomaly_ids[0], anomaly_name="wheel speed divergence",
            weight=0.9, edge_role="direct", evidence_level="E1", evidence_source="basilisk-a",
            source_ref="EXP-001", review_status="approved", provisional=False, knowledge_version="kg-v1",
        )]


class FakeCsvRepository(FakeNeo4jRepository):
    backend_id = "csv"


def test_formal_neo4j_reasoning_records_query_evidence():
    result = Neo4jCausalReasoningService(FakeNeo4jRepository()).reason(["ANOM-RW-SPEED"])
    assert result.backend == "neo4j"
    assert result.read_only_query_executed is True
    assert result.matched_fault_ids == ["ADCS_RW_JAM"]
    assert result.formal_release_accepted is True


def test_projection_repository_cannot_masquerade_as_formal_neo4j():
    try:
        Neo4jCausalReasoningService(FakeCsvRepository()).reason(["ANOM-RW-SPEED"])
    except ValueError as exc:
        assert "Neo4j" in str(exc)
    else:
        raise AssertionError("CSV repository must be rejected")
