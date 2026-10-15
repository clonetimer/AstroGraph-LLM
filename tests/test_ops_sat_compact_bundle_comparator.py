from __future__ import annotations

from scripts.experiments.run_ops_sat_compact_bundle_comparator import (
    CompactBundleRevision,
    _candidate_states,
    _citations,
    summarize_attributable_outcomes,
)


def test_compact_revision_has_only_host_mappable_fields() -> None:
    revision = CompactBundleRevision(decision="ABSTAIN", citation_scope="BOTH")
    assert revision.model_dump() == {"decision": "ABSTAIN", "citation_scope": "BOTH"}
    assert _candidate_states(revision.decision) == ["ANOMALOUS", "NOMINAL"]
    assert _citations(revision.citation_scope, "BASE", "BUNDLE") == ["BASE", "BUNDLE"]


def test_compact_decisions_map_to_registered_binary_candidates() -> None:
    assert _candidate_states("NOMINAL") == ["NOMINAL"]
    assert _candidate_states("ANOMALOUS") == ["ANOMALOUS"]
    assert _candidate_states("KEEP_BOTH") == ["ANOMALOUS", "NOMINAL"]


def test_attributable_outcome_audit_distinguishes_harm_from_change() -> None:
    rows = [{
        "group_id": "healthy-1",
        "offline_truth": "NOMINAL",
        "arm": {
            "active_observation_attributable": True,
            "validated_candidate_states": ["ANOMALOUS"],
            "revision": {"abstain_recommended": False},
        },
    }]
    audit = summarize_attributable_outcomes(rows)
    assert audit["attributable_case_count"] == 1
    assert audit["attributable_correct_count"] == 0
    assert audit["attributable_incorrect_count"] == 1
    assert audit["verdict"] == "ATTRIBUTABLE_HARM_OBSERVED_NO_BENEFIT"
