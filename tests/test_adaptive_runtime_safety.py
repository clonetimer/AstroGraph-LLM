from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from astrograph.diagnosis.case_catalog import FaultCaseCatalog
from astrograph.knowledge.neo4j_projection import Neo4jGraphWriter
from astrograph.tools.online import (
    LEGACY_WHOLE_SPACECRAFT_EVIDENCE_BUILDER_ID,
    _OnlineRuntimeEvidenceBuildTool,
)


def test_runtime_evidence_builder_requires_explicit_selection() -> None:
    tool = _OnlineRuntimeEvidenceBuildTool()
    with pytest.raises(ValueError, match="must explicitly select"):
        tool({"protocol_root": "/unused"})


def test_legacy_runtime_builder_rejects_configured_package() -> None:
    tool = _OnlineRuntimeEvidenceBuildTool()
    with pytest.raises(ValueError, match="must not receive evidence_package_path"):
        tool(
            {
                "runtime_evidence_builder_id": LEGACY_WHOLE_SPACECRAFT_EVIDENCE_BUILDER_ID,
                "evidence_package_path": "/unused/package.json",
                "protocol_root": "/unused",
            }
        )


def test_packaged_fault_case_catalog_is_complete_and_versioned() -> None:
    catalog = FaultCaseCatalog.packaged_default()
    assert catalog.schema_version == "astrograph-fault-case-catalog.v1"
    assert catalog.version == "1.1.0"
    assert len(catalog.mappings) == 26
    assert set(catalog.mappings) == {
        "ADCS_CONTROL_LOOP_AUTHORITY_LOSS",
        "ADCS_GYRO_BIAS_DRIFT",
        "ADCS_IMU_AXIS_DROPOUT",
        "ADCS_IMU_GYRO_NOISE_AMPLIFICATION",
        "ADCS_IMU_OUTPUT_ZERO",
        "ADCS_RW_FRICTION_INCREASE",
        "ADCS_RW_JAM",
        "ADCS_RW_SPEED_SENSOR_BIAS",
        "ADCS_RW_TORQUE_AUTHORITY_LOSS",
        "ADCS_STAR_SOLUTION_BIAS",
        "ADCS_STAR_TRACKER_DROPOUT",
        "ADCS_STAR_TRACKER_STUCK_AT_LAST",
        "ADCS_SUN_SENSOR_CELL_DEGRADATION",
        "ADCS_SUN_SENSOR_SATURATION",
        "ADCS_SUN_VECTOR_UNAVAILABLE",
        "COMM_DOWNLINK_SERVICE_LOSS",
        "COMM_TRANSMITTER_OUTPUT_POWER_LOSS",
        "DATA_STORAGE_WRITE_FAILURE",
        "EPS_BATTERY_CAPACITY_LOSS",
        "EPS_BATTERY_INTERNAL_RESISTANCE_INCREASE",
        "EPS_SOLAR_ARRAY_EFFICIENCY_LOSS",
        "EPS_SOLAR_ARRAY_STRING_FAILURE",
        "THERMAL_HEATER_STUCK_ON",
        "THERMAL_RADIATOR_REJECTION_LOSS",
        "THERMAL_TEMPERATURE_SENSOR_BIAS",
        "THERMAL_TEMPERATURE_SENSOR_DRIFT",
    }
    assert catalog.case_id_for_fault("ADCS_RW_JAM") == "reaction_wheel_jam"
    with pytest.raises(ValueError, match="missing from case catalog"):
        catalog.case_id_for_fault("UNREGISTERED_FAULT")


def test_neo4j_writer_requires_password_before_driver_creation(monkeypatch) -> None:
    called = False

    class FakeGraphDatabase:
        @staticmethod
        def driver(*args, **kwargs):
            nonlocal called
            called = True
            return object()

    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=FakeGraphDatabase))
    monkeypatch.delenv("ASTROGRAPH_NEO4J_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="password is required"):
        Neo4jGraphWriter()
    assert called is False


def test_neo4j_writer_accepts_explicit_credentials(monkeypatch) -> None:
    captured = {}

    class FakeGraphDatabase:
        @staticmethod
        def driver(uri, *, auth):
            captured.update({"uri": uri, "auth": auth})
            return object()

    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=FakeGraphDatabase))
    writer = Neo4jGraphWriter(uri="neo4j://example:7687", user="reader", password="secret")
    assert writer._driver is not None
    assert captured == {"uri": "neo4j://example:7687", "auth": ("reader", "secret")}
