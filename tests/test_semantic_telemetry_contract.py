from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from astrograph.semantics import (
    ChannelMap,
    ChannelMapEntry,
    ChannelMapper,
    POWER,
    QuantityKind,
    SemanticMappingError,
    SemanticSchema,
    TimeAxisMap,
    UnmappedFieldPolicy,
)


def _schema() -> SemanticSchema:
    return SemanticSchema(
        semantic_schema_id="astrograph.core.power.v1",
        version="1.0.0",
        quantity_kinds=[
            QuantityKind(
                quantity_kind_id="electrical.active_power",
                dimension=POWER,
                canonical_unit="W",
            )
        ],
    )


def _mapping(*, source_field: str, source_unit: str, time_unit: str = "s") -> ChannelMap:
    return ChannelMap(
        mapping_id=f"map-{source_field}",
        version="1.0.0",
        source_adapter_id="opaque-adapter-ref",
        source_schema_sha256="a" * 64,
        semantic_schema_id="astrograph.core.power.v1",
        time_axis=TimeAxisMap(source_field="clock", source_unit=time_unit),
        entries=[
            ChannelMapEntry(
                source_field=source_field,
                semantic_channel_id="asset.main_bus.active_power",
                asset_instance_id="asset-under-test",
                component_archetype_id="electrical.power_port",
                port_role_id="power.output",
                quantity_kind_id="electrical.active_power",
                source_unit=source_unit,
            )
        ],
    )


def _signal_values(result) -> list[float | None]:
    return result.frame.signals[0].values


def test_source_field_rename_does_not_change_semantic_frame() -> None:
    first = ChannelMapper(
        semantic_schema=_schema(),
        channel_map=_mapping(source_field="vendor_power", source_unit="W"),
    ).map_frame(pd.DataFrame({"clock": [0.0, 1.0, 2.0], "vendor_power": [10.0, 20.0, 30.0]}))
    renamed = ChannelMapper(
        semantic_schema=_schema(),
        channel_map=_mapping(source_field="renamed_signal", source_unit="W"),
    ).map_frame(pd.DataFrame({"clock": [0.0, 1.0, 2.0], "renamed_signal": [10.0, 20.0, 30.0]}))

    assert first.frame == renamed.frame
    assert first.frame.semantic_data_sha256 == renamed.frame.semantic_data_sha256
    assert first.receipt.source_data_sha256 != renamed.receipt.source_data_sha256
    serialized = first.frame.model_dump_json()
    assert "vendor_power" not in serialized
    assert "opaque-adapter-ref" not in serialized


def test_equivalent_units_and_time_units_produce_identical_semantics() -> None:
    watts = ChannelMapper(
        semantic_schema=_schema(),
        channel_map=_mapping(source_field="power_w", source_unit="W", time_unit="s"),
    ).map_frame(pd.DataFrame({"clock": [0.0, 1.0, 2.0], "power_w": [1000.0, 1500.0, 2000.0]}))
    kilowatts = ChannelMapper(
        semantic_schema=_schema(),
        channel_map=_mapping(source_field="power_kw", source_unit="kW", time_unit="ms"),
    ).map_frame(pd.DataFrame({"clock": [0.0, 1000.0, 2000.0], "power_kw": [1.0, 1.5, 2.0]}))

    assert watts.frame == kilowatts.frame
    assert _signal_values(kilowatts) == [1000.0, 1500.0, 2000.0]
    assert kilowatts.frame.timestamps_s == [0.0, 1.0, 2.0]


def test_sampling_profile_and_missing_mask_are_explicit() -> None:
    result = ChannelMapper(
        semantic_schema=_schema(),
        channel_map=_mapping(source_field="power", source_unit="W"),
    ).map_frame(pd.DataFrame({"clock": [0.0, 1.0, 2.5], "power": [10.0, None, 30.0]}))

    assert result.frame.sampling.irregular is True
    assert result.frame.sampling.nominal_period_s == 1.25
    assert result.frame.sampling.missing_value_count == 1
    assert result.frame.signals[0].missing_mask == [False, True, False]


@pytest.mark.parametrize("field", ["label.fault", "truth.mode", "event.fault", "object.case"])
def test_prohibited_fields_fail_even_when_unmapped(field: str) -> None:
    mapping = _mapping(source_field="power", source_unit="W").model_copy(
        update={"unmapped_field_policy": UnmappedFieldPolicy.IGNORE_CLEAN}
    )
    source = pd.DataFrame({"clock": [0.0, 1.0], "power": [1.0, 2.0], field: [0, 1]})

    with pytest.raises(SemanticMappingError) as captured:
        ChannelMapper(semantic_schema=_schema(), channel_map=mapping).map_frame(source)
    assert captured.value.code == "PROHIBITED_SOURCE_FIELD"
    assert captured.value.fields == [field]


def test_clean_unmapped_fields_require_explicit_policy() -> None:
    source = pd.DataFrame({"clock": [0.0, 1.0], "power": [1.0, 2.0], "quality_flag": [1, 1]})
    with pytest.raises(SemanticMappingError, match="UNMAPPED_SOURCE_FIELD"):
        ChannelMapper(
            semantic_schema=_schema(),
            channel_map=_mapping(source_field="power", source_unit="W"),
        ).map_frame(source)

    relaxed = _mapping(source_field="power", source_unit="W").model_copy(
        update={"unmapped_field_policy": UnmappedFieldPolicy.IGNORE_CLEAN}
    )
    result = ChannelMapper(semantic_schema=_schema(), channel_map=relaxed).map_frame(source)
    assert result.receipt.ignored_clean_fields == ["quality_flag"]
    assert "quality_flag" not in result.frame.model_dump_json()


def test_unit_dimension_mismatch_fails_before_mapping() -> None:
    with pytest.raises(ValueError, match="dimension"):
        ChannelMapper(
            semantic_schema=_schema(),
            channel_map=_mapping(source_field="power", source_unit="s"),
        )


def test_mapping_rejects_duplicate_semantic_channels() -> None:
    payload = _mapping(source_field="power_a", source_unit="W").model_dump(mode="json")
    duplicate = dict(payload["entries"][0])
    duplicate["source_field"] = "power_b"
    payload["entries"].append(duplicate)
    with pytest.raises(ValidationError, match="semantic_channel_id"):
        ChannelMap.model_validate(payload)


def test_frame_and_receipt_have_separate_identity_boundaries() -> None:
    result = ChannelMapper(
        semantic_schema=_schema(),
        channel_map=_mapping(source_field="source_power", source_unit="W"),
    ).map_frame(pd.DataFrame({"clock": [0.0, 1.0], "source_power": [4.0, 5.0]}))

    frame_payload = json.loads(result.frame.model_dump_json())
    receipt_payload = json.loads(result.receipt.model_dump_json())
    assert frame_payload["labels_used"] is False and frame_payload["truth_used"] is False
    assert "source_power" not in json.dumps(frame_payload)
    assert receipt_payload["source_to_semantic"] == {
        "source_power": "asset.main_bus.active_power"
    }
