"""Auditable, conservative bilingual expansion for spacecraft fault retrieval.

These are *search aliases*, never evidence that a component or fault exists.
Unreviewed/model-proposed synonyms must not silently enter this registry.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re

LEXICON_VERSION = "astrograph-spacecraft-lexicon.v1"
# Canonical keys serve as stable concept IDs and are NOT labels in a diagnosed case.
# Acronyms are only expanded at word boundaries.  Keep ambiguous terms out.
_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reaction_wheel", ("反作用飞轮", "反作用轮", "reaction wheel", "reaction wheels", "rw assembly", "rw speed", "飞轮转速")),
    ("wheel_friction", ("飞轮摩擦增大", "轴承摩擦", "轴承阻力", "wheel friction", "bearing friction", "friction increase")),
    ("wheel_saturation", ("飞轮饱和", "角动量饱和", "wheel saturation", "momentum saturation", "momentum dumping")),
    ("star_tracker", ("星敏感器", "星敏", "star tracker", "star sensor")),
    ("attitude_control", ("姿态控制", "姿控", "attitude control", "attitude determination and control", "adcs", "aocs")),
    ("attitude_error", ("姿态误差", "指向误差", "pointing error", "attitude error", "attitude deviation")),
    ("power_bus_undervoltage", ("母线欠压", "母线电压低", "电源母线欠压", "bus undervoltage", "dc bus low voltage")),
    ("power_system", ("电源分系统", "电源系统", "electrical power system", "eps")),
    ("battery_soc", ("电池荷电状态", "剩余电量", "state of charge", "battery soc")),
    ("solar_array", ("太阳电池阵", "太阳能电池阵", "solar array", "photovoltaic array")),
    ("power_converter", ("直流变换器", "电源变换器", "dc-dc converter", "power converter")),
    ("thermal_system", ("热控系统", "热控分系统", "thermal control", "thermal subsystem")),
    ("temperature_excursion", ("温度超限", "温度异常", "temperature excursion", "overtemperature", "thermal excursion")),
    ("heater", ("加热器", "电加热器", "heater", "survival heater")),
    ("propellant_valve", ("推进阀门", "推进剂阀门", "propellant valve", "thruster valve")),
    ("propulsion", ("推进系统", "推进分系统", "propulsion system", "thruster")),
    ("downlink", ("下行链路", "下行通信", "downlink", "telemetry downlink")),
    ("telemetry_telecommand", ("遥测遥控", "测控", "telemetry and telecommand", "tm/tc", "ttc")),
    ("antenna_pointing", ("天线指向", "天线指向偏差", "antenna pointing", "antenna mispointing")),
    ("onboard_computer", ("星载计算机", "星务计算机", "机载计算机", "onboard computer", "obc", "c&dh")),
    ("watchdog_reset", ("看门狗复位", "看门狗重启", "watchdog reset", "watchdog reboot")),
    ("payload", ("有效载荷", "载荷分系统", "payload subsystem", "payload")),
    ("data_storage", ("星载存储", "数据存储", "mass memory", "onboard storage")),
    ("loss_of_signal", ("信号丢失", "通信失锁", "loss of signal", "los")),
)


def _matches(needle: str, haystack: str) -> bool:
    token = needle.casefold()
    text = haystack.casefold()
    if re.search(r"[a-z0-9]", token):
        return re.search(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", text) is not None
    return token in text


@dataclass(frozen=True)
class VocabularyMatch:
    concept_id: str
    matched_alias: str
    variants: tuple[str, ...]


def load_pinned_domain_lexicon(equipment_scope: str | None) -> tuple[tuple[tuple[str, tuple[str, ...]], ...], dict[str, object]]:
    """Optional admin-pinned, independently reviewed mission glossary.

    Reject unsupervised or hash-drifted vocabularies. This does not certify that
    the textual concepts are universally true or their abbreviations unambiguous.
    """
    filename = os.environ.get('ASTROGRAPH_MATERIAL_LEXICON_PATH', '').strip()
    expected = os.environ.get('ASTROGRAPH_MATERIAL_LEXICON_SHA256', '').strip().lower()
    if not filename and not expected:
        return (), {'status': 'BUILT_IN_ONLY', 'scope': str(equipment_scope or 'UNKNOWN')}
    if not filename or not re.fullmatch(r'[a-f0-9]{64}', expected):
        raise ValueError('mission glossary requires both path and SHA-256 digest')
    path = Path(filename).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 512_000:
        raise ValueError('mission glossary is not a bounded local file')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('mission glossary digest mismatch')
    document = json.loads(raw.decode('utf-8'))
    if set(document) != {'schema_version', 'equipment_scope', 'review', 'concepts'}:
        raise ValueError('mission glossary unexpected fields')
    if document['schema_version'] != 'astrograph-reviewed-mission-glossary.v1':
        raise ValueError('mission glossary incompatible schema')
    review = document['review']
    if not isinstance(review, dict) or review.get('status') != 'EXPERT_APPROVED' or not all(
        isinstance(review.get(key), str) and 4 <= len(review[key]) <= 200
        for key in ('reviewer', 'truth_source_ref')):
        raise ValueError('mission glossary is not independently reviewed')
    scope = str(document['equipment_scope'])
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{1,79}', scope):
        raise ValueError('mission glossary equipment_scope invalid')
    if scope != str(equipment_scope or ''):
        return (), {'status': 'SCOPE_MISMATCH_DISABLED',
                    'equipment_scope': scope, 'sha256': expected}
    items = document['concepts']
    if not isinstance(items, list) or not 1 <= len(items) <= 100:
        raise ValueError('mission glossary concept count invalid')
    concepts = []
    used_ids = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {'id', 'aliases'}:
            raise ValueError('mission glossary concept structure invalid')
        cid, aliases = item['id'], item['aliases']
        if not isinstance(cid, str) or not re.fullmatch(r'[a-z][a-z0-9_]{2,79}', cid) or cid in used_ids:
            raise ValueError('mission glossary concept IDs invalid or duplicate')
        if (not isinstance(aliases, list) or not 2 <= len(aliases) <= 12
            or any(not isinstance(a, str) or not 2 <= len(a.strip()) <= 120 for a in aliases)
            or len({a.casefold() for a in aliases}) != len(aliases)):
            raise ValueError('mission glossary aliases invalid')
        concepts.append((cid, tuple(aliases)))
        used_ids.add(cid)
    return tuple(concepts), {'status': 'EXPERT_REVIEWED_PINNED',
                            'equipment_scope': scope, 'sha256': expected,
                            'reviewer': review['reviewer'], 'truth_source_ref': review['truth_source_ref']}


def expand_spacecraft_terms(text: str, *, max_concepts: int = 5, max_variants: int = 12,
                            extra_concepts: tuple[tuple[str, tuple[str, ...]], ...] = ()) -> dict[str, object]:
    """Expand exact, curated aliases with an inspectable bounded audit trail.

    No edit-distance guesses and no LLM vocabulary injection.  A match is only a
    retrieval-routing hint, not a claim that the spacecraft uses this component.
    """
    if not (1 <= max_concepts <= 12 and 1 <= max_variants <= 36):
        raise ValueError("unsupported vocabulary expansion budget")
    matched: list[VocabularyMatch] = []
    alternatives: list[str] = []
    for canonical, aliases in (*extra_concepts, *_CONCEPTS):
        term = next((item for item in aliases if _matches(item, text)), None)
        if term is None:
            continue
        if len(matched) >= max_concepts:
            break
        expansions = tuple(item for item in aliases if item.casefold() != term.casefold())
        matched.append(VocabularyMatch(canonical, term, expansions))
        for alias in expansions:
            if alias.casefold() not in {x.casefold() for x in alternatives}:
                alternatives.append(alias)
            if len(alternatives) >= max_variants:
                break
        if len(alternatives) >= max_variants:
            break
    return {
        "lexicon_version": LEXICON_VERSION,
        "match_policy": "EXACT_CURATED_ALIAS_SEARCH_HINT_NOT_FAULT_ASSERTION",
        "matches": [{"concept_id": m.concept_id, "matched_alias": m.matched_alias} for m in matched],
        "alternatives": alternatives[:max_variants],
        "unknown_vocabulary_not_inferred": not matched,
    }


__all__ = ["expand_spacecraft_terms", "load_pinned_domain_lexicon", "LEXICON_VERSION"]
