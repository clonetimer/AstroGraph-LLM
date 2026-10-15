from __future__ import annotations

import re
from enum import Enum


class ProhibitedFieldRole(str, Enum):
    """Name-derived roles that must not enter model-observation telemetry.

    This classifier is deliberately conservative for untyped source schemas.
    A name that does not match a deny rule is *not* automatically approved: a
    reviewed semantic mapping or feature allowlist is still required before a
    field becomes a model input.
    """

    DIRECT_LABEL = "DIRECT_LABEL"
    INJECTION_METADATA = "INJECTION_METADATA"
    IDENTITY_PROXY = "IDENTITY_PROXY"
    POST_EVENT_DERIVED = "POST_EVENT_DERIVED"


# Prefix checks operate on the case-folded field name while exact checks also
# use a separator-normalized identifier.  This prevents exporters from evading
# the guard with variants such as ``fault-type`` or ``Fault Type``.
PROHIBITED_PREFIXES = (
    "label.",
    "truth.",
    "ground_truth.",
    "hidden_truth.",
    "effect.",
    "event.",
    "injection.",
    "injected.",
    "object.",
    "modifier.",
)
PROHIBITED_EXACT = {
    # Dataset/scenario identity can encode the answer through layout, pairing,
    # ordering, split assignment, or one-to-one scenario-to-fault mappings.
    "task_id",
    "case_id",
    "scenario_id",
    "condition_id",
    "dataset_id",
    "sample_id",
    "record_id",
    "run_id",
    "pair_id",
    "trajectory_pair_id",
    "class_id",
    "split",
    "split_id",
    "fold_id",
    "seed",
    "random_seed",
    "file_path",
    "source_path",
    "mode",
    # Direct answer/target identities.
    "ground_truth",
    "fault_label",
    "target_label",
    "diagnosis_label",
    "diagnostic_label",
    "diagnostic_class",
    "diagnosis_class",
    "fault_type",
    "fault_id",
    "fault_class",
    "failure_mode",
    "health_state",
    "injected_fault",
    "injected_fault_id",
    # Injection/event timing is a high-bandwidth answer proxy even when it is
    # not nested below the historical ``event.`` namespace.
    "event_start_s",
    "event_end_s",
    "event_onset_s",
    "fault_start_s",
    "fault_end_s",
    "fault_onset_s",
    "onset_time_s",
    "activation_time_s",
}
PROHIBITED_CONTAINS = (
    "ground_truth",
    "hidden_truth",
    "fault_label",
    "target_label",
    "diagnosis_label",
    "diagnostic_label",
    "diagnostic_class",
    "diagnosis_class",
    "injected_fault",
    "fault_injection",
    "answer_key",
    "oracle_label",
    "actual_fault",
)

_DIRECT_LABEL_PREFIXES = (
    "label.",
    "truth.",
    "ground_truth.",
    "hidden_truth.",
)
_DIRECT_LABEL_EXACT = {
    "ground_truth",
    "fault_label",
    "target_label",
    "diagnosis_label",
    "diagnostic_label",
    "diagnostic_class",
    "diagnosis_class",
    "fault_type",
    "fault_id",
    "fault_class",
    "failure_mode",
}
_DIRECT_LABEL_CONTAINS = (
    "ground_truth",
    "hidden_truth",
    "fault_label",
    "target_label",
    "diagnosis_label",
    "diagnostic_label",
    "diagnostic_class",
    "diagnosis_class",
    "answer_key",
    "oracle_label",
    "actual_fault",
)
_INJECTION_PREFIXES = ("event.", "injection.", "injected.", "modifier.")
_INJECTION_EXACT = {
    "injected_fault",
    "injected_fault_id",
    "event_start_s",
    "event_end_s",
    "event_onset_s",
    "fault_start_s",
    "fault_end_s",
    "fault_onset_s",
    "onset_time_s",
    "activation_time_s",
}
_INJECTION_CONTAINS = ("injected_fault", "fault_injection")
_POST_EVENT_PREFIXES = ("effect.",)


def _casefolded(field: str) -> str:
    return str(field).lstrip("\ufeff").strip().casefold()


def _normalized_identifier(field: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _casefolded(field)).strip("_")


def _matches_exact(value: str, normalized: str, names: set[str]) -> bool:
    return value in names or normalized in names


def _contains_token(value: str, normalized: str, tokens: tuple[str, ...]) -> bool:
    return any(token in value or token in normalized for token in tokens)


def prohibited_field_role(field: str) -> ProhibitedFieldRole | None:
    """Infer a prohibited information role from a field name.

    ``None`` only means that no name-based deny rule matched.  It does not
    constitute feature approval; callers should still enforce an explicit
    semantic mapping or feature allowlist.
    """

    lowered = _casefolded(field)
    normalized = _normalized_identifier(field)

    if (
        _matches_exact(lowered, normalized, _DIRECT_LABEL_EXACT)
        or lowered.startswith(_DIRECT_LABEL_PREFIXES)
        or _contains_token(lowered, normalized, _DIRECT_LABEL_CONTAINS)
    ):
        return ProhibitedFieldRole.DIRECT_LABEL

    if (
        _matches_exact(lowered, normalized, _INJECTION_EXACT)
        or lowered.startswith(_INJECTION_PREFIXES)
        or _contains_token(lowered, normalized, _INJECTION_CONTAINS)
    ):
        return ProhibitedFieldRole.INJECTION_METADATA

    if lowered.startswith(_POST_EVENT_PREFIXES):
        return ProhibitedFieldRole.POST_EVENT_DERIVED

    if (
        _matches_exact(lowered, normalized, PROHIBITED_EXACT)
        or lowered.startswith(PROHIBITED_PREFIXES)
        or _contains_token(lowered, normalized, PROHIBITED_CONTAINS)
    ):
        return ProhibitedFieldRole.IDENTITY_PROXY

    return None


def is_prohibited_field(field: str) -> bool:
    return prohibited_field_role(field) is not None


__all__ = [
    "PROHIBITED_PREFIXES",
    "PROHIBITED_EXACT",
    "PROHIBITED_CONTAINS",
    "ProhibitedFieldRole",
    "prohibited_field_role",
    "is_prohibited_field",
]
