from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from astrograph.semantics.leakage import prohibited_field_role
from astrograph.lineage import observation_correlation_group_id


class AdmissionError(ValueError):
    """Raised when a model input fails a frozen admission contract."""


class FeatureValueKind(str, Enum):
    NUMBER = "NUMBER"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"
    STRING = "STRING"
    NUMBER_ARRAY = "NUMBER_ARRAY"
    NULLABLE_NUMBER_ARRAY = "NULLABLE_NUMBER_ARRAY"
    INTEGER_ARRAY = "INTEGER_ARRAY"
    BOOLEAN_ARRAY = "BOOLEAN_ARRAY"
    STRING_ARRAY = "STRING_ARRAY"


class InputRole(str, Enum):
    FEATURE = "FEATURE"
    MASK = "MASK"
    CONTEXT = "CONTEXT"


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_token(prefix: str, value: Any, *, length: int = 32) -> str:
    return f"{prefix}-{canonical_sha256(value)[:length].upper()}"


class InputBinding(BaseModel):
    """One explicitly admitted model-visible payload leaf.

    ``payload_path`` is relative to ``ModelInferenceRequest.payload``.  Every leaf
    visible to a strict provider must have exactly one binding; unknown leaves are
    rejected rather than imputed or ignored.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_id: str = Field(min_length=1)
    payload_path: tuple[str, ...] = Field(min_length=1)
    value_kind: FeatureValueKind
    role: InputRole = InputRole.FEATURE
    shape: tuple[int | None, ...] = ()
    source_fields: tuple[str, ...] = Field(min_length=1)
    semantic_signal_ids: tuple[str, ...] = Field(min_length=1)
    transform_chain: tuple[str, ...] = Field(default_factory=tuple)
    expected_value_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "Optional canonical JSON SHA-256 for a literal contract value. "
            "This is used for model-visible context such as a frozen channel order."
        ),
    )
    description: str = ""

    @field_validator("payload_path", "source_fields", "semantic_signal_ids", "transform_chain")
    @classmethod
    def nonempty_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not str(item).strip() for item in value):
            raise ValueError("binding string values must be nonempty")
        if len(value) != len(set(value)):
            raise ValueError("binding tuple values must be unique")
        return tuple(str(item) for item in value)

    @field_validator("shape")
    @classmethod
    def valid_shape(cls, value: tuple[int | None, ...]) -> tuple[int | None, ...]:
        if any(item is not None and item <= 0 for item in value):
            raise ValueError("shape dimensions must be positive integers or null wildcards")
        return value

    @model_validator(mode="after")
    def deny_truth_bindings(self) -> "InputBinding":
        names = [*self.payload_path, *self.source_fields, *self.semantic_signal_ids]
        prohibited = [(name, prohibited_field_role(name)) for name in names]
        hits = [(name, role.value) for name, role in prohibited if role is not None]
        if hits:
            raise ValueError(f"prohibited truth/identity field in feature binding: {hits}")
        is_array = self.value_kind in {
            FeatureValueKind.NUMBER_ARRAY,
            FeatureValueKind.NULLABLE_NUMBER_ARRAY,
            FeatureValueKind.INTEGER_ARRAY,
            FeatureValueKind.BOOLEAN_ARRAY,
            FeatureValueKind.STRING_ARRAY,
        }
        if is_array and not self.shape:
            raise ValueError("array binding requires an explicit rank/shape contract")
        if not is_array and self.shape:
            raise ValueError("scalar binding must not declare an array shape")
        if self.role == InputRole.MASK and self.value_kind != FeatureValueKind.BOOLEAN_ARRAY:
            raise ValueError("MASK role requires BOOLEAN_ARRAY")
        return self


class FeatureContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-feature-admission-contract.v1"
    contract_id: str = Field(min_length=1)
    feature_schema_id: str = Field(min_length=1)
    bindings: tuple[InputBinding, ...] = Field(min_length=1)
    require_observation_scope: bool = True
    require_input_hash_match: bool = True
    allow_additional_payload_leaves: bool = False
    claim_boundary: str = (
        "Default-deny model-input contract. Admission proves conformance to the frozen input interface; "
        "it does not establish model validity or diagnostic accuracy."
    )

    @model_validator(mode="after")
    def unique_bindings(self) -> "FeatureContract":
        ids = [item.binding_id for item in self.bindings]
        paths = [item.payload_path for item in self.bindings]
        if len(ids) != len(set(ids)):
            raise ValueError("feature binding IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("feature binding payload paths must be unique")
        if self.allow_additional_payload_leaves:
            raise ValueError(
                "strict feature-admission contracts cannot allow unbound payload leaves"
            )
        return self

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class ModelAdmissionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-model-admission-manifest.v1"
    admission_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    model_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_model_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_contract_id: str = Field(min_length=1)
    feature_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    supported_fault_ids: tuple[str, ...] = Field(min_length=1)
    normal_class_id: str = "NORMAL"
    enabled: bool = True
    claim_boundary: str = (
        "Content-addressed execution binding only. It does not certify training data, calibration, "
        "fault coverage, or flight readiness."
    )

    @field_validator("supported_fault_ids")
    @classmethod
    def unique_faults(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item for item in value):
            raise ValueError("supported fault IDs must be nonempty")
        if len(value) != len(set(value)):
            raise ValueError("supported fault IDs must be unique")
        return tuple(value)

    @model_validator(mode="after")
    def normal_not_fault(self) -> "ModelAdmissionManifest":
        if self.normal_class_id in set(self.supported_fault_ids):
            raise ValueError("normal_class_id must not appear in supported_fault_ids")
        return self

    @property
    def content_sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class AdmissionReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-model-admission-receipt.v1"
    receipt_id: str = Field(min_length=1)
    admission_id: str = Field(min_length=1)
    admission_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_contract_id: str = Field(min_length=1)
    feature_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_schema_id: str = Field(min_length=1)
    original_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    admitted_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observation_scope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_episode_token: str = Field(min_length=1)
    source_evidence_ids: tuple[str, ...]
    correlation_group_ids: tuple[str, ...]
    signer_key_id: str = Field(min_length=1)
    signature_hmac_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def unsigned_payload(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload.pop("signature_hmac_sha256", None)
        return payload


class AdmissionSigner:
    """Process-local HMAC signer for trusted admission receipts.

    Deployments should provide ``ASTROGRAPH_ADMISSION_HMAC_KEY`` as at least 32
    bytes of secret material.  When absent, a process-ephemeral key is generated;
    receipts remain tamper-evident inside the process but cannot be verified after a
    restart.  The key provenance is explicit through ``key_id``.
    """

    def __init__(self, secret: bytes, *, key_id: str):
        if len(secret) < 32:
            raise ValueError("admission HMAC secret must contain at least 32 bytes")
        if not key_id:
            raise ValueError("admission signer key_id must be nonempty")
        self._secret = bytes(secret)
        self.key_id = key_id

    @classmethod
    def from_environment_or_ephemeral(cls) -> "AdmissionSigner":
        raw = os.environ.get("ASTROGRAPH_ADMISSION_HMAC_KEY", "")
        if raw:
            secret = raw.encode("utf-8")
            if len(secret) < 32:
                raise ValueError("ASTROGRAPH_ADMISSION_HMAC_KEY must be at least 32 bytes")
            key_id = os.environ.get("ASTROGRAPH_ADMISSION_HMAC_KEY_ID", "env-admission-key")
            return cls(secret, key_id=key_id)
        return cls(secrets.token_bytes(32), key_id="process-ephemeral")

    def sign(self, payload: Mapping[str, Any]) -> str:
        return hmac.new(self._secret, canonical_json_bytes(payload), hashlib.sha256).hexdigest()

    def verify(self, receipt: AdmissionReceipt) -> bool:
        return hmac.compare_digest(self.sign(receipt.unsigned_payload()), receipt.signature_hmac_sha256)


def _flatten_payload(value: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    if isinstance(value, Mapping):
        flattened: dict[tuple[str, ...], Any] = {}
        if not value and prefix:
            flattened[prefix] = {}
        for key, item in value.items():
            key_text = str(key)
            flattened.update(_flatten_payload(item, (*prefix, key_text)))
        return flattened
    return {prefix: value}


def _array_shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise AdmissionError("array binding received a non-array value")
    length = len(value)
    if length == 0:
        return (0,)
    child_shapes = [_array_shape(item) if isinstance(item, (list, tuple)) or hasattr(item, "tolist") else () for item in value]
    if len(set(child_shapes)) != 1:
        raise AdmissionError("ragged arrays are forbidden by the feature contract")
    return (length, *child_shapes[0])


def _iter_scalars(value: Any) -> Iterable[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_scalars(item)
    else:
        yield value


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_value(binding: InputBinding, value: Any) -> None:
    kind = binding.value_kind
    if kind == FeatureValueKind.NUMBER:
        ok = _is_number(value)
    elif kind == FeatureValueKind.INTEGER:
        ok = _is_integer(value)
    elif kind == FeatureValueKind.BOOLEAN:
        ok = isinstance(value, bool)
    elif kind == FeatureValueKind.STRING:
        ok = isinstance(value, str)
    else:
        actual_shape = _array_shape(value)
        if len(actual_shape) != len(binding.shape):
            raise AdmissionError(
                f"payload {'.'.join(binding.payload_path)} rank mismatch: "
                f"expected {len(binding.shape)}, got {len(actual_shape)}"
            )
        for expected, actual in zip(binding.shape, actual_shape, strict=True):
            if expected is not None and expected != actual:
                raise AdmissionError(
                    f"payload {'.'.join(binding.payload_path)} shape mismatch: "
                    f"expected {binding.shape}, got {actual_shape}"
                )
        values = list(_iter_scalars(value))
        if kind == FeatureValueKind.NUMBER_ARRAY:
            ok = all(_is_number(item) for item in values)
        elif kind == FeatureValueKind.NULLABLE_NUMBER_ARRAY:
            ok = all(item is None or _is_number(item) for item in values)
        elif kind == FeatureValueKind.INTEGER_ARRAY:
            ok = all(_is_integer(item) for item in values)
        elif kind == FeatureValueKind.BOOLEAN_ARRAY:
            ok = all(isinstance(item, bool) for item in values)
        else:
            ok = all(isinstance(item, str) for item in values)
    if not ok:
        raise AdmissionError(
            f"payload {'.'.join(binding.payload_path)} does not match value kind {binding.value_kind.value}"
        )
    if (
        binding.expected_value_sha256 is not None
        and canonical_sha256(value) != binding.expected_value_sha256
    ):
        raise AdmissionError(
            f"payload {'.'.join(binding.payload_path)} does not match its frozen literal value"
        )


def _scope_payload(request: Any) -> dict[str, Any]:
    scope = getattr(request, "observation_scope", None)
    if scope is None:
        return {}
    return scope.model_dump(mode="json") if isinstance(scope, BaseModel) else _jsonable(scope)


@dataclass(frozen=True)
class AdmittedExecution:
    provider_request: Any
    receipt: AdmissionReceipt
    evidence_lineage_by_fault: dict[str, dict[str, Any]]


def admit_request(
    *,
    request: Any,
    data_model_manifest: Any,
    feature_contract: FeatureContract,
    admission_manifest: ModelAdmissionManifest,
    provider_id: str,
    signer: AdmissionSigner,
) -> AdmittedExecution:
    """Validate and blind a request before an untrusted provider can observe it."""

    if not admission_manifest.enabled:
        raise AdmissionError(f"model admission is disabled: {admission_manifest.admission_id}")
    if request.model_id != admission_manifest.model_id:
        raise AdmissionError("request model_id does not match admission manifest")
    resolved_version = request.model_version or data_model_manifest.model_version
    if resolved_version != admission_manifest.model_version:
        raise AdmissionError("request model_version does not match admission manifest")
    if provider_id != admission_manifest.provider_id:
        raise AdmissionError("provider identity does not match admission manifest")
    if canonical_sha256(data_model_manifest.model_dump(mode="json")) != admission_manifest.data_model_manifest_sha256:
        raise AdmissionError("data-driven model manifest hash mismatch")
    if feature_contract.contract_id != admission_manifest.feature_contract_id:
        raise AdmissionError("feature contract ID does not match admission manifest")
    if feature_contract.content_sha256 != admission_manifest.feature_contract_sha256:
        raise AdmissionError("feature contract hash mismatch")
    if request.feature_schema_id != feature_contract.feature_schema_id:
        raise AdmissionError("request feature schema does not match feature contract")
    if tuple(request.registered_fault_ids) != admission_manifest.supported_fault_ids:
        raise AdmissionError(
            "request registered_fault_ids must exactly equal the frozen admission fault universe"
        )

    scope = _scope_payload(request)
    if feature_contract.require_observation_scope and not scope:
        raise AdmissionError("strict feature admission requires observation_scope")

    flattened = _flatten_payload(request.payload)
    expected = {binding.payload_path: binding for binding in feature_contract.bindings}
    unknown = sorted(set(flattened) - set(expected))
    missing = sorted(set(expected) - set(flattened))
    if missing:
        raise AdmissionError(f"required model inputs are missing: {[ '.'.join(path) for path in missing ]}")
    if unknown and not feature_contract.allow_additional_payload_leaves:
        raise AdmissionError(f"unapproved model inputs are present: {[ '.'.join(path) for path in unknown ]}")

    for path, binding in expected.items():
        for segment in path:
            role = prohibited_field_role(segment)
            if role is not None:
                raise AdmissionError(
                    f"prohibited model-visible payload key {segment!r}: {role.value}"
                )
        _validate_value(binding, flattened[path])

    admitted_payload = deepcopy(_jsonable(request.payload))
    admitted_sha = canonical_sha256(admitted_payload)
    if feature_contract.require_input_hash_match and admitted_sha != request.input_data_sha256:
        raise AdmissionError(
            "input_data_sha256 does not match the canonical admitted model payload"
        )

    scope_sha = canonical_sha256(scope)
    source_ids: set[str] = set()
    groups: set[str] = set()
    for binding in feature_contract.bindings:
        for source_field in binding.source_fields:
            for signal_id in binding.semantic_signal_ids:
                groups.add(
                    observation_correlation_group_id(
                        episode_id=request.episode_id,
                        window_start_s=request.observation_scope.window_start_s,
                        window_end_s=request.observation_scope.window_end_s,
                        source_field=source_field,
                        semantic_signal_id=signal_id,
                    )
                )
        source_ids.add(
            _stable_token(
                "EVID",
                {
                    "scope_sha256": scope_sha,
                    "binding_id": binding.binding_id,
                    "source_fields": binding.source_fields,
                    "semantic_signal_ids": binding.semantic_signal_ids,
                    "transform_chain": binding.transform_chain,
                    "admitted_payload_sha256": admitted_sha,
                },
            )
        )

    unsigned_base = {
        "schema_version": "astrograph-model-admission-receipt.v1",
        "admission_id": admission_manifest.admission_id,
        "admission_manifest_sha256": admission_manifest.content_sha256,
        "feature_contract_id": feature_contract.contract_id,
        "feature_contract_sha256": feature_contract.content_sha256,
        "feature_schema_id": feature_contract.feature_schema_id,
        "original_input_sha256": request.input_data_sha256,
        "admitted_payload_sha256": admitted_sha,
        "observation_scope_sha256": scope_sha,
        "source_evidence_ids": tuple(sorted(source_ids)),
        "correlation_group_ids": tuple(sorted(groups)),
        "signer_key_id": signer.key_id,
    }
    receipt_id = _stable_token("ADMISSION", unsigned_base)
    # Provider-visible identities are keyed commitments rather than raw hashes.
    # An unkeyed payload/episode hash is a high-bandwidth identity side channel
    # when a provider knows the finite replay corpus and can precompute hashes.
    provider_episode_token = (
        "OBS-"
        + signer.sign(
            {
                "purpose": "provider-episode-token",
                "receipt_id": receipt_id,
                "episode_id": str(request.episode_id),
            }
        )[:24].upper()
    )
    provider_input_token = signer.sign(
        {
            "purpose": "provider-input-token",
            "receipt_id": receipt_id,
            "admitted_payload_sha256": admitted_sha,
        }
    )
    unsigned = {
        **unsigned_base,
        "receipt_id": receipt_id,
        "provider_episode_token": provider_episode_token,
    }
    receipt = AdmissionReceipt(
        **unsigned,
        signature_hmac_sha256=signer.sign(unsigned),
    )

    # Import here to keep the assurance module independent from data-driven
    # contracts at import time and avoid a circular dependency.
    provider_request = request.model_copy(
        update={
            "episode_id": provider_episode_token,
            "model_version": admission_manifest.model_version,
            "input_data_sha256": provider_input_token,
            "registered_fault_ids": list(admission_manifest.supported_fault_ids),
            "payload": admitted_payload,
            "metadata": {},
            "observation_scope": None,
        },
        deep=True,
    )
    lineage = {
        fault_id: {
            "source_evidence_ids": list(receipt.source_evidence_ids),
            "correlation_group_ids": list(receipt.correlation_group_ids),
            "lineage_complete": True,
        }
        for fault_id in admission_manifest.supported_fault_ids
    }
    return AdmittedExecution(
        provider_request=provider_request,
        receipt=receipt,
        evidence_lineage_by_fault=lineage,
    )


__all__ = [
    "AdmissionError",
    "AdmissionReceipt",
    "AdmissionSigner",
    "AdmittedExecution",
    "FeatureContract",
    "FeatureValueKind",
    "InputBinding",
    "InputRole",
    "ModelAdmissionManifest",
    "admit_request",
    "canonical_json_bytes",
    "canonical_sha256",
]
