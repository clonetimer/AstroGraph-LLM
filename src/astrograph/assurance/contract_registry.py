from __future__ import annotations

from collections import defaultdict
from typing import Any

from .feature_admission import (
    AdmissionError,
    AdmissionSigner,
    AdmittedExecution,
    FeatureContract,
    ModelAdmissionManifest,
    admit_request,
    canonical_sha256,
)


class FeatureContractRegistry:
    """Default-deny registry binding providers to content-addressed input contracts."""

    def __init__(self) -> None:
        self._contracts: dict[str, FeatureContract] = {}
        self._admissions: dict[tuple[str, str], ModelAdmissionManifest] = {}
        self._versions: dict[str, list[str]] = defaultdict(list)

    def register_contract(self, contract: FeatureContract, *, replace: bool = False) -> None:
        existing = self._contracts.get(contract.contract_id)
        if existing is not None and not replace:
            if existing.content_sha256 != contract.content_sha256:
                raise ValueError(f"feature contract already registered with different content: {contract.contract_id}")
            return
        self._contracts[contract.contract_id] = contract

    def register_admission(
        self, admission: ModelAdmissionManifest, *, replace: bool = False
    ) -> None:
        contract = self._contracts.get(admission.feature_contract_id)
        if contract is None:
            raise ValueError(
                f"feature contract must be registered before admission: {admission.feature_contract_id}"
            )
        if contract.content_sha256 != admission.feature_contract_sha256:
            raise ValueError("admission feature contract hash does not match registered contract")
        key = (admission.model_id, admission.model_version)
        existing = self._admissions.get(key)
        if existing is not None and not replace:
            if existing.content_sha256 != admission.content_sha256:
                raise ValueError(f"model admission already registered with different content: {key}")
            return
        self._admissions[key] = admission
        if admission.model_version not in self._versions[admission.model_id]:
            self._versions[admission.model_id].append(admission.model_version)
            self._versions[admission.model_id].sort()

    def get_contract(self, contract_id: str) -> FeatureContract:
        try:
            return self._contracts[contract_id]
        except KeyError as exc:
            raise KeyError(f"unknown feature contract: {contract_id}") from exc

    def get_admission(self, model_id: str, model_version: str | None = None) -> ModelAdmissionManifest:
        versions = self._versions.get(model_id, [])
        if not versions:
            raise KeyError(f"no model admission registered for: {model_id}")
        resolved = model_version or versions[-1]
        try:
            return self._admissions[(model_id, resolved)]
        except KeyError as exc:
            raise KeyError(f"no model admission registered for: {model_id}@{resolved}") from exc

    def validate_provider_binding(
        self,
        *,
        data_model_manifest: Any,
        provider_id: str,
        model_artifact_sha256: str | None,
    ) -> None:
        try:
            admission = self.get_admission(
                data_model_manifest.model_id, data_model_manifest.model_version
            )
        except KeyError as exc:
            raise AdmissionError(str(exc)) from exc
        if admission.provider_id != provider_id:
            raise AdmissionError("registered provider identity does not match admission manifest")
        if not model_artifact_sha256:
            raise AdmissionError(
                "strict provider must expose the hash-pinned model artifact used by its admission"
            )
        if model_artifact_sha256 != admission.model_artifact_sha256:
            raise AdmissionError("runtime model artifact hash does not match admission manifest")
        manifest_sha = canonical_sha256(data_model_manifest.model_dump(mode="json"))
        if manifest_sha != admission.data_model_manifest_sha256:
            raise AdmissionError("registered data-model manifest does not match admission hash")
        if tuple(data_model_manifest.supported_fault_ids) != admission.supported_fault_ids:
            raise AdmissionError("registered model fault universe differs from admission manifest")
        if not admission.enabled:
            raise AdmissionError("registered model admission is disabled")

    def admit(
        self,
        *,
        request: Any,
        data_model_manifest: Any,
        provider_id: str,
        signer: AdmissionSigner,
    ) -> AdmittedExecution:
        admission = self.get_admission(data_model_manifest.model_id, data_model_manifest.model_version)
        contract = self.get_contract(admission.feature_contract_id)
        return admit_request(
            request=request,
            data_model_manifest=data_model_manifest,
            feature_contract=contract,
            admission_manifest=admission,
            provider_id=provider_id,
            signer=signer,
        )

    def describe(self) -> dict[str, list[dict]]:
        return {
            "contracts": [
                self._contracts[key].model_dump(mode="json") for key in sorted(self._contracts)
            ],
            "admissions": [
                self._admissions[key].model_dump(mode="json") for key in sorted(self._admissions)
            ],
        }


__all__ = ["FeatureContractRegistry"]
