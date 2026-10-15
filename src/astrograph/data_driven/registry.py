from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Sequence

from .adapters import adapt_raw_model_output
from .contracts import DataDrivenModelManifest, ModelInferenceRequest
from .models import DataDrivenPrediction
from .providers import RawDataDrivenModelProvider

if TYPE_CHECKING:
    from astrograph.assurance import AdmissionSigner, FeatureContractRegistry


def provider_identity(provider: RawDataDrivenModelProvider) -> str:
    """Return the stable provider identity used by admission manifests."""

    explicit = getattr(provider, "provider_id", None)
    if explicit:
        return str(explicit)
    provider_type = type(provider)
    return f"python:{provider_type.__module__}.{provider_type.__qualname__}"


class DataDrivenModelRegistry:
    """Versioned registry for heterogeneous data-driven model providers.

    ``require_admission=True`` creates a fail-closed production boundary: every
    provider must have a content-addressed model admission manifest and every
    request is validated, blinded, and signed before provider invocation.

    Unadmitted execution is deliberately *not* the default.  Offline research and
    replay code must opt in with ``allow_unadmitted_research=True`` (or use
    :meth:`for_research`).  This prevents a newly added service entry point from
    silently inheriting the old permissive execution path.
    """

    def __init__(
        self,
        *,
        admission_registry: "FeatureContractRegistry | None" = None,
        admission_signer: "AdmissionSigner | None" = None,
        require_admission: bool = False,
        allow_unadmitted_research: bool = False,
    ) -> None:
        self._providers: dict[tuple[str, str], RawDataDrivenModelProvider] = {}
        self._versions: dict[str, list[str]] = defaultdict(list)
        self._admission_registry = admission_registry
        self._admission_signer = admission_signer
        self.require_admission = bool(require_admission)
        self.allow_unadmitted_research = bool(allow_unadmitted_research)
        if self.require_admission and self.allow_unadmitted_research:
            raise ValueError("strict admission and unadmitted research mode are mutually exclusive")
        if not self.require_admission and not self.allow_unadmitted_research:
            raise ValueError(
                "unadmitted model execution is disabled by default; use strict admission "
                "or explicitly construct DataDrivenModelRegistry.for_research()"
            )
        if self.require_admission and (admission_registry is None or admission_signer is None):
            raise ValueError(
                "strict data-driven registry requires admission_registry and admission_signer"
            )

    @classmethod
    def for_research(cls) -> "DataDrivenModelRegistry":
        """Create an explicitly unadmitted registry for offline research/replay only."""

        return cls(allow_unadmitted_research=True)

    def register(self, provider: RawDataDrivenModelProvider, *, replace: bool = False) -> None:
        manifest = provider.manifest
        if not self.require_admission and bool(getattr(provider, "requires_admission", False)):
            raise ValueError(
                f"model provider requires strict admission: {manifest.manifest_id}"
            )
        if self.require_admission:
            if self._admission_registry is None:
                raise RuntimeError("strict registry lost its admission registry")
            self._admission_registry.validate_provider_binding(
                data_model_manifest=manifest,
                provider_id=provider_identity(provider),
                model_artifact_sha256=getattr(provider, "model_artifact_sha256", None),
            )
        key = (manifest.model_id, manifest.model_version)
        if key in self._providers and not replace:
            raise ValueError(f"model provider already registered: {manifest.manifest_id}")
        self._providers[key] = provider
        if manifest.model_version not in self._versions[manifest.model_id]:
            self._versions[manifest.model_id].append(manifest.model_version)
            self._versions[manifest.model_id].sort()

    def get(self, model_id: str, model_version: str | None = None) -> RawDataDrivenModelProvider:
        versions = self._versions.get(model_id, [])
        if not versions:
            raise KeyError(f"unknown data-driven model: {model_id}")
        resolved = model_version or versions[-1]
        key = (model_id, resolved)
        if key not in self._providers:
            raise KeyError(f"unknown data-driven model version: {model_id}@{resolved}")
        return self._providers[key]

    def _finalize_admitted_prediction(self, *, provider, request, execution, raw) -> DataDrivenPrediction:
        if raw.episode_id != execution.provider_request.episode_id:
            raise ValueError("provider output episode token does not match the admitted request")
        # The true episode identity is restored only after the provider returns.
        rebound = raw.model_copy(update={"episode_id": request.episode_id})
        prediction = adapt_raw_model_output(provider.manifest, request, rebound)
        metadata = dict(prediction.metadata)
        receipt_payload = execution.receipt.model_dump(mode="json")
        metadata.update(
            {
                "model_admission": {
                    "receipt_id": execution.receipt.receipt_id,
                    "admission_id": execution.receipt.admission_id,
                    "feature_contract_id": execution.receipt.feature_contract_id,
                    "admitted_payload_sha256": execution.receipt.admitted_payload_sha256,
                    "observation_scope_sha256": execution.receipt.observation_scope_sha256,
                    "signer_key_id": execution.receipt.signer_key_id,
                    "receipt_signature_hmac_sha256": execution.receipt.signature_hmac_sha256,
                    "receipt": receipt_payload,
                    "handling": (
                        "Trusted audit metadata; never reintroduce this receipt or its hashes "
                        "into a model-visible request."
                    ),
                },
                "evidence_lineage_by_fault": execution.evidence_lineage_by_fault,
            }
        )
        return prediction.model_copy(update={"metadata": metadata})

    def predict(self, request: ModelInferenceRequest) -> DataDrivenPrediction:
        return self.predict_many((request,))[0]

    def predict_many(
        self, requests: Sequence[ModelInferenceRequest]
    ) -> list[DataDrivenPrediction]:
        """Predict an ordered homogeneous batch without weakening admission checks.

        Every strict request receives its own admission receipt and blinded provider
        request.  Providers may expose ``predict_raw_batch`` for efficient vectorized
        execution; otherwise this method falls back to ordered single-request calls.
        A batch may not mix model/provider identities because doing so makes output
        ordering and model-artifact provenance ambiguous.
        """

        batch = list(requests)
        if not batch:
            return []
        provider = self.get(batch[0].model_id, batch[0].model_version)
        for request in batch[1:]:
            resolved = self.get(request.model_id, request.model_version)
            if resolved is not provider:
                raise ValueError("predict_many requires one homogeneous model provider")

        batch_predict = getattr(provider, "predict_raw_batch", None)
        if not self.require_admission:
            if bool(getattr(provider, "requires_admission", False)):
                raise RuntimeError(
                    f"registered provider now requires strict admission: {provider.manifest.manifest_id}"
                )
            if callable(batch_predict):
                raw_outputs = list(batch_predict(batch))
                if len(raw_outputs) != len(batch):
                    raise ValueError("provider batch output count does not match request count")
                return [
                    adapt_raw_model_output(provider.manifest, request, raw)
                    for request, raw in zip(batch, raw_outputs, strict=True)
                ]
            return [
                adapt_raw_model_output(provider.manifest, request, provider.predict_raw(request))
                for request in batch
            ]

        # Constructor and registration have already established these invariants;
        # explicit runtime checks avoid relying on ``assert`` in production code.
        if self._admission_registry is None or self._admission_signer is None:
            raise RuntimeError("strict registry lost its admission dependencies")
        self._admission_registry.validate_provider_binding(
            data_model_manifest=provider.manifest,
            provider_id=provider_identity(provider),
            model_artifact_sha256=getattr(provider, "model_artifact_sha256", None),
        )
        executions = [
            self._admission_registry.admit(
                request=request,
                data_model_manifest=provider.manifest,
                provider_id=provider_identity(provider),
                signer=self._admission_signer,
            )
            for request in batch
        ]
        provider_requests = [execution.provider_request for execution in executions]
        if callable(batch_predict):
            raw_outputs = list(batch_predict(provider_requests))
        else:
            raw_outputs = [provider.predict_raw(request) for request in provider_requests]
        if len(raw_outputs) != len(batch):
            raise ValueError("provider batch output count does not match request count")
        return [
            self._finalize_admitted_prediction(
                provider=provider, request=request, execution=execution, raw=raw
            )
            for request, execution, raw in zip(batch, executions, raw_outputs, strict=True)
        ]

    def provider_lifecycle(self, model_id: str, model_version: str | None = None) -> dict:
        provider = self.get(model_id, model_version)
        profile = getattr(provider, "lifecycle_profile", None)
        if profile is None:
            from astrograph.provider_lifecycle import BUILTIN_PROVIDER_QUALIFICATIONS

            profile = BUILTIN_PROVIDER_QUALIFICATIONS.get("inprocess-model-runtime.v1")
        return profile.model_dump(mode="json")

    def manifests(self) -> list[DataDrivenModelManifest]:
        return [self._providers[key].manifest for key in sorted(self._providers)]

    def describe(self) -> list[dict]:
        return [manifest.model_dump(mode="json") for manifest in self.manifests()]


__all__ = ["DataDrivenModelRegistry", "provider_identity"]
