from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .contracts import DataDrivenModelManifest, ModelInferenceRequest, RawModelOutput
from astrograph.provider_lifecycle import BUILTIN_PROVIDER_QUALIFICATIONS, ProviderLifecycleProfile


class RawDataDrivenModelProvider(Protocol):
    manifest: DataDrivenModelManifest

    def predict_raw(self, request: ModelInferenceRequest) -> RawModelOutput: ...


@dataclass
class CallableRawModelProvider:
    """Model-family-neutral provider for PyTorch, TensorFlow, ONNX, or custom code.

    The callable remains outside the AstroGraph core. It receives a typed request and must
    return a strict RawModelOutput. This avoids importing heavyweight runtimes into the core.
    """

    manifest: DataDrivenModelManifest
    predict_fn: Callable[[ModelInferenceRequest], RawModelOutput | dict[str, Any]]
    requires_admission: bool = False
    model_artifact_sha256: str | None = None

    @property
    def lifecycle_profile(self) -> ProviderLifecycleProfile:
        return BUILTIN_PROVIDER_QUALIFICATIONS.get("inprocess-model-runtime.v1")

    @property
    def provider_id(self) -> str:
        function = self.predict_fn
        return f"callable:{getattr(function, '__module__', 'unknown')}.{getattr(function, '__qualname__', type(function).__qualname__)}"

    def predict_raw(self, request: ModelInferenceRequest) -> RawModelOutput:
        result = self.predict_fn(request)
        return result if isinstance(result, RawModelOutput) else RawModelOutput.model_validate(result)


@dataclass
class SklearnProbabilityProvider:
    """Duck-typed adapter for estimators exposing classes_ and predict_proba.

    It supports HistGradientBoosting, RandomForest, LogisticRegression, SVC with
    probability=True, and compatible XGBoost/LightGBM estimators without making any of
    those model families mandatory AstroGraph dependencies.
    """

    manifest: DataDrivenModelManifest
    estimator: Any
    feature_order: list[str]
    requires_admission: bool = False
    model_artifact_sha256: str | None = None

    @property
    def lifecycle_profile(self) -> ProviderLifecycleProfile:
        return BUILTIN_PROVIDER_QUALIFICATIONS.get("inprocess-model-runtime.v1")

    @property
    def provider_id(self) -> str:
        estimator_type = type(self.estimator)
        return f"sklearn-compatible:{estimator_type.__module__}.{estimator_type.__qualname__}"

    def predict_raw(self, request: ModelInferenceRequest) -> RawModelOutput:
        if not hasattr(self.estimator, "predict_proba") or not hasattr(self.estimator, "classes_"):
            raise TypeError("estimator must expose predict_proba and classes_")
        features = request.payload.get("features")
        if not isinstance(features, dict):
            raise ValueError("sklearn provider expects payload.features mapping")
        expected = list(self.feature_order)
        missing = [name for name in expected if name not in features]
        unknown = sorted(set(features) - set(expected))
        if missing or unknown:
            raise ValueError(
                f"sklearn provider feature contract mismatch; missing={missing}, unknown={unknown}"
            )
        row_values = [float(features[name]) for name in expected]
        if any(not math.isfinite(value) for value in row_values):
            raise ValueError("sklearn provider features must be finite")
        row = [row_values]
        probabilities = self.estimator.predict_proba(row)
        values = probabilities[0] if hasattr(probabilities, "__getitem__") else probabilities
        class_values = {
            str(class_id): float(probability)
            for class_id, probability in zip(self.estimator.classes_, values, strict=True)
        }
        return RawModelOutput(
            episode_id=request.episode_id,
            model_id=self.manifest.model_id,
            model_version=self.manifest.model_version,
            output_semantics=self.manifest.output_semantics,
            class_values=class_values,
            metadata={"provider": "SklearnProbabilityProvider"},
        )


Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]


def urllib_json_transport(endpoint: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - explicit configured endpoint
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("external model service must return a JSON object")
    return parsed


@dataclass
class ExternalServiceModelProvider:
    """Provider for model servers hosting TCN/LSTM/Transformer/ONNX or other models."""

    manifest: DataDrivenModelManifest
    endpoint: str
    timeout_s: float = 30.0
    transport: Transport = urllib_json_transport
    requires_admission: bool = False
    model_artifact_sha256: str | None = None

    @property
    def lifecycle_profile(self) -> ProviderLifecycleProfile:
        if self.transport is urllib_json_transport:
            return BUILTIN_PROVIDER_QUALIFICATIONS.get("http-json-provider.v1")
        return BUILTIN_PROVIDER_QUALIFICATIONS.get("inprocess-model-runtime.v1")

    @property
    def provider_id(self) -> str:
        return f"external-json:{self.endpoint}"

    def predict_raw(self, request: ModelInferenceRequest) -> RawModelOutput:
        payload = {
            "model_id": self.manifest.model_id,
            "model_version": self.manifest.model_version,
            "request": request.model_dump(mode="json"),
        }
        result = self.transport(self.endpoint, payload, self.timeout_s)
        return RawModelOutput.model_validate(result)
