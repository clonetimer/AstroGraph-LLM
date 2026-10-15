"""Online-safe data-driven contracts and provider registry.

This namespace intentionally excludes training campaigns, benchmark pipelines, and
research model implementations. The root ``astrograph.data_driven`` package retains
lazy compatibility exports for older callers.
"""

from ..activity_axes import (
    ActivityAxis,
    ActivityAxisGate,
    ActivityAxisGateDecision,
    ActivityAxisGateStatus,
    RuntimeActivityAxisCapability,
    RuntimeActivityAxisCapabilityManifest,
    default_v0522_activity_axis_manifest,
    load_activity_axis_manifest,
    manifest_sha256,
)
from ..adapters import adapt_raw_model_output, ranked_class_ids
from ..contracts import (
    DataDrivenModelManifest,
    DataDrivenTaskType,
    ModelCapability,
    ModelInferenceRequest,
    ModelOutputSemantics,
    ModelRuntime,
    ProbabilityCalibrationReceipt,
    ProbabilityCalibrationQualificationReceipt,
    RawModelOutput,
)
from ..credibility import OpenSetAssessment, OpenSetPolicy, OpenSetStatus, assess_open_set
from ..models import ClassProbability, DataDrivenPrediction, DataDrivenRankingProvider, StaticDataDrivenRankingProvider
from ..providers import (
    CallableRawModelProvider,
    ExternalServiceModelProvider,
    RawDataDrivenModelProvider,
    SklearnProbabilityProvider,
    urllib_json_transport,
)
from ..registry import DataDrivenModelRegistry

__all__ = [
    "ActivityAxis",
    "ActivityAxisGate",
    "ActivityAxisGateDecision",
    "ActivityAxisGateStatus",
    "RuntimeActivityAxisCapability",
    "RuntimeActivityAxisCapabilityManifest",
    "default_v0522_activity_axis_manifest",
    "load_activity_axis_manifest",
    "manifest_sha256",
    "adapt_raw_model_output",
    "ranked_class_ids",
    "DataDrivenModelManifest",
    "DataDrivenTaskType",
    "ModelCapability",
    "ModelInferenceRequest",
    "ModelOutputSemantics",
    "ModelRuntime",
    "ProbabilityCalibrationReceipt",
    "ProbabilityCalibrationQualificationReceipt",
    "RawModelOutput",
    "OpenSetAssessment",
    "OpenSetPolicy",
    "OpenSetStatus",
    "assess_open_set",
    "ClassProbability",
    "DataDrivenPrediction",
    "DataDrivenRankingProvider",
    "StaticDataDrivenRankingProvider",
    "CallableRawModelProvider",
    "ExternalServiceModelProvider",
    "RawDataDrivenModelProvider",
    "SklearnProbabilityProvider",
    "urllib_json_transport",
    "DataDrivenModelRegistry",
]
