"""Software-assurance controls for model admission, lineage, and artifacts.

The package contains deterministic controls used by production-capable paths.  It
must remain independent of heavyweight machine-learning runtimes.
"""

from importlib import import_module

from .artifact_paths import ArtifactPathError, ArtifactRoot
from .contract_registry import FeatureContractRegistry
from .feature_admission import (
    AdmissionError,
    AdmissionReceipt,
    AdmissionSigner,
    AdmittedExecution,
    FeatureContract,
    FeatureValueKind,
    InputBinding,
    InputRole,
    ModelAdmissionManifest,
    canonical_sha256,
)
from .temporal_sealed_campaign import (
    CampaignState,
    FaultConditionMapping,
    OneTimeExecutionLedger,
    SealedTemporalCampaignPreregistration,
    TemporalCampaignPrecisionPlan,
    TemporalConfirmatoryAnalysisPlan,
    clopper_pearson_lower_one_sided,
    clopper_pearson_upper_one_sided,
)


_LAZY_EXPORTS = {
    "build_flat_feature_mapping_contract_registry": (
        ".model_admission_helpers",
        "build_flat_feature_mapping_contract_registry",
    ),
    "build_temporal_window_contract_registry": (
        ".model_admission_helpers",
        "build_temporal_window_contract_registry",
    ),
}


def __getattr__(name: str):
    """Load provider-binding helpers only when requested.

    The assurance package is imported by low-level data and evidence modules.
    Eagerly importing model-admission helpers pulled the data-driven registry
    back into that path and made ``astrograph.data_driven`` imports dependent on
    incidental module-import order.  Keeping these two helpers lazy preserves
    the public API while removing the circular initialization edge.
    """

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "AdmissionError",
    "AdmissionReceipt",
    "AdmissionSigner",
    "AdmittedExecution",
    "ArtifactPathError",
    "ArtifactRoot",
    "FeatureContract",
    "FeatureContractRegistry",
    "FeatureValueKind",
    "InputBinding",
    "InputRole",
    "ModelAdmissionManifest",
    "canonical_sha256",
    "build_flat_feature_mapping_contract_registry",
    "build_temporal_window_contract_registry",
    "CampaignState",
    "FaultConditionMapping",
    "OneTimeExecutionLedger",
    "SealedTemporalCampaignPreregistration",
    "TemporalCampaignPrecisionPlan",
    "TemporalConfirmatoryAnalysisPlan",
    "clopper_pearson_lower_one_sided",
    "clopper_pearson_upper_one_sided",
]
