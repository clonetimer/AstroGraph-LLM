"""Generated deterministic feature-service scaffold for generated.whole_spacecraft.thermal_heater.

Direct and DAG-composed registry-backed features are executable after generation. Online plugin
registration remains disabled until the object data adapter, evidence tests,
and approval record are complete. Custom formulas are never executed here.
"""

from pathlib import Path

from astrograph.plugin_factory import ConfigDrivenFeatureService, ObjectPluginBuildSpec

PLUGIN_ID = 'generated.whole_spacecraft.thermal_heater'
PLUGIN_VERSION = '0.5.13'
ONLINE_REGISTRATION_ALLOWED = False
REQUIRED_OPERATOR_IDS = ['JOINT_THRESHOLD_FRACTION', 'MAX_ABS_PAIRED_RESIDUAL', 'MEAN_VALUE']

_ROOT = Path(__file__).resolve().parent
_SPEC = ObjectPluginBuildSpec.model_validate_json(
    (_ROOT / "build_spec.json").read_text(encoding="utf-8")
)
_FEATURES = {item.feature_id: item for item in _SPEC.features}
_SERVICE = ConfigDrivenFeatureService()


def compute_feature(feature_id, *, current_signals, baseline_signals=None, time_s=None):
    """Execute one approved direct or DAG-composed feature deterministically."""
    try:
        feature = _FEATURES[feature_id]
    except KeyError as exc:
        raise KeyError(f"unknown generated feature: {feature_id}") from exc
    if feature.custom_formula is not None:
        raise RuntimeError(
            f"feature {feature_id} requires reviewed custom code and cannot execute "
            "in the configuration runtime"
        )
    return _SERVICE.compute(
        feature,
        current_signals=current_signals,
        baseline_signals=baseline_signals,
        time_s=time_s,
    ).model_dump(mode="json")


def register(registry):
    raise RuntimeError(
        "Generated feature service is not an approved online object plugin. "
        "Complete the object adapter, evidence tests, and approval record first."
    )
