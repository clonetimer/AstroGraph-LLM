from __future__ import annotations

from typing import Any

from .models import BiRRFConfig, CandidateFusionPackage


class OpenSetFusionPolicy:
    """Translate an upstream open-set assessment into fusion-only effects."""

    suppressed_action = "DATA_BRANCH_SUPPRESSED_BY_OPEN_SET_POLICY"

    @staticmethod
    def effective_config(config: BiRRFConfig, assessment: Any | None) -> BiRRFConfig:
        if assessment is not None and not assessment.data_branch_eligible_for_fusion:
            return config.model_copy(update={"data_weight": 0.0})
        return config

    @classmethod
    def apply_assessment(
        cls,
        package: CandidateFusionPackage,
        assessment: Any | None,
    ) -> CandidateFusionPackage:
        if assessment is None:
            return package
        assessment_payload = assessment.model_dump(mode="json")
        fusion_metadata = dict(package.metadata)
        fusion_metadata["open_set_assessment"] = assessment_payload
        fused = package.birrf_result
        legacy_flags = list(fused.conflict_flags)
        actions = list(fused.open_set_actions)
        if not assessment.data_branch_eligible_for_fusion:
            actions.append(cls.suppressed_action)
            legacy_flags.append(cls.suppressed_action)
        fused_metadata = dict(fused.metadata)
        fused_metadata["open_set_assessment"] = assessment_payload
        return package.model_copy(
            update={
                "metadata": fusion_metadata,
                "birrf_result": fused.model_copy(
                    update={
                        "conflict_flags": list(dict.fromkeys(legacy_flags)),
                        "open_set_actions": list(dict.fromkeys(actions)),
                        "metadata": fused_metadata,
                    }
                ),
            }
        )
