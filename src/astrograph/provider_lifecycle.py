from __future__ import annotations

from enum import Enum
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderQualificationStatus(str, Enum):
    UNQUALIFIED = "UNQUALIFIED"
    PARTIAL = "PARTIAL"
    QUALIFIED = "QUALIFIED"


class ProviderTimeoutCapability(str, Enum):
    NONE = "NONE"
    COOPERATIVE = "COOPERATIVE"
    PROVIDER_BOUNDED = "PROVIDER_BOUNDED"
    PARENT_PROCESS_ENFORCED = "PARENT_PROCESS_ENFORCED"


class ProviderCleanupCapability(str, Enum):
    NONE = "NONE"
    CONTEXT_MANAGED = "CONTEXT_MANAGED"
    RESOURCE_CLOSED = "RESOURCE_CLOSED"
    PROCESS_BOUNDARY_CLEANED = "PROCESS_BOUNDARY_CLEANED"
    PROCESS_GROUP_OR_TREE_CLEANED = "PROCESS_GROUP_OR_TREE_CLEANED"


class ProviderCrashContainment(str, Enum):
    NONE = "NONE"
    EXCEPTION_BOUNDARY = "EXCEPTION_BOUNDARY"
    PROCESS_BOUNDARY = "PROCESS_BOUNDARY"


class ProviderLifecycleProfile(BaseModel):
    """Machine-checkable lifecycle/qualification statement for a provider family.

    The profile deliberately separates bounded I/O/resource cleanup from permission to
    enter Stable parallel execution. A provider can be PARTIAL even when it has a socket
    or subprocess timeout if request-scoped cancellation, descendant cleanup, or replay
    evidence remains incomplete.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "astrograph-provider-lifecycle-profile.v1"
    provider_id: str = Field(min_length=1)
    provider_family: str = Field(min_length=1)
    qualification_status: ProviderQualificationStatus
    timeout_capability: ProviderTimeoutCapability
    cleanup_capability: ProviderCleanupCapability
    crash_containment: ProviderCrashContainment
    stable_parallel_eligible: bool = False
    qualification_id: str | None = None
    platform_scope: str = "portable-contract"
    evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_qualification(self) -> "ProviderLifecycleProfile":
        if self.stable_parallel_eligible:
            if self.qualification_status != ProviderQualificationStatus.QUALIFIED:
                raise ValueError("stable_parallel_eligible requires QUALIFIED status")
            if not self.qualification_id:
                raise ValueError("stable_parallel_eligible requires qualification_id")
            if self.timeout_capability not in {
                ProviderTimeoutCapability.PROVIDER_BOUNDED,
                ProviderTimeoutCapability.PARENT_PROCESS_ENFORCED,
            }:
                raise ValueError("stable parallel provider requires enforced bounded execution")
            if self.crash_containment == ProviderCrashContainment.NONE:
                raise ValueError("stable parallel provider requires crash containment")
        return self


class ProviderQualificationRegistry:
    def __init__(self, profiles: Iterable[ProviderLifecycleProfile] = ()) -> None:
        self._profiles: dict[str, ProviderLifecycleProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: ProviderLifecycleProfile, *, replace: bool = False) -> None:
        if profile.provider_id in self._profiles and not replace:
            raise ValueError(f"provider lifecycle profile already registered: {profile.provider_id}")
        self._profiles[profile.provider_id] = profile

    def get(self, provider_id: str) -> ProviderLifecycleProfile:
        try:
            return self._profiles[provider_id]
        except KeyError as exc:
            raise KeyError(f"unknown provider lifecycle profile: {provider_id}") from exc

    def snapshot(self) -> list[dict]:
        return [
            self._profiles[key].model_dump(mode="json")
            for key in sorted(self._profiles)
        ]

    def stable_parallel_eligible(self, provider_ids: Iterable[str]) -> bool:
        ids = tuple(provider_ids)
        if not ids:
            return True
        return all(self.get(provider_id).stable_parallel_eligible for provider_id in ids)


BUILTIN_PROVIDER_QUALIFICATIONS = ProviderQualificationRegistry(
    (
        ProviderLifecycleProfile(
            provider_id="local-deterministic-isolated.v1",
            provider_family="local deterministic stage",
            qualification_status=ProviderQualificationStatus.QUALIFIED,
            timeout_capability=ProviderTimeoutCapability.PARENT_PROCESS_ENFORCED,
            cleanup_capability=ProviderCleanupCapability.PROCESS_BOUNDARY_CLEANED,
            crash_containment=ProviderCrashContainment.PROCESS_BOUNDARY,
            stable_parallel_eligible=True,
            qualification_id="post8.whole_spacecraft.local-branch.hard-isolation.v1",
            evidence=(
                "post8 hard timeout of non-cooperative worker",
                "post8 crash containment",
                "serial/isolated semantic replay equivalence",
            ),
            limitations=("Qualification applies only to frozen local stage adapters.",),
        ),
        ProviderLifecycleProfile(
            provider_id="external-cli-managed.v1",
            provider_family="optional external Python CLI",
            qualification_status=ProviderQualificationStatus.PARTIAL,
            timeout_capability=ProviderTimeoutCapability.PARENT_PROCESS_ENFORCED,
            cleanup_capability=ProviderCleanupCapability.PROCESS_GROUP_OR_TREE_CLEANED,
            crash_containment=ProviderCrashContainment.PROCESS_BOUNDARY,
            stable_parallel_eligible=False,
            platform_scope="POSIX runtime-qualified; Windows process-tree implementation present but requires native certification",
            evidence=(
                "managed process-group timeout/termination boundary",
                "temporary-directory context cleanup in online audit adapters",
            ),
            limitations=(
                "External tool semantic replay equivalence is provider/version specific.",
                "Windows descendant-process cleanup is not certified by this Linux build environment.",
            ),
        ),
        ProviderLifecycleProfile(
            provider_id="neo4j-readonly.v1",
            provider_family="Neo4j read-only KG backend",
            qualification_status=ProviderQualificationStatus.PARTIAL,
            timeout_capability=ProviderTimeoutCapability.NONE,
            cleanup_capability=ProviderCleanupCapability.RESOURCE_CLOSED,
            crash_containment=ProviderCrashContainment.EXCEPTION_BOUNDARY,
            stable_parallel_eligible=False,
            evidence=("driver.close() in finally", "read-only query compiler/executor"),
            limitations=("No reviewed query-level deadline contract is currently enforced." ,),
        ),
        ProviderLifecycleProfile(
            provider_id="kg-csv-reference.v1",
            provider_family="versioned CSV KG fallback",
            qualification_status=ProviderQualificationStatus.QUALIFIED,
            timeout_capability=ProviderTimeoutCapability.PARENT_PROCESS_ENFORCED,
            cleanup_capability=ProviderCleanupCapability.CONTEXT_MANAGED,
            crash_containment=ProviderCrashContainment.PROCESS_BOUNDARY,
            stable_parallel_eligible=False,
            evidence=("deterministic local file replay", "post8 isolated worker boundary available"),
            limitations=("KG stage can dynamically select Neo4j, so the stage as a whole is not promoted." ,),
        ),
        ProviderLifecycleProfile(
            provider_id="http-json-provider.v1",
            provider_family="urllib HTTP JSON provider",
            qualification_status=ProviderQualificationStatus.PARTIAL,
            timeout_capability=ProviderTimeoutCapability.PROVIDER_BOUNDED,
            cleanup_capability=ProviderCleanupCapability.CONTEXT_MANAGED,
            crash_containment=ProviderCrashContainment.EXCEPTION_BOUNDARY,
            stable_parallel_eligible=False,
            evidence=("urllib request timeout", "response context manager cleanup"),
            limitations=(
                "No request-scoped hard cancellation before the socket/provider timeout.",
                "Remote provider resource cleanup is outside AstroGraph control.",
            ),
        ),
        ProviderLifecycleProfile(
            provider_id="inprocess-model-runtime.v1",
            provider_family="in-process sklearn/PyTorch/TensorFlow/ONNX/custom model",
            qualification_status=ProviderQualificationStatus.UNQUALIFIED,
            timeout_capability=ProviderTimeoutCapability.NONE,
            cleanup_capability=ProviderCleanupCapability.NONE,
            crash_containment=ProviderCrashContainment.EXCEPTION_BOUNDARY,
            stable_parallel_eligible=False,
            limitations=(
                "GPU/accelerator memory lifecycle is provider specific.",
                "Opaque callables cannot be hard-cancelled in-process.",
            ),
        ),
    )
)
