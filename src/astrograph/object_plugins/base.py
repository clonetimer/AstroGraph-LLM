from __future__ import annotations

from abc import ABC, abstractmethod

from astrograph.object_plugins.models import (
    ObjectEvidenceBuildRequest,
    ObjectEvidenceBuildResult,
    ObjectPluginManifest,
)


class ObjectDiagnosticPlugin(ABC):
    """Interface implemented by an object-specific deterministic service package."""

    @property
    @abstractmethod
    def manifest(self) -> ObjectPluginManifest: ...

    @abstractmethod
    def build_evidence(self, request: ObjectEvidenceBuildRequest) -> ObjectEvidenceBuildResult: ...
