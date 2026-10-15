from __future__ import annotations

from astrograph.object_plugins.base import ObjectDiagnosticPlugin
from astrograph.object_plugins.models import ObjectPluginManifest, PluginReviewStatus
from astrograph.operators import BUILTIN_OPERATOR_REGISTRY, DeterministicOperatorRegistry


class ObjectPluginRegistry:
    """Explicit allow-list for versioned object plugins.

    Registration validates all referenced numerical operators.  Dynamic import
    from user-supplied paths is intentionally not supported by the online
    runtime; generated plugins must pass review and be registered explicitly.
    """

    def __init__(self, *, operator_registry: DeterministicOperatorRegistry | None = None) -> None:
        self.operator_registry = operator_registry or BUILTIN_OPERATOR_REGISTRY
        self._plugins: dict[str, ObjectDiagnosticPlugin] = {}

    def register(self, plugin: ObjectDiagnosticPlugin) -> None:
        manifest = plugin.manifest
        if manifest.plugin_id in self._plugins:
            raise ValueError(f"object plugin already registered: {manifest.plugin_id}")
        missing = [item for item in manifest.required_operator_ids if not self.operator_registry.has(item)]
        if missing:
            raise ValueError(
                f"object plugin {manifest.plugin_id} references unknown operators: {sorted(missing)}"
            )
        if manifest.review_status == PluginReviewStatus.RETIRED:
            raise ValueError(f"retired object plugin cannot be registered: {manifest.plugin_id}")
        self._plugins[manifest.plugin_id] = plugin

    def get(self, plugin_id: str) -> ObjectDiagnosticPlugin:
        try:
            return self._plugins[plugin_id]
        except KeyError as exc:
            raise KeyError(f"unknown object plugin: {plugin_id}") from exc

    def list_manifests(self) -> list[ObjectPluginManifest]:
        return [self._plugins[key].manifest for key in sorted(self._plugins)]
