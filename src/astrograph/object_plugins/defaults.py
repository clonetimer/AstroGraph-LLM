from __future__ import annotations

from astrograph.object_plugins.reaction_wheel import D3ReactionWheelObjectPlugin
from astrograph.object_plugins.thermal_heater import D3ThermalHeaterObjectPlugin
from astrograph.object_plugins.registry import ObjectPluginRegistry


def build_builtin_object_plugin_registry() -> ObjectPluginRegistry:
    registry = ObjectPluginRegistry()
    registry.register(D3ReactionWheelObjectPlugin())
    registry.register(D3ThermalHeaterObjectPlugin())
    return registry
