from __future__ import annotations

from pathlib import Path

from astrograph.plugin_factory.models import ObjectPluginBuildSpec
from astrograph.plugin_factory.planner import PluginGenerationPlanner
from astrograph.plugin_factory.scaffold import ObjectPluginScaffolder
from astrograph.plugin_factory.validator import GeneratedPluginValidator


class ObjectPluginFactoryService:
    """Deterministic boundary behind future object-modeling/code-generation Agents."""

    def __init__(self) -> None:
        self.planner = PluginGenerationPlanner()
        self.scaffolder = ObjectPluginScaffolder()
        self.validator = GeneratedPluginValidator()

    def generate(self, spec: ObjectPluginBuildSpec, output_root: str | Path) -> dict[str, object]:
        plan = self.planner.plan(spec)
        package_root = self.scaffolder.scaffold(spec, plan, output_root)
        validation = self.validator.validate(package_root)
        return {
            "package_root": str(package_root),
            "plan": plan.model_dump(mode="json"),
            "validation": validation,
        }
