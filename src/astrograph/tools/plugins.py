"""Governed importlib runtime for optional Agent tool plugins.

Design goals:
- administrator-owned trusted root only; never an arbitrary untrusted-code sandbox;
- explicit BaseClass + manifest contract;
- direct source compilation and digest-based module identity for deterministic hot reload;
- per-plugin contract validation before activation;
- managed-install manifest, atomic update and rollback on failed updates;
- running/sealed task registries are immutable; reload affects future registry snapshots only;
- dynamic tool execution remains isolated-process + SHA-256 pinned.

The install/reload lifecycle intentionally mirrors mature plugin-manager patterns used by
engineering evaluation software: descriptor/contract validation, immediate discovery,
rollback on failed update, protected built-ins, and a capability snapshot consumable by
both GUI and headless clients.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astrograph.tools.executor import validate_tool_definition
from astrograph.tools.registry import ToolDefinition, ToolRegistry, ToolSpec, WorkerTarget


WorkMode = Literal["normal", "thinking", "creative"]
TaskKind = Literal["diagnose", "research", "build_knowledge", "train_model", "infer_candidate", "replay"]

_INSTALL_MANIFEST = ".astrograph_plugin_installations.json"
_PLUGIN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]+$")


class AgentToolPluginManifest(BaseModel):
    """Reviewable metadata for one trusted Agent-tool plugin package."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["astrograph-agent-tool-plugin.v1"] = "astrograph-agent-tool-plugin.v1"
    plugin_id: str = Field(min_length=3, pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    plugin_version: str = Field(min_length=1)
    display_name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1200)
    allowed_tasks: list[TaskKind] = Field(default_factory=lambda: ["diagnose"])
    allowed_work_modes: list[WorkMode] = Field(default_factory=lambda: ["normal", "thinking", "creative"])
    review_status: Literal["DEVELOPMENT", "APPROVED", "RETIRED"] = "DEVELOPMENT"
    source_refs: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    config_schema: dict[str, Any] = Field(default_factory=dict)
    contribution_point: Literal["agent.tools"] = "agent.tools"
    activation_events: list[str] = Field(default_factory=lambda: ["on:task-run"])

    @model_validator(mode="after")
    def validate_manifest(self) -> "AgentToolPluginManifest":
        if len(self.allowed_tasks) != len(set(self.allowed_tasks)):
            raise ValueError("allowed_tasks must be unique")
        if len(self.allowed_work_modes) != len(set(self.allowed_work_modes)):
            raise ValueError("allowed_work_modes must be unique")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError("dependencies must be unique")
        if len(self.activation_events) != len(set(self.activation_events)):
            raise ValueError("activation_events must be unique")
        if self.review_status == "APPROVED" and not self.source_refs:
            raise ValueError("approved Agent tool plugin requires source_refs")
        return self


@dataclass(frozen=True)
class AgentToolExport:
    """One isolated worker function exposed by a plugin."""

    spec: ToolSpec
    function_name: str
    display_name: str = ""
    purpose: str = ""


class AgentToolPlugin(ABC):
    """Base class implemented by trusted, versioned Agent-tool plugins."""

    @property
    @abstractmethod
    def manifest(self) -> AgentToolPluginManifest: ...

    @abstractmethod
    def exports(self) -> list[AgentToolExport]: ...


@dataclass(frozen=True)
class LoadedAgentToolPlugin:
    path: Path
    sha256: str
    plugin: AgentToolPlugin
    definitions: tuple[ToolDefinition, ...]
    exports: tuple[AgentToolExport, ...]
    missing_dependencies: tuple[str, ...] = ()
    managed_install: bool = False


class TrustedAgentToolPluginManager:
    """Import, validate, install and snapshot trusted plugins from one local root."""

    def __init__(self, root: str | Path | None = None, *, enabled: bool | None = None) -> None:
        configured = str(root) if root is not None else os.environ.get("ASTROGRAPH_AGENT_TOOL_PLUGIN_DIR", "")
        self.root = Path(configured).expanduser().resolve() if configured else None
        if enabled is None:
            enabled = os.environ.get("ASTROGRAPH_AGENT_TOOL_PLUGINS_ENABLED", "").strip().lower() in {
                "1", "true", "yes", "on"
            }
        self.enabled = bool(enabled and self.root is not None)
        self._plugins: dict[str, LoadedAgentToolPlugin] = {}
        self._errors: list[dict[str, Any]] = []
        self._generation = 0

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @property
    def _manifest_path(self) -> Path | None:
        return self.root / _INSTALL_MANIFEST if self.root else None

    def _load_install_manifest(self) -> dict[str, Any]:
        path = self._manifest_path
        if path is None or not path.exists():
            return {"version": 1, "plugins": {}}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "plugins": {}}
        plugins = data.get("plugins") if isinstance(data, dict) else None
        return {"version": 1, "plugins": plugins if isinstance(plugins, dict) else {}}

    def _save_install_manifest(self, data: dict[str, Any]) -> None:
        path = self._manifest_path
        if path is None:
            raise RuntimeError("trusted plugin root is not configured")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _record_managed(self, plugin_id: str, path: Path, digest: str) -> None:
        data = self._load_install_manifest()
        data["plugins"][plugin_id] = {
            "plugin_id": plugin_id,
            "file": path.name,
            "sha256": digest,
        }
        self._save_install_manifest(data)

    def _remove_managed(self, plugin_id: str) -> dict[str, Any] | None:
        data = self._load_install_manifest()
        entry = data["plugins"].pop(plugin_id, None)
        self._save_install_manifest(data)
        return dict(entry) if isinstance(entry, dict) else None

    def _candidate_paths(self) -> list[Path]:
        if not self.enabled or self.root is None or not self.root.is_dir():
            return []
        return sorted(
            path.resolve()
            for path in self.root.glob("*.py")
            if path.is_file() and not path.name.startswith("_") and path.resolve().parent == self.root
        )

    @staticmethod
    def _dependency_health(names: list[str]) -> tuple[str, ...]:
        missing: list[str] = []
        for name in names:
            module_name = str(name).strip()
            if not module_name:
                continue
            try:
                if importlib.util.find_spec(module_name) is None:
                    missing.append(module_name)
            except Exception:
                missing.append(module_name)
        return tuple(sorted(set(missing)))

    def _load_one(self, path: Path, *, managed_install: bool | None = None) -> LoadedAgentToolPlugin:
        if self.root is None or path.resolve().parent != self.root:
            raise PermissionError("plugin path is outside the trusted plugin root")
        importlib.invalidate_caches()
        source = path.read_bytes()
        # Compile current source bytes before import. This catches syntax failures and
        # bypasses timestamp-based bytecode cache ambiguity during same-second edits.
        compile(source, str(path), "exec")
        digest = hashlib.sha256(source).hexdigest()
        safe_stem = re.sub(r"[^A-Za-z0-9_]", "_", path.stem)
        module_name = f"astrograph_trusted_tool_plugin_{safe_stem}_{digest[:16]}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError("unable to create importlib spec")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            code = compile(source, str(path), "exec")
            exec(code, module.__dict__)
        finally:
            sys.modules.pop(module_name, None)
        plugin = getattr(module, "PLUGIN", None)
        if plugin is None and callable(getattr(module, "create_plugin", None)):
            plugin = module.create_plugin()
        if not isinstance(plugin, AgentToolPlugin):
            raise TypeError("plugin module must export PLUGIN or create_plugin() returning AgentToolPlugin")
        manifest = plugin.manifest
        missing_dependencies = self._dependency_health(manifest.dependencies)
        if missing_dependencies:
            raise ImportError("missing plugin dependencies: " + ", ".join(missing_dependencies))
        exports = tuple(plugin.exports())
        if not exports:
            raise ValueError("plugin exports no tools")
        definitions: list[ToolDefinition] = []
        seen: set[str] = set()
        for export in exports:
            tool_id = export.spec.tool_id
            if tool_id in seen:
                raise ValueError(f"duplicate tool_id in plugin: {tool_id}")
            seen.add(tool_id)
            prefix = manifest.plugin_id + "."
            if not tool_id.startswith(prefix):
                raise ValueError(f"plugin tool_id must start with {prefix!r}: {tool_id}")
            if export.spec.execution_mode != "ISOLATED_PROCESS":
                raise ValueError("dynamic Agent tool plugins must use ISOLATED_PROCESS")
            definition = ToolDefinition(
                spec=export.spec,
                handler=None,
                worker_target=WorkerTarget(
                    module_file=str(path), function_name=export.function_name, artifact_sha256=digest
                ),
            )
            validate_tool_definition(definition)
            definitions.append(definition)
        if managed_install is None:
            managed = self._load_install_manifest().get("plugins", {}).get(manifest.plugin_id)
            managed_install = bool(isinstance(managed, dict) and managed.get("file") == path.name)
        return LoadedAgentToolPlugin(
            path=path,
            sha256=digest,
            plugin=plugin,
            definitions=tuple(definitions),
            exports=exports,
            missing_dependencies=missing_dependencies,
            managed_install=bool(managed_install),
        )

    def reload(self) -> dict[str, Any]:
        """Transactional hot reload. Running sealed registries are untouched.

        Each candidate is validated before it replaces the prior version. If an edited
        plugin fails validation, the previous healthy in-memory version is retained and
        the error is surfaced. Deleted files are removed from future snapshots.
        """
        previous = dict(self._plugins)
        next_plugins: dict[str, LoadedAgentToolPlugin] = {}
        errors: list[dict[str, Any]] = []
        if not self.enabled:
            self._plugins = {}
            self._errors = []
            self._generation += 1
            return self.status()
        if self.root is None or not self.root.is_dir():
            self._errors = [{"file": "", "error": "configured plugin directory does not exist"}]
            self._generation += 1
            return self.status()

        previous_by_file = {loaded.path.name: loaded for loaded in previous.values()}
        for path in self._candidate_paths():
            try:
                loaded = self._load_one(path)
                plugin_id = loaded.plugin.manifest.plugin_id
                if plugin_id in next_plugins:
                    raise ValueError(f"duplicate plugin_id: {plugin_id}")
                next_plugins[plugin_id] = loaded
            except Exception as exc:
                retained = previous_by_file.get(path.name)
                if retained is not None:
                    next_plugins[retained.plugin.manifest.plugin_id] = retained
                errors.append({
                    "file": path.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "retained_previous": retained is not None,
                })
        self._plugins = next_plugins
        self._errors = errors
        self._generation += 1
        return self.status()

    def install_source(self, *, filename: str, source: str) -> dict[str, Any]:
        """Install/update a reviewed Python plugin with validate-before-commit rollback."""
        if not self.enabled or self.root is None:
            raise RuntimeError("trusted plugin runtime is not enabled")
        name = Path(filename).name
        if name != filename or not re.fullmatch(r"[A-Za-z0-9_.-]+\.py", name) or name.startswith("_"):
            raise ValueError("plugin filename must be a simple .py filename")
        if not source.strip():
            raise ValueError("plugin source must not be empty")
        self.root.mkdir(parents=True, exist_ok=True)
        target = (self.root / name).resolve()
        if target.parent != self.root:
            raise PermissionError("plugin target is outside trusted root")
        compile(source.encode("utf-8"), name, "exec")
        prior = target.read_bytes() if target.exists() else None
        tmp = (self.root / f"_installing_{re.sub(r'[^A-Za-z0-9_.-]', '_', target.stem)}.py").resolve()
        tmp.write_text(source, encoding="utf-8")
        try:
            candidate = self._load_one(tmp, managed_install=True)
            plugin_id = candidate.plugin.manifest.plugin_id
            if prior is not None:
                prior_loaded = self._load_one(target)
                prior_id = prior_loaded.plugin.manifest.plugin_id
                if prior_id != plugin_id:
                    raise ValueError(f"managed update cannot change plugin_id: {prior_id} -> {plugin_id}")
            if not _PLUGIN_ID_RE.fullmatch(plugin_id):
                raise ValueError("invalid plugin_id")
            # Commit only after the temporary source has fully passed the contract.
            tmp.replace(target)
            committed = self._load_one(target, managed_install=True)
            self._record_managed(plugin_id, target, committed.sha256)
            # Reload all candidates to detect collisions, retaining any prior healthy
            # versions if another plugin is malformed.
            status = self.reload()
            active = self._plugins.get(plugin_id)
            if active is None or active.sha256 != committed.sha256:
                raise RuntimeError("installed plugin did not become the active future snapshot")
            return {
                "status": "SUCCEEDED",
                "plugin_id": plugin_id,
                "plugin_version": active.plugin.manifest.plugin_version,
                "artifact_sha256": active.sha256,
                "managed_install": True,
                "reload_generation": status["reload_generation"],
                "reload_scope": status["reload_scope"],
            }
        except Exception:
            tmp.unlink(missing_ok=True)
            if prior is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(prior)
            self.reload()
            raise

    def uninstall(self, plugin_id: str, *, force: bool = False) -> dict[str, Any]:
        """Preview or remove a managed plugin; built-in tools are not part of this runtime."""
        self.ensure_loaded()
        plugin_id = str(plugin_id or "").strip()
        loaded = self._plugins.get(plugin_id)
        if loaded is None:
            raise FileNotFoundError(f"unknown plugin: {plugin_id}")
        manifest = self._load_install_manifest()
        entry = manifest.get("plugins", {}).get(plugin_id)
        if not isinstance(entry, dict):
            raise PermissionError("plugin was not installed by AstroGraph and will not be deleted automatically")
        preview = {
            "status": "PREVIEW" if not force else "SUCCEEDED",
            "plugin_id": plugin_id,
            "file": loaded.path.name,
            "tool_ids": [item.spec.tool_id for item in loaded.definitions],
            "managed_install": True,
            "running_registries_affected": False,
            "future_registry_snapshots_affected": True,
        }
        if not force:
            return preview
        loaded.path.unlink(missing_ok=True)
        self._remove_managed(plugin_id)
        status = self.reload()
        return {**preview, "reload_generation": status["reload_generation"]}

    def ensure_loaded(self) -> None:
        if self.enabled and not self._plugins and not self._errors:
            self.reload()

    def register_for_task(self, registry: ToolRegistry, *, task: str, work_mode: str) -> list[str]:
        """Register APPROVED matching tools into a not-yet-sealed task registry."""
        self.ensure_loaded()
        registered: list[str] = []
        for loaded in self._plugins.values():
            manifest = loaded.plugin.manifest
            if manifest.review_status != "APPROVED":
                continue
            if task not in manifest.allowed_tasks or work_mode not in manifest.allowed_work_modes:
                continue
            for definition in loaded.definitions:
                registry.register(definition)
                registered.append(definition.spec.tool_id)
        return sorted(registered)

    def is_allowed(self, tool_id: str, *, task: str, work_mode: str) -> bool:
        self.ensure_loaded()
        for loaded in self._plugins.values():
            manifest = loaded.plugin.manifest
            if manifest.review_status != "APPROVED":
                continue
            if task not in manifest.allowed_tasks or work_mode not in manifest.allowed_work_modes:
                continue
            if tool_id in {item.spec.tool_id for item in loaded.definitions}:
                return True
        return False

    def catalog(self) -> list[dict[str, Any]]:
        self.ensure_loaded()
        rows: list[dict[str, Any]] = []
        for plugin_id, loaded in sorted(self._plugins.items()):
            manifest = loaded.plugin.manifest
            export_by_id = {item.spec.tool_id: item for item in loaded.exports}
            for definition in loaded.definitions:
                export = export_by_id[definition.spec.tool_id]
                rows.append({
                    "source": "trusted_plugin",
                    "plugin_id": plugin_id,
                    "plugin_version": manifest.plugin_version,
                    "plugin_display_name": manifest.display_name,
                    "plugin_review_status": manifest.review_status,
                    "plugin_managed_install": loaded.managed_install,
                    "plugin_dependencies": manifest.dependencies,
                    "plugin_config_schema": manifest.config_schema,
                    "contribution_point": manifest.contribution_point,
                    "activation_events": manifest.activation_events,
                    "tool_id": definition.spec.tool_id,
                    "display_name": export.display_name or definition.spec.tool_id,
                    "purpose": export.purpose or manifest.description,
                    "allowed_tasks": manifest.allowed_tasks,
                    "allowed_work_modes": manifest.allowed_work_modes,
                    "spec": definition.spec.model_dump(mode="json"),
                    "implementation": {
                        "module_file": loaded.path.name,
                        "function_name": definition.worker_target.function_name if definition.worker_target else None,
                        "artifact_sha256": loaded.sha256,
                        "execution": "isolated process / SHA-256 pinned",
                    },
                })
        return rows

    def status(self) -> dict[str, Any]:
        self.ensure_loaded() if self.enabled and self._generation == 0 and not self._plugins and not self._errors else None
        return {
            "schema_version": "astrograph-agent-tool-plugin-status.v2",
            "enabled": self.enabled,
            "trusted_root_configured": self.root is not None,
            "trusted_root": str(self.root) if self.root else None,
            "loaded_plugin_count": len(self._plugins),
            "loaded_tool_count": sum(len(item.definitions) for item in self._plugins.values()),
            "reload_generation": self._generation,
            "contribution_point": "agent.tools",
            "activation_events": ["on:task-run"],
            "plugins": [
                {
                    **loaded.plugin.manifest.model_dump(mode="json"),
                    "module_file": loaded.path.name,
                    "artifact_sha256": loaded.sha256,
                    "tool_ids": [definition.spec.tool_id for definition in loaded.definitions],
                    "managed_install": loaded.managed_install,
                    "healthy": not loaded.missing_dependencies,
                    "missing_dependencies": list(loaded.missing_dependencies),
                }
                for _, loaded in sorted(self._plugins.items())
            ],
            "errors": list(self._errors),
            "reload_scope": "FUTURE_REGISTRY_SNAPSHOTS_ONLY",
            "dynamic_execution_mode": "ISOLATED_PROCESS_SHA256_PINNED",
            "failed_update_policy": "RETAIN_PREVIOUS_HEALTHY_VERSION",
            "managed_install_policy": "ATOMIC_VALIDATE_THEN_COMMIT_WITH_ROLLBACK",
        }


_DEFAULT_MANAGER: TrustedAgentToolPluginManager | None = None
_DEFAULT_KEY: tuple[str, str] | None = None


def get_trusted_agent_tool_plugin_manager() -> TrustedAgentToolPluginManager:
    """Environment-keyed singleton so tests/runtime configuration can be refreshed safely."""
    global _DEFAULT_MANAGER, _DEFAULT_KEY
    key = (
        os.environ.get("ASTROGRAPH_AGENT_TOOL_PLUGIN_DIR", ""),
        os.environ.get("ASTROGRAPH_AGENT_TOOL_PLUGINS_ENABLED", ""),
    )
    if _DEFAULT_MANAGER is None or _DEFAULT_KEY != key:
        _DEFAULT_MANAGER = TrustedAgentToolPluginManager()
        _DEFAULT_KEY = key
    return _DEFAULT_MANAGER

# ---------------------------------------------------------------------------
# Legacy governed-descriptor compatibility
# ---------------------------------------------------------------------------
# The candidate dual-source DAG and its implementation acceptance tests predate
# AgentToolPlugin.  v0.7 initially replaced this descriptor API, which made those
# opt-in orchestration modules unimportable even though the new runtime worked.
# Keep both contracts: descriptor snapshots for the research/candidate DAG and
# BaseClass/importlib plugins for the operator Agent runtime.

class PluginSpec(BaseModel):
    """Immutable descriptor for one reviewed isolated-process tool artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["astrograph-plugin-spec.v1"] = "astrograph-plugin-spec.v1"
    plugin_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    module_file: str = Field(min_length=1)
    function_name: str = Field(min_length=1)
    artifact_sha256: str = Field(min_length=64, max_length=64)
    tool: ToolSpec

    def approval_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class PluginSnapshot:
    """Sealed registry snapshot consumed by opt-in candidate orchestration."""

    snapshot_sha256: str
    tool_versions: dict[str, str]
    registry: ToolRegistry


def discover_plugin_specs(root: str | Path) -> list[PluginSpec]:
    """Read descriptor JSON only; plugin Python source is deliberately not imported."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return []
    out: list[PluginSpec] = []
    for path in sorted(base.glob("*.plugin.json")):
        if not path.is_file() or path.resolve().parent != base:
            continue
        out.append(PluginSpec.model_validate_json(path.read_text(encoding="utf-8")))
    return out


def load_approved_snapshot(
    root: str | Path,
    specs: list[PluginSpec] | tuple[PluginSpec, ...],
    *,
    approved_hashes: frozenset[str] | set[str],
    approved_manifest_hashes: frozenset[str] | set[str],
) -> PluginSnapshot:
    """Build a sealed registry from explicitly approved artifact + descriptor hashes.

    This is intentionally separate from TrustedAgentToolPluginManager: descriptor
    snapshots are deterministic release inputs, while hot-loaded Agent plugins are
    administrator-managed future-task extensions.
    """
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"plugin root does not exist: {base}")
    registry = ToolRegistry()
    manifest_rows: list[dict[str, Any]] = []
    seen_plugins: set[str] = set()
    seen_tools: set[str] = set()
    for spec in specs:
        if spec.plugin_id in seen_plugins:
            raise ValueError(f"duplicate plugin_id: {spec.plugin_id}")
        seen_plugins.add(spec.plugin_id)
        if spec.tool.tool_id in seen_tools:
            raise ValueError(f"duplicate tool_id: {spec.tool.tool_id}")
        seen_tools.add(spec.tool.tool_id)
        candidate = (base / spec.module_file).resolve()
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise ValueError("plugin module_file must remain inside plugin root") from exc
        if candidate.parent != base:
            raise ValueError("plugin module_file must name a file inside plugin root")
        if spec.artifact_sha256 not in approved_hashes:
            raise PermissionError(f"plugin artifact is not approved: {spec.plugin_id}")
        descriptor_hash = spec.approval_sha256()
        if descriptor_hash not in approved_manifest_hashes:
            raise PermissionError(f"plugin descriptor is not approved: {spec.plugin_id}")
        if not candidate.is_file():
            raise ValueError("plugin module_file must name a file inside plugin root")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != spec.artifact_sha256:
            raise PermissionError(f"plugin artifact hash mismatch: {spec.plugin_id}")
        if spec.tool.execution_mode != "ISOLATED_PROCESS":
            raise ValueError("approved plugin descriptors must use ISOLATED_PROCESS")
        definition = ToolDefinition(
            spec=spec.tool,
            handler=None,
            worker_target=WorkerTarget(
                module_file=str(candidate),
                function_name=spec.function_name,
                artifact_sha256=spec.artifact_sha256,
            ),
        )
        validate_tool_definition(definition)
        registry.register(definition)
        manifest_rows.append({
            "plugin_id": spec.plugin_id,
            "version": spec.version,
            "artifact_sha256": spec.artifact_sha256,
            "descriptor_sha256": descriptor_hash,
            "tool_id": spec.tool.tool_id,
            "tool_version": spec.tool.tool_version,
        })
    registry.seal()
    canonical = json.dumps(sorted(manifest_rows, key=lambda x: (x["plugin_id"], x["tool_id"])), sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return PluginSnapshot(
        snapshot_sha256=digest,
        tool_versions={item.tool_id: item.tool_version for item in registry.list_specs()},
        registry=registry,
    )
