# Trusted Agent Tool Plugins

AstroGraph exposes an optional administrator-owned extension point for Mission Agent tools. The runtime is deliberately narrower than a general Python package manager: it loads only reviewed `.py` files from one trusted root and never treats arbitrary uploaded code as a sandboxed tool.

## Design adopted for v0.7

The v0.7 lifecycle was revised after reviewing the uploaded **智能诊断算法评估软件** plugin runtime. The useful patterns retained are:

- explicit contribution point and activation event metadata;
- one host runtime that owns discovery, install/update, capability snapshots and uninstall;
- `importlib.invalidate_caches()` / current-source execution semantics rather than relying on stale timestamp bytecode caches;
- validate implementation and contract before the installed copy becomes active;
- managed-install manifest so only host-managed files can be automatically deleted;
- update rollback: a malformed update does not replace the last healthy version;
- hot reload updates future registry snapshots instead of mutating an already running task;
- built-in AstroGraph tools remain outside the managed plugin uninstall path.

AstroGraph keeps its stricter existing governance boundaries: dynamic tools execute only as `ISOLATED_PROCESS` worker targets pinned to the plugin file SHA-256, and normal ToolSpec permission / schema / timeout / audit checks still apply.

## Security boundary

1. Plugins are disabled by default.
2. The loader scans only `ASTROGRAPH_AGENT_TOOL_PLUGIN_DIR`.
3. A plugin must implement `AgentToolPlugin` and return `AgentToolPluginManifest`.
4. Each exported `ToolSpec.tool_id` must begin with `<plugin_id>.`.
5. Only `APPROVED` plugins can enter future Mission Agent task registries.
6. Dynamic tools must declare `ISOLATED_PROCESS`.
7. The worker target is SHA-256 pinned to the exact reviewed source file.
8. Running/sealed `ToolRegistry` instances are immutable; reload affects future task snapshots only.
9. This is **not an untrusted Python sandbox**. Administrators must review plugin source before installation.

## Enable

```bash
export ASTROGRAPH_AGENT_TOOL_PLUGINS_ENABLED=1
export ASTROGRAPH_AGENT_TOOL_PLUGIN_DIR=/srv/astrograph/reviewed-agent-tools
```

Windows PowerShell example:

```powershell
$env:ASTROGRAPH_AGENT_TOOL_PLUGINS_ENABLED = "1"
$env:ASTROGRAPH_AGENT_TOOL_PLUGIN_DIR = "C:\\AstroGraph\\reviewed-agent-tools"
```

The directory can be empty initially. A managed install will create it when required.

## Base class and manifest

See `examples/agent_tool_plugin_example.py`.

A module exports either:

```python
PLUGIN = MyPlugin()
```

or:

```python
def create_plugin():
    return MyPlugin()
```

`AgentToolPluginManifest` includes the plugin identity/version, review status, source references, allowed tasks/work modes, dependencies, optional config schema, contribution point (`agent.tools`) and activation events (`on:task-run`).

## Managed install / update

Administrator API:

```http
POST /v1/platform/tool-plugins/install
Content-Type: application/json

{
  "filename": "my_tools.py",
  "source": "...python source..."
}
```

The host lifecycle is:

```text
source received
  -> syntax compile
  -> temporary current-source load
  -> BaseClass / manifest / ToolSpec contract validation
  -> dependency check
  -> commit file
  -> SHA-256 + managed-install manifest
  -> discovery / reload
  -> active for future task registries
```

If validation or activation fails, the new file is removed or the previous file is restored, then the previous healthy runtime is reloaded. The GUI reports the failure and whether the previous version was retained.

## Hot reload

```http
POST /v1/platform/tool-plugins/reload
```

Hot reload executes current source bytes using digest-based module identity. Same-second edits therefore do not rely on `.pyc` timestamp invalidation.

Reload is per-plugin transactional: when an edited plugin is invalid, the current reload reports an error and keeps the previous healthy in-memory version for future snapshots. Deleted files disappear from future snapshots.

**Running tasks are not hot-mutated.** This is intentional; a task should keep a stable tool set for audit and reproducibility.

## Managed uninstall

Preview first:

```http
DELETE /v1/platform/tool-plugins/<plugin_id>
```

Then commit:

```http
DELETE /v1/platform/tool-plugins/<plugin_id>?force=true
```

Only plugins recorded in `.astrograph_plugin_installations.json` can be removed automatically. A plugin that merely exists in the administrator directory is treated as externally managed and will not be deleted by AstroGraph.

## GUI

`Agent 中心 -> 插件管理` provides:

- runtime/reload generation;
- installed plugin list, versions, review status and contributed tools;
- reload errors and rollback status;
- `.py` managed install/update;
- preview + confirmation uninstall;
- explicit hot reload.

The UI does not imply that every registered plugin is automatically invoked by the upload-to-diagnosis adapter. A tool becomes executable only when it is both permitted in the selected task/mode and actually selected by a reviewed plan/workflow.
