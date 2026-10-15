import hashlib
import json
from pathlib import Path
import threading
import time
import pytest
from astrograph.tools.registry import ToolRegistry, ToolDefinition, ToolSpec, WorkerTarget
from astrograph.tools.executor import ToolExecutor, ToolExecutionError
from astrograph.tools.plugins import PluginSpec, load_approved_snapshot, discover_plugin_specs
from astrograph.orchestration.models import PermissionScope, WorkflowContext


def context(**changes):
    data=dict(workflow_id="tests",run_id="run",permissions=[PermissionScope.READ_DATA]);data.update(changes)
    return WorkflowContext(**data)

SCHEMA={"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":False}

def test_registry_checks_input_output_and_keeps_private_copy():
    registry=ToolRegistry()
    registry.register(ToolDefinition(ToolSpec(tool_id="echo",input_schema=SCHEMA,output_schema=SCHEMA),lambda x:x))
    payload={"x":2};assert registry.invoke("echo",payload,context())==payload
    with pytest.raises(ToolExecutionError,match="INPUT_SCHEMA"):registry.invoke("echo",{"x":"bad"},context())
    registry.register(ToolDefinition(ToolSpec(tool_id="bad",output_schema=SCHEMA),lambda x:{"x":"bad"}))
    with pytest.raises(ToolExecutionError,match="OUTPUT_SCHEMA"):registry.invoke("bad",{},context())
    registry.get("echo").spec.input_schema.clear()
    with pytest.raises(ToolExecutionError):registry.invoke("echo",{"x":"still bad"},context())
    assert registry.executor.receipts()[-1]["status"]=="FAILED"


def test_permissions_and_version_pin_checked_before_handler():
    called=[];registry=ToolRegistry()
    registry.register(ToolDefinition(ToolSpec(tool_id="read",required_permissions=[PermissionScope.READ_DATA]),lambda x:called.append(1) or {}))
    with pytest.raises(PermissionError):registry.invoke("read",{},context(permissions=[]))
    with pytest.raises(ToolExecutionError,match="VERSION"):registry.invoke("read",{},context(metadata={"tool_versions":{"read":"wrong"}}))
    assert called==[]


def test_deployment_permissions_cannot_be_expanded_by_task():
    registry=ToolRegistry(executor=ToolExecutor(deployment_permissions=frozenset()))
    registry.register(ToolDefinition(ToolSpec(tool_id="read",required_permissions=[PermissionScope.READ_DATA]),lambda x:{}))
    with pytest.raises(PermissionError):registry.invoke("read",{},context())


def test_inprocess_deadline_is_detected_but_not_claimed_killable():
    registry=ToolRegistry();registry.register(ToolDefinition(ToolSpec(tool_id="slow",timeout_s=.001),lambda x:time.sleep(.01) or {}))
    with pytest.raises(ToolExecutionError,match="DEADLINE"):registry.invoke("slow",{},context())
    assert registry.executor.receipts()[0]["hard_timeout_enforced"] is False
    strict=ToolExecutor(require_hard_timeout=True)
    with pytest.raises(ToolExecutionError,match="HARD_TIMEOUT"):strict.invoke(registry.get("slow"),{},context())


def test_schema_remote_references_rejected_without_network():
    with pytest.raises(ValueError,match="external"):
        ToolRegistry().register(ToolDefinition(ToolSpec(tool_id="bad",input_schema={"$ref":"https://invalid.example/schema"}),lambda x:{}))


def plugin(tmp_path,source="def run(payload):\n    return {'x':payload['x']+1}\n",timeout=3):
    path=tmp_path/"plugin.py";path.write_text(source)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    spec=PluginSpec(plugin_id="test",version="1.0",module_file="plugin.py",function_name="run",artifact_sha256=digest,
        tool=ToolSpec(tool_id="plugin.run",tool_version="1.0",input_schema=SCHEMA,output_schema=SCHEMA,
                      execution_mode="ISOLATED_PROCESS",timeout_s=timeout,required_permissions=[PermissionScope.READ_DATA]))
    return path,spec


def test_metadata_discovery_does_not_execute_plugin(tmp_path):
    path,spec=plugin(tmp_path,"raise RuntimeError('must not import during discovery')\n")
    (tmp_path/"example.plugin.json").write_text(spec.model_dump_json())
    assert discover_plugin_specs(tmp_path)[0].plugin_id=="test"
    with pytest.raises(PermissionError):load_approved_snapshot(tmp_path,[spec],approved_hashes=frozenset(),approved_manifest_hashes=frozenset())


def test_approved_worker_success_and_snapshot_sealed(tmp_path):
    path,spec=plugin(tmp_path)
    snapshot=load_approved_snapshot(tmp_path,[spec],approved_hashes=frozenset({spec.artifact_sha256}),approved_manifest_hashes=frozenset({spec.approval_sha256()}))
    assert snapshot.registry.invoke("plugin.run",{"x":2},context(metadata={"tool_versions":snapshot.tool_versions}))=={"x":3}
    assert snapshot.registry.executor.receipts()[0]["hard_timeout_enforced"] is True
    with pytest.raises(RuntimeError,match="sealed"):snapshot.registry.register(ToolDefinition(ToolSpec(tool_id="late"),lambda x:{}))
    path.write_text("def run(x):return {'x':99}\n")
    with pytest.raises(ToolExecutionError,match="HASH"):snapshot.registry.invoke("plugin.run",{"x":2},context())


def test_timeout_really_stops_worker_not_just_waiting_thread(tmp_path):
    marker=tmp_path/"late-effect.txt"
    path,spec=plugin(tmp_path,f"import time\nfrom pathlib import Path\ndef run(payload):\n    time.sleep(.7)\n    Path({str(marker)!r}).write_text('bad')\n    return {{'x':1}}\n",timeout=.1)
    snapshot=load_approved_snapshot(tmp_path,[spec],approved_hashes=frozenset({spec.artifact_sha256}),approved_manifest_hashes=frozenset({spec.approval_sha256()}))
    start=time.monotonic()
    with pytest.raises(ToolExecutionError,match="DEADLINE"):snapshot.registry.invoke("plugin.run",{"x":1},context())
    assert time.monotonic()-start < .65
    time.sleep(.75)
    assert not marker.exists()


def test_worker_failure_and_cancel_are_not_normal_predictions(tmp_path):
    path,spec=plugin(tmp_path,"def run(payload):\n    raise RuntimeError('private-secret')\n")
    snapshot=load_approved_snapshot(tmp_path,[spec],approved_hashes=frozenset({spec.artifact_sha256}),approved_manifest_hashes=frozenset({spec.approval_sha256()}))
    with pytest.raises(ToolExecutionError,match="WORKER_FAILED"):snapshot.registry.invoke("plugin.run",{"x":1},context())
    assert "private-secret" not in json.dumps(snapshot.registry.executor.receipts())
    cancelled=threading.Event();cancelled.set()
    with pytest.raises(ToolExecutionError,match="CANCELLED"):
        snapshot.registry.executor.invoke(snapshot.registry.get("plugin.run"),{"x":1},context(),cancel=cancelled)


def test_path_traversal_and_failed_load_do_not_change_old_snapshot(tmp_path):
    path,spec=plugin(tmp_path)
    snapshot=load_approved_snapshot(tmp_path,[spec],approved_hashes=frozenset({spec.artifact_sha256}),approved_manifest_hashes=frozenset({spec.approval_sha256()}))
    outside=tmp_path.parent/"outside.py";outside.write_bytes(path.read_bytes())
    bad=spec.model_copy(update={"module_file":"../outside.py"})
    with pytest.raises(ValueError,match="inside"):load_approved_snapshot(tmp_path,[bad],approved_hashes=frozenset({spec.artifact_sha256}),approved_manifest_hashes=frozenset({spec.approval_sha256()}))
    assert snapshot.registry.invoke("plugin.run",{"x":1},context())=={"x":2}


def test_artifact_approval_does_not_authorize_descriptor_permission_changes(tmp_path):
    path,spec=plugin(tmp_path)
    changed=spec.model_copy(update={"tool":spec.tool.model_copy(update={"required_permissions":[]})})
    with pytest.raises(PermissionError,match="descriptor"):
        load_approved_snapshot(tmp_path,[changed],approved_hashes=frozenset({spec.artifact_sha256}),
                               approved_manifest_hashes=frozenset({spec.approval_sha256()}))
