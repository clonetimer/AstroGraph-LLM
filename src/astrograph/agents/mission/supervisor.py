"""Process-supervised Mission execution: task deadline, per-tool deadline, cancel.

The public CLI uses this path by default. Arbitrary injected in-process Python
backends are test/embedding APIs and cannot be serialized into this worker.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from .models import AgentRequest, AgentSettings, AgentResult
from .storage import Workspace, file_sha, sha
from astrograph.integrations.process_control import _terminate_process_tree


def _descendants(pid):
    """Linux descendant snapshot, including children that created new sessions."""
    if not Path('/proc').is_dir():return []
    table={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            fields=(path/'stat').read_text(encoding="utf-8").rsplit(')',1)[1].split()
            table[int(path.name)]=int(fields[1])
        except (OSError,ValueError,IndexError):pass
    result=[];front={pid}
    while front:
        children={p for p,parent in table.items() if parent in front and p not in result and p!=pid}
        result.extend(children);front=children
    return result


def _kill(process):
    children=_descendants(process.pid)
    _terminate_process_tree(process)
    # A nested provider may have its own session; do not leave it running.
    if os.name=='posix':
        for pid in reversed(children):
            try:os.kill(pid,signal.SIGKILL)
            except (ProcessLookupError,PermissionError):pass
    process.wait()


def _recover(settings, request, reason):
    root=Path(settings.output_root).resolve()/request.task_id
    identity_path=root/'request_identity.json'
    if not identity_path.exists():
        raise TimeoutError(reason+': no task identity committed')
    identity=json.loads(identity_path.read_text(encoding="utf-8"))
    # Do not return another request's existing result just because IDs collide.
    if identity['request_sha256']!=sha(request.model_dump(mode='json')) or identity['settings_sha256']!=sha(settings.model_dump(mode='json')):
        raise ValueError('supervisor task identity mismatch')
    ws=Workspace(settings.output_root,settings.read_roots,request.task_id)
    ws.verify_events()
    output={'error_type':'SupervisorStopped','message':reason};core_status='NOT_COMPLETED'
    if (root/'core-result.json').exists() and (root/'core-result-receipt.json').exists():
        if file_sha(root/'core-result.json')!=json.loads((root/'core-result-receipt.json').read_text(encoding="utf-8"))['sha256']:
            raise ValueError('core result integrity mismatch')
        output['diagnosis_result']=json.loads((root/'core-result.json').read_text(encoding="utf-8"))
        output['attachment_status']='INTERRUPTED_CORE_PRESERVED';core_status='SUCCEEDED'
    elif (root/'diagnostic-snapshot-receipt.json').exists():
        receipt=json.loads((root/'diagnostic-snapshot-receipt.json').read_text(encoding="utf-8"))
        path=root/receipt['name']
        if path.resolve().parent!=root or file_sha(path)!=receipt['sha256']:
            raise ValueError('snapshot integrity mismatch')
        snapshot=json.loads(path.read_text(encoding="utf-8"))
        output['diagnosis_result']=snapshot['diagnosis_result']
        output['investigation_completion']='INTERRUPTED_SNAPSHOT_ONLY';core_status='PARTIAL_SNAPSHOT'
    result=AgentResult(task_id=request.task_id,task=request.task,work_mode=request.work_mode,
        status='PARTIAL' if 'diagnosis_result' in output else 'FAILED',
        request_sha256=identity['request_sha256'],settings_sha256=identity['settings_sha256'],
        plan={'supervisor_stopped':True},output=output,tool_receipts=[],limitations=[reason,'No uncommitted LLM claims or memory writes are promoted.'],
        capability_status={'core_diagnosis':core_status,'supervisor':{'status':'STOPPED','reason':reason},'llm':{'status':'INTERRUPTED_OR_UNKNOWN'}},
        execution_budget={'task_timeout_s':settings.task_timeout_s,'enforcement':'PROCESS_SUPERVISOR'})
    # Existing partial finalization is not overwritten; inspect interrupted work.
    if (root/'result.json').exists() or (root/'artifacts.json').exists():
        ws.write_json('supervisor-recovery.json',result.model_dump(mode='json'))
    else:
        ws.event('SUPERVISOR_STOPPED',reason=reason)
        ws.finish(result.model_dump(mode='json'))
    return result


def run_supervised(settings: AgentSettings, request: AgentRequest, *, cancel=None):
    from .run_lock import task_lease
    request=AgentRequest.model_validate_json(request.model_dump_json())
    with task_lease(settings.output_root,request.task_id):
        return _run_supervised(settings,request,cancel=cancel)


def _run_supervised(settings: AgentSettings, request: AgentRequest, *, cancel=None):
    request=AgentRequest.model_validate_json(request.model_dump_json())
    if request.work_mode not in settings.allowed_work_modes:
        raise PermissionError('WORK_MODE_NOT_AUTHORIZED')
    task_root=Path(settings.output_root).resolve()/request.task_id
    deadline=time.monotonic()+settings.task_timeout_s
    with tempfile.TemporaryDirectory(prefix='astrograph-mission-supervisor-') as temp:
        temp=Path(temp);inp=temp/'request.json';out=temp/'worker-result.json'
        inp.write_text(json.dumps({'settings':settings.model_dump(mode='json'),'request':request.model_dump(mode='json')}), encoding='utf-8')
        os.chmod(inp,0o600)
        env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[3])+os.pathsep+env.get('PYTHONPATH','')
        env['ASTROGRAPH_PROCESS_SUPERVISED']='1'
        with (temp/'stdout.log').open('wb') as stdout,(temp/'stderr.log').open('wb') as stderr:
            p=subprocess.Popen([sys.executable,'-m','astrograph.agents.mission.supervisor','--worker',str(inp),str(out)],
                stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,env=env,start_new_session=os.name=='posix')
            reason=None
            try:
                while p.poll() is None:
                    if cancel is not None and cancel.is_set():reason='TASK_CANCELLED';break
                    now=time.monotonic()
                    if now>=deadline:reason='TASK_TOTAL_DEADLINE_EXCEEDED';break
                    if any((temp/name).stat().st_size>2_000_000 for name in ('stdout.log','stderr.log')):
                        reason='WORKER_LOG_SIZE_EXCEEDED';break
                    progress=task_root/'execution-progress.json'
                    try:
                        if progress.exists():
                            state=json.loads(progress.read_text(encoding="utf-8"))
                            if any(now>=float(tool['deadline']) for tool in state.get('active_tools',[])):
                                reason='TOOL_PROCESS_DEADLINE_EXCEEDED';break
                    except (OSError,ValueError,KeyError):pass
                    time.sleep(0.02)
                if reason:
                    _kill(p)
                    return _recover(settings,request,reason)
                if not out.exists() or out.stat().st_size>settings.max_artifact_bytes:
                    return _recover(settings,request,'WORKER_FAILED_OR_RESPONSE_OVERSIZE')
                response=json.loads(out.read_text(encoding="utf-8"))
                if response.get('error'):
                    kinds={'ValueError':ValueError,'PermissionError':PermissionError,'RuntimeError':RuntimeError,'TimeoutError':TimeoutError}
                    raise kinds.get(response['error_type'],RuntimeError)(response['error'])
                return AgentResult.model_validate(response['result'])
            except BaseException:
                if p.poll() is None:_kill(p)
                raise


def _worker(inp,out):
    from .service import MissionAgent
    try:
        data=json.loads(Path(inp).read_text(encoding="utf-8"))
        agent=MissionAgent(AgentSettings.model_validate(data['settings']))
        result=agent.run_inline(AgentRequest.model_validate(data['request']))
        payload={'result':result.model_dump(mode='json')}
    except Exception as exc:
        payload={'error_type':type(exc).__name__,'error':str(exc)[:1500]}
    path=Path(out);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(payload,ensure_ascii=False), encoding='utf-8');os.replace(temp,path)

if __name__=='__main__':
    if len(sys.argv)!=4 or sys.argv[1]!='--worker':raise SystemExit(2)
    _worker(sys.argv[2],sys.argv[3])
