import { spawn } from 'node:child_process';
import readline from 'node:readline';
import crypto from 'node:crypto';

export class AstroGraphCapabilityClient {
  constructor(config={}) { this.config=config; this.pending=new Map(); this.seq=0; this.child=null; }
  async start() {
    if (this.child) return this;
    const python=this.config.pythonExecutable || process.env.PYTHON || 'python3';
    const projectRoot=this.config.projectRoot || process.cwd();
    const args=['-m','astrograph.agentization.server','--project-root',projectRoot,'--allowed-root',projectRoot];
    for (const root of (this.config.allowedRoots || [])) args.push('--allowed-root',root);
    for (const root of (this.config.externalAuditRoots || [])) args.push('--external-audit-root',root);
    this.child=spawn(python,args,{cwd:projectRoot,env:{...process.env,PYTHONPATH:[`${projectRoot}/src`,process.env.PYTHONPATH].filter(Boolean).join(process.platform==='win32'?';':':')},stdio:['pipe','pipe','pipe']});
    const rl=readline.createInterface({input:this.child.stdout});
    rl.on('line',line=>{ let msg; try { msg=JSON.parse(line); } catch { return; } const p=this.pending.get(msg.request_id); if (p) { this.pending.delete(msg.request_id); msg.status==='OK'?p.resolve(msg):p.reject(Object.assign(new Error(msg.error?.message||'AstroGraph capability error'),{response:msg})); }});
    this.child.on('exit',code=>{ for (const p of this.pending.values()) p.reject(new Error(`AstroGraph sidecar exited ${code}`)); this.pending.clear(); this.child=null; });
    await this.request('health',{}); return this;
  }
  async request(method,params={},profile=this.config.profile||'MAINTAINER') {
    if (!this.child) await this.start();
    const request_id=`cordis-${Date.now()}-${++this.seq}-${crypto.randomBytes(4).toString('hex')}`;
    const payload={protocol_version:'astrograph-capability-protocol.v1',request_id,method,profile,params};
    return new Promise((resolve,reject)=>{ this.pending.set(request_id,{resolve,reject}); this.child.stdin.write(JSON.stringify(payload)+'\n'); });
  }
  async list(profile=this.config.profile||'MAINTAINER') { return (await this.request('capabilities.list',{},profile)).result; }
  async invoke(capability,arguments_={},profile=this.config.profile||'MAINTAINER') { return (await this.request('capabilities.invoke',{capability,arguments:arguments_},profile)).result; }
  async stop() { if (!this.child) return; try { await this.request('shutdown',{},this.config.profile||'MAINTAINER'); } catch {} this.child?.kill(); this.child=null; }
}
