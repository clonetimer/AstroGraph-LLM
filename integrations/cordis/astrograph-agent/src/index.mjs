import { AstroGraphCapabilityClient } from './client.mjs';
import { registerCordisService, registerCordisTool, onDispose } from './compat.mjs';
import { PRESETS } from './presets.mjs';
import { createTrajectorySink } from './trajectory.mjs';

function toolName(capability){ return `astrograph__${capability.replaceAll('.','__')}`; }
export async function apply(ctx,config={}) {
  const preset=PRESETS[config.preset||'shadow']; if(!preset) throw new Error(`Unknown AstroGraph preset ${config.preset}`);
  const client=new AstroGraphCapabilityClient({...config,profile:preset.profile}); await client.start();
  const trajectory=createTrajectorySink(); const dispose=[];
  dispose.push(registerCordisService(ctx,'astrograph',{client,profile:preset.profile,trajectory}));
  const manifest=await client.list(preset.profile); const byName=new Map(manifest.capabilities.map(x=>[x.name,x]));
  for(const cap of preset.tools){ const meta=byName.get(cap); if(!meta) continue; const invoke=async args=>{ const start=Date.now(); try { const result=await client.invoke(cap,args||{},preset.profile); trajectory.append({capability:cap,profile:preset.profile,status:'OK',duration_ms:Date.now()-start,result_sha256:null}); return result; } catch(error){ trajectory.append({capability:cap,profile:preset.profile,status:'ERROR',duration_ms:Date.now()-start,error:String(error)}); throw error; }}; dispose.push(registerCordisTool(ctx,{name:toolName(cap),description:meta.description,parameters:meta.input_schema,inputSchema:meta.input_schema,execute:invoke,handler:invoke,metadata:{astrographCapability:cap,profile:preset.profile,sideEffect:meta.side_effect,labelAccess:meta.label_access,projectStatus:'B009_WIP_DEVELOPMENT_EVIDENCE'}})); }
  const cleanup=async()=>{ for(const d of dispose.reverse()) try{await d();}catch{} await client.stop(); };
  onDispose(ctx,cleanup); return cleanup;
}
export default { name:'astrograph-agent', apply };
export { AstroGraphCapabilityClient, PRESETS };
