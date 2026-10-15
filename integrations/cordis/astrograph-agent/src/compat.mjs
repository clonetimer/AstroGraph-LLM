export function registerCordisService(ctx,name,value) {
  if (typeof ctx?.provide === 'function') { const disposer=ctx.provide(name,value); return typeof disposer==='function'?disposer:()=>{}; }
  if (ctx?.services && typeof ctx.services.set === 'function') { ctx.services.set(name,value); return ()=>ctx.services.delete(name); }
  const old=ctx[name]; ctx[name]=value; return ()=>{ if (old===undefined) delete ctx[name]; else ctx[name]=old; };
}
export function registerCordisTool(ctx,definition) {
  const registry=ctx?.tools || ctx?.toolRegistry || ctx?.tool;
  if (!registry || typeof registry.register!=='function') throw new Error('Cordis/Harness tool registry not found');
  const result=registry.register(definition);
  if (typeof result==='function') return result;
  if (result && typeof result.dispose==='function') return ()=>result.dispose();
  return ()=>{ if (typeof registry.unregister==='function') registry.unregister(definition.name); };
}
export function onDispose(ctx,fn) {
  if (typeof ctx?.effect==='function') { ctx.effect(()=>fn); return; }
  if (typeof ctx?.on==='function') { ctx.on('dispose',fn); return; }
  if (!ctx.__astrographDisposers) ctx.__astrographDisposers=[]; ctx.__astrographDisposers.push(fn);
}
