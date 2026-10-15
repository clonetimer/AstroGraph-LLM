export type AstroGraphPreset = 'diagnosis'|'shadow'|'development'|'posthoc'|'evidence'|'confirmatory'|'maintainer';
export interface AstroGraphPluginConfig { preset?: AstroGraphPreset; projectRoot?: string; pythonExecutable?: string; allowedRoots?: string[]; externalAuditRoots?: string[]; }
export declare class AstroGraphCapabilityClient { constructor(config?:AstroGraphPluginConfig); start():Promise<this>; list(profile?:string):Promise<unknown>; invoke(capability:string,args?:Record<string,unknown>,profile?:string):Promise<unknown>; stop():Promise<void>; }
export declare function apply(ctx:unknown,config?:AstroGraphPluginConfig):Promise<()=>Promise<void>>;
declare const plugin:{name:string;apply:typeof apply}; export default plugin;
