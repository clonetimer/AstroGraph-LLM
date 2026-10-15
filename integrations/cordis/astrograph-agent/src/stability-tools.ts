export type AstroGraphToolRegistrar = {
  register: (name: string, schema: unknown, handler: (input: unknown) => Promise<unknown>) => void | (() => void)
}
export type AstroGraphSidecarClient = { invoke: (capability: string, payload: unknown) => Promise<unknown> }

const tools = [
  {
    name: "lineage.step6.assess",
    schema: { type: "object", additionalProperties: true },
    profiles: new Set(["MAINTAINER", "DEVELOPMENT", "SHADOW"]),
  },
  {
    name: "release.stability.assess",
    schema: { type: "object", additionalProperties: true },
    profiles: new Set(["MAINTAINER"]),
  },
] as const

export function applyAstroGraphStabilityTools(ctx: { tools: AstroGraphToolRegistrar }, client: AstroGraphSidecarClient, profile: string): () => void {
  const disposers: Array<() => void> = []
  for (const tool of tools) {
    if (!tool.profiles.has(profile as never)) continue
    const maybeDispose = ctx.tools.register(tool.name, tool.schema, async (payload) => client.invoke(tool.name, payload))
    if (typeof maybeDispose === "function") disposers.push(maybeDispose)
  }
  return () => { for (const dispose of disposers.reverse()) dispose() }
}
