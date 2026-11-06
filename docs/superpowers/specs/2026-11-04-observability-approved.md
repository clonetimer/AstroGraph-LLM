# Approved AstroGraph observability upgrade

User approved the PilotDeck assessment and instructed implementation. Source of design: attached AstroGraph_PilotDeck_源码升级评估_20261104.md and conversation.

Scope: external diagnosis text -> LLM analysis -> cited report and UI, without diagnosis computation. Preserve Win10 / Python 3.8 Win7 portable profiles. First reconcile shipped V2.7.1 and Win7 fixes into canonical source, preserving full repository exports. Then add durable stage events, per-HTTP-attempt usage (unknown is not zero), input inventory and immutable per-run report artifacts. Preserve original files, successful reports, credentials-in-memory, local loopback / online HTTPS restrictions. No imported PilotDeck code, Node runtime, autonomous tasks or new knowledge memory in this slice.

Compatibility: old jobs remain readable; calls without observability retain behavior. Request and response text, provider errors, keys and endpoint paths never enter safe telemetry. A received HTTP response is not a successful analysis. Only actual provider usage is counted as known. No invented cost without pricing. New runs receive new IDs; explicit retry never silently changes model. No new packaging claim for Win7 until runtime validated.
