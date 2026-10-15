# AstroGraph Agentization Architecture — R001

状态：`CURRENT_POST49_REFERENCE`

AstroGraph remains the deterministic scientific core. Cordis/Harness receives only typed, profile-scoped capabilities through a long-running NDJSON sidecar. The LLM may plan, retrieve, draft and explain; it is not the numerical evaluator, evidence approver or policy selector.

## Atomic diagnosis execution boundary

The `DIAGNOSIS` profile may invoke `runtime.diagnosis.execute` with a typed `astrograph-runtime-request.v1`. The capability admits only the three canonical runtime pipelines, resolves every path-like field against configured roots, and returns the complete `RuntimeResult` plus content hashes and stage receipts. The Agent cannot invoke individual scientific stages or replace their outputs. In particular, `whole_spacecraft.online.v1` remains a single governed Agent action whose internal 19-node DAG is executed exclusively by `AstroGraphRuntime`.

`SHADOW` may inspect `runtime.describe`, read the reference workflow and validate a proposed plan, but cannot invoke `runtime.diagnosis.execute`. This separation makes an Agent-loadable end-to-end diagnosis available without transferring scientific execution authority to an LLM or plugin host.

## Physical profile separation

- `DIAGNOSIS`: no truth/evaluator tools.
- `DEVELOPMENT`: groups 3/8 only; prediction freeze precedes offline truth join.
- `POSTHOC_AUDIT`: groups 4/9 replay only; no policy mutation.
- `EVIDENCE_CURATION`: proposal/source/compile/unit-test; agent cannot freeze evidence.
- `CONFIRMATORY_SEALED`: verify and execute a frozen one-time DAG; no selection tools.
- `MAINTAINER`: source, migration, testing and packaging.
- `SHADOW`: validates an LLM plan but grants execution authority only to the deterministic reference workflow.

## Trust boundary

Cordis validates tool routing and lifecycle. AstroGraph validates scientific semantics, hashes, data splits, label access, statistical units and claim scope. DAG/schema validity must never be described as physical or causal correctness.
