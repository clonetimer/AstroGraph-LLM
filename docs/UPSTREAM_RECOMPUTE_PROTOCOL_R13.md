# R13 Optional Upstream Recompute Protocol

## Purpose

R13 removes the architectural implication that AstroGraph must install or execute `sat_agent_platform`, MATLAB or Simulink in order to continue a PIPELINE_RUNTIME evidence-gap workflow.

AstroGraph core now treats every automatic upstream recompute backend as optional. The built-in fallback is a zero-extra-dependency manual request/response exchange.

The design rule is:

> Capability may degrade from automatic to manual, but the data/evidence contract may not degrade.

## Provider model

The provider capability contract is:

`astrograph-upstream-provider-capabilities.v1`

Provider states are deliberately small:

- `AVAILABLE` — an executable automatic provider is actually available;
- `MANUAL` — the built-in offline request/response exchange is available;
- `UNAVAILABLE` — the optional provider is absent or not executable.

R13 registers:

- `manual-import` — built in, status `MANUAL`, no extra dependency;
- `sat-agent-platform` — optional, not imported by AstroGraph core;
- `matlab-simulink` — optional external backend, not a Python dependency.

If both optional providers are unavailable, AstroGraph diagnosis, maintenance reanalysis, OAG reruns, PUBLIC_CASE raw-input reruns and the manual PIPELINE_RUNTIME recompute protocol remain usable.

## PIPELINE_RUNTIME request

A completed PIPELINE_RUNTIME source job can emit:

`astrograph-upstream-recompute-request.v1`

The request freezes:

- source job ID;
- source runtime SHA256;
- source mapping-bundle SHA256;
- input/required runtime schema;
- required mapping schema;
- current maintenance evidence-gap input-contract IDs;
- `derived_runtime_patch_allowed=false`;
- request ID and request SHA256.

The request is content-bound. Editing the request after export invalidates its hash.

## Manual request bundle

The built-in `manual-import` provider exports a ZIP containing exactly:

- `request.json`;
- `response_template.json`;
- `README.txt`.

This ZIP can be moved to an air-gapped MATLAB workstation, simulation server, ground-data workstation, `sat_agent_platform` installation or any other producer.

No AstroGraph automatic provider needs to be installed on the source machine.

## Manual response bundle

The return ZIP contains exactly:

- `response.json`;
- `runtime.json`;
- `mapping.json`.

`response.json` binds the return artifacts to one request ID/SHA256 and freezes the runtime/mapping SHA256 values.

AstroGraph rejects:

- unsafe ZIP paths or symlinks;
- untracked extra files;
- oversized members/expanded content;
- request ID/SHA mismatch;
- runtime/mapping hash mismatch;
- unsupported schemas;
- runtime/mapping cross-project identity failure;
- in-place derived feature/runtime patching.

A successful admission produces:

`astrograph-upstream-recompute-admission-receipt.v1`

## Child job

The complete platform/UI distribution can import the admitted response and create a new immutable PIPELINE_RUNTIME child job.

The source task remains unchanged. The child must pass ordinary PIPELINE_RUNTIME preflight and the ordinary pipeline diagnosis path.

R13 freezes a PREPARED receipt before execution and a COMPLETED receipt after execution. Completion re-checks:

- frozen source report identity;
- source runtime SHA256;
- source mapping SHA256;
- actual child runtime/mapping SHA256;
- child artifact family and successful execution.

Equivalent manual responses are idempotently blocked by a content-identity claim until an earlier child is explicitly failed/cancelled/missing.

## Provenance boundary

Manual exchange verifies the returned artifacts, but AstroGraph cannot independently prove how an external machine generated them.

Therefore R13 explicitly records:

- `upstream_recompute_received=true`;
- `raw_telemetry_recompute_verified=false`;
- `formal_release_eligible=false`.

An upstream producer may add its own raw-input provenance receipt hash to the response. AstroGraph preserves that reference, but does not manufacture raw-telemetry provenance on the producer's behalf.

## CLI

The core source distribution adds the zero-extra-dependency CLI entry point:

```bash
astrograph-upstream providers
```

Build a request bundle:

```bash
astrograph-upstream build-request \
  --source-job-id JOB-... \
  --runtime source_runtime.json \
  --mapping source_mapping.json \
  --contract-id EIC-... \
  --out recompute-request.zip
```

After external recomputation, package returned artifacts without manually computing hashes:

```bash
astrograph-upstream package-response \
  --request-json request.json \
  --runtime runtime.json \
  --mapping mapping.json \
  --out recompute-response.zip
```

Verify/materialize a returned bundle:

```bash
astrograph-upstream verify-response \
  --request-json request.json \
  --response-zip recompute-response.zip \
  --out-dir admitted/
```

## Platform API

Complete platform/UI distribution:

- `GET /v1/platform/upstream/providers`
- `GET /v1/platform/jobs/{job_id}/maintenance/upstream-recompute`
- `GET /v1/platform/jobs/{job_id}/maintenance/upstream-recompute/request-bundle`
- `POST /v1/platform/jobs/{job_id}/maintenance/upstream-recompute/manual-response?start=true`
- `GET /v1/platform/jobs/{job_id}/maintenance/upstream-recompute/history`

The UI shows manual exchange as usable even when Sat Agent Platform and MATLAB/Simulink are unavailable.

## Optional automatic providers

R13 intentionally does **not** install or import `sat_agent_platform`, MATLAB or Simulink.

Future connectors may implement the same request/response contract through local execution or remote HTTP without changing AstroGraph core diagnosis semantics.

The optional packages supplied alongside development are therefore reference/back-end candidates, not mandatory runtime dependencies.

## Claim boundary

R13 can claim that PIPELINE_RUNTIME recomputation no longer requires an installed automatic upstream provider: a content-bound manual request can be exported, a complete recomputed runtime/mapping bundle can be imported and strictly admitted, and a new immutable child can re-enter the ordinary PIPELINE_RUNTIME diagnosis path.

R13 does **not** claim that manual exchange proves raw-telemetry provenance, that MATLAB/Sat Agent automatic connectors are implemented, that returned artifacts are flight-qualified, or that a child gains formal release authority.

## R13 delivery portability and analysis hardening

The optional-provider/manual-exchange contract is unchanged by the final R13 stability pass. AstroGraph Core still starts and the `manual-import` provider still works when MATLAB/Simulink and Sat Agent Platform are both unavailable.

For desktop/Windows deployments, internal JSON/text reads used by the maintained runtime are explicitly UTF-8, and final source ZIPs are created with the ZIP UTF-8 filename flag for non-ASCII documentation names. A fresh extraction must therefore succeed with the standard Python `zipfile` decoder; no platform-specific metadata override is part of the product contract.

Maintenance statistics and temporal localization are no longer computed from the display-density curve payload. They consume a separate truth-safe analysis series; display downsampling preserves selected-channel extrema and publishes source-row coverage. This change remains presentation/maintenance analytics only and does not promote those statistics into formal EvidenceFact or release evidence.

