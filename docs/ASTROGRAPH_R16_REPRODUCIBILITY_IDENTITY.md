<!-- CURRENT_DOCUMENT: 0.7.1.post49 -->
# AstroGraph R16 Confirmatory Reproducibility Identity

Status: **CURRENT_ENGINEERING_CONFIRMATORY_IDENTITY**  
Date: 2026-09-18  
Baseline: PATCH-R15 unified source

## 1. Scope

PATCH-R16 closes the engineering identity gap recorded as `R14-REP-001`: a formal A0–A7 confirmatory run must no longer be identified only by dataset hashes and a model name advertised by `/v1/models`.

R16 does **not** run the real 38×8 campaign and does not create new performance evidence. It strengthens pre/post execution identity admission so that a future result can be tied to the source, inference profile, Python environment, model/tokenizer bytes, model-server runtime and accelerator that actually produced it.

R16 also repairs two inherited execution-path defects:

- `BASE-R16-01`: R15 froze `group_id` in the case manifest, but the live ablation runner still populated result rows with `group_id=case_id`. The runner now preserves the manifest `group_id`, so the R15 group-aware inferential design survives the real execution path.
- `BASE-R16-02`: the formal runner previously appended `admitted_at` to `execution_receipt.json` after `receipt_sha256` had already been computed. R16 generates the timestamp before hashing and the post-run verifier first validates the execution receipt self-hash.

## 2. Frozen orchestrator identity

The AstroGraph-side runtime identity binds:

- `SOURCE_MANIFEST.json` SHA256;
- `SHA256SUMS.txt` SHA256;
- `src/astrograph/llm/openai_compatible.py` and `task_prompts.py` content identities;
- frozen `configs/evaluation/llm_method_confirmatory_inference.v1.json` file and canonical hashes;
- the complete public `OpenAICompatibleConfig` snapshot used by the confirmatory backend;
- Python implementation/version/ABI/platform/machine;
- Python executable SHA256 when it is a regular file;
- the exact observed installed distribution name/version manifest;
- a deterministic CycloneDX 1.7 SBOM derived from that distribution manifest.

`source integrity = PASS` and `environment lock = STABLE` are required before the formal runner is admitted.

The SBOM is a software inventory/identity artifact. It is not a package-signature or supply-chain authenticity proof, and dependency name/version identity is not represented as byte-for-byte identity of every installed wheel.

## 3. Frozen inference configuration

R16 moves the confirmatory OpenAI-compatible backend settings into one frozen profile:

`configs/evaluation/llm_method_confirmatory_inference.v1.json`

It includes temperature, seed, token limits, timeout, HTTP retry/response limits, structured-response strategy, prompt profile, repair count, candidate Top-K policy, context/budget policy and tokenizer behavior.

The model-server launch contract separately requires:

`generation_config_policy = vllm`

and explicit `--generation-config vllm` in the frozen vLLM argv. This prevents an unrecorded model-repository `generation_config.json` from silently overriding sampling defaults. vLLM's current server documentation explicitly states that repository generation configuration is applied by default and recommends `--generation-config vllm` to disable that behavior.

## 4. Model-host runtime attestation

Schema:

`astrograph-confirmatory-model-runtime-attestation.v1`

A freshly generated attestation freezes:

### Model / tokenizer

- exact declared model ID;
- model revision;
- recursive regular-file size/SHA256 manifest;
- aggregate model content identity;
- tokenizer revision;
- recursive tokenizer size/SHA256 manifest;
- aggregate tokenizer content identity.

Artifact symlinks are rejected rather than followed into an external mutable path.

### Model server

At least one strong server binary identity must be present:

- content-hashed server artifact tree; or
- content-hashed installed `vllm` distribution; or
- container image digest.

The exact launch-config object and its SHA256 are frozen. Container deployment additionally requires an image digest.

### Python / SBOM / accelerator

The model-host attestation also records:

- model-host Python environment lock;
- CycloneDX 1.7 SBOM identity;
- NVIDIA GPU UUID/name/index;
- driver version;
- total GPU memory;
- installed torch version;
- torch-observed CUDA runtime version;
- deployment kind;
- optional model/tokenizer read-only mount audit.

Formal R16 model-runtime verification fails closed when NVIDIA/CUDA identity is not observable. This is intentionally stricter than ordinary AstroGraph startup; it applies only to the frozen confirmatory execution path.

## 5. Local and remote attestation modes

Two acquisition modes are supported:

1. `LOCAL_RECOMPUTED`: the formal runner receives a model-runtime spec and directly rehashes model/tokenizer/server artifacts before and after A0–A7.
2. `HTTP_REOBSERVED`: a model-host sidecar recomputes and serves an attestation; the formal runner fetches it independently before and after A0–A7.

The HTTP mode intentionally records:

`origin_authentication.status = UNVERIFIED`

A self-hash proves that an attestation object is internally intact. It does **not** prove which remote machine/organization produced it. TLS identity, signed provider attestations, TPM/HSM roots, nonce/challenge freshness and authorization remain outside R16 and align with the existing `R14-UP-001` backlog.

## 6. Formal runner sequence

The R16 formal runner now requires:

- frozen dataset snapshot receipt;
- frozen client environment lock;
- exactly one model-runtime attestation source;
- live `/models` advertisement of the requested exact model ID;
- valid R16 runtime identity admission.

Execution order:

1. acquire/recompute model-runtime attestation;
2. recompute AstroGraph client/source/environment identity;
3. create `runtime_identity_admission_pre.json`;
4. run confirmatory readiness with runtime identity enforcement;
5. revalidate the dataset snapshot receipt;
6. create execution receipt v4;
7. launch A0–A7 with the frozen inference profile;
8. acquire/recompute model-runtime attestation again;
9. recompute client/source/environment identity again;
10. rerun readiness and dataset snapshot checks;
11. run execution-identity and runtime-identity drift checks;
12. permit promotion evaluation only when the combined identity result is `STABLE`.

The timestamped attestation object's own SHA256 is retained as audit metadata but is **not** required to remain identical pre/post. Stability is decided from content-bound model/client/inference runtime identities, because a freshly regenerated attestation legitimately has a different generation timestamp.

## 7. BASE-R16-01 — group identity propagation

R15's statistical layer correctly treats `group_id` as the inferential unit, but the live runner previously emitted each `LLMMethodAblationRow` with its own `case_id` as `group_id`.

R16 changes `_row_from_result()` to preserve the frozen manifest `group_id` and makes `group_id` a required manifest field in the ablation runner admission.

A regression asserts that a case such as:

`case_id=case-1, group_id=trajectory-pair-42`

produces an ablation row with:

`group_id=trajectory-pair-42`.

This repair is required for the R15 cluster-aware promotion statistics to be meaningful in the eventual real campaign.

## 8. BASE-R16-02 — execution receipt self-hash

The pre-R16 runner created a self-hashed execution receipt and then mutated it by appending `admitted_at`. That made the persisted bytes semantically inconsistent with `receipt_sha256`.

R16 moves `admitted_at` into `build_confirmatory_execution_receipt()` before the canonical hash is calculated. Audit-only attestation-source paths are kept in separate readiness/admission artifacts rather than mutating the frozen execution receipt.

`verify_confirmatory_execution_identity()` now recomputes the execution-receipt canonical SHA256 before any post-run comparison. A mutated receipt produces:

`EXECUTION_RECEIPT_SHA256_INVALID`.

This is an inherited receipt-integrity defect, not a performance-model change.

## 9. Schemas / artifacts

R16 active design:

`astrograph-llm-method-acceptance-campaign.v4`

Active campaign:

`llm-method-acceptance-v4-confirmatory`

New/advanced governed artifacts include:

- `astrograph-llm-confirmatory-inference-profile.v1`;
- `astrograph-vllm-server-launch-config.v1`;
- `astrograph-confirmatory-python-environment-lock.v1`;
- `astrograph-confirmatory-model-runtime-spec.v1`;
- `astrograph-confirmatory-model-runtime-attestation.v1`;
- `astrograph-confirmatory-runtime-identity-admission.v1`;
- `astrograph-llm-method-confirmatory-readiness.v6`;
- `astrograph-llm-method-confirmatory-execution-receipt.v4`;
- `astrograph-llm-method-confirmatory-identity-check.v4`;
- `astrograph-llm-method-confirmatory-launch-plan.v6`.

## 10. Research boundary

R16 improves **reproducibility identity**, not statistical power, external validity or safety qualification.

A `STABLE` R16 identity means the pre/post observed identities match the frozen engineering contract. It does not prove:

- deterministic GPU kernels or bit-identical floating-point output;
- remote producer authenticity;
- absence of a compromised runtime;
- model calibration;
- cross-simulator/HIL/flight validity;
- A0–A7 performance improvement;
- Probe formal EvidenceFact authority.

The real confirmatory campaign remains unexecuted until real frozen ADCS/PHM snapshots and the governed model runtime are available.

## 11. External standards / implementation rationale

- CycloneDX current specification overview (v1.7): https://cyclonedx.org/specification/overview/
- CycloneDX 1.7 JSON reference: https://cyclonedx.org/docs/1.7/json/
- vLLM OpenAI-compatible server documentation and `--generation-config vllm` behavior: https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/

These references justify inventory/configuration identity choices. They do not imply that AstroGraph has received certification from those projects or standards bodies.
