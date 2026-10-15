# R11 Governed Maintenance Rerun

## Supported scope

R11 supports one deliberately narrow rerun mode:

`OAG_EVIDENCE_BRANCH_REPLAY`

It creates a new immutable OAG child job from a previously completed OAG job after a maintenance supplemental input has passed the current EvidenceInputContract and deterministic server-side admission path.

The child uses the admitted maintenance evidence overlay while preserving the source job. It is **not** a raw-telemetry production rerun and it is **not** formal release evidence.

## Admission gates

A child can be created only when all of the following are true:

1. the source artifact is `OAG_DIAGNOSTIC`;
2. the source job is `SUCCEEDED`;
3. the original OAG input is available and validates;
4. a supplemental overlay exists;
5. the overlay preserves the original episode, feature definitions, evidence definitions and component identity;
6. every admission receipt in the current overlay chain is covered by a valid supplemental provenance receipt;
7. each provenance receipt binds the same source job, frozen report and original EvidenceBundle;
8. provenance parent chains exactly match the overlay admission order;
9. the original OAG input does not carry a native `evidence_projection` bound to raw paired telemetry/model input;
10. the generated child passes the normal platform preflight.

Any failed gate returns `BLOCKED` and no child is started.

## Why native evidence projection is blocked

A native projection binds the original EvidenceBundle, feature hashes, model input, model artifact and paired telemetry window. A maintenance numeric capsule cannot lawfully regenerate those identities. Such cases must return to the raw governed telemetry/model pipeline.

## Immutable lineage

The source job remains unchanged. The child records:

- source job ID;
- source report SHA256;
- original EvidenceBundle SHA256;
- overlay EvidenceBundle SHA256;
- rerun OAG input SHA256;
- supplemental admission receipt chain;
- supplemental provenance receipt chain;
- child preflight SHA256;
- a self-verifying governed-rerun receipt SHA256.

Tampering with the governed-rerun receipt invalidates `receipt_sha256`.

## Result authority

Even when the child OAG execution succeeds and produces `READY_FOR_EXPLANATION`, the platform keeps:

- `raw_telemetry_rerun_performed=false`;
- `formal_release_eligible=false`;
- `formal_diagnosis_publishable=false`.

A formal release-changing rerun still requires the original production input family: raw telemetry/profile admission for PUBLIC_CASE, or the governed runtime/mapping path for PIPELINE_RUNTIME.


## Completion receipt and post-run recheck

A prepared child receipt is not treated as proof that the child actually executed the prepared bytes. After the child finishes, R11 re-reads and re-hashes:

- the frozen source report;
- the original source OAG input;
- the actual child OAG input;
- the child report and OAG output;
- the prepared receipt/source-child lineage.

A completion receipt is emitted only if all frozen identities still match and the child report remains non-publishable. The completion receipt freezes the child diagnosis status, selected fault (if any), child report/output hashes and `FORMAL_RELEASE_BLOCKED` state. Source-report drift, source-OAG-input drift, child-input drift, lost lineage, or a publishable child report fail closed and no trusted completion receipt is produced.

Schema: `astrograph-maintenance-governed-rerun-completion-receipt.v1`.

## Equivalent-rerun idempotency

The complete UI/platform distribution prevents accidental or concurrent duplicate child jobs for the same frozen source report/OAG input, exact rerun input, admission chain and supplemental-provenance chain. An atomic content-identity claim is acquired before child creation so two concurrent POST requests cannot both materialize an equivalent child. While an equivalent child is `CREATED`, `UPLOADING`, `RUNNING` or `SUCCEEDED`, readiness returns `BLOCKED` with `EQUIVALENT_GOVERNED_RERUN_EXISTS` and exposes the existing child job. A prior equivalent child in `FAILED`, `CANCELLED` or missing state does not permanently block a retry.

The source job also exposes a verified governed-rerun history index that joins each prepared receipt, child status and completion receipt when available. This history is audit metadata only and does not change release authority.
