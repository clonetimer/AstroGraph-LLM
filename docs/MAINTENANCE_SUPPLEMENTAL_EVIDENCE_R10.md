# Maintenance Supplemental Evidence R10

R10 connects a current `EvidenceInputContract` to a narrow, server-owned deterministic reanalysis path. It is designed for a maintenance operator who has the exact missing channel-window data required by a current contract.

## What the client may submit

The client submits an `astrograph-maintenance-supplemental-feature-input.v1` capsule containing only:

- current contract ID;
- already-resolved canonical source field;
- unit;
- frozen episode window;
- timestamped numeric samples;
- paired baseline values only when the frozen operator requires them;
- an optional source note.

The client cannot submit a FeatureResult, EvidenceFact, threshold result, diagnosis status, or release decision. Extra fields are rejected.

## When direct admission is allowed

Direct supplemental admission is fail-closed and requires all of the following:

- a current (not stale) EvidenceInputContract;
- resolved source field;
- episode-scoped window;
- exact unit match;
- exact frozen builtin algorithm version match;
- an allow-listed deterministic operator;
- valid sample shapes and finite admitted samples.

Operators whose input/output unit semantics are ambiguous, whose parameters are not completely frozen, or whose processing needs unrestricted telemetry access remain `UNSUPPORTED` and must use the normal telemetry/profile admission pipeline.

## Server-owned recomputation

For an admitted capsule the server:

1. revalidates the current contract;
2. rejects identical capsule replay for the same contract;
3. executes the allow-listed deterministic operator;
4. creates the FeatureResult server-side;
5. recomputes EvidenceFacts using the frozen EvidenceDefinitions;
6. rebuilds deterministic hypotheses;
7. stores an auditable maintenance reanalysis overlay and receipt.

The original frozen report is not modified. Formal release eligibility remains false. If the overlay should become release evidence, the supplemented governed input must re-enter the full production pipeline.

## Platform endpoints

- `GET /v1/platform/jobs/{job_id}/maintenance/input-contracts`
- `GET /v1/platform/jobs/{job_id}/maintenance/input-contracts/{contract_id}/template`
- `POST /v1/platform/jobs/{job_id}/maintenance/input-contracts/{contract_id}/admit-feature-input`
- `GET /v1/platform/jobs/{job_id}/maintenance/supplemental-admissions`

The operator UI exposes “复制补采模板 JSON” and, only for supported current contracts, “导入补充输入 JSON”.
