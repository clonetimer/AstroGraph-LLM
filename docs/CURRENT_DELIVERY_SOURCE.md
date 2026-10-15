<!-- CURRENT_DOCUMENT: 0.7.1.post49 -->
# AstroGraph-Agent current delivery source

This is the **current-source delivery profile** for AstroGraph-Agent `0.7.1.post49`.

## Included

- Current runtime and Agent source (`src/`), current configuration, models, data assets, reference assets and benchmarks.
- Replay, generic-series detection, admitted profile-gated classification/localization, knowledge reasoning, document/RAG processing, continuous diagnosis, expert-feedback and controlled-evolution code paths.
- Tests, acceptance harnesses and protocol definitions that are executable code rather than past run records.
- Current post49 governance contract/readiness metadata.
- Empty mutable ledgers so feedback/evolution/governance functions start from a clean state.
- The minimal frozen gate inputs listed by `governance/evidence/CURRENT_RUNTIME_ASSETS.json`. Their directory names retain the release in which the gate input was created because current evaluator code is hash/path bound to those assets; they are operational dependencies, not a bundled history archive.

## Excluded

- `docs/history/` and archived release/dev/date reports.
- Prior `post48_dev*` implementation ledgers, receipts and run outputs.
- Prior release contracts/change requests/task boards/rollback manifests.
- Historical research-result ledgers and old acceptance result workspaces.
- Historical mutable failure-memory/passport/problem-stack events.

Historical reproduction therefore requires a separately retained audit/archive package. It is intentionally not possible to reconstruct old metrics merely by reading this delivery package.

## Frozen provenance inside functional assets

A small number of current, hash-pinned model/evidence packages contain provenance strings that refer to the release in which they were created. Those strings are part of the current governed artifact identity and are retained unchanged. The referenced historical files themselves are not included unless explicitly listed in `CURRENT_RUNTIME_ASSETS.json` as a current operational gate dependency.

## Claim boundary

Removing historical records does **not** expand diagnostic qualification. Scenario/profile/provider boundaries remain those in `governance/implementation/post49/RELEASE_READINESS.json`.
