# AstroGraph P2/P3 continuation — source development delivery

Base: P0/P1 4e5d3331a8d346e98f7f90be6d90ae9f61225c17. The unavailable previous P2/P3 worktree was reconstructed from its recorded design and integration boundaries; the old unverified commit hashes are not reused.

## Implemented source paths
- Reviewed workspace context: scoped notes/hypotheses, optimistic revision checks, explicit review state, explicit selection, immutable task snapshot, original source registry unchanged.
- Same context is supplied to analysis and follow-up; no silent cross-workspace recall. Review means permission to use, not truth certification.
- Context backup export/import is transactional and rejects non-identical workspace histories rather than merging silently.
- Shared explicit phase router: primary analysis, chunk analysis, synthesis, follow-up. Omitted overrides inherit the primary configuration, never fall back after a model failure. Cross-endpoint overrides do not inherit API keys.
- Portable service: bounded opt-in queue, maximum 8 waiting items and one executing item; default busy behavior retained. Cancellation is cooperative. A request already sent may finish or be billed, but cancelled task output is not published. Queued credentials remain in memory and callbacks are discarded on completion/cancellation.
- Restart changes persisted queued/running/cancel-requested tasks to interrupted; manual resume requires the original input, context snapshot and execution configuration. No automatic paid replay.
- Portable UI: workspace selection/edit/review/history, explicit note checkboxes, per-stage model IDs on the same endpoint, queue checkbox and cancel action. Six navigation pages retained.
- Full platform: reviewed-context API with existing roles (viewer/operator/reviewer), task binding before preflight and model input integration. Mission/Legacy upload views use the shared context component.

## Explicit boundaries
This is SOURCE ONLY, not a rebuilt Windows portable ZIP. Old V2.7.1 packages are unchanged.
The bounded queue and editable stage controls are wired to the portable service. Complete-platform multiprocess scheduling and phase-profile UI are NOT included in this batch. Do not describe them as complete.
The 2026-11-05 recovery ran the full portable test suite under actual Linux Python 3.8.20 and Python 3.13.5, plus the assembled app self-test. This is not Win7/Win10 execution. No production-model or independent-reviewer validation is claimed.
Current results are in P2P3_RECOVERY_VERIFICATION_20261105.md/json; P2P3_CONTINUATION_VERIFICATION.json is the preserved, failed historical checkpoint. Detailed current logs are provided separately. Incomplete or failed test commands remain disclosed; this development snapshot is not Stable or a claim that the entire research repository passes.
No PilotDeck code was copied or translated; project license is unchanged. SQLite files/snapshots and report versions may contain user data and must be protected accordingly.

## Development
Use the existing project environment. Run `python -B -m unittest discover -s portable -p 'test*.py' -v` and affected full-platform pytest modules. Assemble the portable source with portable.build.stage_app; model services remain external.
Do not overwrite existing user data or mix Windows7/Windows10 runtime directories.
