# Portable V2 delivery closeout

Approved scope: freeze business features; make the delivered external-diagnosis workbench reproducible and usable. Continue the approved V2 plan rather than redesign it.

1. Recover the exact V2.5 app, canonical core delta, tests and builder from the delivered archive into an isolated Git worktree. Run baseline regression and record any failures.
2. Add failure-first tests for launch/resume state races, backups during active analysis, restoring RUNNING jobs, Windows path aliases in imported backups, and corruption of saved resume input. Fix only these usability/data-integrity faults.
3. Add a package verifier and persisted self-check diagnostics so a Windows user can report a concrete failure without a transient console. Keep runtime startup entirely offline; do not write API keys into diagnostics.
4. Recover a source-only update payload whose file hashes are recorded. Synchronize ordinary source paths to the existing GitHub feature branch with a guarded write; use CI for Windows-native runtime checks if available. Never treat Windows Server CI as Windows 10 validation.
5. Rebuild from verified local runtime/dependencies; run portable regression, HTTP/self-test, JS checks, manifest and archive checks. Emit one versioned candidate with instructions and exact evidence. Do not claim Stable or production-model quality without corresponding runs.

Review focus: restored task state, archive path aliases, incomplete snapshots, stale/mutated source state, and test environment isolation. Dependency upgrade is allowed only with downloaded and verified official bytes; no guessed provenance.
