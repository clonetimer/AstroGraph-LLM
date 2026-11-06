# Execution ledger — 2026-11-04 portable-v27-navigation

Baseline: delivered V2.6.2 ZIP SHA-256 7e765a7ce62a810f427bd92f14c678f9d4af36f298711acf5d6ec8129a490fea, extracted source in an isolated local Git branch.
Recovery: only README and two screenshots from interrupted V2.7 turn were mounted, not its modified source. Reconstruct the agreed navigation against recoverable code; do not count prior screenshots as current verification.
Pre-flight: navigation toggles panels, app.js owns source/task/session state; static backend must expose the extra navigation script. No inference-core changes.
Baseline: three Chromium component scenarios PASS. Full suite attempt exceeded 200s and exposed a package-only ROOT assumption in test_check_startup.py; preserve in baseline-tests.log, correct harness for source use, then rerun full suite.

Task 1 complete: 6 browser scenarios first failed on missing navigation controls; reproduced in nav-red.log.
Task 2 complete: six persistent panels and navigation shell, history filters/opening and chat source routing. Initial six and three additional scenarios pass. Three original component scenarios also pass after adapting route selection.
Task 3 complete: source-root harness fixed, full suite 49 collected / 48 passed / 1 packaged-only skip / 0 failure. API static script + missing generic file route verified; check quote fix preserved.
Task 4: builder rejects wrong pinned baseline before output, first failed then passed. Source docs and version updated; original runtime/dependencies unchanged by design.
Final review: self-review (no subagent tool). Reviewed client diff, shell state lifetime, stale-request checks, output escaping, CSP asset route, backup/resume contracts and source staging. Removed an inline style in favor of same-origin CSS. No unresolved new important finding.
Ruling: The screenshot from the interrupted turn included an extra overview page. Keep the six explicitly approved routes; no extra dashboard or empty business page.
Ruling: Full-source shared-frontend integration is not claimed; this turn delivers a reusable shell plus the portable front-end. Full repository is not modified.
Task 4 complete: first offline V2.7 build and package manifest verification passed. Runtime is inherited from SHA-pinned V2.6.2. Final deliverable is rebuilt twice after this record is frozen; external package verification report records the final ZIP hashes and packaged-suite outcome.
Task 5 delivery staging: sources, docs, test cases and reproducible builder are frozen. Final executable verification runs on the staged package; receipts are separate to avoid modifying the package being verified.
