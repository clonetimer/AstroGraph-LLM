# Portable V2.7 — verification record

Date: 2026-11-04. Author self-review, no independent reviewer or native Windows runner available.

## Source regression (executed)

`python -m unittest discover -s portable -p 'test*.py' -v`

49 tests discovered, 48 passed, 1 skipped, 0 failed/error; 41.704 seconds. The skipped test requires a packaged MANIFEST in the parent directory and explicitly skips in a standalone source checkout. It must be rerun in the final extracted distribution. No model-quality test is implied.

Baseline recovery initially exposed test_check_startup.py assuming the source always resides inside a distribution. The test harness now tests the actual portable CHECK script and uses a real temporary manifest/asset from another working directory. No manifest protection was removed. The V2.6.2 CHECK.bat is unchanged.

## Real Chromium components (executed)

- `python portable/browser_navigation.py --output <artifacts>`: 9 behavioral scenarios passed, 0 failed.
- `python portable/browser_smoke.py --chromium /usr/bin/chromium --output <artifacts>`: 3 previous regression scenarios passed, 0 failed.

Production HTML, CSS and JavaScript execute in Chromium; only API responses are fixtures. Covers six routes, preserved drafts/files/model/key, actual file upload serialization, history filtering/opening, cited chat submission, source navigation, failure retry, browser back/forward, keyboard focus, ongoing analysis without route hijacking, and 390/900/1024/1280/1440px layouts. Screenshots were visually inspected. No remote resources or model request happens on navigation alone.

`node --check portable/assets/navigation.js`, `node --check portable/assets/app.js`, `python -m compileall -q portable src`, `git diff --check`: passed.

## Live-browser attempt (not passed)

`python portable/browser_live_smoke.py --output <artifacts>` attempted direct browser navigation to an actual local workbench. The environment returned `net::ERR_BLOCKED_BY_ADMINISTRATOR` on the loopback URL. Do not count this as end-to-end success; no attempt to disable the environment policy was made. Independent HTTP tests run the actual local server/model worker and passed.

## Packaging checks

Use `portable/refresh_v27.py` with the exact V2.6.2 baseline ZIP. It checks the baseline SHA/manifest, retains runtime/packages, restages source/UI, fixes timestamps, and validates AMD64 binaries and output CRC. Final archive-specific hashes, count, repeated-build comparison and packaged test results are delivered separately in AstroGraph_Portable_V2.7_PackageVerification.json, avoiding a self-referential hash.

## Preserved boundaries

No change to the src/astrograph analysis core, no migration of task schema, no runtime/dependency upgrade. pypdf remains inherited 5.9.0; only trusted PDFs. Native Windows CHECK/START/python.exe and real LLM quality are unverified. The shell is reusable but the full repository has not been switched to it, and no GitHub write/merge/release was performed.
