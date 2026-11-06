# Win7 Python 3.8 compatibility preview

Approved scope: produce an actual offline portable ZIP for Windows 7 SP1 x64, based on V2.7.1. Preserve six-page UI, external-text diagnosis analysis, citation validation, long reports, follow-up, history/backup/restore. Do not add diagnosis computation or change Win10 distribution.

Runtime: official CPython 3.8.10 embedded x64; pinned Pydantic v2-compatible dependency closure; certifi CA store without disabling TLS validation. Same-machine local model or online HTTPS model modes retained; no new LAN-HTTP mode in this trial. Browser baseline Chromium 109 / Firefox ESR 115; IE unsupported.

Offline means no target-machine pip, Python/Node install, or CDN; external model service and weights remain prerequisites. Win7 SP1 + KB2533623/superseding update + Universal CRT may be necessary system prerequisites. These legacy components are EOL; not a supported-secure enterprise OS promise.

Validation must distinguish Python3.8 Linux logic tests, Windows-host runtime checks, static Win7 PE import checks, and actual Win7 execution (unavailable unless obtained). Keep original hashes and data contracts; no bypass of manifest, TLS, citation or path checks.
