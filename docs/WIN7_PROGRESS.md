# Win7 compatibility execution ledger

Baseline: V2.7.1 source archive and Win10 ZIP ff12ff5c565ec5086e5a73c43b9a155f6c8988b829b90804aad997fbc9f364a1.
Ruling: isolated local source checkout, no modifications to Win10 deliverable. Remote materials workflow on feat/portable-win7-py38, based on cdfb2a2d29abf2b6c96ab8acb697fd05f8c17a41; this is not equivalent to remote application source synchronization.
Materials acquired through GitHub Actions (container DNS unavailable). Artifact SHA256 73110168ea1e9619176b93e5a3ba984ae50367ed0102923692df85dc43be7d5a verified.

Task 1 complete: official Python3.8.10 embed SHA256 abbe314e9b41603dde0a823b76f5bbbe17b3de3e5ac4ef06b759da5466711271; 7 Windows runtime wheels checked against PyPI receipts. pydantic-core2.27.2 upstream pins Rust1.77 Windows builds; actual wheel static imports inspected (not native Win7 proof).
Task 2: actual locally built Python3.8.20 discovered missing Annotated/importlib.resources.files, runtime-evaluated annotation in verifier, Path.is_relative_to/hashlib.file_digest. Focused regression RED→GREEN and full suite passed 54/55 with one source-only manifest skip. Fixed two test-fixture incompatibilities (str.removesuffix and ReportLab md5 usedforsecurity) without reducing production assertions. PDF fixture now stdlib-only.
Ruling: use Pydantic2.10.6/core2.27.2 + official CP38 wheel and eval-type-backport rather than rewriting validators or downgrading to Pydantic1; maintains strict schema behavior. Native Win7 load remains pending.
Ruling: keep LAN HTTP out of trial. Preserve existing loopback-only/HTTPS endpoint guards. UI preset labels now explicitly do not promise current LM Studio/Ollama Win7 support.
Task 3: SSL context adds verified certifi roots to system trust, keeps hostname/chain validation and TLS>=1.2. Environment guard checks3.8,x64,SP1,AddDllDirectory,UCRT; conditional Windows-only ctypes import (Linux test interpreter lacks _ctypes). CHECK quotes remain safe, delayed expansion disabled.
Windows-host evidence: GitHub run41821203263 succeeded; WindowsServer2022 actually launched embedded3.8.10/core2.27.2 via CMD path with spaces and validated model schema+TLS. Not Chinese-path native test, not full app, not Win7.
Task 4: new locked offline builder's 3 regression tests first failed (builder absent), then passed; pending complete build and final gates.

Task 4: completed on trial build. 61/61 tests in distribution; source 60pass+1manifest-skip. Isolated Python3.8 self-test PASS. UI 3+9+10=22 scenarios PASS. 29 PE files x64, chosen post-Win7 static import scan clear (non-exhaustive).
Final review: author self-review (no subagent tool). Scope preserved; no schema/path/TLS weakening. Historical Win10 scripts are clearly marked non-production for this package.
Ruling: retain official 3.8.10 Windows binary rather than mislabel the compiled Linux3.8.20 as Windows; security fixes after3.8.10 remain an explicit risk. No native Win7/old browser/real model validation claim.
Ruling: use bundled certifi roots in addition to OS store; organization CA deployment still follows user administrator policy. Does not update OpenSSL binaries.
Deferred: native Win7 acceptance, production model quality, old-browser rendering, Python/OpenSSL/PDF security-maintenance upgrade; unchanged from trial boundaries.
