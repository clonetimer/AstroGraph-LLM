# Win7 portable implementation plan

Goal: self-contained Win7 SP1 x64 + Python3.8 preview ZIP.
Spec: WIN7_SPEC.md.

- [x] Acquire official runtime + pinned Windows and Linux test wheels, verify publisher hashes and native-import baseline.
- [x] Run existing suite under Python3.8; add regression tests for Annotated/runtime type aliases, path containment, subprocess PDF/DOCX parsing, hashing, UTF-8 paths and TLS trust. Observe failures before compatibility edits.
- [x] Adapt canonical minimal source and deployment scripts; keep UI layout/business contract unchanged, add explicit legacy requirements and self-test logs.
- [x] Build from locked local materials; validate no cp314 or newer-only wheel leakage, all PE x64, exact manifest, no font/model files.
- [x] Re-run whole portable suite, isolated real HTTP self-test and UI regression, validate repeat build; supply ZIP + source + Chinese steps + hashes.

Review focus: startup quoting/Chinese paths, Python3.8 runtime annotations, unsafe paths in document attachments, HTTPS verification/roots, accidental heavy dependency import.
