# AstroGraph-LLM Portable V2.6.1 — manifest diagnostics (Preview)

This is a diagnostics and packaging update on top of the SHA-pinned V2.6 preview. No model/analysis code or dependencies change.

- `CHECK.bat` still stops when the SHA-256 manifest fails; it now points to a specific log for troubleshooting instead of labeling all errors as corrupt downloads.
- The verifier reports the failing manifest path, relative file path, actual exception message and errno/WinError. A single inaccessible file no longer produces only `OSError`.
- Extra tests cover injected file permission errors, file stat errors, missing manifest and unmodified package integrity.
- The Windows runtime is inherited and not executed in the Linux test environment; Windows 10 real-device acceptance remains pending.

For old V2.6 installations, use the separate `CHECK_DIAGNOSTIC.bat` + `verify_manifest_diagnostic.py` addition if you want to diagnose without replacing any manifest-protected file. If checksum errors remain, fully re-extract.
