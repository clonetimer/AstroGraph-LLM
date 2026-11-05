# Convergence ledger
Baseline delivered V2.5, SHA e0f6a0d274b4570b94dfeff142702cdd918a68d0a2a9224575b418d9dc5cb80d.
- Baseline: 28 tests passed, fresh Linux run.
- Synthesis/provenance/retry/backup: regression first failed, then 36 tests passed.
- Bounded document parser: 2 tests failed, root cause was empty PDF elements reaching Pydantic; after fix 38 tests passed.
- Chromium component run: PDF/DOCX rejected and duplicate comparison sections reproduced; fixed; 3 scenarios passed.
- Offline refresh and integrity checker: 2 tests failed because functions absent; implemented; 40 tests passed.
- Ruling: preserve verified Windows runtime/packages because no network download is available; do not claim PDF dependency upgraded.
- Ruling: current GitHub actions expose only reads; remote write/sync remains incomplete. Deliver source and base-verified patch instead of empty PR.
- Final review: author self-review (no subagent tool). Not a native Windows or production LLM acceptance.
