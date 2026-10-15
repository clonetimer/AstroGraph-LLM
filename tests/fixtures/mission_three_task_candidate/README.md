# Mission three-task candidate test fixture

Current neutral test fixture for strict-admission and generic three-task contract tests.

This directory contains only the minimal immutable package assets required by the tests (`manifest.json` and `weights.pt`). It is not a historical run record and does not include the original training report, run directory, acceptance receipt, or development ledger.

The copied weight bytes and manifest are preserved so the existing runtime/admission contract is tested against the same package identity without depending on an archived `post48_dev*` path.
