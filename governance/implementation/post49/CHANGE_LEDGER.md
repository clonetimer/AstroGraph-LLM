<!-- CURRENT_DOCUMENT: 0.7.1.post49 -->
# post49 current-source delivery ledger

- Profile: `post49-current-delivery-source-r1`
- Historical records bundled: **no**.
- Product/runtime functions intentionally preserved; prior reports/receipts/release contracts removed.
- Mutable failure/passport/problem ledgers initialized empty.
- Current operational gate inputs retained only as enumerated in `governance/evidence/CURRENT_RUNTIME_ASSETS.json`.
- Validation discovered that docs-r1 had changed hash-pinned ADCS evidence JSON provenance strings. The delivery restores the four governed ADCS config files to their original `0.7.1.post49` stable bytes; no whitelist or qualification rule was broadened.
- The neutral three-task test package lives under `tests/fixtures/mission_three_task_candidate/`, so functional tests no longer depend on a `post48_dev4` run directory.
- Historical audit/reproduction requires a separate archive package.
