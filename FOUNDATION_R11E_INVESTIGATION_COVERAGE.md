# R11-E — audited model-selected evidence investigation (development candidate)

## Why
R11-A–D prove the host can receive a bounded model-selected tool plan, run task-bound read-only RAG / GraphRAG / R9 / R10 checks, and return their receipts to the LLM. A valid tool call is **not** evidence that the model asked a useful question, inspected available counterevidence, or settled the right fault. R11-E adds deterministic *investigation coverage accounting*; it does not score diagnostic correctness.

## Deterministic, non-voting interface
`astrograph.tools.material_review.material_investigation_opportunities(material_review)` constructs a bounded inventory **only from the already reviewed, frozen host snapshot**:

- A comparison check if multiple frozen candidates and host-reviewed competition material exist.
- A causal path / blocker search if a candidate has reviewed chains or blocked paths.
- Per-path context (R9) and temporal (R10) checks only when those reviewed verdicts actually exist.
- No model-generated filenames, source URLs, equipment models, observations, edges or new candidate IDs.

The inventory is advisory (up to 32 visible opportunities; `total_opportunities` and `truncated` reveal overflow), with a separate maximum model tool budget of **three calls**. It does **not** assert the flight observation is present.

`audit_material_investigation_plan(plan, opportunities, calls)` compares the model's selected and successfully executed checks to the host inventory. It reports:

- `checks_matching_opportunities`, omitted checks, denied calls, repeated calls and calls with no host-observable reason.
- `NO_REVIEWED_TOOL_OPPORTUNITIES` when approved material exposes nothing that tools could check.
- `MODEL_DECLINED_AVAILABLE_REVIEW` when the model chose zero calls despite known review opportunities.
- `REVIEW_COVERAGE_INCOMPLETE`, `REVIEW_OPPORTUNITY_LIST_TRUNCATED`, or `REVIEWED_TOOL_CHECKS_COVERED_NOT_FLIGHT_VERIFIED` as appropriate.

The `review_coverage_audit` receipt is returned as part of the existing `tool_investigation` payload and therefore reaches the model's final structured explanation request. It is not a diagnosis vote, a root-cause confirmation, or an assumption that absent documentation is contrary flight evidence.

## Tests
- `tests/test_foundation_r11e_investigation_quality.py`: host identity, deliberately skipped review, partial budget, bad IDs/repeats, no toolable paths, unnecessary calls, truncated checklist.
- `tests/test_foundation_r11d_model_wire_protocol.py`: scripted loopback HTTP two-pass protocol now checks the coverage receipt reaches the second request, including the zero-tool-selection case.
- Local R7–R11-E focused regression: **63 passed**, 0 failed on an extracted R11-A base with matching local production edits. Full latest GitHub branch coverage is adjudicated separately by expanded GitHub Actions.

## Boundaries / next acceptance
- Scripted responses and reviewer-authored synthetic bench data do not demonstrate a live LM Studio model's judgment.
- Checklist coverage is a **process check**, not a precision/recall or calibrated diagnostic metric. Model choice remains advisory and never changes frozen candidates.
- Next: actual LM Studio backend integration on a user-owned deployed endpoint; blinded evaluation with no disclosed SADA retrospective answer, contrasting no tools / RAG / GraphRAG / audited tool use. Score tool selection, evidence attribution, justified abstention, contradiction awareness and root-cause correctness separately.
- Keep PR Draft until full project and real-model acceptance.
