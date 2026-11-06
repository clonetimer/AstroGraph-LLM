# External diagnosis analysis implementation plan

Spec: `docs/superpowers/specs/2026-11-02-external-diagnosis-llm.md`

Base: `0cd07bdef83df0f34a6fa8530161210459586d5f` on
`dev/foundation-r11-integration`; isolated branch `feat/external-diagnosis-llm`.

## Global constraints

No diagnosis calculation, telemetry classifiers, R20/BRB migration, new heavyweight
dependencies or whole-archive replacement. Keep legacy manifests and existing
frozen-diagnosis contracts compatible. Do not merge the broad draft foundation PR
as part of this narrow change. Use a stacked pull request against its branch.

## Tasks and interfaces

1. **Import and dispatch (root).** Test single-report manifests and nontruncated
   text first. Make log/reference lists optional, add the explicit analysis mode,
   and dispatch it to a dedicated report path. Add `analyze-report` CLI and API
   capability metadata. Expected: missing required diagnosis still fails; a
   declared-but-missing optional file still fails; legacy mode remains unchanged.
2. **Model contract (delegated).** Test and implement
   `load_external_diagnosis_input` and `run_external_diagnosis_analysis` in a new
   module. Output is `generation` plus `model_explanation`, with task-bound sources
   and separate success/failure states. Expected: full source reaches the model;
   incorrect binding/citations are rejected; failures expose fixed reason codes.
3. **Presentation (delegated UI; root export).** UI manifests select the new mode;
   both report views and Markdown render summary, findings, hypotheses, gaps,
   suggestions and source locators. Expected: model text is preserved, escaped
   and never confused with deterministic legacy output. All output fields use the
   same schema from task 2; source IDs match the registry, not legacy catalogue IDs.
4. **Integration and review (root + independent reviewer).** Exercise API upload,
   preflight, HTTP model call, stored JSON and Markdown with a scripted local
   provider; execute the actual JS renderers in Node. Run the affected regression
   suite and foundation gate, inspect the diff, fix material review findings,
   commit and create a stacked PR with test evidence.

## Review focus

Check ordinary prose without catalogue matches, optional-file role separation,
long lines, incorrect/missing source IDs, source tampering and task mismatches,
provider failures containing secrets, HTML in uploaded/model text, empty reports,
report status versus model status, both UI variants, optional logs/references,
and whether the external route accidentally invokes diagnosis or graph code.

## Decisions

- Explicit mode with legacy default preserves existing contracts and avoids
  weakening the old material guard globally.
- Source-bound free analysis is displayed as model interpretation/hypotheses;
  citation validity is not represented as semantic verification.
- A visible input limit is implemented now; generic long-document chunking and
  production-model quality evaluation are outside this first change.

## Independent review and fixes

- The independent review found that the native DOCX parser kept headings outside
  its text elements. Added opt-in heading elements for the external route, keeping
  legacy parsing defaults unchanged. Heading + body, heading-only, and interleaved
  headings/tables now retain source order and unique original paragraph locators.
  All three new regressions failed before the fix and passed afterward.
- The legacy job diagnostics still claimed core diagnosis ran when no LLM was
  configured. Replaced that label with an accurate model-only status; an actual
  JS diagnostics-renderer assertion failed first and then passed.
- Targeted review-fix and document-compatibility verification: 33 tests passed.
- Before these review fixes, the complete 33-file foundation gate passed 318 tests
  with 3 expected skips (two unavailable archived evidence fixtures and one
  Windows-only check). The expanded gate includes the DOCX compatibility tests.
- Final expanded 36-file foundation gate on the reviewed fixes: **340 passed,
  3 skipped**, exit 0, 152.01 seconds. The skipped cases are the same existing
  archive/Windows conditions; all added external-analysis tests ran and passed.
- Live production-model reasoning quality and browser pixel/accessibility review
  remain unverified. Protocol and executable-renderer tests do not claim either.
  Broader historical diagnosis changes remain outside the user's requested scope.
