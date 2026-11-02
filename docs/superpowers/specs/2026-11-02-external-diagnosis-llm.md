# External diagnosis text → LLM analysis → report

## User-approved scope

Import an externally produced diagnosis as readable text, let the configured LLM
analyse that text, and present its actual structured response in the interface and
JSON/Markdown downloads. How the upstream diagnosis was calculated is outside
this project change. One diagnosis file is sufficient; logs and references are
optional. Unknown fault names and ordinary narrative do not require catalogue IDs.

## Contract

New UI submissions and the `analyze-report` CLI command explicitly select
`analysis_mode: EXTERNAL_DIAGNOSIS_ANALYSIS` in the existing upload manifest.
Manifests without this field retain the existing `MATERIAL_REVIEW` behaviour.
The external mode does not run detection, fault rules, graph construction, case
learning, or the legacy extract-only material answer guard.

The host binds each source segment to the current task, original file hash,
filename, role and locator. All parsed text reaches the model within an explicit
60,000-character aggregate limit; over-limit inputs receive a visible status and
are never silently shortened. A model provider may impose a smaller context limit.

The model output contains summary, source-cited findings, source-cited cause
hypotheses, information gaps and review suggestions. A finding is the model's
reading of the report, not a newly verified measurement. Hypotheses may use new
fault descriptions without changing the imported diagnosis. The response schema
and every citation are validated against the current source registry. This checks
structure and provenance, not the truth of the model's reasoning.

Success, missing configuration, provider/validation failure and excessive input
size remain distinct in JSON, Markdown and both report interfaces. Source text
and model text are escaped in HTML. Public output contains no raw provider errors,
credentials or unrestricted backend receipts.

## Acceptance

- A single ordinary TXT/MD report can be submitted from the UI, API and CLI.
- The complete source, including text after character 2,500 on a long line,
  reaches the real OpenAI-compatible HTTP transport.
- The model response survives into the stored report, both renderers and Markdown.
- Fabricated/cross-task citations fail closed; missing configuration and failed
  model calls cannot be shown as successful LLM analysis.
- Existing frozen diagnosis and material-review guard tests remain green.
- Automated transport tests use a local scripted HTTP server and are identified
  as protocol tests, not evidence of real-model analysis quality.
