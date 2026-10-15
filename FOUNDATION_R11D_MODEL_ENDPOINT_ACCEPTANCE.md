# R11-D: real model vs HTTP protocol verification

## What counts as an LLM invocation?

ChatGPT in the interactive conversation **is** an LLM and may review code, design tests and critique synthetic diagnoses. However, a response produced in this conversation is not proof that the AstroGraph application itself called LM Studio, an OpenAI-compatible endpoint or its configured local model.

This project distinguishes:

1. **Host/algorithm tests:** the application invokes deterministic functions and model-shaped Python protocol doubles.
2. **HTTP contract tests (this phase):** the real `OpenAICompatibleStructuredBackend` sends requests to an actual loopback `/v1/chat/completions` HTTP server and parses responses. The loopback server returns **preprogrammed responses, not neural-network inference**. These tests exercise the genuine JSON/HTTP/schema/dispatcher path, but **not** model reasoning quality.
3. **Live model tests (still required):** AstroGraph must connect to a running real model endpoint, select the allow-listed tool, accept actual tool feedback, generate a structured explanation, and pass independent evidence/claim and cross-task tests.

## Implemented test coverage

`tests/test_foundation_r11d_model_wire_protocol.py` exercises:

- A two-request HTTP exchange with model-shaped selection and explanation, including JSON-object `response_format`, request IDs, tool receipts and nested results.
- R9 condition check and R10 temporal check as read-only returned evidence. Static R9 context conflict does not by itself overrule reviewed R10 sequential compatibility. Neither constitutes observed flight proof.
- A model-supplied fault ID outside the frozen candidate set is denied.
- A model-supplied unregistered tool name fails closed.
- A generated candidate ID that tries to override the upstream frozen decision is rejected.
- A reviewed material bundle with an unrelated source group does not reach the model's tool selector.

All test material concerns fictional pump faults and synthetic bench paths. No already-disclosed retrospective SADA root cause is included in the inference corpus.

## Real local model acceptance gate (not passed)

To run an actual acceptance test, start a local OpenAI-compatible server with a known model; for example, enable LM Studio's Local Server and provide the model ID, base URL and local access restrictions. Never commit API keys or private diagnosis data. Configure AstroGraph's existing backend, enable `ASTROGRAPH_PLATFORM_MATERIAL_TOOL_CALLS_ENABLED=1` in a controlled test environment and use task-frozen, independently reviewed material.

Acceptance evidence should capture:
- Exact model ID, quantization/build, provider version, response format and context budget.
- Model response to at least five independent synthetic tasks including cross-task and contradictory reference data, not simply a seeded known-answer case.
- Whether tool selector returns valid schema and only task-authorized IDs; number of executed calls and denied requests; tool receipts and response hashes.
- Whether final model claims are source-supported, distinguish missing evidence from refutation, and preserve upstream candidate and no-control boundaries.
- Timeout, truncated/incomplete JSON, unavailable model and retry/failure behavior.

The CI does **not** run a real model, so this phase may be called **R11-D HTTP integration candidate**, not a verified real-model release. Windows/LM Studio and independent reviewer acceptance remain open gates.
