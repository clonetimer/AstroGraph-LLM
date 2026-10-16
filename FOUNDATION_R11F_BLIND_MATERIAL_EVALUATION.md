# R11-F — Truth-isolated four-arm evaluation of **post-diagnosis explanations**

## Why this is a new evaluation
R11-A–E provide RAG, GraphRAG, expert-reviewed R9/R10 causal checks, task-frozen read-only tools, HTTP protocol checks, and investigation coverage receipts. **The LLM is not authorized to change frozen diagnosis candidates.** Accordingly, counting identical candidate IDs across these four post-diagnosis arms as four different classification accuracies would be a misleading experiment.

This campaign evaluates **analysis quality**: source entailment, contradiction/negative evidence handling, justified abstention, actionability, evidence attribution, useful/denied/repeated tool checks and selective failure. Frozen upstream classification accuracy is external and cannot be attributed to an explanation-only method.

## Paired arm contract

| Arm | Frozen diagnosis | Local RAG documents | Same-task reviewed GraphRAG | LLM tool selector |
|---|---|---|---|---|
| `LLM_ONLY` | yes | no | no | off |
| `LLM_RAG` | yes | yes | no | off |
| `LLM_RAG_GRAPHRAG` | yes | yes | yes, if supplied | off |
| `LLM_FULL_TOOLS` | yes | yes | yes, if supplied | on, maximum three calls |

Every arm receives **identical** frozen fault candidates, task identity and evidence. No arm can modify the upstream decision. PaperSeek/A2O/live external retrieval is disabled across all arms to avoid arm-specific changing reference corpora. The models are configured by the operator; the host should pin model ID, weights/template identity, temperature, sampling parameters, retrieval bundle and dataset/group holdout.

A case without approved GraphRAG material does *not* establish the benefit of graph augmentation: `LLM_RAG_GRAPHRAG` and `LLM_FULL_TOOLS` may then have no extra reference to use. Do not aggregate such cases as positive evidence of a graph effect.

## Inputs and leakage boundary
`scripts/run_r11f_blind_material_campaign.py` accepts a JSON manifest:
```json
{
  "schema_version": "astrograph-blind-inputs.v1",
  "cases": [
    {
      "case_id": "BENCH-01",
      "group_id": "BENCH-FAMILY-01",
      "task": "Explain the frozen fault candidates without making a root-cause claim.",
      "frozen_diagnosis": {
        "candidate_fault_id": "FAULT_A",
        "candidate_fault_ids": ["FAULT_A", "FAULT_B"],
        "material_source_group_id": "BENCH-01",
        "evidence": []
      },
      "reviewed_material": null,
      "document_paths": ["bench-reference.md"]
    }
  ]
}
```
The referenced document must be under the manifest's directory, in TXT/Markdown/DOCX/PDF form. Fill `reviewed_material` only with same-task, independently approved material reasoning (matching `source_group_id` and *exact* candidate set). Otherwise leave it `null`, never synthesize expert approval. Real holdout labels, answer keys, retrospective root-cause notes and evaluator judgments belong outside the model inputs and retrieval corpus. Known SADA retrospective answers are excluded. Key-level guards catch many explicit contamination patterns but **cannot detect semantic label leakage inside arbitrary prose, document text, pretraining or model memory**; independent dataset review is still required.

### Safe preflight (no LLM calls)
```bash
python scripts/run_r11f_blind_material_campaign.py --manifest path/to/inputs.json --output path/to/new-campaign
```

### Operator-authorized real local model execution
Set a real OpenAI-compatible endpoint and model accessible **from the machine running AstroGraph**; the sample uses LM Studio local loopback:
```bash
export ASTROGRAPH_PLATFORM_LLM_BASE_URL=http://127.0.0.1:1234/v1
export ASTROGRAPH_PLATFORM_LLM_MODEL=<operator-installed-model-id>
python scripts/run_r11f_blind_material_campaign.py \
  --manifest path/to/inputs.json --output path/to/new-campaign --allow-live
```
A fresh `--output` directory is mandatory. There is no hidden default live model invocation. If no configured model is reachable, a generation failure is **not** a scored model-quality success.

## Independent after-run scoring
Reviewer ratings **must be produced after the four model cells finish**, in a separate JSON file:
```json
{
  "schema_version": "astrograph-material-llm-human-ratings.v1",
  "reviews": [
    {
      "case_id": "BENCH-01",
      "arm": "LLM_ONLY",
      "evidence_entailment": 0,
      "counterevidence_recognition": 0,
      "abstention_appropriateness": 0,
      "review_actionability": 0
    }
  ]
}
```
Example zeroes are placeholders, not adjudicated scores; supply all case×arm cells, using integer 0..4 for every rubric. Execute:
```bash
python scripts/score_r11f_blind_material_campaign.py \
  --campaign-root path/to/new-campaign --ratings path/to/separate-ratings.json
```
The scoring step checks the hash of the frozen campaign manifest, rejects missing/duplicate/foreign ratings and never calls a model. It writes a single `human_scores.json` without overwriting existing scores.

**Limitation:** this initial reviewer format exposes arm names, so this is a **label-blinded model-input experiment, not a fully assessor-blinded study**. A later registered protocol can export anonymized answer packets and conceal arms from independent reviewers. Do not make a statistical superiority claim from one or a few synthetic cases.

## Engineering verification
- `tests/test_foundation_r11f_blind_material_campaign.py`: four-arm isolation, truth-key rejection, frozen candidate binding, non-overwrite, reviewer-only scores, no-graph case.
- `tests/test_foundation_r11f_cli_and_separate_scoring.py`: no-live preflight, malicious path and answer-key tests, separate post-run scoring, manifest tampering.
- No current LM Studio model or independent real holdout labels were accessed by GitHub CI. CI test doubles are **not evidence that a real LLM improved diagnostics**.
- Existing `src/astrograph/evaluation/llm_method_ablation.py` measures a distinct upstream-controlled agentic experiment; this R11-F campaign specifically isolates downstream explanation.
