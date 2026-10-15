"""Prepare independent adapters for extraction, retrieval planning and diagnosis.

The existing fine_tune trainer remains the execution backend. Nothing in a
training artifact authorizes publication of model output or ontology changes.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from astrograph.oag.contracts import OAGExplanation, OntologyProposal, content_hash

TASKS = ('ONTOLOGY_EXTRACTION', 'RETRIEVAL_QUERY', 'DIAGNOSTIC_REPORT')


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    query: str = Field(min_length=1, max_length=500)
    source_ids: list[str] = Field(min_length=1)


OUTPUTS = {'ONTOLOGY_EXTRACTION': OntologyProposal, 'RETRIEVAL_QUERY': RetrievalQuery,
           'DIAGNOSTIC_REPORT': OAGExplanation}


class TrainingExample(BaseModel):
    """Source-group split and named review travel with each supervised example."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    example_id: str = Field(min_length=1)
    task: Literal['ONTOLOGY_EXTRACTION', 'RETRIEVAL_QUERY', 'DIAGNOSTIC_REPORT']
    source_group: str = Field(min_length=1)
    split: Literal['train', 'validation', 'test']
    input: dict
    expected: dict
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_artifact_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    reviewer: str = Field(min_length=1)
    review_status: Literal['APPROVED']

    @model_validator(mode='after')
    def validate_reviewed_content(self):
        if not self.reviewer.strip() or not self.source_group.strip():
            raise ValueError('named review and source group are required')
        if content_hash(self.input) != self.source_sha256:
            raise ValueError('training source hash mismatch')
        validate_task_target(self.task, self.input, self.expected)
        return self


def validate_task_target(task: str, inputs: dict, expected: dict) -> None:
    """Validate output structure and grounding without assigning review authority."""
    OUTPUTS[task].model_validate(expected)
    if task == 'ONTOLOGY_EXTRACTION':
        documents = {d['document_id']: d['text'] for d in inputs.get('documents', [])}
        for t in expected.get('triples', []):
            if t['source_document_id'] not in documents or t['source_quote'] not in documents[t['source_document_id']]:
                raise ValueError('extraction target is not source-grounded')
            if any(term.casefold() not in t['source_quote'].casefold() for term in (t['subject'], t['object'])):
                raise ValueError('extraction entity must occur in the quoted source')
    if task == 'DIAGNOSTIC_REPORT':
        from astrograph.oag.runtime import _validate_explanation
        from astrograph.oag.contracts import LiteratureDocument
        from astrograph.oag.assessment import assess_summary, validate_assessment
        expected = OAGExplanation.model_validate(expected)
        summary = inputs['summary']
        docs = [LiteratureDocument.model_validate(d) for d in inputs.get('documents', [])]
        _validate_explanation(expected, summary, docs)
        if expected.assessment is None:
            raise ValueError('diagnosis training target must include the evidence assessment')
        validate_assessment(expected.assessment, assess_summary(summary))
    if task == 'RETRIEVAL_QUERY':
        available = set(inputs.get('source_ids', []))
        if not set(expected['source_ids']) <= available:
            raise ValueError('retrieval target cites an unknown source')


def prepare_training(examples: list[TrainingExample], *, output: Path, task: str,
                     tool_root: Path, model: Path, gpu: int = 0, python_executable: Path | None = None) -> dict:
    """Export train-only messages with held-out groups and exact trainer arguments."""
    if task not in TASKS or not examples or any(e.task != task for e in examples):
        raise ValueError('prepare exactly one nonempty task-specific adapter dataset')
    groups, identities, contents, artifacts = {}, set(), {}, {}
    for e in examples:
        if e.example_id in identities:
            raise ValueError('duplicate example ID')
        identities.add(e.example_id)
        if groups.setdefault(e.source_group, e.split) != e.split:
            raise ValueError('source group crosses train/validation/test')
        if contents.setdefault(e.source_sha256, e.split) != e.split:
            raise ValueError('duplicate input crosses train/validation/test')
        if artifacts.setdefault(e.source_artifact_sha256, e.split) != e.split:
            raise ValueError('source artifact crosses train/validation/test under different group names')
    if set(groups.values()) != {'train', 'validation', 'test'}:
        raise ValueError('independent train, validation and test source groups are required')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for split in ('train', 'validation', 'test'):
        rows = [e for e in examples if e.split == split]
        path = output / (split + '.jsonl')
        path.write_text(''.join(json.dumps(e.model_dump(mode='json'), ensure_ascii=False) + '\n' for e in rows), encoding='utf-8')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    messages = output / 'train_messages.jsonl'
    schema = OUTPUTS[task].model_json_schema()
    messages.write_text(''.join(json.dumps({'messages': [
        {'role': 'system', 'content': 'Task: ' + task + '. Return only grounded JSON matching: ' + json.dumps(schema)},
        {'role': 'user', 'content': json.dumps(e.input, ensure_ascii=False)},
        {'role': 'assistant', 'content': json.dumps(e.expected, ensure_ascii=False)},
    ]}, ensure_ascii=False) + '\n' for e in examples if e.split == 'train'), encoding='utf-8')
    hashes[messages.name] = hashlib.sha256(messages.read_bytes()).hexdigest()
    command = [str(python_executable.absolute() if python_executable else tool_root.resolve() / '.venv/bin/python'), str(tool_root.resolve() / 'train_sft.py'),
               '--model', str(model.resolve()), '--data', str(messages), '--output', str(output / 'adapter'),
               '--gpu', str(gpu), '--batch-size', '1', '--grad-accum', '8', '--epochs', '1', '--seed', '3407']
    receipt = {'schema_version': 'astrograph-small-model-training-plan.v1', 'task': task,
               'status': 'PREPARED_NOT_TRAINED', 'example_count': len(examples), 'group_splits': groups,
               'working_directory': str(output),
               'files': hashes, 'command': command, 'model_path': str(model.resolve()),
               'selection_policy': 'Compare unchanged base versus adapter on validation; evaluate test once after selection.',
               'promotion_policy': 'No automatic registration. Require held-out schema/citation/abstention non-regression.'}
    (output / 'plan.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


def preflight(tool_root: Path, model: Path, python_executable: Path | None = None) -> dict:
    """Probe the supplied environment without installing packages or loading weights."""
    python = python_executable or tool_root / '.venv/bin/python'
    script = tool_root / 'train_sft.py'
    missing = [str(p) for p in (python, script, model / 'config.json') if not p.is_file()]
    result = {'schema_version': 'astrograph-small-model-preflight.v1', 'missing_paths': missing,
              'model': str(model), 'trainer': str(script), 'status': 'BLOCKED'}
    if missing:
        return result
    code = ("import importlib.util,json,sys; names=['torch','unsloth','trl','datasets','peft']; "
            "r={n:importlib.util.find_spec(n) is not None for n in names}; "
            "import torch; r['cuda_available']=torch.cuda.is_available(); "
            "r['gpu_count']=torch.cuda.device_count(); "
            "from transformers.models.auto.configuration_auto import CONFIG_MAPPING; "
            "c=json.load(open(sys.argv[1])); r['model_architecture_supported']=c['model_type'] in CONFIG_MAPPING; "
            "print(json.dumps(r))")
    p = subprocess.run([str(python), '-c', code, str(model / 'config.json')], capture_output=True, text=True, timeout=60)
    if p.returncode:
        result['reason'] = 'TRAINING_ENVIRONMENT_PROBE_FAILED'
        result['exit_code'] = p.returncode
        return result
    result['environment'] = json.loads(p.stdout.splitlines()[-1])
    result['trainer_sha256'] = hashlib.sha256(script.read_bytes()).hexdigest()
    result['status'] = 'READY_FOR_REVIEWED_DATA' if all(result['environment'].values()) else 'BLOCKED'
    result['training_performed'] = False
    return result


def evaluate_predictions(examples: list[TrainingExample], predictions: dict[str, dict]) -> dict:
    """Fixed denominator: missing, malformed and ungrounded outputs count as failures."""
    if not examples or any(e.split == 'train' for e in examples):
        raise ValueError('evaluate a nonempty held-out set')
    ids = [e.example_id for e in examples]
    if len(ids) != len(set(ids)) or set(predictions) - set(ids):
        raise ValueError('duplicate examples or extra predictions')
    schema_ok = grounded = exact = 0
    for e in examples:
        p = predictions.get(e.example_id)
        if p is None:
            continue
        try:
            OUTPUTS[e.task].model_validate(p)
            schema_ok += 1
            TrainingExample.model_validate({**e.model_dump(), 'expected': p})
            grounded += 1
            exact += p == e.expected
        except (ValueError, KeyError, TypeError):
            pass
    n = len(examples)
    return {'n': n, 'schema_valid_rate': schema_ok / n, 'grounded_rate': grounded / n,
            'exact_target_rate': exact / n, 'missing_predictions': n - len(predictions),
            'claim_boundary': 'Contract/citation compliance only; semantic correctness requires independent review.'}
