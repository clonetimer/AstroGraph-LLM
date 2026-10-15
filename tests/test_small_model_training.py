from pathlib import Path
import json
import pytest
from astrograph.training.small_model import TrainingExample, prepare_training, evaluate_predictions, preflight
from astrograph.oag.contracts import content_hash


def example(i=0, split='train', group=None):
    inputs = {'source_ids': [f'source-{i}'], 'symptom': f'observed feature {i}'}
    return TrainingExample(example_id=str(i), task='RETRIEVAL_QUERY', source_group=group or f'group-{i}',
                           split=split, input=inputs, expected={'query': f'feature {i}', 'source_ids': [f'source-{i}']},
                           source_sha256=content_hash(inputs), source_artifact_sha256=content_hash(['source',i]),
                           reviewer='named-test-reviewer', review_status='APPROVED')


def test_prepare_keeps_heldout_examples_out_of_trainer_input(tmp_path):
    examples = [example(0), example(1, 'validation'), example(2, 'test')]
    r = prepare_training(examples, output=tmp_path/'plan', task='RETRIEVAL_QUERY', tool_root=tmp_path/'tools', model=tmp_path/'model')
    rows = [json.loads(s) for s in (tmp_path/'plan/train_messages.jsonl').read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert 'feature 0' in rows[0]['messages'][1]['content']
    assert r['status'] == 'PREPARED_NOT_TRAINED'
    assert '--data' in r['command'] and '--model' in r['command']
    assert 'train_messages.jsonl' in r['files']


@pytest.mark.parametrize('kind', ['group', 'duplicate', 'missing_split', 'task', 'artifact'])
def test_training_split_and_task_leakage_rejected(tmp_path, kind):
    rows = [example(0), example(1, 'validation'), example(2, 'test')]
    if kind == 'group': rows[1] = example(1, 'validation', 'group-0')
    if kind == 'duplicate': rows.append(rows[0])
    if kind == 'missing_split': rows.pop()
    if kind == 'artifact': rows[1] = rows[1].model_copy(update={'source_artifact_sha256': rows[0].source_artifact_sha256})
    task = 'ONTOLOGY_EXTRACTION' if kind == 'task' else 'RETRIEVAL_QUERY'
    with pytest.raises(ValueError): prepare_training(rows, output=tmp_path/'plan', task=task, tool_root=tmp_path, model=tmp_path)


def test_review_and_source_hash_are_required():
    row = example().model_dump()
    row['reviewer'] = ' '
    with pytest.raises(ValueError): TrainingExample.model_validate(row)
    row = example().model_dump()
    row['input']['source_ids'] = ['tampered']
    with pytest.raises(ValueError): TrainingExample.model_validate(row)


def test_evaluation_uses_fixed_denominator_and_citation_checks():
    rows = [example(0, 'test'), example(1, 'test')]
    r = evaluate_predictions(rows, {'0': rows[0].expected})
    assert r['n'] == 2 and r['missing_predictions'] == 1 and r['grounded_rate'] == .5
    r = evaluate_predictions(rows, {'0': {'query': 'test', 'source_ids': ['invented']}})
    assert r['schema_valid_rate'] == .5 and r['grounded_rate'] == 0
    with pytest.raises(ValueError): evaluate_predictions([example()], {})


def test_preflight_reports_missing_inputs_without_training(tmp_path):
    r = preflight(tmp_path, tmp_path/'model')
    assert r['status'] == 'BLOCKED' and r['missing_paths']
