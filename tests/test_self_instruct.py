"""Generated authorization must not become human approval or source leakage."""
import hashlib
import json
import pytest
from astrograph.training.self_instruct import GeneratedExample, filter_tasks, prepare_generated
from astrograph.training.small_model import TrainingExample


def examples(root):
    result=[]
    for i, split in enumerate(('train','validation','test')):
        text=f'component_{i} field_{i} source text'
        path=root/f'{i}.txt';path.write_text(text)
        result.append(GeneratedExample(example_id=str(i),task_id='teacher-task',task='RETRIEVAL_QUERY',
            instruction='Retrieve the source component and field.',source_group=f'g{i}',
            source_artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_path=path.name,
            source_excerpt=text,split=split,input={'source_ids':[str(i)],'source_text':text},
            expected={'query':text,'source_ids':[str(i)]},teacher='session assistant',generation_round=1))
    return result


def test_generated_export_is_train_only_and_not_human_approved(tmp_path):
    rows=examples(tmp_path)
    result=prepare_generated(rows,output=tmp_path/'out',source_root=tmp_path)
    assert result['counts']=={'train':1,'validation':1,'test':1}
    assert result['human_reviewed'] is False
    exported=(tmp_path/'out/train_messages.jsonl').read_text(encoding="utf-8")
    assert 'component_0' in exported and 'component_1' not in exported and 'component_2' not in exported
    with pytest.raises(ValueError): TrainingExample.model_validate(rows[0].model_dump())


@pytest.mark.parametrize('update',[{'review_status':'APPROVED'},{'human_reviewed':True},{'teacher':' '},
    {'source_excerpt':'tampered'},{'expected':{'query':'x','source_ids':['invented']}}])
def test_generated_identity_and_citations_are_checked(tmp_path,update):
    row=examples(tmp_path)[0].model_dump()
    row.update(update)
    with pytest.raises(ValueError): GeneratedExample.model_validate(row)


@pytest.mark.parametrize('kind',['group','artifact','source','escape','forbidden','duplicate','conflict','missing_split'])
def test_source_and_split_guards(tmp_path,kind):
    rows=examples(tmp_path); blocked=set()
    if kind=='group': rows[1]=rows[1].model_copy(update={'source_group':rows[0].source_group})
    if kind=='artifact': rows[1]=rows[1].model_copy(update={'source_artifact_sha256':rows[0].source_artifact_sha256})
    if kind=='source': (tmp_path/'0.txt').write_text('changed')
    if kind=='escape': rows[0]=rows[0].model_copy(update={'source_path':'../outside.txt'})
    if kind=='forbidden': blocked={rows[0].source_artifact_sha256}
    if kind=='duplicate': rows.append(rows[0].model_copy(update={'example_id':'other'}))
    if kind=='conflict': rows.append(rows[0].model_copy(update={'example_id':'other','expected':{'query':'different','source_ids':['0']}}))
    if kind=='missing_split': rows.pop()
    with pytest.raises(ValueError): prepare_generated(rows,output=tmp_path/'out',source_root=tmp_path,forbidden_training_hashes=blocked)
    assert not (tmp_path/'out').exists()


def test_task_filter_rejects_paraphrase_duplicates_and_unsupported_tasks():
    first={'task_id':'one','task':'RETRIEVAL_QUERY','instruction':'Generate a query using the source component and parameter names.'}
    result=filter_tasks([first,{**first,'task_id':'two'}, {**first,'task_id':'three','task':'DIAGNOSTIC_REPORT'}])
    assert len(result['accepted'])==1
    assert [r['reason'] for r in result['rejected']]==['NEAR_DUPLICATE_INSTRUCTION','UNSUPPORTED_TASK']


def test_empty_ontology_still_requires_a_real_document(tmp_path):
    row=examples(tmp_path)[0].model_dump()
    row.update(task='ONTOLOGY_EXTRACTION',input={'documents':[]},expected={'triples':[]})
    with pytest.raises(ValueError): GeneratedExample.model_validate(row)
