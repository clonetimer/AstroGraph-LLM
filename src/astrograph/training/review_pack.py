"""Content-bound review handoff; generated answers are never human approvals."""
from __future__ import annotations
import csv,json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator
from astrograph.oag.contracts import content_hash
from .small_model import TrainingExample,TASKS


class ReviewDraft(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True)
    example_id:str=Field(min_length=1)
    task:Literal['ONTOLOGY_EXTRACTION','RETRIEVAL_QUERY','DIAGNOSTIC_REPORT']
    source_group:str=Field(min_length=1)
    source_artifact_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    source_url:str=Field(min_length=1)
    split:Literal['train','validation','test']
    input:dict
    candidate_output:dict
    candidate_author:str=Field(min_length=1)
    origin:Literal['SOURCE_DERIVED_DRAFT','SYNTHETIC']= 'SOURCE_DERIVED_DRAFT'
    review_status:Literal['REVIEW_REQUIRED']='REVIEW_REQUIRED'


def write_review_pack(drafts:list[ReviewDraft], output:Path)->dict:
    if not drafts or len({x.example_id for x in drafts})!=len(drafts):
        raise ValueError('nonempty unique review examples required')
    groups,artifacts={},{}
    for d in drafts:
        if groups.setdefault(d.source_group,d.split)!=d.split or artifacts.setdefault(d.source_artifact_sha256,d.split)!=d.split:
            raise ValueError('one original source cannot cross frozen splits')
    body={'schema_version':'astrograph-training-review-pack.v1','drafts':[d.model_dump(mode='json') for d in drafts],
          'review_status':'REVIEW_REQUIRED','split_policy':'Frozen by original source before model evaluation; do not reuse existing evaluated gold as training data.'}
    pack={**body,'pack_sha256':content_hash(body)}
    output.mkdir(parents=True,exist_ok=False)
    (output/'pack.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2)+'\n')
    fields=['example_id','pack_sha256','input_sha256','candidate_sha256','task','frozen_split','source_url',
            'review_status','reviewer','review_comment','corrected_output_json']
    with (output/'review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for d in drafts:
            writer.writerow({'example_id':d.example_id,'pack_sha256':pack['pack_sha256'],
                'input_sha256':content_hash(d.input),'candidate_sha256':content_hash(d.candidate_output),
                'task':d.task,'frozen_split':d.split,'source_url':d.source_url,'review_status':'REVIEW_REQUIRED',
                'reviewer':'','review_comment':'','corrected_output_json':json.dumps(d.candidate_output,ensure_ascii=False)})
    return pack


def export_reviewed(pack_path:Path,reviews_path:Path,output:Path)->dict:
    pack=json.loads(pack_path.read_text(encoding='utf-8'));digest=pack.pop('pack_sha256')
    if content_hash(pack)!=digest:raise ValueError('review pack integrity failure')
    drafts=[ReviewDraft.model_validate(x) for x in pack['drafts']];by_id={d.example_id:d for d in drafts}
    if len(by_id)!=len(drafts):raise ValueError('duplicate source draft ID')
    with reviews_path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=len(by_id) or {x['example_id'] for x in rows}!=set(by_id):
        raise ValueError('review must cover the exact pack without duplicate or missing rows')
    approved=[];decisions={}
    for row in rows:
        d=by_id[row['example_id']]
        checks={'pack_sha256':digest,'input_sha256':content_hash(d.input),'candidate_sha256':content_hash(d.candidate_output),
                'task':d.task,'frozen_split':d.split,'source_url':d.source_url}
        if any(row[k]!=v for k,v in checks.items()):raise ValueError('review identity, source or frozen split changed')
        decision=row['review_status'];decisions[decision]=decisions.get(decision,0)+1
        if decision not in {'APPROVED','REJECTED','DEFERRED','REVIEW_REQUIRED'}:raise ValueError('unknown review decision')
        if decision!='APPROVED':continue
        if not row['reviewer'].strip() or not row['review_comment'].strip():
            raise ValueError('approval needs a named reviewer and review rationale')
        if d.origin=='SYNTHETIC':raise ValueError('synthetic smoke data cannot be exported as domain-reviewed training data')
        approved.append(TrainingExample(example_id=d.example_id,task=d.task,source_group=d.source_group,
            split=d.split,input=d.input,expected=json.loads(row['corrected_output_json']),
            source_sha256=content_hash(d.input),source_artifact_sha256=d.source_artifact_sha256,
            reviewer=row['reviewer'].strip(),review_status='APPROVED'))
    if not approved:raise ValueError('no approved domain examples; do not treat drafts as training labels')
    output.mkdir(parents=True,exist_ok=False)
    files={}
    for task in TASKS:
        selected=[x for x in approved if x.task==task]
        if not selected:continue
        name='reviewed-'+task+'.jsonl'
        (output/name).write_text(''.join(x.model_dump_json()+'\n' for x in selected));files[task]=name
    receipt={'schema_version':'astrograph-review-export.v1','pack_sha256':digest,'decisions':decisions,
             'approved_examples':len(approved),'files':files,'training_performed':False}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
