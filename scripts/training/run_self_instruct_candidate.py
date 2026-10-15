"""Train one generated-data candidate and compare frozen source-held-out outputs.

Uses the supplied fine_tune environment; never registers the resulting adapter.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from astrograph.training.self_instruct import GeneratedExample, messages
from astrograph.training.small_model import OUTPUTS, validate_task_target


def score(example, raw):
    result = {'schema_valid': False, 'grounded': False, 'exact_target': False}
    try:
        parsed = json.loads(raw)
        OUTPUTS[example.task].model_validate(parsed)
        result['schema_valid'] = True
        validate_task_target(example.task, example.input, parsed)
        if example.task == 'RETRIEVAL_QUERY':
            # Exact target is the strict reproducible query criterion; source IDs
            # alone do not establish semantic grounding of the query text.
            result['grounded'] = parsed == example.expected
        else:
            result['grounded'] = True
        def normalize(x):
            x = dict(x)
            if 'triples' in x: x['triples'] = sorted(x['triples'], key=lambda t: json.dumps(t, sort_keys=True))
            return x
        result['exact_target'] = normalize(parsed) == normalize(example.expected)
    except (ValueError, KeyError, TypeError):
        pass
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--base-validation-dir', type=Path, help='Reuse a completed identical-settings base evaluation after an execution failure')
    a=p.parse_args()
    a.model=a.model.resolve(); a.corpus=a.corpus.resolve(); a.output=a.output.resolve()
    if a.base_validation_dir: a.base_validation_dir=a.base_validation_dir.resolve()
    a.output.mkdir(parents=True, exist_ok=False); os.chdir(a.output)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_DISABLED='true')
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    torch.set_num_threads(4)
    rows={s: [GeneratedExample.model_validate_json(line) for line in (a.corpus/(s+'.jsonl')).read_text().splitlines()] for s in ('train','validation','test')}
    hashes={s: hashlib.sha256((a.corpus/(s+'.jsonl')).read_bytes()).hexdigest() for s in rows}
    receipt=json.loads((a.corpus/'receipt.json').read_text())
    if any(hashes[s] != receipt['files'][s+'.jsonl'] for s in rows): raise ValueError('frozen corpus changed')
    if any(e.split != s for s in rows for e in rows[s]): raise ValueError('split label mismatch')
    for left, right in (('train','validation'), ('train','test'), ('validation','test')):
        if {e.source_artifact_sha256 for e in rows[left]} & {e.source_artifact_sha256 for e in rows[right]}: raise ValueError('source leakage')
    settings={'model': str(a.model), 'corpus_hashes': hashes, 'seed':3407, 'epochs':3, 'learning_rate':2e-4,
        'lora_r':16, 'lora_alpha':32, 'target_modules':['q_proj','k_proj','v_proj','o_proj'], 'max_length':2048,
        'batch_size':2, 'gradient_accumulation':4, 'max_new_tokens':768, 'do_sample':False,
        'selection_policy':'One fixed candidate. Validation comparison precedes final test. No tuning after test.',
        'loss':'completion_only', 'human_reviewed':False, 'production_promotion_allowed':False}
    (a.output/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    model, tokenizer=FastLanguageModel.from_pretrained(model_name=str(a.model), max_seq_length=2048,
        load_in_4bit=False, load_in_16bit=True, local_files_only=True)
    if hasattr(tokenizer,'tokenizer'): tokenizer=tokenizer.tokenizer
    results={}
    def evaluate(phase, split):
        FastLanguageModel.for_inference(model)
        outputs=[]
        for e in rows[split]:
            batch=tokenizer.apply_chat_template(messages(e, include_answer=False), tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors='pt', return_dict=True).to('cuda:0')
            with torch.inference_mode(): generated=model.generate(**batch, max_new_tokens=768, do_sample=False)
            raw=tokenizer.decode(generated[0,batch['input_ids'].shape[1]:], skip_special_tokens=True)
            outputs.append({'example_id':e.example_id,'task':e.task,'raw_output':raw,**score(e,raw)})
            (a.output/f'{phase}-{split}-predictions.jsonl').write_text(''.join(json.dumps(o,ensure_ascii=False)+'\n' for o in outputs))
            print(f'{phase} {split}: {len(outputs)}/{len(rows[split])}', flush=True)
        def aggregate(items):
            return {'n':len(items), **{k:sum(o[k] for o in items)/len(items) for k in ('schema_valid','grounded','exact_target')}}
        summary={**aggregate(outputs), 'by_task':{t:aggregate([o for o in outputs if o['task']==t]) for t in OUTPUTS if any(o['task']==t for o in outputs)}}
        results[f'{phase}_{split}']=summary
        (a.output/'evaluation.json').write_text(json.dumps({'results':results,'production_promotion_allowed':False,
            'claim_boundary':'Generated simulator-source held-out contract tasks only. Exact target includes requested selection; grounded ontology alone may still select wrong effects. No human semantic or diagnostic qualification.'},indent=2)+'\n')
        return summary
    if a.base_validation_dir:
        if json.loads((a.base_validation_dir/'settings.json').read_text()) != settings:
            raise ValueError('cannot reuse evaluation with different settings')
        path=a.base_validation_dir/'base-validation-predictions.jsonl'
        cached=[json.loads(line) for line in path.read_text().splitlines()]
        if [o['example_id'] for o in cached] != [e.example_id for e in rows['validation']]:
            raise ValueError('incomplete or different base validation')
        for e, o in zip(rows['validation'], cached):
            if any(o[k] != v for k,v in score(e,o['raw_output']).items()):
                raise ValueError('cached score mismatch')
        results['base_validation']=json.loads((a.base_validation_dir/'evaluation.json').read_text())['results']['base_validation']
        (a.output/path.name).write_bytes(path.read_bytes())
        (a.output/'reused-base-validation.json').write_text(json.dumps({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2)+'\n')
    else:
        evaluate('base','validation')
    # Clear inference flags before wrapping: PEFT delegates attribute lookup to
    # the base model, while Unsloth's flag deletion acts on the wrapper itself.
    FastLanguageModel.for_training(model)
    model=FastLanguageModel.get_peft_model(model,r=16,target_modules=settings['target_modules'],lora_alpha=32,
        lora_dropout=0,bias='none',use_gradient_checkpointing=False,random_state=3407)
    data=[]; lengths=[]
    for e in rows['train']:
        prompt=tokenizer.apply_chat_template(messages(e,include_answer=False),tokenize=False,add_generation_prompt=True,enable_thinking=False)
        full=tokenizer.apply_chat_template(messages(e,include_answer=True),tokenize=False,add_generation_prompt=False,enable_thinking=False)
        if not full.startswith(prompt): raise ValueError('chat template completion boundary mismatch')
        length=len(tokenizer(full,add_special_tokens=False)['input_ids']); lengths.append(length)
        if length > 2048: raise ValueError('target would be truncated')
        data.append({'prompt':prompt,'completion':full[len(prompt):]})
    FastLanguageModel.for_training(model)
    trainer=SFTTrainer(model=model,processing_class=tokenizer,train_dataset=Dataset.from_list(data),args=SFTConfig(
        output_dir=str(a.output/'checkpoints'),per_device_train_batch_size=2,gradient_accumulation_steps=4,
        num_train_epochs=3,max_length=2048,learning_rate=2e-4,warmup_steps=3,optim='adamw_torch',bf16=True,fp16=False,
        logging_steps=1,save_strategy='no',report_to='none',seed=3407,completion_only_loss=True,dataloader_num_workers=0))
    trained=trainer.train()
    model.save_pretrained(a.output/'adapter');tokenizer.save_pretrained(a.output/'adapter')
    (a.output/'training-receipt.json').write_text(json.dumps({'status':'TRAINED_GENERATED_CANDIDATE','training_performed':True,
        'global_steps':trainer.state.global_step,'metrics':trained.metrics,'max_training_tokens':max(lengths),
        'training_examples':len(data),'human_reviewed':False,'production_promotion_allowed':False},indent=2)+'\n')
    candidate=evaluate('adapter','validation')
    base=results['base_validation']
    (a.output/'selection.json').write_text(json.dumps({'candidate_retained_for_research':candidate['exact_target']>base['exact_target'] and candidate['schema_valid']>=base['schema_valid'],
        'validation_base':base,'validation_candidate':candidate,'production_promotion_allowed':False},indent=2)+'\n')
    evaluate('adapter','test')
    with model.disable_adapter(): evaluate('base','test')
    print(json.dumps(results), flush=True)

if __name__=='__main__': main()
