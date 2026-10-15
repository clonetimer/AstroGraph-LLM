"""Bounded LoRA plumbing check using explicitly synthetic ontology sentences.

Uses the supplied fine_tune environment and local model. The resulting adapter
is an experiment, never a reviewed diagnosis or production model.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--steps',type=int,default=4)
    args=p.parse_args()
    if not 1 <= args.steps <= 20:p.error('smoke steps must be between 1 and 20')
    args.model=args.model.resolve()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    os.chdir(args.output)  # Keep compiler/cache/temporary trainer state outside source.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_DISABLED='true')
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer,SFTConfig
    import torch
    torch.set_num_threads(4)
    examples=[]
    for i in range(16):
        text=f'Unit_{i} is part of Module_{i}.'
        expected={'triples':[{'subject':f'Unit_{i}','predicate':'part_of','object':f'Module_{i}',
                  'source_document_id':f'SYNTHETIC-{i}','source_quote':text}],
                  'lifecycle':'REVIEW_REQUIRED','active_kg_modified':False}
        examples.append({'messages':[{'role':'system','content':'Extract only explicitly stated part_of triples. Return JSON containing triples, lifecycle REVIEW_REQUIRED and active_kg_modified false. Each triple needs subject, predicate, object, source_document_id and exact source_quote.'},
            {'role':'user','content':json.dumps({'document_id':f'SYNTHETIC-{i}','text':text})},
            {'role':'assistant','content':json.dumps(expected)}]})
    data=args.output/'synthetic_train.jsonl';data.write_text(''.join(json.dumps(x)+'\n' for x in examples))
    model,tokenizer=FastLanguageModel.from_pretrained(model_name=str(args.model.resolve()),max_seq_length=512,
        load_in_4bit=False,load_in_16bit=True,local_files_only=True)
    model=FastLanguageModel.get_peft_model(model,r=4,target_modules=['q_proj','k_proj','v_proj','o_proj'],
        lora_alpha=8,lora_dropout=0,bias='none',use_gradient_checkpointing=False,random_state=3407)
    dataset=Dataset.from_list([{'text':tokenizer.apply_chat_template(x['messages'],tokenize=False,add_generation_prompt=False,enable_thinking=False)} for x in examples])
    trainer=SFTTrainer(model=model,processing_class=tokenizer,train_dataset=dataset,args=SFTConfig(
        output_dir=str(args.output/'checkpoints'),per_device_train_batch_size=1,gradient_accumulation_steps=1,
        max_steps=args.steps,max_length=512,learning_rate=1e-4,warmup_steps=0,optim='adamw_torch',
        bf16=True,fp16=False,logging_steps=1,save_strategy='no',report_to='none',seed=3407,
        dataset_text_field='text',dataloader_num_workers=0))
    result=trainer.train()
    model.save_pretrained(args.output/'adapter');tokenizer.save_pretrained(args.output/'adapter')
    receipt={'schema_version':'astrograph-synthetic-training-smoke.v1','status':'TRAINED_SYNTHETIC_SMOKE_ONLY',
             'model':str(args.model.resolve()),'steps':args.steps,'training_performed':True,
             'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'metrics':result.metrics,
             'human_reviewed':False,'production_promotion_allowed':False,
             'claim_boundary':'Optimizer/load/save plumbing only. Synthetic symbols provide no satellite diagnosis skill or independent quality evidence.'}
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
