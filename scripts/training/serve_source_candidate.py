"""Loopback source-task adapter server; base/adapter hashes and scope are explicit."""
import argparse,hashlib,json,os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--adapter-sha256',required=True);p.add_argument('--gpu',type=int,default=2)
    p.add_argument('--port',type=int,default=18083);p.add_argument('--work-dir',type=Path,required=True)
    a=p.parse_args();a.model=a.model.resolve();a.adapter=a.adapter.resolve();a.work_dir.mkdir(parents=True,exist_ok=True);os.chdir(a.work_dir)
    if hashlib.sha256((a.adapter/'adapter_model.safetensors').read_bytes()).hexdigest()!=a.adapter_sha256:raise ValueError('adapter hash mismatch')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(a.gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from unsloth import FastLanguageModel
    import torch,threading
    from peft import PeftModel
    from fastapi import FastAPI,HTTPException
    import uvicorn
    from astrograph.training.source_tools import SourceTaskRequest,source_target,source_messages,checked_source_result
    torch.set_num_threads(4)
    model,tokenizer=FastLanguageModel.from_pretrained(model_name=str(a.model),max_seq_length=2048,load_in_4bit=False,load_in_16bit=True,local_files_only=True)
    model=PeftModel.from_pretrained(model,str(a.adapter));FastLanguageModel.for_inference(model)
    if hasattr(tokenizer,'tokenizer'):tokenizer=tokenizer.tokenizer
    app=FastAPI();lock=threading.Lock()
    @app.get('/health')
    def health():return {'status':'READY','adapter_sha256':a.adapter_sha256,'scope':'LITERAL_PARAMETER_EFFECT_SOURCE_ONLY'}
    @app.post('/v1/source-tools')
    def run(request:SourceTaskRequest):
        try:source_target(request)
        except (ValueError,SyntaxError) as e:raise HTTPException(422,str(e))
        tokens=tokenizer.apply_chat_template(source_messages(request),tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)
        if tokens['input_ids'].shape[1]>1200:raise HTTPException(422,'Source prompt exceeds qualified context')
        with lock,torch.inference_mode():result=model.generate(**tokens.to('cuda:0'),max_new_tokens=768,do_sample=False)
        raw=tokenizer.decode(result[0,tokens['input_ids'].shape[1]:],skip_special_tokens=True)
        return {**checked_source_result(request,raw),'adapter_sha256':a.adapter_sha256}
    uvicorn.run(app,host='127.0.0.1',port=a.port)

if __name__=='__main__':main()
