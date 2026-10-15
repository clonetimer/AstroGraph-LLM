"""Loopback-only validation server using existing local weights, no remote code."""
import os,argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model",required=True)
parser.add_argument("--gpu",type=int,default=0)
parser.add_argument("--port",type=int,default=18081)
args=parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import time,uuid,threading
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText,AutoTokenizer
from fastapi import FastAPI,HTTPException
import uvicorn
model_path=args.model
model_id=Path(model_path).name
torch.set_num_threads(4)
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
model=AutoModelForImageTextToText.from_pretrained(model_path,dtype=torch.bfloat16,local_files_only=True).to('cuda:0').eval()
app=FastAPI(); lock=threading.Lock()
@app.get('/v1/models')
def models():return {'object':'list','data':[{'id':model_id,'object':'model','owned_by':'local'}]}
@app.post('/v1/chat/completions')
def chat(body:dict):
 if body.get('stream'):raise HTTPException(400,'Streaming is not supported by this validation server')
 tokens=tokenizer.apply_chat_template(body['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=True)
 n=tokens['input_ids'].shape[1]
 if n>24000:raise HTTPException(400,'Validation context budget exceeded')
 limit=min(int(body.get('max_tokens') or body.get('max_completion_tokens') or 2048),4096)
 with lock,torch.inference_mode():
  result=model.generate(**tokens.to('cuda:0'),max_new_tokens=limit,do_sample=False)
 content=tokenizer.decode(result[0,n:],skip_special_tokens=True)
 count=result.shape[1]-n
 return {'id':'chatcmpl-'+uuid.uuid4().hex,'object':'chat.completion','created':int(time.time()),'model':model_id,'choices':[{'index':0,'message':{'role':'assistant','content':content},'finish_reason':'length' if count>=limit else 'stop'}],'usage':{'prompt_tokens':n,'completion_tokens':count,'total_tokens':n+count}}
if __name__=='__main__':uvicorn.run(app,host='127.0.0.1',port=args.port)
