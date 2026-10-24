"""Exploratory Colibri INT8 vs AirLLM BF16; not precision-matched."""
import argparse,json,os,subprocess,sys,time,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'results/colibri_airllm_20261014'
MODEL=ROOT/'public_models/olmoe'
CONVERTED=ROOT/'public_models/olmoe_colibri_int8'
def save(name,obj):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2),encoding='utf8')
def main():
 global OUT
 p=argparse.ArgumentParser();p.add_argument('--air-worker',action='store_true');p.add_argument('--tokens',type=int,default=16);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args()
 if not 1 <= a.tokens <= 256:p.error('--tokens must be between 1 and 256')
 OUT=a.output.resolve()
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 prompts=['Explain what a gene is in two sentences.','What is the derivative of x squared?','Write Python code to sort the list [3, 1, 2].']
 ids=[tok(tok.apply_chat_template([{'role':'user','content':s}],add_generation_prompt=True,tokenize=False),add_special_tokens=False)['input_ids'] for s in prompts]
 if a.air_worker:
  import torch
  from asi.models.airllm_olmoe import load_airllm
  from asi.experiments.public_moe import decode
  start=time.perf_counter();model=load_airllm(MODEL,device='cuda:0',dtype=torch.bfloat16,max_seq_len=512,layer_shards_saving_path=str(MODEL/'airllm_layers'),prefetching=False,delete_original=False)
  model.model.set_attn_implementation('eager');model.model.eval()
  report={'completed':False,'backend':'airllm','precision':'BF16','load_seconds':time.perf_counter()-start,'records':[]}
  for i,row in enumerate(ids):
   torch.cuda.reset_peak_memory_stats()
   result=decode(model,torch.tensor([row],device='cuda'),a.tokens,'cuda')
   result.update(prompt=prompts[i],input_ids=row,text=tok.decode(result['token_ids']),gpu_peak_bytes=torch.cuda.max_memory_allocated())
   report['records'].append(result);save('airllm.json',report)
  report['completed']=True;save('airllm.json',report);return
 OUT.mkdir(parents=True,exist_ok=True)
 existing=json.loads((OUT/'airllm.json').read_text()) if (OUT/'airllm.json').exists() else {}
 if not existing.get('completed'):
  save('status.json',{'stage':'airllm'})
  with (OUT/'airllm.log').open('a') as log:subprocess.run([sys.executable,'-u',__file__,'--air-worker','--tokens',str(a.tokens),'--output',str(OUT)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 air=json.loads((OUT/'airllm.json').read_text());records=[]
 assert len(air['records'])==len(ids) and all(r['input_ids']==row and len(r['token_ids'])==a.tokens for r,row in zip(air['records'],ids)), 'Existing AirLLM protocol mismatch'
 save('status.json',{'stage':'converting_colibri'})
 converter=ROOT/'external/colibri/c/tools/convert_olmoe_merged.py'
 env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 with (OUT/'conversion.log').open('a') as log:
  subprocess.run([sys.executable,'-u',str(converter),'--model',str(MODEL),'--out',str(CONVERTED),'--flush-every','32'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 save('conversion_provenance.json',{'source':str(MODEL),'output':str(CONVERTED),'converter_sha256':hashlib.sha256(converter.read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 import psutil
 for backend in ['cpu','vulkan']:
  save('status.json',{'stage':'colibri_'+backend})
  for i,row in enumerate(ids):
   ref=OUT/f'ref_{i}.json';save(ref.name,{'prompt_ids':row,'full_ids':row+air['records'][i]['token_ids']})
   env=os.environ.copy();env.update(SNAP=str(CONVERTED),OMP_NUM_THREADS='4',COLI_VULKAN='1' if backend=='vulkan' else '0',PILOT='0',HOT='0')
   engine=ROOT/'external/colibri_release/bin/olmoe.exe'
   logpath=OUT/f'colibri_{backend}_{i}.log';start=time.perf_counter();peak=0
   if logpath.exists():logpath.rename(logpath.with_name(logpath.stem+'_'+str(time.time_ns())+'.log'))
   with logpath.open('w') as log:
    proc=subprocess.Popen([str(engine),'8','8',str(ref)],env=env,cwd=engine.parent,stdout=log,stderr=subprocess.STDOUT)
    while proc.poll() is None:
     try:peak=max(peak,psutil.Process(proc.pid).memory_info().rss)
     except psutil.NoSuchProcess:pass
     if time.perf_counter()-start>1800:proc.kill();raise RuntimeError('Colibri pilot timeout')
     time.sleep(.2)
   text=logpath.read_text(errors='replace');match=re.search(r'C engine\s*:\s*([\d ]+)',text);speed=re.search(r'TUNE decode: (\d+) tokens in ([\d.]+)s',text)
   if proc.returncode or not match or not speed:raise RuntimeError('Colibri failed: '+str(logpath))
   tokens=list(map(int,match[1].split()));assert len(tokens)==a.tokens
   records.append({'backend':backend,'prompt':prompts[i],'input_ids':row,'token_ids':tokens,'text':tok.decode(tokens),'generation_seconds_including_prefill':float(speed[2]),'process_seconds':time.perf_counter()-start,'sampled_peak_rss_bytes':peak,'agreement_with_airllm':sum(x==y for x,y in zip(tokens,air['records'][i]['token_ids']))/a.tokens})
   save('colibri.json',{'completed':False,'precision':'INT8 experts, FP32 dense','records':records})
 save('colibri.json',{'completed':True,'precision':'INT8 experts, FP32 dense','records':records})
 save('status.json',{'stage':'completed','limitations':['Not precision matched','Single run per prompt, setup/order/OS cache confounded','Token agreement is not correctness','Colibri timing includes prefill; compare AirLLM generation_seconds not decode rate']})
if __name__=='__main__':
 try:main()
 except Exception as e:save('status.json',{'stage':'failed','error':repr(e)});raise
