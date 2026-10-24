"""Matched base-model generation diagnostic; no training or sealed-test access."""
from recovery_v2.sdpa_four_model_audit import *
import hashlib, re, traceback

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=Path('configs/four_model_quality_20261024.json')); p.add_argument('--output',type=Path,required=True); p.add_argument('--resume',action='store_true'); a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True); path=a.output/'report.json'; status=a.output/'STATUS.json'
 cfg=json.loads(a.config.read_text()); prior=json.loads(Path('results/v2/F0/sdpa_four_models_20261023/report.json').read_text())
 hist=json.loads(Path('results/deepseek_19072_reference_20261010/retry_cache/generation/report.json').read_text()); paths={m['variant']:Path(m['checkpoint']) for m in hist['models']}
 audit=json.loads(Path('results/v2/F0/bank_active_matched_20261022/audit.json').read_text()); bank={r['pool']:Path(r['checkpoint']) for r in audit['models']}
 names=list(json.loads(Path('configs/experiment_124m_v1/expert_pools.json').read_text())['pools'])
 settings={'config_sha256':file_sha256(a.config),'script_sha256':file_sha256(Path(__file__)),'prior_sdpa_sha256':file_sha256(Path('results/v2/F0/sdpa_four_models_20261023/report.json')),'helpers':{str(x):file_sha256(x) for x in [Path('recovery_v2/dense_inference_audit.py'),Path('recovery_v2/sdpa_four_model_audit.py'),Path('asi/experiments/local_generation.py'),Path('asi/models/domain.py'),Path('../Creating-GPT-2-From-0/eval.py'),DEFAULT_ARCHITECTURE]}}
 if path.exists() and not a.resume: p.error('Use --resume')
 report=json.loads(path.read_text()) if path.exists() else {'status':'partial','settings':settings,'config':cfg,'results':{},'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'versions':dict(prior['versions'],transformers=__import__('transformers').__version__),'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),'limitations':['Exploratory base-model instruction diagnostic, not an independent test or SFT comparison.','Preassigned category; classifier excluded. No generated code is executed.','Variable EOS lengths: timing descriptive, not the fixed-length performance benchmark.','NVML null means unavailable. Semantic rubric requires separate manual review.']}
 if report['settings']!=settings: p.error('Resume identity mismatch; use new output directory')
 torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; enc=tiktoken.get_encoding('gpt2'); total=4*len(cfg['prompts'])
 try:
  for family in ['asi','deepseek','bank','gpt2']:
   for q in cfg['prompts']:
    key=family+'/'+q['id']
    if key in report['results']: continue
    durable_save(status,dict(status='running',current=key,complete=len(report['results']),total=total,pid=os.getpid()))
    ck=bank[q['pool']] if family=='bank' else paths['domain_selected' if family=='asi' else family]
    sha=file_sha256(ck); assert sha==prior['results'][family+'/'+q['pool']]['checkpoint_sha256']
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    model,meta=load_model(ck) if family=='bank' else load_variant('domain_selected' if family=='asi' else family,ck,DEFAULT_ARCHITECTURE,Path('../Creating-GPT-2-From-0/eval.py'))
    cache=None
    if family=='asi':
     validate_pool_identity(model,meta,names)
     for layer in model.layers:
      if hasattr(layer.ffn,'moe_backend'): layer.ffn.moe_backend='bmm'; layer.ffn.uniform_fastpath=True
     configure_mode(model,'classified',model.config.routing_mode); cache=ExpertCacheManager('cuda',1); cache.initialize(model); cache.prepare(model,[names.index(q['pool'])]); model.set_active_pools([names.index(q['pool'])])
    else: model.to('cuda')
    model.eval()
    with mode(model,'last' if family=='gpt2' else 'sdpa_last'):
     timed_generate(model,enc,q['prompt'],'cuda',2)
     torch.cuda.reset_peak_memory_stats(); before=cache.snapshot() if cache else {}
     with memory_monitor() as peaks: run=timed_generate(model,enc,q['prompt'],'cuda',cfg['max_new_tokens'])
     after=cache.snapshot() if cache else {}
    run.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),sampled_peaks=peaks,expert_h2d_bytes=after.get('host_to_device_bytes',0)-before.get('host_to_device_bytes',0),checkpoint_sha256=sha,prompt_sha256=hashlib.sha256(q['prompt'].encode()).hexdigest(),output_sha256=hashlib.sha256(json.dumps(run['output_ids']).encode()).hexdigest())
    text=run['text'].split('<|endoftext|>')[0].strip(); words=re.findall(r"\b\w+\b",text.lower()); grams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+',text) if s.strip()]
    if q['check']=='exact': passed=text==q['reference']
    elif q['check']=='two_bullets': passed=len(text.splitlines())==2 and all(x.strip().startswith('- ') for x in text.splitlines())
    else: passed=len(sentences)==1 and text.endswith(('.','!','?'))
    run['automatic']={'word_count':len(words),'repeated_word_4gram_fraction':(len(grams)-len(set(grams)))/len(grams) if grams else None,'format_or_exact_proxy_pass':passed,'check':q['check'],'eos':run['stop_reason']=='eos','limit_reached':run['stop_reason']=='token_limit'}
    report['results'][key]=run; durable_save(path,report); print(f'{len(report["results"])}/{total} {key}',flush=True)
    del model; cache=None; gc.collect(); torch.cuda.empty_cache()
  report['status']='complete'; durable_save(path,report); durable_save(status,dict(status='complete',complete=total,total=total,pid=os.getpid()))
 except Exception:
  report['status']='failed'; report['error']=traceback.format_exc(); durable_save(path,report); durable_save(status,dict(status='failed',complete=len(report['results']),total=total,error=report['error'])); raise
if __name__=='__main__': main()
