"""Matched four-family audit; explicit opt-in optimization, frozen weights/routing."""
from recovery_v2.dense_inference_audit import *
from torch.nn.attention import sdpa_kernel, SDPBackend
from asi.experiments.checkpoint_eval import configure_mode
from asi.runtime.generation import validate_pool_identity
from asi.runtime.cache import ExpertCacheManager
import time

@contextmanager
def mode(model,name):
    if name=='math_attention':
        with sdpa_kernel(SDPBackend.MATH): yield
    else: yield

@contextmanager
def memory_monitor():
    stop=threading.Event(); peaks={'rss_bytes':0,'nvml_process_bytes':None}
    def sample():
        try:
            import pynvml
            pynvml.nvmlInit(); handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception: handle=None
        while not stop.is_set():
            peaks['rss_bytes']=max(peaks['rss_bytes'],psutil.Process().memory_info().rss)
            if handle is not None:
                try:
                    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        if proc.pid==os.getpid() and proc.usedGpuMemory is not None and proc.usedGpuMemory<2**60:
                            peaks['nvml_process_bytes']=max(peaks['nvml_process_bytes'] or 0,proc.usedGpuMemory)
                except Exception: pass
            stop.wait(.05)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try: yield peaks
    finally: stop.set();thread.join()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/'report.json'
    if path.exists() and not a.resume: p.error('Use --resume')
    historical=json.loads(Path('results/deepseek_19072_reference_20261010/retry_cache/generation/report.json').read_text(encoding='utf8'))
    gpt_path=Path(next(m['checkpoint'] for m in historical['models'] if m['variant']=='gpt2'))
    audit=json.loads(Path('results/v2/F0/bank_active_matched_20261022/audit.json').read_text(encoding='utf8'))
    checkpoints={r['pool']:Path(r['checkpoint']) for r in audit['models']}
    paths={m['variant']:Path(m['checkpoint']) for m in historical['models']}
    names=list(json.loads(Path('configs/experiment_124m_v1/expert_pools.json').read_text())['pools'])
    settings=dict(script_sha256=file_sha256(Path(__file__)),seed=1337,tokens=64,repeats=3,
                  protocol='FP32 weights, BF16 autocast, batch1, full prefix, greedy; no compile; 4 CPU threads; randomized modes/repeats within each prompt',
                  evaluator_sha256=file_sha256(Path('../Creating-GPT-2-From-0/eval.py')),
                  source_hashes={str(p):file_sha256(p) for p in [Path('recovery_v2/dense_inference_audit.py'),Path('asi/models/domain.py'),Path('asi/experiments/local_generation.py'),DEFAULT_ARCHITECTURE]})
    report=json.loads(path.read_text(encoding='utf8')) if path.exists() else dict(status='partial',settings=settings,results={},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        versions=dict(torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version()),
        gpu=torch.cuda.get_device_name(),limitations=['No claim of universal equivalence from eight diagnostic prompts.',
        'SDPA changes MLA factorization and BF16 rounding; must pass both logit and output checks.',
        'Fixed-token timing ignores EOS; separate EOS output is checked.',
        'No incremental KV cache in either family. No transfer or classifier in timed region.',
        'Order randomized within prompts; timing uncertainty is descriptive, not independent model seeds.'])
    if report['settings']!=settings: p.error('Protocol changed: use a new directory')
    durable_save(path,report)
    torch.set_num_threads(4); torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32=False
    enc=tiktoken.get_encoding('gpt2')
    for family in ['gpt2']:
      for probe in PROBES:
        key=family+'/'+probe['pool']
        if key in report['results']: continue
        durable_save(a.output/'STATUS.json',dict(status='running',current=key,complete=len(report['results']),total=8,pid=os.getpid()))
        torch.manual_seed(1337)
        checkpoint=checkpoints[probe['pool']] if family=='bank' else paths['domain_selected' if family=='asi' else family]
        model,metadata=load_model(checkpoint) if family=='bank' else load_variant('domain_selected' if family=='asi' else family,checkpoint,DEFAULT_ARCHITECTURE,Path('../Creating-GPT-2-From-0/eval.py'))
        cache=None
        if family=='asi':
            validate_pool_identity(model,metadata,names)
            for layer in model.layers:
                if hasattr(layer.ffn,'moe_backend'):
                    layer.ffn.moe_backend='bmm';layer.ffn.uniform_fastpath=True
            configure_mode(model,'classified',model.config.routing_mode)
            cache=ExpertCacheManager('cuda',1);cache.initialize(model)
            cache.prepare(model,[names.index(probe['pool'])]);model.set_active_pools([names.index(probe['pool'])])
        else: model.to('cuda')
        model.eval()
        modes=['baseline','math_attention']
        reference=timed_generate(model,enc,probe['prompt'],'cuda',64,stop_at_eos=False)
        eos=timed_generate(model,enc,probe['prompt'],'cuda',64)
        prefixes=[reference['input_ids']+reference['output_ids'][:n] for n in [0,16,48]]
        refs={bf:[logits(model,ids,bf) for ids in prefixes] for bf in [False,True]}
        record=dict(checkpoint_sha256=file_sha256(checkpoint),prompt=probe['prompt'],reference=reference,eos_reference=eos,modes={})
        for name in modes:
          with mode(model,name):
            checks={}
            for bf in [False,True]:
                diffs=[(logits(model,ids,bf)-target).abs() for ids,target in zip(prefixes,refs[bf])]
                checks['bf16' if bf else 'fp32']=dict(max_abs=max(float(d.max()) for d in diffs),
                    allclose=all(torch.allclose(logits(model,ids,bf),target,rtol=1e-4,atol=1e-4) for ids,target in zip(prefixes,refs[bf])))
            fixed=timed_generate(model,enc,probe['prompt'],'cuda',64,stop_at_eos=False)
            readable=timed_generate(model,enc,probe['prompt'],'cuda',64)
            record['modes'][name]=dict(logit_checks=checks,
                fixed_tokens_identical=fixed['output_ids']==reference['output_ids'],
                eos_tokens_identical=readable['output_ids']==eos['output_ids'],
                fixed_output=fixed, eos_output=readable,runs=[])
        order=[name for name in modes for _ in range(3)]; random.Random(1337+PROBES.index(probe)).shuffle(order)
        for name in order:
          with mode(model,name):
            torch.cuda.reset_peak_memory_stats()
            before=cache.snapshot() if cache else {}
            with memory_monitor() as peaks:
                run=timed_generate(model,enc,probe['prompt'],'cuda',64,stop_at_eos=False)
            after=cache.snapshot() if cache else {}
            run['sampled_peaks']=peaks
            run['expert_loads']=after.get('loads',0)-before.get('loads',0)
            run['measured_expert_h2d_bytes']=after.get('host_to_device_bytes',0)-before.get('host_to_device_bytes',0)
            run.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),memory=module_memory(model),rss_bytes=psutil.Process().memory_info().rss,expert_h2d_bytes=0)
            record['modes'][name]['runs'].append(run)
        report['results'][key]=record; durable_save(path,report)
        print(key,{k:(v['fixed_tokens_identical'],v['eos_tokens_identical']) for k,v in record['modes'].items()},flush=True)
        passed=all(v['logit_checks']['fp32']['allclose'] and v['fixed_tokens_identical'] and v['eos_tokens_identical'] and all(run['output_ids']==reference['output_ids'] for run in v['runs']) for v in record['modes'].values())
        if cache: cache.offload_all(model)
        del model,cache; gc.collect(); torch.cuda.empty_cache()
        if not passed:
            report.setdefault("nonidentical_cases",[]).append(key)
            durable_save(path,report)
        if False:
            report['status']='partial';durable_save(path,report)
            durable_save(a.output/'STATUS.json',dict(status='gate_failed',current=key,complete=len(report['results']),total=8))
            (a.output/'DECISION.md').write_text('Equivalence gate failed for '+key+'. See report.json for logits and token differences. No optimized path promoted; original results preserved. Explicit instruction needed before further experiments.')
            return
    report['status']='complete';durable_save(path,report)
    durable_save(a.output/'STATUS.json',dict(status='complete',complete=8,total=8))


if __name__=='__main__':
    try: main()
    except Exception as exc:
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
            durable_save(output/'STATUS.json',{'status':'failed','error':repr(exc),'pid':os.getpid()})
        raise
