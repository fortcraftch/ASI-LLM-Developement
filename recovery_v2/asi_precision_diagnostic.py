"""Precision diagnosis on the already exposed ASI prompt; no quality selection."""
import argparse, json, os
from pathlib import Path
from contextlib import contextmanager
import types
from recovery_v2.dense_inference_audit import *
from asi.experiments.checkpoint_eval import configure_mode
import asi.experiments.local_generation as lg

@contextmanager
def precision_mode(model,name):
    original_context=lg.inference_context
    originals=[]
    try:
        if name.startswith('fp32_'):
            lg.inference_context=lambda device: torch.autocast('cuda',enabled=False)
        with mode(model,'sdpa_last' if 'sdpa' in name else 'baseline'):
            if name=='attention_fp32':
                for layer in model.layers:
                    attn=layer.attn; originals.append((attn,attn.forward))
                    def forward(self,x,*args,**kwargs):
                        with torch.autocast('cuda',enabled=False):
                            return sdpa_forward(self,x.float(),*args,**kwargs).to(x.dtype)
                    attn.forward=types.MethodType(forward,attn)
            yield
    finally:
        for attn,f in originals: attn.forward=f
        lg.inference_context=original_context

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    dest=a.output/'report.json'
    if dest.exists() and not a.resume: raise ValueError('Use --resume')
    old_path=Path('results/v2/F0/four_model_inference_20261023/report.json')
    old=json.loads(old_path.read_text());row=old['results']['asi/education_language'];prompt=row['prompt']
    model,meta=load_variant('domain_selected',Path('model_37595.pt'),DEFAULT_ARCHITECTURE,Path('../Creating-GPT-2-From-0/eval.py'))
    assert meta['checkpoint_sha256']==row['checkpoint_sha256']
    for layer in model.layers:
        if hasattr(layer.ffn,'moe_backend'):layer.ffn.moe_backend='bmm';layer.ffn.uniform_fastpath=True
    configure_mode(model,'classified',model.config.routing_mode)
    names=list(json.loads(Path('configs/experiment_124m_v1/expert_pools.json').read_text())['pools'])
    model.set_active_pools([names.index('education_language')]);model.cuda().eval()
    torch.set_num_threads(4);torch.manual_seed(1337);torch.backends.cuda.matmul.allow_tf32=False
    enc=tiktoken.get_encoding('gpt2');prefix=row['reference']['input_ids']+row['reference']['output_ids'][:14]
    report=json.loads(dest.read_text()) if dest.exists() else dict(status='partial',source_sha256=file_sha256(old_path),checkpoint_sha256=meta['checkpoint_sha256'],script_sha256=file_sha256(__file__),results={},versions={'torch':torch.__version__,'cuda':torch.version.cuda},gpu=torch.cuda.get_device_name())
    for name in ['baseline','sdpa_last','attention_fp32','fp32_baseline','fp32_sdpa']:
        if name in report['results']:continue
        durable_save(a.output/'STATUS.json',dict(status='running',current=name,pid=os.getpid()))
        with precision_mode(model,name):
            with lg.inference_context('cuda'),torch.inference_mode():
                out=model(torch.tensor([prefix],device='cuda'));scores=(out[0] if isinstance(out,tuple) else out)[0,-1,:enc.n_vocab].float()
                vals,ids=scores.topk(5)
            fixed=lg.timed_generate(model,enc,prompt,'cuda',64,stop_at_eos=False)
            eos=lg.timed_generate(model,enc,prompt,'cuda',64)
        report['results'][name]=dict(top_tokens=[{'id':i,'text':enc.decode([i]),'logit':v} for i,v in zip(ids.tolist(),vals.tolist())],margin=float(vals[0]-vals[1]),fixed=fixed,eos=eos,identical_historical=fixed['output_ids']==row['reference']['output_ids'],eos_identical_historical=eos['output_ids']==row['eos_reference']['output_ids'])
        durable_save(dest,report); print(name,report['results'][name]['identical_historical'],flush=True)
    report['fp32_pair_identical']=report['results']['fp32_baseline']['fixed']['output_ids']==report['results']['fp32_sdpa']['fixed']['output_ids']
    report['status']='complete';durable_save(dest,report);durable_save(a.output/'STATUS.json',{'status':'complete'})
if __name__=='__main__':main()
