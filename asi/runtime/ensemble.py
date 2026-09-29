"""Whole-model selection for independent, category-trained dense specialists."""
import argparse
import json
from pathlib import Path
import time

from asi.runtime.generation import load_model, generate, validate_pool_identity
from asi.runtime.routing import DomainSessionRouter


class IndependentModelRouter:
    def __init__(self, checkpoints, pool_names, device='cpu', total_budget=124_000_000):
        if set(checkpoints) != set(pool_names) or not checkpoints:
            raise ValueError('One independent checkpoint is required per pool')
        self.checkpoints=checkpoints;self.names=pool_names;self.device=device
        self.budget=total_budget;self.model=None;self.pool=None;self.loads=0

    def select(self, pool):
        if pool not in self.checkpoints: raise ValueError('Unknown pool')
        if self.pool == pool: return self.model, False
        path=self.checkpoints[pool]
        if not path or not Path(path).is_file(): raise FileNotFoundError('Specialist not trained/registered: '+pool)
        if self.model is not None:
            self.model.to('cpu');self.model=None;self.pool=None
        model,metadata=load_model(Path(path))
        validate_pool_identity(model,metadata,self.names)
        recipe=metadata.get('training_recipe') or {}
        if recipe.get('family')!='router' or recipe.get('pool')!=pool or model.config.n_dense_layers!=model.config.n_layer:
            raise ValueError('Checkpoint is not the declared independent dense specialist')
        # Equal per-model allotments bound the collection even before all are loaded.
        if sum(p.numel() for p in model.parameters()) > self.budget//len(self.names):
            raise ValueError('Specialist exceeds its share of the total collection budget')
        self.model=model.to(self.device).eval();self.pool=pool;self.loads+=1
        return self.model, True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True,help='JSON mapping pool name to checkpoint path; paths relative to bank')
    p.add_argument('--pool-manifest',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--max-new-tokens',type=int,default=64)
    args=p.parse_args()
    import tiktoken
    manifest=json.loads(args.pool_manifest.read_text(encoding='utf-8'));names=list(manifest['pools'])
    bank=json.loads(args.bank.read_text(encoding='utf-8'))
    paths={k:str((args.bank.parent/v).resolve()) if v else None for k,v in bank.items()}
    manager=IndependentModelRouter(paths,names,args.device)
    router=DomainSessionRouter(names,max_pools=1,pool_manifest=manifest)
    enc=tiktoken.get_encoding('gpt2')
    while True:
        prompt=input('> ').strip()
        if prompt.lower() in ('exit','quit'): break
        if not prompt: continue
        label=router.route(prompt).ranked_pools[0]
        start=time.perf_counter();model,changed=manager.select(label)
        print(json.dumps({'pool':label,'backbone_changed':changed,'load_seconds':time.perf_counter()-start}))
        print(generate(model,enc,prompt,args.device,args.max_new_tokens,0.8,50))


if __name__=='__main__':main()
