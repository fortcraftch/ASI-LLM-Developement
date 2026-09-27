"""Portable training recipes with an exact total-parameter budget, no training."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from asi import ROOT


def pool_manifest(source, taxonomy):
    categories = source['categories']; assigned = set(); pools = {}
    for name, domains in taxonomy.items():
        if not domains or len(domains) != len(set(domains)):
            raise ValueError('Empty or duplicated broad-domain assignment')
        chosen = sorted(c for c in categories if c.split('__')[0] in domains)
        if not chosen or assigned.intersection(chosen):
            raise ValueError('Empty pool or overlapping categories: '+name)
        assigned.update(chosen)
        pools[name] = {'categories': chosen, 'broad_domains':domains,
                       'tokens':sum(categories[c]['tokens'] for c in chosen),
                       'documents':sum(categories[c]['documents'] for c in chosen)}
    if assigned != set(categories):
        raise ValueError('Taxonomy must cover every category exactly once: '+str(sorted(set(categories)-assigned)))
    if sum(p['tokens'] for p in pools.values()) != source['total_tokens']:
        raise ValueError('Manifest totals disagree with its categories')
    if source.get('tokenizer') != 'gpt2' or source.get('dtype') != 'uint16':
        raise ValueError('This training adapter expects GPT-2 uint16 token shards')
    return {'pools':pools, 'unassigned_categories':{}, 'tokenizer':'gpt2',
            'summary':{'pool_count':len(pools),'assigned_categories':len(assigned),'assigned_tokens':source['total_tokens']}}


def count_parameters(config):
    import torch
    from asi.models.domain import GPT, GPTConfig
    with torch.device('meta'):
        model = GPT(GPTConfig(**config))
    routed = sum(p.numel() for name,p in model.named_parameters() if '.ffn.experts.' in name)
    total = sum(p.numel() for p in model.parameters())
    return {'total':total, 'routed':routed, 'common_including_shared':total-routed}


def fit_config(experts, pools=8, budget=124_000_000, family='domain', pool_mode='learned'):
    from asi.models.domain import GPTConfig
    if experts % pools or experts < pools:
        raise ValueError('Equal pools require an expert count divisible by the pool count')
    per_pool = experts//pools
    config = asdict(GPTConfig(n_routed_experts=experts,n_pools=pools,experts_per_pool=per_pool,
                             n_activated_experts=min(2,per_pool),max_batch_size=2,routing_mode=pool_mode))
    if family == 'router':
        # Independent dense specialists; the entire collection shares the total cap.
        config.update(n_embd=128,n_layer=4,n_dense_layers=4,n_head=4,kv_lora_rank=64,
                      qk_nope_head_dim=32,qk_rope_head_dim=16,v_head_dim=32,routing_mode='learned')
        budget //= pools
        field = 'inter_dim'
    else:
        field = 'moe_inter_dim'
    low, high, best = 1, 512, None  # Widths aligned to 8, actual graph is counted.
    while low <= high:
        mid=(low+high)//2; config[field]=mid*8
        counts=count_parameters(config)
        if counts['total'] <= budget:
            best=(dict(config),counts); low=mid+1
        else: high=mid-1
    if best is None:
        raise ValueError('Backbone alone exceeds the requested parameter budget')
    return best


def validate_recipe(recipe, names):
    from asi.models.domain import GPTConfig
    if recipe.get('schema') != 1 or recipe.get('family') not in ('base','domain','router'):
        raise ValueError('Unsupported training recipe family')
    config = GPTConfig(**recipe['model'])
    if recipe['pool_order'] != names or config.n_pools != len(names):
        raise ValueError('Recipe pool identity/order differs from the manifest')
    if config.n_routed_experts != config.n_pools*config.experts_per_pool:
        raise ValueError('Invalid pool dimensions')
    if not 1 <= config.n_activated_experts <= config.experts_per_pool:
        raise ValueError('Too many activated experts')
    if config.routing_mode == 'uniform_pool' and config.experts_per_pool != config.n_activated_experts:
        raise ValueError('Uniform routing requires N/N')
    if recipe['family'] != 'domain' and config.routing_mode != 'learned':
        raise ValueError('Only the categorized family may force a fixed pool')
    if recipe['family']=='router' and (recipe.get('pool') not in names or config.n_dense_layers != config.n_layer):
        raise ValueError('Router specialists need a selected pool and a fully dense backbone')
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'manifest.json')
    parser.add_argument('--taxonomy',type=Path,default=ROOT/'configs/taxonomy_8.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budget',type=int,default=124_000_000)
    args=parser.parse_args()
    source=json.loads(args.manifest.read_text(encoding='utf-8'))
    taxonomy=json.loads(args.taxonomy.read_text(encoding='utf-8'))
    pools=pool_manifest(source,taxonomy); names=list(pools['pools'])
    source_hash=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    pools_hash=hashlib.sha256(json.dumps(pools,indent=2).encode('utf-8')).hexdigest()
    recipes={}; summary=[]
    training={'total_batch_size':262144,'batch_size':2,'seq_len':1024,
              'max_steps':math.ceil(source['splits']['train']['tokens']/262144),
              'pool_sampling':'sqrt','data_order':'random','max_lr':0.0003,
              'warmup_steps':1000,'val_interval':500,'save_interval':500,'val_steps_per_pool':16}
    for experts in (8,16,32,64):
        for family, mode in [('base','learned'),('domain','learned')]+([('domain','uniform_pool')] if experts//len(names)<=2 else []):
            config,counts=fit_config(experts,len(names),args.budget,family,mode)
            name=f'{family}_{experts}_{mode}'
            recipe={'schema':1,'family':family,'model':config,'pool_order':names,
                    'parameter_budget':args.budget,'parameter_counts':counts,'training':training,
                    'source_manifest_sha256':source_hash,'pool_manifest_sha256':pools_hash}
            validate_recipe(recipe,names);recipes[name]=recipe
            summary.append({'name':name,'experts_per_layer':experts,'experts_per_pool':experts//len(names),
                            'active_per_layer':config['n_activated_experts'],'expert_hidden_width':config['moe_inter_dim'],**counts})
    config,counts=fit_config(16,len(names),args.budget,'router')
    sampling_total=sum(math.sqrt(pool['tokens']) for pool in pools['pools'].values())
    for name in names:
        exposure=training['max_steps']*training['total_batch_size']*math.sqrt(pools['pools'][name]['tokens'])/sampling_total
        local_training={**training,'total_batch_size':32768,
                        'max_steps':math.ceil(exposure/32768)}
        recipes['router_'+name]={'schema':1,'family':'router','pool':name,'model':config,'pool_order':names,
            'parameter_budget':args.budget//len(names),'collection_parameter_budget':args.budget,
            'parameter_counts':counts,'training':local_training,'source_manifest_sha256':source_hash,'pool_manifest_sha256':pools_hash}
    summary.append({'name':'router_collection','models':len(names),'per_model':counts['total'],
                    'total':counts['total']*len(names),'note':'Complete independent dense backbones; total cap applies to all models together'})
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'expert_pools.json').write_text(json.dumps(pools,indent=2),encoding='utf-8')
    (args.output/'router_bank.json').write_text(json.dumps(dict.fromkeys(names),indent=2),encoding='utf-8')
    for name,recipe in recipes.items():
        (args.output/(name+'.json')).write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    report={'source_manifest_sha256':source_hash,'state':'designed_not_trained','architectures':summary,
            'train_tokens':source['splits']['train']['tokens'],'val_tokens':source['splits']['val']['tokens'],
            'posthoc':'Classified base reuses a trained base checkpoint; it is not a fourth training recipe.',
            'training_note':'Random token windows with replacement; token budget is not an exact corpus epoch.',
            'recommended':'domain_16_uniform_pool.json'}
    (args.output/'design.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
