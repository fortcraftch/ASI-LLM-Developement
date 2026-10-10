"""Prepare capacity controls without allocating or training language-model weights."""
import argparse
import copy
import json
import hashlib
from pathlib import Path
from asi.models.design import count_parameters
from asi.models.design import validate_recipe


def freeze_recipes(source, rows, manifest_path, output):
    """Deduplicate axes and export executable recipes; never start training."""
    payload=manifest_path.read_bytes();manifest=json.loads(payload)
    names=list(manifest['pools'])
    if names!=source['pool_order']:raise ValueError('Source pool order differs')
    output.mkdir(parents=True,exist_ok=False)
    (output/'expert_pools.json').write_bytes(payload)
    seen={};recipes=[]
    for row in rows:
        key=json.dumps(row['model'],sort_keys=True)
        if key in seen:continue
        name=row['name'];seen[key]=name
        family='domain' if row['model']['routing_mode']=='uniform_pool' else 'base'
        recipe=copy.deepcopy(source)
        recipe.update(family=family,model=row['model'],parameter_budget=row['total'],
                      parameter_counts={k:row[k] for k in ('total','routed','common_including_shared')},
                      pool_manifest_sha256=hashlib.sha256(payload).hexdigest(),
                      capacity_study={'axis':row['axis'],'status':'prepared_not_trained',
                                      'reference_budget':source['parameter_budget'],
                                      'exceeds_reference_budget':row['total']>source['parameter_budget']})
        # Checkpoint often; preserve a manageable number under the supervisor.
        recipe['training'].update(save_interval=25,keep_checkpoints=3)
        validate_recipe(recipe,names)
        (output/(name+'.json')).write_text(json.dumps(recipe,indent=2),encoding='utf8')
        recipes.append({'name':name,'family':family,'parameters':row['total'],
                        'over_124m':row['total']>124_000_000})
    commands=['#!/usr/bin/env bash','set -euo pipefail',
              '# Templates only. Choose one reviewed recipe; this file does not launch the sweep.',
              '# Run from the ASI repository root. Copy both recipes and expert_pools.json to homelab.']
    for r in recipes:
        for seed in (1337,2027,31415):
            commands.append(f"# ASI_MOE_BACKEND=bmm python -m asi train --recipe {output.as_posix()}/{r['name']}.json --pool-manifest {output.as_posix()}/expert_pools.json --seed {seed} --auto-restart --log-dir results/capacity/{r['name']}_{seed}")
    (output/'commands.sh').write_text('\n'.join(commands)+'\n',encoding='utf8')
    return recipes


def variants(recipe):
    base=recipe['model']; rows=[]
    # Keep the shared FFN width unchanged when changing expert granularity.
    shared_width=base['moe_inter_dim']*base['n_shared_experts']
    pools=base['n_pools']
    cases=[('capacity',h,2) for h in (60,120,240)]
    cases += [('granularity',h,k) for h,k in ((240,1),(120,2),(60,4))]
    for axis,h,k in cases:
        if shared_width % h: raise ValueError('Shared width cannot be held constant')
        for routing in ('uniform_pool','learned'):
            config=copy.deepcopy(base)
            config.update(moe_inter_dim=h,n_shared_experts=shared_width//h,
                          n_routed_experts=pools*k,experts_per_pool=k,
                          n_activated_experts=k,routing_mode=routing)
            counts=count_parameters(config)
            expert=3*config['n_embd']*h
            name=f'{axis}_h{h}_k{k}_{routing}'
            rows.append({'name':name,'axis':axis,'model':config,**counts,
                'parameters_per_expert':expert,'routed_active_per_moe_layer':expert*k,
                'shared_per_moe_layer':3*config['n_embd']*shared_width,
                'routing_control':'category restricted' if routing=='uniform_pool' else 'unrestricted learned top-k',
                'note':'Model configuration only; requires a matching pool manifest and frozen training recipe. Learned control must see all experts.'})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,default=Path('configs/experiment_124m_v1/domain_16_uniform_pool.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--freeze',action='store_true',help='Also export executable recipes and commented commands, not training runs')
    p.add_argument('--pool-manifest',type=Path,default=Path('configs/experiment_124m_v1/expert_pools.json'))
    a=p.parse_args()
    if a.output.exists():p.error('Use a fresh output directory')
    source=json.loads(a.recipe.read_text(encoding='utf8'))
    rows=variants(source)
    a.output.mkdir(parents=True)
    plan={'status':'design_only_not_trained','source_recipe':str(a.recipe),'seeds':[1337,2027,31415],
          'claim':'No universal language-comprehension threshold can be inferred from isolated FFNs.',
          'controls':['Same tokenizer, backbone, train/validation/test examples and order per seed',
                      'Same tokens first; separate equal-compute comparison',
                      'Same OASST SFT exposure for every model; compare before and after',
                      'Evaluate native trained routing first, then resident expert restrictions',
                      'Track expert token counts, routing collapse, held-out NLL and task correctness'],
          'variants':rows}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
    lines=['# Diseño de capacidad (sin entrenamientos ni resultados)', '',
           ('Las recetas ejecutables se exportan en `recipes/`, con manifiesto y presupuestos propios. No se ha lanzado ningún entrenamiento.' if a.freeze else 'Estas configuraciones NO son recetas ejecutables de `asi train`. Los manifiestos y el presupuesto deben congelarse antes de entrenar; no reutilizar el límite 124M en variantes que lo superen.'), '',
           '| Variante | Parámetros/experto | Activos MoE/capa | Compartidos/capa | Total modelo |',
           '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['parameters_per_expert']} | {r['routed_active_per_moe_layer']} | {r['shared_per_moe_layer']} | {r['total']} |")
    lines+=['','Las variantes repetidas entre ejes son el mismo control y deben reutilizarse, no entrenarse dos veces.',
            'En granularity se conserva k × ancho y el ancho compartido; en capacity solo aumenta la capacidad enrutada. Los parámetros del router varían con el número de expertos. El total no mide FLOPs ni memoria temporal.',
            'No se puede convertir esta tabla en un mínimo de parámetros para comprender lenguaje sin entrenar y evaluar modelos completos.']
    (a.output/'plan.md').write_text('\n'.join(lines),encoding='utf8')
    if a.freeze:
        plan['frozen_recipes']=freeze_recipes(source,rows,a.pool_manifest,a.output/'recipes')
        plan['status']='recipes_prepared_not_trained'
        (a.output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf8')
        with (a.output/'plan.md').open('a',encoding='utf8') as f:
            f.write('\n\nActualización: `recipes/` contiene diez recetas ejecutables validadas y comandos comentados. El barrido no se ha entrenado. Las variantes de 154,8M tienen presupuesto explícito propio, superior al control original de 124M; no se han lanzado.\n')
    print(a.output/'plan.md')


if __name__=='__main__':main()
