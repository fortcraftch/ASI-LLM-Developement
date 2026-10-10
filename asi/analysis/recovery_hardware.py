"""Offline OLMoE residency calibration from dev quality and standalone measurements."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''): h.update(chunk)
    return h.hexdigest()


def build_profiles(training, evaluations):
    if training.get('action')!='train' or not training.get('completed'):
        raise ValueError('Completed training report required')
    step=training['selected_step']
    dev=training['dev_before'] if step==0 else next(v for v in training['dev_checks'] if v['step']==step)
    groups={}
    for row in dev['records']:
        value=row.get('kl')
        if not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError('Measured dev KL to native teacher required')
        groups.setdefault(row['mask'],{}).setdefault(row['category'],[]).append(value)
    profiles=[]; checkpoint=None; generated=None
    for report in evaluations:
        if report.get('action')!='evaluate' or not report.get('completed'):
            raise ValueError('Completed standalone evaluation required; training memory is not inference memory')
        if report['identity']!=training['identity'] or report['selected_step']!=step:
            raise ValueError('Evaluation model, subset or selected step differs')
        current=report.get('checkpoint_sha256')
        if not current or (checkpoint is not None and current!=checkpoint): raise ValueError('Checkpoints differ')
        checkpoint=current
        rows=[r for r in report['generations'] if r['stage']=='after']
        outputs={(r['id'],r['mask']):r['token_ids'] for r in rows}
        if not outputs or len(outputs)!=len(rows): raise ValueError('Missing or duplicated generations')
        if generated is not None and outputs!=generated: raise ValueError('RAM runs changed evaluated prompts or generated tokens')
        generated=outputs
        for mask in sorted({r['mask'] for r in rows}):
            sample=[r for r in rows if r['mask']==mask]
            if mask not in groups: raise ValueError('No validation quality for this mask')
            if any(r.get('memory_scope')!='standalone evaluation process' for r in sample):
                raise ValueError('Inference memory scope not recorded')
            kl_by_category={k:statistics.mean(v) for k,v in groups[mask].items()}
            profiles.append({'resident_per_layer':int(mask),'worst_category_dev_kl':max(kl_by_category.values()),
                'dev_kl_by_category':kl_by_category,
                'ram_cache_cap_bytes':int(report['settings']['expert_ram_gib']*2**30),
                'measured_peak_cuda_bytes':max(r['peak_cuda_allocated_bytes'] for r in sample),
                'mean_prepare_seconds':statistics.mean(r['prepare_seconds'] for r in sample),
                'mean_generation_seconds':statistics.mean(r['generation_seconds'] for r in sample),
                'generation_expert_loads':sum(r['cache_after']['loads']-r['cache_before']['loads'] for r in sample),
                'generation_h2d_bytes':sum(r['cache_after']['host_to_device_bytes']-r['cache_before']['host_to_device_bytes'] for r in sample),
                'rss_at_report_end_bytes':report.get('rss_bytes'),
                'measured_gpu':report['hardware']['gpu']['name'],'generated_examples':len(sample),
                'generation_token_budget':report['settings']['new_tokens']})
    return {'identity':training['identity'],'checkpoint_sha256':checkpoint,'selected_step':step,'profiles':profiles}


def choose_profile(profiles,gpu_bytes,ram_cache_bytes,max_dev_kl,reserve_bytes):
    if any(not math.isfinite(v) or v<0 for v in (gpu_bytes,ram_cache_bytes,max_dev_kl,reserve_bytes)):
        raise ValueError('Budgets and KL tolerance must be finite and nonnegative')
    eligible=[r for r in profiles if r['measured_peak_cuda_bytes']+reserve_bytes<=gpu_bytes
              and r['ram_cache_cap_bytes']<=ram_cache_bytes and r['worst_category_dev_kl']<=max_dev_kl]
    if not eligible: return {'status':'blocked','reason':'No measured profile satisfies these GPU/cache/validation constraints'}
    # Prefer measured fidelity, then the lower RAM cap; do not rank single-pass latency as a proven speed benefit.
    best=min(eligible,key=lambda r:(r['worst_category_dev_kl'],r['ram_cache_cap_bytes']))
    return {'status':'estimated_candidate','profile':best,'executed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-report',type=Path,required=True)
    p.add_argument('--reports',type=Path,nargs='+',required=True)
    p.add_argument('--gpu-gib',type=float,nargs='+',default=[4,6,8,12])
    p.add_argument('--ram-cache-gib',type=float,nargs='+',default=[.25,2])
    p.add_argument('--max-dev-kl',type=float,nargs='+',default=[.1,.5,1.,2.])
    p.add_argument('--reserve-mib',type=float,default=512)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Use a fresh output directory')
    training=json.loads(a.training_report.read_text(encoding='utf8'))
    result=build_profiles(training,[json.loads(path.read_text(encoding='utf8')) for path in a.reports])
    checkpoint=Path(training['settings']['output'])/'adaptation_best.pt'
    if file_hash(checkpoint)!=result['checkpoint_sha256']: raise ValueError('Evaluated checkpoint is not this training run best adapter')
    result['sources']={str(path):file_hash(path) for path in [a.training_report,*a.reports]}
    result['decisions']=[{'gpu_gib':g,'ram_cache_gib':r,'max_dev_kl':q,
        **choose_profile(result['profiles'],g*2**30,r*2**30,q,a.reserve_mib*2**20)}
        for g in a.gpu_gib for r in a.ram_cache_gib for q in a.max_dev_kl]
    result['limitations']=['Offline estimates, not execution on smaller GPUs.',
        'RAM cache cap is not total process RAM. RSS snapshot is not a peak.',
        'Dev KL measures imitation, not task correctness. Tolerances are illustrative.',
        'One pass and three prompts per mask do not establish latency differences.',
        'Single-category contexts; mixed-label questions require separate validation.']
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    lines=['# Residencia OLMoE: perfiles medidos y decisiones estimadas','',
        '| Residentes/capa | Caché RAM (GiB) | Pico VRAM (GiB) | Peor KL dev por categoría | Preparación media (s) | Generación media (s) | Cargas durante generación |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['profiles']:
        lines.append(f"| {r['resident_per_layer']} | {r['ram_cache_cap_bytes']/2**30:.2f} | {r['measured_peak_cuda_bytes']/2**30:.2f} | {r['worst_category_dev_kl']:.3f} | {r['mean_prepare_seconds']:.2f} | {r['mean_generation_seconds']:.2f} | {r['generation_expert_loads']} |")
    lines+=['','La selección usa calidad de validación y memoria medida; nunca usa las respuestas test para escoger el perfil.',
        'Una decisión estimated_candidate no carga ni ejecuta el modelo. No garantiza caber en otra GPU o contexto.',
        'La caché RAM no equivale a la RAM total. Las latencias son una pasada sin control del estado de la caché del sistema operativo.',
        'Los umbrales KL son ilustrativos; no certifican coherencia o corrección.']
    (a.output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':main()
