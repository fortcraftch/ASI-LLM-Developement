"""Vulkan timing screen for the experimental Colibri candidate-mask build."""
import json,os,subprocess,time,re,random,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/colibri_candidates_vulkan_20261014'
ENGINE=ROOT/'external/colibri/c/olmoe_asi_vulkan.exe'
def save(name,data):
 (OUT/name).write_text(json.dumps(data,indent=2),encoding='utf8')
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'runs.json').exists():raise RuntimeError('Existing run: do not overwrite')
 source=ROOT/'results/recovery_calibration_batched_v3/report.json'
 mappings=json.loads(source.read_text())['mappings']
 cats=['life_sciences_biology','mathematics_statistics','computer_science_software_engineering']
 masks={}
 for n in [16,32,64]:
  for cat in cats:
   rows=[list(range(64)) if n==64 else mappings[str(n)][cat][str(l)]['experts'] for l in range(16)]
   assert all(len(row)==n and len(set(row))==n and all(0<=x<64 for x in row) for row in rows)
   path=OUT/f'{cat}_{n}.txt';path.write_text(f'16 64 {n}\n'+'\n'.join(' '.join(map(str,row)) for row in rows))
   masks[n,cat]=(path,rows)
 save('provenance.json',{'engine_sha256':hashlib.sha256(ENGINE.read_bytes()).hexdigest(),'mapping_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'backend':'Vulkan hybrid CPU/GPU','expert_tier_gib':2,'dense_weights_additional':True,'tokens':64,'active':8,'cache_per_layer':8,'repeats':3,'seed':20261014,'limitations':['OS file cache uncontrolled','quality not evaluated','expert tier capped at 2 GiB; dense GPU weights are additional']})
 runs=[]
 def run(n,i,label):
  env=os.environ.copy()
  for key in ['ASI_CANDIDATES','ROUTE_TRACE','COLI_USAGE','USAGE_SAVE']:env.pop(key,None)
  env.update(SNAP=str(ROOT/'public_models/olmoe_colibri_int8'),COLI_VULKAN='1',COLI_VK_TIER_GB='2',COLI_VK_SHADERS=str(ROOT/'external/colibri_release/bin/shaders'),PILOT='0',HOT='0',OMP_NUM_THREADS='4',COLI_VK_TIER_WARM='0')
  trace=OUT/f'{label}.trace';env['ROUTE_TRACE']=str(trace)
  if n:env['ASI_CANDIDATES']=str(masks[n,cats[i]][0])
  save('status.json',{'stage':'running','current':label,'finished_runs':len(runs)})
  start=time.perf_counter()
  with (OUT/f'{label}.log').open('w') as log:
   subprocess.run([str(ENGINE),'8','8',str(ROOT/f'results/colibri_quality_20261014/ref_{i}.json')],cwd=ENGINE.parent,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=True)
  text=(OUT/f'{label}.log').read_text();assert '[VK] ready:' in text and re.search(r'device [1-9]\d* of \d+ routed experts',text), 'GPU execution not confirmed';tokens=list(map(int,re.search(r'C engine\s*:\s*([\d ]+)',text)[1].split()))
  assert len(tokens)==64
  rows=0
  for line in trace.read_text().splitlines():
   if not line or line.startswith('#'):continue
   fields=line.split()
   if len(fields)<11:continue
   layer=int(fields[2]);selected=[int(x.split(':')[0]) for x in fields[3:]]
   assert len(selected)==8 and len(set(selected))==8
   if n:assert set(selected)<=set(masks[n,cats[i]][1][layer])
   rows+=1
  assert rows>0,'Missing routing audit'
  result={'label':label,'candidates':n or 64,'prompt_index':i,'tokens':tokens,'seconds':float(re.search(r'TUNE decode: \d+ tokens in ([\d.]+)s',text)[1]),'process_seconds':time.perf_counter()-start,'audited_rows':rows}
  runs.append(result);save('runs.json',runs);return result
 # Warm-up plus same-build semantic equivalence check.
 for i in range(3):
  free=run(0,i,f'validation_free_{i}');full=run(64,i,f'validation_all64_{i}')
  assert free['tokens']==full['tokens'],'All64 mask changed output'
 jobs=[(r,n,i) for r in range(3) for n in [0,32,16] for i in range(3)]
 random.Random(20261014).shuffle(jobs)
 for r,n,i in jobs:run(n,i,f'repeat{r}_n{n or 64}_prompt{i}')
 save('status.json',{'stage':'completed','finished_runs':len(runs),'all64_equivalence':True})
if __name__=='__main__':
 try:main()
 except Exception as e:
  OUT.mkdir(parents=True,exist_ok=True);save('status.json',{'stage':'failed','error':repr(e)});raise
