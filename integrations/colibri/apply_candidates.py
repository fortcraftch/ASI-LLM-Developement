"""Apply a guarded experimental patch to an external Colibri checkout."""
import argparse, hashlib, json, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
src=a.source/'olmoe.c';text=src.read_text(encoding='utf8');before=hashlib.sha256(src.read_bytes()).hexdigest()
if '#include "asi_candidates.h"' in text:raise SystemExit('Already patched; no changes')
edits=[('static void load_cfg(Cfg *c, const char *snap) {','#include "asi_candidates.h"\n\nstatic void load_cfg(Cfg *c, const char *snap) {'),('    load_cfg(&m->c, snap);','    load_cfg(&m->c, snap);\n    asi_load_candidates(m->c.n_layers, m->c.n_experts, m->c.topk);'),('if (!taken && pr[e] > bv)', 'if (!taken && (!asi_candidates || asi_candidates[layer * E + e]) && pr[e] > bv)')]
for old,new in edits:
 if text.count(old)!=1:raise SystemExit('Unsupported source: anchor count '+repr(old))
 text=text.replace(old,new)
backup=src.with_suffix('.c.asi_original')
if backup.exists():raise SystemExit('Backup exists; inspect before changing')
shutil.copy2(src,backup);shutil.copy2(Path(__file__).with_name('asi_candidates.h'),a.source/'asi_candidates.h')
src.write_text(text,encoding='utf8')
(a.source/'asi_patch_provenance.json').write_text(json.dumps({'before_sha256':before,'after_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'scope':'fixed candidates; original softmax, top8; experimental, not validated'},indent=2))
print('Patched source; compile and validate before benchmarking')
