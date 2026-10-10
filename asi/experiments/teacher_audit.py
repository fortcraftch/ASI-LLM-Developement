"""Read-only verification of teacher shards, including interrupted preparations.

An orphan shard without embedded identity cannot prove its teacher provenance.
This command deliberately does not manufacture a trusted resume manifest.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from asi.experiments.recovery import read_examples, supervised_rows
from asi.experiments.public_moe import sha256, read_lock
from asi.experiments.recovery_protocol import teacher_identity
from asi.taxonomy import digest


def validate_shard(row, expected, vocab_size):
    for key,value in expected.items():
        if row.get(key)!=value:
            raise ValueError('Mismatch in '+key)
    logp=row['teacher_logp']
    if logp.shape!=(len(expected['tokens'])-expected['prompt_length'],vocab_size):
        raise ValueError('Wrong probability matrix shape')
    if not torch.isfinite(logp).all():
        raise ValueError('Nonfinite probabilities')
    if logp.max()>1e-5 or not torch.allclose(logp.float().logsumexp(-1),torch.zeros(logp.shape[0]),atol=2e-4,rtol=0):
        raise ValueError('Unnormalized log probabilities')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher-dir',type=Path,required=True)
    p.add_argument('--examples',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--max-length',type=int,default=384)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): p.error('Use a fresh audit output')
    torch.set_num_threads(2)
    corpus=read_examples(a.examples)
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    expected={r['id']:r for r in supervised_rows(corpus,tokenizer,a.max_length)}
    identity=teacher_identity({'source':digest(read_lock(a.model_dir)), 'corpus':digest(corpus)},True)
    vocab=json.loads((a.model_dir/'config.json').read_text())['vocab_size']
    manifest=a.teacher_dir/'manifest.json'
    report={'expected_examples':len(expected),'identity':identity,'files':[],
            'manifest_sha256':sha256(manifest) if manifest.exists() else None}
    try:
        previous=json.loads(manifest.read_text())
        report['manifest_valid_json']=True
        report['manifest_identity_matches']=previous.get('identity')==identity
    except (OSError,ValueError) as error:
        previous={}
        report.update(manifest_valid_json=False,manifest_error=str(error))
    entries={e['id']:e for e in previous.get('entries',[])}
    for path in sorted(a.teacher_dir.glob('*.pt')):
        item={'id':path.stem,'bytes':path.stat().st_size,'sha256':sha256(path)}
        try:
            row=torch.load(path,map_location='cpu',weights_only=True)
            validate_shard(row,expected[path.stem],vocab)
            item.update(valid_content=True,split=row['split'],category=row['category'])
            item['provenance_verified']=row.get('teacher_identity')==identity or (
                previous.get('identity')==identity and entries.get(path.stem,{}).get('sha256')==item['sha256'])
        except Exception as error:
            item.update(valid_content=False,error=str(error),provenance_verified=False)
        report['files'].append(item)
    counts=Counter(('verified' if r['provenance_verified'] else 'orphan_valid') if r['valid_content'] else 'invalid' for r in report['files'])
    report.update(counts=dict(counts),total_bytes=sum(r['bytes'] for r in report['files']),
        missing_examples=sorted(set(expected)-{r['id'] for r in report['files'] if r['valid_content']}),
        note='Valid content checks structure/tokens/normalization, not agreement with the original teacher. Legacy orphans require teacher recomputation or an intact historical manifest before trusted reuse.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['files','missing_examples']},indent=2))


if __name__=='__main__': main()
