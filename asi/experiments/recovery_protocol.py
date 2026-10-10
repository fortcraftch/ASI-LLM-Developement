"""Memory-bounded references and explicit recovery evaluation contracts."""
import ast
from decimal import Decimal, InvalidOperation
import random
import re
import torch
import os
from pathlib import Path


def durable_save(path, payload, tensor=False):
    """Flush before atomic replacement; an interrupted save preserves the old file."""
    import json
    path=Path(path)
    pending=path.with_suffix(path.suffix+'.tmp')
    try:
        with pending.open('wb') as handle:
            if tensor:
                torch.save(payload,handle)
            else:
                handle.write(json.dumps(payload,indent=2).encode('utf-8'))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending,path)
    finally:
        pending.unlink(missing_ok=True)


class TeacherRow(dict):
    """Keep text/tokens in RAM, load one distribution only when the loss needs it."""
    def __init__(self, metadata, path):
        super().__init__(metadata)
        self.path = path
        self['teacher_logp'] = None

    def __getitem__(self, key):
        if key == 'teacher_logp':
            return torch.load(self.path, map_location='cpu', weights_only=True)['teacher_logp']
        return super().__getitem__(key)


def padded_batch(rows, pad_token, device):
    width=max(len(r['tokens']) for r in rows)
    tokens=torch.full((len(rows),width),pad_token,dtype=torch.long,device=device)
    attention=torch.zeros_like(tokens)
    for i,row in enumerate(rows):
        length=len(row['tokens'])
        tokens[i,:length]=torch.tensor(row['tokens'],device=device)
        attention[i,:length]=1
    return tokens,attention


def teacher_identity(identity, with_eos):
    if not with_eos:
        return identity  # Historical v1 artifacts remain compatible.
    return {'source': identity['source'], 'corpus': identity['corpus'],
            'objective': 'native_full_distribution_assistant_eos_v3'}


def select_teacher_entries(manifest, identity, expected_ids, allow_partial=False):
    if manifest.get('identity')!=identity or (not manifest.get('completed') and not allow_partial):
        raise ValueError('Teacher identity differs or preparation is incomplete')
    entries=manifest['entries']; ids=[e['id'] for e in entries]; expected=set(expected_ids)
    if len(ids)!=len(set(ids)) or not expected<=set(ids) or (not allow_partial and set(ids)!=expected):
        raise ValueError('Teacher examples differ from requested corpus')
    return [e for e in entries if e['id'] in expected]


def training_order(count, masks, steps, seed, warmup32=0, paired_examples=False, only_mask=None):
    if count < 1 or steps < 1 or not masks:
        raise ValueError('Positive example count, steps and masks required')
    if only_mask is not None and (only_mask not in masks or warmup32 or not paired_examples):
        raise ValueError('Fixed training mask requires paired examples and no curriculum')
    if warmup32 and (set(masks) != {'32','16'} or not 0 < warmup32 < steps):
        raise ValueError('Curriculum requires masks 32/16 and 0 < warmup < steps')
    if paired_examples:
        # Example order is independent of residency: the treatment changes masks only.
        examples_rng=random.Random(seed)
        mask_rng=random.Random(seed ^ 0x5A17)
        indices=list(range(count)); mask_cycle=[]; result=[]
        for step in range(steps):
            if step % count == 0: examples_rng.shuffle(indices)
            if only_mask is not None: mask=only_mask
            elif step < warmup32: mask='32'
            else:
                if not mask_cycle:
                    mask_cycle=list(masks); mask_rng.shuffle(mask_cycle)
                mask=mask_cycle.pop()
            result.append((indices[step % count],mask))
        return result
    rng=random.Random(seed)
    phases=[(warmup32,['32']), (steps-warmup32,list(masks))] if warmup32 else [(steps,list(masks))]
    result=[]
    for length, names in phases:
        order=[(i,name) for name in names for i in range(count)]
        for step in range(length):
            if step % len(order)==0: rng.shuffle(order)
            result.append(order[step % len(order)])
    return result


def task_metrics(text, row):
    metrics={'task_correct':None,'format_pass':None,'python_syntax_pass':None,
             'answer_phrase_present':None,'manual_review_required':True}
    if 'expected_number' in row:
        found=re.findall(r'Final answer:\s*([-+]?\d[\d,]*(?:\.\d+)?)',text,flags=re.I)
        # Presence of the final marker is the format contract; explanations can be brief.
        metrics['format_pass']=len(found)==1 and bool(re.search(r'Final answer:\s*[-+]?\d[\d,]*(?:\.\d+)?\s*$',text,re.I))
        if not found and re.fullmatch(r'\s*[-+]?\d[\d,]*(?:\.\d+)?\s*',text):
            found=[text.strip()]  # Correct bare number can still fail the requested format.
        try:
            metrics['task_correct']=len(found)==1 and Decimal(found[0].replace(',',''))==Decimal(row['expected_number'].replace(',',''))
        except InvalidOperation:
            metrics['task_correct']=False
    if row.get('evaluation')=='python_syntax_and_manual_tests':
        match=re.fullmatch(r'\s*```(?:python)?\s*\n(.*?)\n```\s*',text,re.S)
        metrics['format_pass']=match is not None
        try:
            ast.parse(match.group(1) if match else text)
            metrics['python_syntax_pass']=True
        except (SyntaxError, ValueError, TypeError):
            metrics['python_syntax_pass']=False
        # Never execute arbitrary generated code in the host process.
    if 'expected_phrase' in row:
        phrase=re.escape(row['expected_phrase'].strip())
        metrics['answer_phrase_present']=bool(re.search(r'(?<!\w)'+phrase+r'(?!\w)',text,re.I))
        # A word mention is not a correctness judgment.
    return metrics
