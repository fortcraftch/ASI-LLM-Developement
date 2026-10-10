"""Batch model requests while retaining the external benchmark's scoring rules.

The first pass records requests, not predictions. The second replays actual model
results through the same evaluator. LAMBADA needs a dedicated path because it
requests next-token logits directly. Routes never see completion/target tokens.
"""
from array import array
from collections import defaultdict
import math

import torch
import torch.nn.functional as F

from asi.runtime.generation import inference_context


def pack(ids):
    return array('I', ids).tobytes()


def unpack(data):
    values = array('I'); values.frombytes(data)
    return list(values)


class Requests:
    def __init__(self, enc):
        self.enc = enc
        self.scores = {}
        self.generations = {}
        self.recording = True

    def completion_losses(self, ids, completion_start):
        key = (pack(ids), completion_start)
        if self.recording:
            self.scores[key] = None
            return 0., 0.
        return self.scores[key]

    def greedy_generate(self, prompt, max_new_tokens=20):
        key = (pack(self.enc.encode(prompt)), max_new_tokens, True)
        if self.recording:
            self.generations[key] = None
            return ''
        return self.enc.decode(self.generations[key]).strip()


def batches(rows, batch_size, token_budget, length):
    batch = []; longest = 0
    for row in rows:
        n = max(1, length(row))
        if batch and (len(batch) >= batch_size or max(longest, n) * (len(batch)+1) > token_budget):
            yield batch
            batch = []; longest = 0
        batch.append(row); longest = max(longest, n)
    if batch: yield batch


@torch.no_grad()
def score_requests(adapter, requests, batch_size, token_budget):
    if hasattr(adapter,'prepare_routes'):
        adapter.prepare_routes(unpack(raw)[:start] for raw,start in requests)
    groups = defaultdict(list)
    for key in requests:
        raw, start = key
        ids = unpack(raw)
        if len(ids) < 2:
            requests[key] = (float('inf'), float('inf')); continue
        route = adapter.route_key(ids[:start])
        if len(ids) > adapter.block_size:
            start = max(1, start - (len(ids)-adapter.block_size))
            ids = ids[-adapter.block_size:]
        groups[route].append((key, pack(ids), max(0, start-1)))
    for route, rows in groups.items():
        adapter.activate(route)
        rows.sort(key=lambda r: len(r[1]))
        for batch in batches(rows, batch_size, token_budget, lambda r: len(r[1])//4-1):
            length = max(len(r[1])//4-1 for r in batch)
            x = torch.full((len(batch), length), adapter.enc.eot_token, device=adapter.device, dtype=torch.long)
            y = x.clone(); mask = torch.zeros_like(x, dtype=torch.bool)
            for i, (_, raw, start) in enumerate(batch):
                ids = unpack(raw); n = len(ids)-1
                x[i,:n] = torch.tensor(ids[:-1], device=adapter.device)
                y[i,:n] = torch.tensor(ids[1:], device=adapter.device)
                mask[i,start:n] = True
            logits = adapter.logits(x)
            losses = F.cross_entropy(logits.reshape(-1,logits.size(-1)), y.reshape(-1), reduction='none').view_as(y)
            total = (losses*mask).sum(-1); count = mask.sum(-1)
            sums = total.masked_fill(count==0,float('inf')).tolist()
            means = (total/count.clamp_min(1)).masked_fill(count==0,float('inf')).tolist()
            for row, s, m in zip(batch, sums, means): requests[row[0]] = (s,m)
            del logits, losses


@torch.no_grad()
def generate_requests(adapter, requests, batch_size, token_budget):
    if hasattr(adapter,'prepare_routes'):
        adapter.prepare_routes(unpack(raw) for raw,_,_ in requests)
    # Equal prompt lengths avoid left-padding/position changes to this architecture.
    groups = defaultdict(list)
    for key in requests:
        raw, steps, stop_eos = key
        ids = unpack(raw)
        if not ids: raise ValueError('Benchmark generation requires a nonempty prompt')
        route = adapter.route_key(ids)
        groups[(route, min(len(ids),adapter.block_size))].append(key)
    for (route, length), rows in groups.items():
        adapter.activate(route)
        for batch in batches(rows,batch_size,token_budget,lambda _:length):
            histories = [unpack(key[0])[-adapter.block_size:] for key in batch]
            outputs = [[] for _ in batch]; done = [key[1]<=0 for key in batch]
            x = torch.tensor(histories,device=adapter.device,dtype=torch.long)
            position = length
            with inference_context(adapter.device):
                logits = adapter.model(x,start_pos=0,logits_to_keep=1)[0]
            for step in range(max(key[1] for key in batch)):
                tokens = logits[:,-1,:].argmax(-1).tolist()
                for i, (token, key) in enumerate(zip(tokens,batch)):
                    histories[i].append(token)
                    if not done[i]:
                        outputs[i].append(token)
                        done[i] = step+1 >= key[1] or (key[2] and token==adapter.enc.eot_token)
                if all(done): break
                with inference_context(adapter.device):
                    if getattr(adapter,'supports_kv_cache',True) and position < adapter.block_size:
                        x = torch.tensor(tokens,device=adapter.device).unsqueeze(1)
                        logits = adapter.model(x,start_pos=position,logits_to_keep=1)[0]
                        position += 1
                    else:
                        # Preserve the reference's rolling window and reset positions.
                        x = torch.tensor([h[-adapter.block_size:] for h in histories],device=adapter.device)
                        logits = adapter.model(x,start_pos=0,logits_to_keep=1)[0]
                # logits contain all vocabulary entries, exactly like reference argmax.
            for key, output in zip(batch,outputs): requests[key] = output


def lambada(bench, adapter, limit, batch_size, token_budget):
    data = bench.limited(bench.ds('EleutherAI/lambada_openai','default','test'),limit)
    scores = {}; generations = {}; examples = []
    for ex in data:
        text = ex.get('text') or ex.get('sentence') or ex.get('context')
        if not text or len(text.rstrip().split()) < 2: continue
        words = text.rstrip().split()
        c = adapter.enc.encode(' '.join(words[:-1])); a = adapter.enc.encode(' '+words[-1])
        sk = (pack(c+a),len(c)); gk = (pack(c),len(a),False)
        scores[sk] = None; generations[gk] = None; examples.append((sk,gk,a))
    score_requests(adapter,scores,batch_size,token_budget)
    generate_requests(adapter,generations,batch_size,token_budget)
    correct = sum(generations[gk]==a for _,gk,a in examples)
    nll = sum(scores[sk][0] for sk,_,_ in examples); count = sum(len(a) for _,_,a in examples)
    return {'accuracy':correct/len(examples) if examples else float('nan'),
            'perplexity':math.exp(nll/count) if count else float('nan'),'total':len(examples)}


def evaluate(bench, task, adapter, limit, batch_size=8, token_budget=2048):
    if task == 'lambada': return lambada(bench,adapter,limit,batch_size,token_budget)
    requests = Requests(adapter.enc)
    # Suppress the planning pass's progress bar; the replay retains the original UI.
    progress = bench.tqdm
    try:
        bench.tqdm = lambda iterable, **kwargs: iterable
        bench.TASKS[task](requests,limit)
    finally:
        bench.tqdm = progress
    print(f'{task}: {len(requests.scores)} scoring requests, {len(requests.generations)} generation requests; grouping by pool',flush=True)
    score_requests(adapter,requests.scores,batch_size,token_budget)
    generate_requests(adapter,requests.generations,batch_size,token_budget)
    requests.recording = False
    return bench.TASKS[task](requests,limit)
