"""Opt-in routing diagnostics. Cross-layer edges are associations, not calls."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from asi.models.original import native_moes
from asi.models.domain import MoE


class RoutingTrace:
    def __init__(self, model, pool_names, max_token_rows=4096):
        self.model = model
        self.pool_names = pool_names
        self.max_token_rows = max_token_rows
        self.observers = []
        self.handle = None
        self.reset()

    def reset(self):
        self.layers = {}
        self.rows = []
        self.transitions = Counter()
        self.previous = None
        self.call_id = -1
        self.dropped_rows = 0
        self.cross_pool_pairs = 0
        self.depth_pairs = 0

    def _begin(self, model, args, kwargs):
        idx = args[0] if args else kwargs['idx']
        self.batch, self.length = idx.shape
        self.start_pos = kwargs.get('start_pos', args[2] if len(args) > 2 else 0)
        self.token_ids = idx.detach().cpu().reshape(-1).tolist()
        self.call_id += 1
        self.previous = None

    @torch.no_grad()
    def _record(self, layer_id, gate, x, logits, weights, indices, mask):
        count = gate.weight.shape[0]
        if layer_id not in self.layers:
            self.layers[layer_id] = {'tokens': 0, 'selections': [0]*count,
                                     'weight_sum': [0.0]*count, 'mask_violations': 0,
                                     'shadow_outside_mask': 0, 'shadow_selections': [0]*count}
        stats = self.layers[layer_id]
        ids = indices.detach().cpu()
        w = weights.detach().float().cpu()
        _, shadow = gate.route_logits(logits.detach(), None)
        shadow = shadow.cpu()
        stats['tokens'] += ids.shape[0]
        counts = torch.bincount(ids.flatten(), minlength=count).tolist()
        shadow_counts = torch.bincount(shadow.flatten(), minlength=count).tolist()
        sums = torch.zeros(count).scatter_add_(0, ids.flatten(), w.flatten()).tolist()
        for i in range(count):
            stats['selections'][i] += counts[i]
            stats['shadow_selections'][i] += shadow_counts[i]
            stats['weight_sum'][i] += sums[i]
        if mask is not None:
            allowed = mask.detach().cpu()
            stats['mask_violations'] += int((~allowed[ids]).sum())
            stats['shadow_outside_mask'] += int((~allowed[shadow]).sum())
        pool_ids = self.model.layers[layer_id].ffn.expert_pool_ids.cpu()
        pools = pool_ids[ids]
        if self.previous is not None:
            prev_layer, prev_ids, prev_pools = self.previous
            self.cross_pool_pairs += int((prev_pools.unsqueeze(2) != pools.unsqueeze(1)).sum())
            self.depth_pairs += ids.shape[0] * prev_ids.shape[1] * ids.shape[1]
            for prev, current in zip(prev_ids.tolist(), ids.tolist()):
                for source in prev:
                    for target in current:
                        self.transitions[f'{prev_layer}:{source}->{layer_id}:{target}'] += 1
        self.previous = (layer_id, ids, pools)
        for t, (experts, scores, token_pools) in enumerate(zip(ids.tolist(), w.tolist(), pools.tolist())):
            if len(self.rows) >= self.max_token_rows:
                self.dropped_rows += 1
                continue
            self.rows.append({'call': self.call_id, 'layer': layer_id,
                              'batch': t // self.length, 'position': self.start_pos + t % self.length,
                              'token_id': self.token_ids[t], 'experts': experts, 'weights': scores,
                              'pools': [self.pool_names[p] for p in token_pools]})

    def __enter__(self):
        self.handle = self.model.register_forward_pre_hook(self._begin, with_kwargs=True)
        for layer_id, layer in enumerate(self.model.layers):
            if isinstance(layer.ffn, MoE):
                gate = layer.ffn.gate
                self.observers.append((gate, gate.routing_observer))
                gate.routing_observer = lambda *args, lid=layer_id: self._record(lid, *args)
        return self

    def __exit__(self, *args):
        self.handle.remove()
        for gate, previous in self.observers:
            gate.routing_observer = previous
        self.observers.clear()

    def report(self):
        identity = []
        for layer_id, layer in enumerate(self.model.layers):
            if not isinstance(layer.ffn, MoE):
                continue
            for expert, pool in enumerate(layer.ffn.expert_pool_ids.tolist()):
                identity.append({'layer': layer_id, 'expert': expert, 'pool': self.pool_names[pool]})
        return {'expert_identity': identity, 'layers': self.layers, 'token_routes': self.rows,
                'dropped_token_rows': self.dropped_rows,
                'cross_layer_cooccurrence': dict(self.transitions),
                'cross_pool_depth_pair_fraction': self.cross_pool_pairs/self.depth_pairs if self.depth_pairs else None,
                'interpretation': 'Edges pair top-k experts for the same token in successive MoE layers. They are not direct expert calls or proof of causal influence. Shadow routes use current hidden states, not a full unrestricted forward.',
                'shared_experts': 'Always active; cross-domain information also flows through attention and residuals.'}

    def write(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.report(), indent=2), encoding='utf-8')


class ExpertCalibrator:
    """Collect native selections and E dot hidden-state evidence by input labels."""
    def __init__(self, model):
        self.moes=native_moes(model)
        self.labels=[]
        self.data={}
        self.handles=[]

    def set_labels(self, labels):
        self.labels=list(dict.fromkeys(labels))
        if not self.labels: raise ValueError('Provide at least one corpus label')

    @torch.no_grad()
    def observe(self, layer, gate, args, output):
        if not self.labels: raise ValueError('Call set_labels before calibration forward')
        weights,indices=output
        hidden=args[0].detach().reshape(-1,gate.weight.shape[1]).float()
        E=gate.weight.detach().float()
        affinities=F.linear(hidden,E).mean(0).cpu()
        counts=torch.bincount(indices.flatten(),minlength=E.shape[0]).cpu()
        mass=torch.zeros(E.shape[0],device=weights.device).scatter_add_(0,indices.flatten(),weights.float().flatten()).cpu()
        for label in self.labels:
            key=(layer,label)
            if key not in self.data:
                self.data[key]={'tokens':0,'counts':torch.zeros_like(counts),'mass':torch.zeros_like(mass),'affinity_sum':torch.zeros_like(affinities)}
            row=self.data[key]
            row['tokens']+=hidden.shape[0]
            row['counts']+=counts
            row['mass']+=mass
            row['affinity_sum']+=affinities*hidden.shape[0]

    def __enter__(self):
        for layer,moe in self.moes.items():
            self.handles.append(moe.gate.register_forward_hook(lambda gate,args,out,lid=layer:self.observe(lid,gate,args,out)))
        return self

    def __exit__(self,*args):
        for h in self.handles: h.remove()
        self.handles=[]

    def report(self):
        rows=[]
        for (layer,label),r in self.data.items():
            for expert in range(len(r['counts'])):
                rows.append({'layer':layer,'expert':expert,'label':label,'tokens':r['tokens'],
                             'selection_rate':float(r['counts'][expert])/r['tokens'],
                             'mean_routing_weight':float(r['mass'][expert])/r['tokens'],
                             'mean_E_dot_hidden':float(r['affinity_sum'][expert])/r['tokens']})
        return {'schema':1,'evidence':rows,
                'notes':'Use balanced labeled calibration inputs, batch=1 or no padding. E is a layer router row, not the input embedding or the expert FFN weights. Selection is correlation, not proof of knowledge. Validate labels on held-out inputs and with ablations.'}

    def write(self,path):
        Path(path).write_text(json.dumps(self.report(),indent=2),encoding='utf-8')

    def propose_mapping(self, experts_per_label=16):
        """Draft for manual review; rates are normalized by each label's tokens."""
        if experts_per_label < 1:
            raise ValueError('experts_per_label must be positive')
        layers={}
        for (layer,label),r in self.data.items():
            counts=r['counts'].float()/r['tokens']
            ids=counts.argsort(descending=True).tolist()
            ids=[i for i in ids if counts[i]>0][:experts_per_label]
            layers.setdefault(str(layer),{})[label]=ids
        return {'schema':1,'reviewed':False,'layers':layers,
                'note':'Draft ranked by selection rate. Review on balanced held-out data; experts can have multiple labels.'}

    def export_router_vectors(self,directory):
        """One E matrix per layer, retaining expert row identities for inspection."""
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        for layer,moe in self.moes.items():
            torch.save({'layer':layer,'E':moe.gate.weight.detach().cpu(),
                        'bias':moe.gate.bias.detach().cpu() if moe.gate.bias is not None else None},
                       directory/f'layer_{layer:03d}_router.pt')


