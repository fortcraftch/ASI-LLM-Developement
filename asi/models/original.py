"""Load the user's original eight-expert architecture without changing its weights."""
import hashlib
import importlib.util
import sys
from pathlib import Path
import torch
from asi import ROOT

DEFAULT_ARCHITECTURE=ROOT.parent/'Creating-DeepSeek-V3-From-0/train_deepseek_v3.py'


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): digest.update(chunk)
    return digest.hexdigest()


def import_architecture(architecture):
    architecture=Path(architecture).resolve()
    if not architecture.is_file(): raise FileNotFoundError(f'Pass --architecture with the original training .py: {architecture}')
    module_name='original_deepseek_'+file_sha256(architecture)[:12]
    spec=importlib.util.spec_from_file_location(module_name,architecture)
    module=importlib.util.module_from_spec(spec)
    sys.modules[module_name]=module
    # The original file imports its adjacent kernel and HellaSwag helpers.
    sys.path.insert(0,str(architecture.parent))
    try: spec.loader.exec_module(module)
    finally: sys.path.pop(0)
    config_class=getattr(module,'DeepSeekV3Config',None)
    model_class=getattr(module,'DeepSeekV3',None)
    if config_class is None or model_class is None:
        raise ValueError('Expected original DeepSeekV3Config and DeepSeekV3 classes')
    return module, config_class, model_class


def load_original_model(checkpoint,architecture=DEFAULT_ARCHITECTURE):
    _, config_class, model_class = import_architecture(architecture)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    config=config_class(**ckpt['config'])
    model=model_class(config)
    model.load_state_dict(ckpt['model'],strict=True)
    model.eval()
    metadata={'checkpoint':str(Path(checkpoint).resolve()),'checkpoint_sha256':file_sha256(checkpoint),
              'architecture':str(architecture),'architecture_sha256':file_sha256(architecture),
              'step':ckpt.get('step'),'val_loss':ckpt.get('val_loss'),'config':ckpt['config']}
    return model,metadata


def load_streamed_model(directory, architecture=DEFAULT_ARCHITECTURE):
    """Instantiate on meta; load backbone only, leaving routed weights as placeholders.

    Attach an ExpertDiskStore-backed NativeExpertSessionCache before any forward.
    Supported native adapter only; no claim of universal HF/distributed loading.
    """
    import json
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    metadata = manifest['metadata']
    if manifest.get('schema') != 1 or file_sha256(architecture) != metadata['architecture_sha256']:
        raise ValueError('Store architecture/schema mismatch')
    if file_sha256(directory/'backbone.pt') != manifest['backbone_sha256']:
        raise ValueError('Backbone checksum mismatch')
    module, config_class, model_class = import_architecture(architecture)
    config = config_class(**metadata['config'])
    with torch.device('meta'):
        model = model_class(config)
    parameters = dict(model.named_parameters())
    for entry in manifest['experts']:
        for name, spec in entry['parameters'].items():
            full_name = f'layers.{entry["key"][0]}.ffn.experts.{entry["key"][1]}.{name}'
            parameter = parameters.get(full_name)
            if parameter is None or list(parameter.shape) != spec['shape'] or str(parameter.dtype) != spec['dtype']:
                raise ValueError('Store expert specification differs from architecture: ' + full_name)
    expected = {f'layers.{entry["key"][0]}.ffn.experts.{entry["key"][1]}.{name}'
                for entry in manifest['experts'] for name in entry['parameters']}
    backbone = torch.load(directory/'backbone.pt', map_location='cpu', mmap=True, weights_only=True)
    incompatible = model.load_state_dict(backbone, strict=False, assign=True)
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError('Backbone/expert manifest does not match original architecture')
    with torch.no_grad():
        for parent in model.modules():
            for name, parameter in list(parent.named_parameters(recurse=False)):
                if parameter.is_meta:
                    parent._parameters[name] = torch.nn.Parameter(torch.empty(0, dtype=parameter.dtype), requires_grad=False)
            for name, buffer in list(parent.named_buffers(recurse=False)):
                if not buffer.is_meta:
                    continue
                if name == 'freqs_cis':
                    source_module = module if hasattr(module, 'precompute_freqs_cis') else sys.modules[type(model).__module__]
                    value = source_module.precompute_freqs_cis(config)
                elif name == 'active_expert_mask':
                    value = torch.ones(buffer.shape, dtype=buffer.dtype)
                elif name == 'expert_pool_ids' and hasattr(config, 'experts_per_pool'):
                    value = torch.arange(config.n_routed_experts, dtype=buffer.dtype) // config.experts_per_pool
                elif name in ('kv_cache', 'pe_cache', 'k_cache', 'v_cache'):
                    value = torch.zeros(buffer.shape, dtype=buffer.dtype)
                else:
                    raise ValueError('Unsupported nonpersistent meta buffer: ' + name)
                setattr(parent, name, value)
    model.eval()
    return model, dict(metadata, expert_store=str(directory.resolve()))


def sync(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


def autocast(device):
    from contextlib import nullcontext
    return torch.autocast('cuda',dtype=torch.bfloat16) if torch.device(device).type=='cuda' else nullcontext()


@torch.inference_mode()
def generate_original(model,enc,prompt,device,max_new_tokens=32,temperature=0.,top_k=50):
    """Original forward has no start_pos: recompute each retained context exactly."""
    ids=enc.encode(prompt)
    if not ids: ids=[enc.eot_token]
    generated=[]
    for _ in range(max_new_tokens):
        x=torch.tensor([ids[-model.config.block_size:]],device=device)
        with autocast(device): logits,_=model(x)
        logits=logits[:,-1,:enc.n_vocab].float()
        if temperature<=0: token=logits.argmax(-1,keepdim=True)
        else:
            logits=logits/temperature
            if top_k>0:
                threshold=logits.topk(min(top_k,logits.shape[-1])).values[:,-1:]
                logits=logits.masked_fill(logits<threshold,float('-inf'))
            token=torch.multinomial(logits.softmax(-1),1)
        value=token.item(); ids.append(value); generated.append(value)
        if value==enc.eot_token: break
    return enc.decode(generated)


def native_moes(model):
    result={i:layer.ffn for i,layer in enumerate(model.layers)
            if hasattr(layer.ffn,'experts') and hasattr(layer.ffn,'gate')}
    if not result:
        raise ValueError('Expected native DeepSeek model.layers[*].ffn.gate/experts')
    return result


class IncrementalDecoder:
    """Batch-one adapter over the original cache-aware blocks, without weight changes.

    Serial prefill keeps a top-2 model's demanded union at two experts per layer.
    A reset logically invalidates KV: each prefix position is overwritten before
    attention reads it. No sliding-window positional reset is performed silently.
    """
    def __init__(self, model):
        if model.training:
            raise ValueError('Incremental decoding requires eval mode')
        for name in ('embed', 'layers', 'norm', 'head', 'freqs_cis', 'config'):
            if not hasattr(model, name):
                raise ValueError('Unsupported native decoder interface: ' + name)
        self.model = model
        self.position = 0

    def reset(self):
        self.position = 0

    @torch.inference_mode()
    def step(self, token):
        if self.model.training:
            raise ValueError('Incremental decoding is inference-only')
        if self.position >= self.model.config.block_size:
            raise ValueError('KV context is full; explicitly reset and prefill a retained context')
        device = self.model.embed.weight.device
        idx = torch.tensor([[int(token)]], device=device)
        x = self.model.embed(idx)
        freqs = self.model.freqs_cis[self.position:self.position + 1].to(device)
        for layer in self.model.layers:
            x = layer(x, self.position, freqs, None)
        logits = self.model.head(self.model.norm(x))
        self.position += 1
        return logits


@torch.inference_mode()
def generate_incremental(model, enc, prompt, device, max_new_tokens=32, temperature=0., top_k=50):
    if not 1 <= max_new_tokens < model.config.block_size:
        raise ValueError('Generation length must fit the model context')
    ids = enc.encode(prompt) or [enc.eot_token]
    ids = ids[-(model.config.block_size - max_new_tokens + 1):]
    decoder = IncrementalDecoder(model)
    for token in ids:
        with autocast(device):
            logits = decoder.step(token)
    generated = []
    for step in range(max_new_tokens):
        scores = logits[0, -1, :enc.n_vocab].float()
        if temperature <= 0:
            token = int(scores.argmax())
        else:
            scores = scores / temperature
            if top_k > 0:
                scores = scores.masked_fill(scores < scores.topk(min(top_k, len(scores))).values[-1], -float('inf'))
            token = int(torch.multinomial(scores.softmax(-1), 1))
        generated.append(token)
        if token == enc.eot_token or step == max_new_tokens - 1:
            break
        with autocast(device):
            logits = decoder.step(token)
    return enc.decode(generated)


