"""OLMoE checkpoint-layout bridge; leaves AirLLM's layer streaming intact."""
import re
from pathlib import Path
import gc
import torch


def pack_experts(state, num_experts):
    prefixes = {match.group(1) for name in state
                if (match := re.fullmatch(r'(.*\.mlp\.experts)\.\d+\.gate_proj\.weight', name))}
    if not prefixes:
        return state
    converted = dict(state)
    for prefix in prefixes:
        def stack(projection):
            return torch.stack([converted.pop(f'{prefix}.{e}.{projection}.weight') for e in range(num_experts)])
        gate, up = stack('gate_proj'), stack('up_proj')
        converted[prefix + '.gate_up_proj'] = torch.cat((gate, up), dim=1)
        converted[prefix + '.down_proj'] = stack('down_proj')
    return converted


def load_airllm(directory, **kwargs):
    from airllm import AirLLMBaseModel
    from safetensors.torch import load_file, save_file
    air = AirLLMBaseModel(str(directory), **kwargs)
    if air.config.model_type != 'olmoe' or air.compression is not None:
        raise ValueError('Bridge only supports uncompressed OLMoE')
    # Convert derived shards once, during setup, instead of penalizing each token
    # with CPU repacking. Original checkpoint files are never modified.
    for layer in range(air.config.num_hidden_layers):
        path = Path(air.checkpoint_path) / f'model.layers.{layer}.safetensors'
        state = load_file(str(path), device='cpu')
        converted = pack_experts(state, air.config.num_experts)
        if converted is not state:
            pending = path.with_suffix('.packing')
            save_file(converted, str(pending))
            del state, converted
            gc.collect()
            pending.replace(path)
        else:
            del state, converted
    return air
