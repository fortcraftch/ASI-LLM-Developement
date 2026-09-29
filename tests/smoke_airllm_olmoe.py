"""Optional real CUDA/AirLLM integration smoke, with random tiny weights only."""
import json
from pathlib import Path
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from test_public_moe import tiny
from asi.models.airllm_olmoe import load_airllm


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        model = tiny().to(dtype=torch.bfloat16)
        model.save_pretrained(folder)
        tokenizer = Tokenizer(models.WordLevel({f't{i}': i for i in range(64)}, unk_token='t2'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='t0', eos_token='t1', unk_token='t2').save_pretrained(folder)
        model.cuda()
        ids = torch.tensor([[3, 4, 7]], device='cuda')
        with torch.inference_mode():
            expected = model(ids, use_cache=True)
            expected_next = model(torch.tensor([[8]], device='cuda'), past_key_values=expected.past_key_values)
        model.cpu()
        air = load_airllm(folder, device='cuda:0', dtype=torch.bfloat16,
            layer_shards_saving_path=str(Path(folder) / 'split'), prefetching=False, delete_original=False)
        air.model.set_attn_implementation('eager')
        air.model.eval()
        with torch.inference_mode():
            actual = air(ids, use_cache=True)
            actual_next = air(torch.tensor([[8]], device='cuda'), past_key_values=actual.past_key_values)
        torch.testing.assert_close(actual.logits, expected.logits, atol=.003, rtol=.03)
        torch.testing.assert_close(actual_next.logits, expected_next.logits, atol=.003, rtol=.03)
        report = {'passed': True, 'scope': 'Random tiny OLMoE only; not the public 7B checkpoint',
            'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'prefill_max_abs_difference': float((actual.logits - expected.logits).abs().max()),
            'decode_max_abs_difference': float((actual_next.logits - expected_next.logits).abs().max())}
        output = Path('results/public_moe_smoke/airllm.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
