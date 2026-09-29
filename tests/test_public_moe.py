import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import OlmoeConfig, OlmoeForCausalLM
from asi.models.hf_olmoe import OlmoeExpertCache, CategoryCalibration
from asi.models.airllm_olmoe import pack_experts
from asi.experiments.public_moe import sample_examples, choose_capacity, read_lock, sha256, calibrate, compare, MODEL, REVISION, CATEGORIES
from asi.taxonomy import Taxonomy
from asi import ROOT
from asi.models.olmoe_storage import load_disk_backed
import json
from unittest.mock import patch
from types import SimpleNamespace


def tiny():
    torch.manual_seed(23)
    config = OlmoeConfig(vocab_size=64, hidden_size=32, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=64,
        pad_token_id=0, eos_token_id=1, norm_topk_prob=False)
    config._attn_implementation = 'eager'
    return OlmoeForCausalLM(config).eval()


class PublicMoeTests(unittest.TestCase):
    def test_continuation_scores_exclude_prompt_and_self_kl_is_zero(self):
        from asi.experiments.routing_study import reference_scores
        class Model:
            def __call__(self, sequence, **kwargs):
                logits = torch.zeros(1, sequence.shape[1], 8)
                logits[0, 0, 7] = 1000  # Unscored prompt position.
                logits[0, 1, 3] = 2
                logits[0, 2, 4] = 2
                return SimpleNamespace(logits=logits)
        ids = torch.tensor([[1, 2]])
        scores, teacher = reference_scores(Model(), ids, [3, 4], 'cpu')
        self.assertEqual(teacher.shape, (2, 8))
        self.assertAlmostEqual(scores['native_continuation_nll'], float(torch.log(torch.exp(torch.tensor(2.)) + 7) - 2), places=6)
        repeated, _ = reference_scores(Model(), ids, [3, 4], 'cpu', teacher)
        self.assertAlmostEqual(repeated['teacher_kl'], 0., places=6)

    def test_restricted_all_experts_equals_native_and_never_loads(self):
        for normalize in (False, True):
            reference = tiny()
            for layer in reference.model.layers:
                layer.mlp.gate.norm_topk_prob = normalize
            reference.config.norm_topk_prob = normalize
            model = copy.deepcopy(reference)
            cache = OlmoeExpertCache(model, 'cpu', capacity=8)
            mapping = {str(i): {'experts': [0, 1, 2, 3], 'weights': [.25] * 4} for i in range(2)}
            for policy in ['restricted', 'restricted_native_mass']:
                cache.set_context(mapping, mixture=policy, active=2)
                loads = cache.stats['loads']
                with torch.inference_mode():
                    ids = torch.tensor([[3, 5, 8]])
                    torch.testing.assert_close(model(ids).logits, reference(ids).logits)
                self.assertEqual(cache.stats['loads'], loads)
            cache.close()

    def test_restricted_respects_candidates_and_active_count(self):
        model = tiny()
        cache = OlmoeExpertCache(model, 'cpu', capacity=4)
        mapping = {str(i): {'experts': [1, 3], 'weights': [.25, .25]} for i in range(2)}
        for policy in ['restricted', 'restricted_native_mass']:
            cache.set_context(mapping, mixture=policy, active=1)
            def check(layer, indices, weights):
                self.assertEqual(indices.shape[-1], 1)
                self.assertTrue(set(indices.flatten().tolist()) <= {1, 3})
            cache.callbacks = [check]
            before = cache.stats['loads']
            with torch.inference_mode():
                model(torch.tensor([[3, 4, 5]]))
            self.assertEqual(cache.stats['loads'], before)
        with self.assertRaises(ValueError):
            cache.set_context(mapping, mixture='restricted', active=3)
        cache.close()

    def test_disk_backed_meta_experts_and_zero_ram_cache(self):
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast
        reference = tiny().to(dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as folder:
            reference.save_pretrained(folder)
            tokenizer = Tokenizer(models.WordLevel({'pad': 0, 'eos': 1, 'unk': 2}, unk_token='unk'))
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token='pad', eos_token='eos', unk_token='unk').save_pretrained(folder)
            model, _, store = load_disk_backed(folder, 0, 32 * 2**30, device='cpu')
            self.assertTrue(model.model.layers[0].mlp.experts.gate_up_proj.is_meta)
            cache = OlmoeExpertCache(model, 'cpu', capacity=2, store=store)
            ids = torch.tensor([[3, 4, 7]])
            with torch.inference_mode():
                torch.testing.assert_close(model(ids).logits, reference(ids).logits, atol=.003, rtol=.03)
            self.assertEqual(store.bytes, 0)
            self.assertGreater(store.stats['ram_misses'], 0)
            self.assertGreater(store.stats['logical_read_bytes'], 0)
            store.limit = 100000
            pair = store.get((0, 0))
            hits = store.stats['ram_hits']
            self.assertIs(store.get((0, 0)), pair)
            self.assertEqual(store.stats['ram_hits'], hits + 1)
            with patch('psutil.virtual_memory', return_value=SimpleNamespace(available=0)):
                store.trim()
            self.assertEqual(store.bytes, 0)
            cache.close()

    def test_calibration_trains_head_with_frozen_embeddings(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return [3, 4, 5]
        model = tiny()
        cache = OlmoeExpertCache(model, 'cpu', capacity=4)
        before = model.model.embed_tokens.weight.detach().clone()
        rows = [{'text': name, 'category': name} for name in CATEGORIES]
        mapping, head = calibrate(model, Tokenizer(), cache, rows, 2)
        self.assertEqual(set(mapping), set(CATEGORIES))
        self.assertEqual(head(torch.randn(1, 32)).shape, (1, 3))
        torch.testing.assert_close(before, model.model.embed_tokens.weight)
        cache.close()

    def test_comparison_rejects_different_prompts(self):
        row = {'repeat': 0, 'prompt_id': 'one', 'input_ids': [1, 2], 'token_ids': [3, 4],
               'ttft_seconds': 1., 'generation_seconds': 2., 'prepare_seconds': .5,
               'classifier_seconds': 0., 'peak_cuda_allocated_bytes': 10, 'rss_bytes': 20}
        native = {'executed': True, 'backend': 'native', 'routing': 'oracle', 'records': [row],
                  'load_seconds': 1., 'hardware': {'gpu': {'name': 'test', 'total_bytes': 100}}}
        fixed = copy.deepcopy(native)
        fixed['backend'] = 'fixed'
        fixed['records'][0]['token_ids'] = [3, 5]
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / f'{i}.json' for i in range(2)]
            for p, report in zip(paths, [native, fixed]):
                p.write_text(json.dumps(report))
            self.assertEqual(compare(paths)['runs'][1]['token_position_agreement_with_native'], .5)
            fixed['records'][0]['input_ids'] = [1, 3]
            paths[1].write_text(json.dumps(fixed))
            with self.assertRaises(ValueError):
                compare(paths)

    def test_checkpoint_layout_bridge(self):
        state = {f'model.layers.0.mlp.experts.{i}.{name}.weight': torch.full((2, 2), float(i + value))
            for i in range(2) for name, value in [('gate_proj', 1), ('up_proj', 3), ('down_proj', 5)]}
        packed = pack_experts(state, 2)
        self.assertEqual(packed['model.layers.0.mlp.experts.gate_up_proj'].shape, (2, 4, 2))
        torch.testing.assert_close(packed['model.layers.0.mlp.experts.gate_up_proj'][1, 2:], torch.full((2, 2), 4.))
        self.assertIs(pack_experts(packed, 2), packed)

    def test_budget_and_disjoint_examples(self):
        with self.assertRaises(ValueError):
            choose_capacity(tiny(), 2, 1, 0)
        taxonomy = Taxonomy.read(ROOT / 'configs/knowledge_v1.json')
        path = ROOT / 'data/classifier_seed_v1/train_val.jsonl'
        train = sample_examples(path, taxonomy, 'train', 8)
        val = sample_examples(path, taxonomy, 'val', 1)
        self.assertEqual(len(train), 24)
        self.assertFalse({r['id'] for r in train} & {r['id'] for r in val})

    def test_lock_detects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / 'fake.safetensors'
            file.write_bytes(b'abc')
            lock = {'model': MODEL, 'revision': REVISION, 'files': {file.name: 3}, 'sha256': {file.name: sha256(file)}}
            (root / 'asi_source.json').write_text(json.dumps(lock))
            self.assertEqual(read_lock(root), lock)
            file.write_bytes(b'abd')
            with self.assertRaises(ValueError):
                read_lock(root)

    def test_native_cache_preserves_logits_and_kv(self):
        reference = tiny()
        model = copy.deepcopy(reference)
        cache = OlmoeExpertCache(model, 'cpu', capacity=2)
        ids = torch.tensor([[3, 4, 7]])
        with torch.inference_mode():
            expected = reference(ids, use_cache=True)
            actual = model(ids, use_cache=True)
            torch.testing.assert_close(actual.logits, expected.logits)
            next_ids = torch.tensor([[8]])
            torch.testing.assert_close(model(next_ids, past_key_values=actual.past_key_values).logits,
                reference(next_ids, past_key_values=expected.past_key_values).logits)
        self.assertLessEqual(cache.snapshot()['resident_experts'], 2)
        self.assertGreater(cache.snapshot()['evictions'], 0)
        cache.close()

    def test_fixed_context_never_loads_during_generation(self):
        model = tiny()
        cache = OlmoeExpertCache(model, 'cpu', capacity=4)
        mapping = {str(i): {'experts': [0, 1], 'weights': [.25, .25]} for i in range(2)}
        cache.set_context(mapping)
        before = cache.snapshot()['loads']
        with torch.inference_mode():
            model(torch.tensor([[2, 3, 4]]))
            model(torch.tensor([[5]]))
        self.assertEqual(cache.snapshot()['loads'], before)
        cache.set_context(mapping)
        self.assertEqual(cache.snapshot()['loads'], before)
        changed = {str(i): {'experts': [1, 2], 'weights': [.25, .25]} for i in range(2)}
        cache.set_context(changed)
        self.assertEqual(cache.snapshot()['loads'], before + 2)
        with self.assertRaises(ValueError):
            cache.set_context({'0': mapping['0']})
        cache.close()

    def test_calibration_preserves_native_routing_mass(self):
        collector = CategoryCalibration(1, 4)
        collector.labels = ['math']
        collector.observe(0, torch.tensor([[0, 1], [0, 2]]), torch.tensor([[.2, .1], [.3, .2]]))
        entry = collector.mapping(['math'], 2)['math']['0']
        self.assertEqual(entry['experts'], [0, 1])
        self.assertAlmostEqual(sum(entry['weights']), .4, places=6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_bf16_adapter(self):
        reference = tiny().to(device='cuda', dtype=torch.bfloat16)
        model = copy.deepcopy(reference).cpu()
        cache = OlmoeExpertCache(model, 'cuda', capacity=4)
        ids = torch.tensor([[3, 4, 7]], device='cuda')
        with torch.inference_mode():
            torch.testing.assert_close(model(ids).logits, reference(ids).logits, atol=.003, rtol=.03)
        self.assertGreater(cache.snapshot()['host_to_device_bytes'], 0)
        self.assertEqual(model.model.layers[0].mlp.experts.gate_up_proj.device.type, 'cpu')
        cache.close()


if __name__ == '__main__':
    unittest.main()
