import json
import hashlib
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict

import numpy as np
import torch
from asi.models.domain import GPT
from asi.models.design import count_parameters, pool_manifest, validate_recipe
from asi.experiments.train import RandomWindowStream, activate_training_pool
from asi.analysis.experts import RoutingTrace
from asi.runtime.routing import DomainSessionRouter
from asi.runtime.ensemble import IndependentModelRouter
from asi.data.identity import manifest_matches, manifest_sha256
from test_expert_runtime import tiny


class DesignTests(unittest.TestCase):
    def test_manifest_identity_survives_checkout_line_endings_only(self):
        lf = b'{\n  "pools": {"a": 1, "b": 2}\n}\n'
        crlf = lf.replace(b'\n', b'\r\n')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            for data in (lf, crlf):
                path.write_bytes(data)
                for original in (lf, crlf):
                    self.assertTrue(manifest_matches(path, hashlib.sha256(original).hexdigest()))
                self.assertEqual(manifest_sha256(path), hashlib.sha256(lf).hexdigest())
            for changed in (lf.replace(b'"a": 1', b'"a": 3'), lf.replace(b'"a": 1, "b": 2', b'"b": 2, "a": 1')):
                path.write_bytes(changed)
                self.assertFalse(manifest_matches(path, hashlib.sha256(crlf).hexdigest()))

    def test_uniform_training_bypasses_E_and_freezes_inactive_experts(self):
        config=tiny().config;config.routing_mode='uniform_pool'
        model=GPT(config).train();model.set_active_pools([1])
        with RoutingTrace(model,['a','b']) as trace, patch('asi.models.domain.Gate.route_logits',side_effect=AssertionError('E must not execute')):
            x=torch.randint(0,64,(1,4));_,loss=model(x,x);loss.backward()
        self.assertTrue(all(not layer['shadow_available'] for layer in trace.layers.values()))
        for layer in model.layers[1:]:
            self.assertIsNone(layer.ffn.gate.weight.grad)
            self.assertIsNone(layer.ffn.experts[0].w1.weight.grad)
            self.assertIsNotNone(layer.ffn.experts[2].w1.weight.grad)
        model.set_active_pools([0,1])
        with self.assertRaisesRegex(ValueError,'exactly N'): model(x)

    def test_random_windows_mix_files_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[]
            for i in (0,100):
                p=Path(directory)/f'{i}.npy';np.save(p,np.arange(i,i+50,dtype=np.uint16));paths.append(p)
            stream=RandomWindowStream(paths,16,4,42)
            x,y=stream.next_batch();self.assertTrue(torch.equal(x[:,1:],y[:,:-1]))
            self.assertTrue((x[:,0]<100).any() and (x[:,0]>=100).any())
            state=stream.state();expected=stream.next_batch()
            stream.load_state(state);actual=stream.next_batch()
            self.assertTrue(torch.equal(expected[0],actual[0]))

    def test_taxonomy_coverage_and_classifier_identity(self):
        source={'tokenizer':'gpt2','dtype':'uint16','total_tokens':30,
                'categories':{'medicine_health__a':{'tokens':10,'documents':1},
                              'mathematics_statistics__b':{'tokens':20,'documents':2}}}
        pools=pool_manifest(source,{'health':['medicine_health'],'math':['mathematics_statistics']})
        self.assertEqual(pools['summary']['assigned_tokens'],30)
        with self.assertRaises(ValueError): pool_manifest(source,{'health':['medicine_health']})
        router=DomainSessionRouter(['health','math'],max_pools=1,load_classifier=False,pool_manifest=pools)
        with patch.object(router,'_broad_scores',return_value={'medicine_health':.9}):
            self.assertEqual(router.route('input').ranked_pools,['health'])

    def test_base_and_domain_differ_in_permissions_not_size(self):
        model=tiny();config=asdict(model.config)
        self.assertEqual(count_parameters(config)['total'],sum(p.numel() for p in model.parameters()))
        recipe={'schema':1,'family':'base','model':config,'pool_order':['a','b']}
        validate_recipe(recipe,['a','b']);model.training_recipe=recipe;activate_training_pool(model,0)
        self.assertEqual(int(model.active_expert_mask.sum()),4)
        recipe['family']='domain';activate_training_pool(model,0)
        self.assertEqual(int(model.active_expert_mask.sum()),2)

    def test_independent_models_reuse_and_change_whole_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            names=['a','b'];paths={}
            for name in names:
                config=tiny().config;config.n_dense_layers=config.n_layer
                model=GPT(config)
                path=Path(directory)/(name+'.pt');paths[name]=path
                torch.save({'config':asdict(config),'model':model.state_dict(),'pool_names':names,
                            'training_recipe':{'family':'router','pool':name}},path)
            router=IndependentModelRouter(paths,names,total_budget=1_000_000)
            first,changed=router.select('a');self.assertTrue(changed)
            again,changed=router.select('a');self.assertFalse(changed);self.assertIs(first,again)
            other,changed=router.select('b');self.assertTrue(changed);self.assertIsNot(first,other)
            self.assertEqual(router.loads,2)

    def test_recipe_training_checkpoint_resume_matches_uninterrupted(self):
        from asi.experiments.train import main
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'manifest.json';source.write_text('{}')
            manifest=root/'pools.json'
            manifest.write_text(json.dumps({'pools':{name:{'categories':[name],'tokens':128} for name in ['a','b']}}))
            for name in ['a','b']:
                (root/name).mkdir()
                for split in ['train','val']:
                    np.save(root/name/(split+'_0.npy'),np.arange(128,dtype=np.uint16)%64)
            config=asdict(tiny().config);config['routing_mode']='uniform_pool'
            recipe={'schema':1,'family':'domain','model':config,'pool_order':['a','b'],
                    'parameter_budget':1_000_000,'source_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'pool_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    'training':{'batch_size':1,'seq_len':8,'total_batch_size':8,'max_steps':3,
                                'warmup_steps':1,'val_interval':1,'save_interval':1,'val_steps_per_pool':1,
                                'data_order':'random','pool_sampling':'sqrt'}}
            recipe_path=root/'recipe.json';recipe_path.write_text(json.dumps(recipe))
            args=['train','--recipe',str(recipe_path),'--data-root',str(root),'--pool-manifest',str(manifest),
                  '--device','cpu','--log-dir',str(root/'full')]
            with contextlib.redirect_stdout(io.StringIO()),patch('sys.argv',args):main()
            resumed=args[:-1]+[str(root/'resumed'),'--resume',str(root/'full/model_00001.pt')]
            with contextlib.redirect_stdout(io.StringIO()),patch('sys.argv',resumed):main()
            full=torch.load(root/'full/model_00002.pt',weights_only=False)
            resumed=torch.load(root/'resumed/model_00002.pt',weights_only=False)
            self.assertEqual(full['train_state'],resumed['train_state'])
            self.assertEqual(full['pool_rng_state'],resumed['pool_rng_state'])
            for key,tensor in full['model'].items():
                self.assertTrue(torch.equal(tensor,resumed['model'][key]),key)


if __name__=='__main__':unittest.main()
