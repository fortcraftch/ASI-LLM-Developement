"""Tests for deterministic post-hoc data sampling and original checkpoint loading."""
import sys
from pathlib import Path
import tempfile
import unittest
from dataclasses import asdict
import numpy as np
import torch
from asi.experiments.posthoc import sample_windows,read_window,coverage_curves
from asi.models.original import load_original_model,file_sha256
from test_expert_runtime import tiny

class PosthocTests(unittest.TestCase):
    def test_sampling_is_disjoint_reproducible_and_reports_missing_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'a').mkdir(); (root/'b').mkdir()
            np.save(root/'a/train_000.npy',np.arange(60,dtype=np.uint16))
            np.save(root/'a/val_000.npy',np.arange(20,dtype=np.uint16))
            np.save(root/'b/train_000.npy',np.arange(20,dtype=np.uint16))
            manifest={'pools':{'math':{'categories':['a']},'code':{'categories':['b']}}}
            rows,coverage=sample_windows(root,manifest,'train',4,8,123)
            self.assertEqual(rows,sample_windows(root,manifest,'train',4,8,123)[0])
            self.assertEqual(coverage['code']['windows'],4)
            for shard in {w['shard'] for w in rows}:
                occupied=[]
                for w in rows:
                    if w['shard']==shard: occupied.extend(range(w['start'],w['start']+w['length']))
                self.assertEqual(len(occupied),len(set(occupied)))
            val,coverage=sample_windows(root,manifest,'val',4,8,124)
            self.assertEqual(coverage['code']['windows'],0)
            x,y=read_window(val[0],'cpu')
            self.assertTrue(torch.equal(x[:,1:],y[:,:-1]))

    def test_original_loader_is_strict_and_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'original.py'; checkpoint=root/'model.pt'
            source.write_text('from asi.models.domain import GPT as DeepSeekV3, GPTConfig as DeepSeekV3Config\nif __name__ == "__main__": raise RuntimeError("must not train")\n')
            model=tiny()
            torch.save({'config':asdict(model.config),'model':model.state_dict(),'step':4750,'val_loss':3.4},checkpoint)
            loaded,meta=load_original_model(checkpoint,source)
            self.assertEqual(meta['step'],4750)
            self.assertEqual(meta['checkpoint_sha256'],file_sha256(checkpoint))
            self.assertFalse(loaded.training)
            for name,value in model.state_dict().items(): self.assertTrue(torch.equal(value,loaded.state_dict()[name]))

    def test_coverage_uses_selected_slots_not_number_of_prompts(self):
        base=[({'label':'math'},{'routes':{'1':[[0,1],[1,2]]}})]
        curves=coverage_curves(base,{'layers':{'1':{'math':[1,2,0]}}},[1,2,3])
        self.assertEqual([r['recall'] for r in curves],[.5,.75,1.])

if __name__=='__main__': unittest.main()
