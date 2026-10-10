import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from asi.models.domain import GPT, GPTConfig
from asi.experiments.benchmark_batch import Requests, pack, score_requests, generate_requests, evaluate
from asi.experiments.checkpoint_eval import load_benchmarks
from asi.runtime.routing import DomainSessionRouter
from asi.runtime.cache import ExpertCacheManager


class BatchBenchmarkTest(unittest.TestCase):
    def test_classifier_batch_preserves_sequential_inertia(self):
        names=['expert_00_programming','expert_03_math']
        serial=DomainSessionRouter(names,max_pools=1,load_classifier=False)
        batched=DomainSessionRouter(names,max_pools=1,load_classifier=False)
        texts=['Python source code','calculus derivative','unrelated','Python']
        self.assertEqual([serial.route(t) for t in texts],batched.route_many(texts,2))

    @torch.no_grad()
    def test_all_resident_does_not_activate_all_experts(self):
        cache=ExpertCacheManager('cpu',self.model.config.n_pools)
        cache.initialize(self.model);cache.prepare(self.model,list(range(self.model.config.n_pools)))
        self.model.set_active_pools([1]);seen=[]
        for layer in self.model.layers:
            if hasattr(layer.ffn,'gate'):
                layer.ffn.gate.routing_observer=lambda gate,x,logits,weights,indices,mask:seen.append(indices.clone())
        self.model(torch.tensor([[1,2,3]]))
        self.assertEqual(len(cache.resident_pools),self.model.config.n_pools)
        self.assertTrue(seen)
        for ids in seen:self.assertTrue(((ids>=2)&(ids<4)).all())
    def setUp(self):
        torch.manual_seed(123)
        cfg=GPTConfig(n_layer=2,n_embd=32,n_head=2,inter_dim=64,moe_inter_dim=8,
            kv_lora_rank=16,qk_nope_head_dim=8,qk_rope_head_dim=4,v_head_dim=8,
            block_size=32,max_seq_len=32,original_seq_len=32,max_batch_size=4,
            vocab_size=32,routing_mode='uniform_pool')
        self.model=GPT(cfg).eval()
        enc=SimpleNamespace(encode=lambda s:[ord(c)%30 for c in s],
                            decode=lambda ids:''.join(chr(97+i%26) for i in ids),eot_token=31)
        self.active=[]
        def activate(route):
            self.active.append(route);self.model.set_active_pools(route)
        self.adapter=SimpleNamespace(model=self.model,enc=enc,block_size=32,device='cpu',
            route_key=lambda ids:(sum(ids)%2,),activate=activate,
            logits=lambda ids:self.model(ids)[0].float())

    @torch.no_grad()
    def test_scores_mixed_lengths_pools_and_empty_targets(self):
        requests={(pack(ids),start):None for ids,start in
                  [([1,2,3],2),([2,4,5,6],1),([1]*40,36),([1,3],2),([1],1)]}
        score_requests(self.adapter,requests,4,48)
        for (raw,start),result in requests.items():
            from asi.experiments.benchmark_batch import unpack
            ids=unpack(raw);self.adapter.activate(self.adapter.route_key(ids[:start]))
            if len(ids)<2: self.assertTrue(math.isinf(result[0]));continue
            if len(ids)>32:start=max(1,start-(len(ids)-32));ids=ids[-32:]
            logits=self.model(torch.tensor([ids[:-1]]))[0]
            losses=torch.nn.functional.cross_entropy(logits[0],torch.tensor(ids[1:]),reduction='none')[max(0,start-1):]
            if not losses.numel():self.assertTrue(math.isinf(result[0]));continue
            torch.testing.assert_close(torch.tensor(result),torch.stack((losses.sum(),losses.mean())),rtol=1e-5,atol=1e-5)
        self.assertTrue(all(len(route)==1 for route in self.active))

    @torch.no_grad()
    def test_kv_decode_matches_full_context_including_rollover(self):
        requests={(pack(ids),steps,stop):None for ids,steps,stop in
                  [([1,2],5,True),([2,1],8,False),([4,5,6],3,True),([1]*31,5,False)]}
        generate_requests(self.adapter,requests,4,128)
        for (raw,steps,stop),actual in requests.items():
            from asi.experiments.benchmark_batch import unpack
            ids=unpack(raw);self.adapter.activate(self.adapter.route_key(ids));expected=[]
            for _ in range(steps):
                logits=self.model(torch.tensor([ids[-32:]]))[0]
                token=int(logits[0,-1].argmax());expected.append(token);ids.append(token)
                if stop and token==31:break
            self.assertEqual(actual,expected)

    @torch.no_grad()
    def test_last_logits_and_capture_replay(self):
        self.adapter.activate((0,))
        x=torch.tensor([[1,2,3],[4,5,6]])
        torch.testing.assert_close(self.model(x)[0][:,-1:],self.model(x,logits_to_keep=1)[0])
        def task(m,limit):
            scores=[m.completion_losses([1,2,3],2)[1],m.completion_losses([1,2,4],2)[1]]
            text=m.greedy_generate('hi',2)
            return {'choice':scores.index(min(scores)),'text':text}
        bench=SimpleNamespace(TASKS={'test':task},tqdm=lambda x,**kw:x)
        result=evaluate(bench,'test',self.adapter,None,4,128)
        self.assertIn(result['choice'],(0,1));self.assertEqual(len(result['text']),2)

    @unittest.skipUnless(Path('../Creating-GPT-2-From-0/eval.py').exists(),'Sibling benchmark integration')
    @torch.no_grad()
    def test_all_external_tasks_keep_scoring_protocol(self):
        bench=load_benchmarks(Path('../Creating-GPT-2-From-0/eval.py'))
        bench.tqdm=lambda x,**kw:x
        class Data(list):
            def select(self,indices):return Data(self[i] for i in indices)
        def dataset(path,name=None,split=None):
            row={'ctx':'context','endings':[' a',' b',' c',' d'],'label':0,
                 'text':'some text last','goal':'goal','sol1':'one','sol2':'two',
                 'question_stem':'stem','question':'cause','premise':'premise',
                 'sentence':'prefix _ suffix','option1':'one','option2':'two',
                 'passage':'passage','sentence1':'first','sentence2':'second',
                 'span1_text':'one','span2_text':'two','answer':True,
                 'choice1':'one','choice2':'two','article':'article',
                 'options':['one','two'],'answerKey':'A',
                 'choices':{'text':['one','two'],'label':['A','B']},
                 'answers':['one'],'answer_right_ending':'1',
                 'sentence_quiz1':'one','sentence_quiz2':'two',
                 **{f'input_sentence_{i}':'story' for i in range(1,5)},
                 'mc1_targets':{'choices':['one','two'],'labels':[1,0]}}
            if 'winogrande' in path:row['answer']='1'
            if 'trivia' in path:row['answer']={'aliases':['one'],'value':'one'}
            if 'race' in path:row['answer']='A'
            if 'gsm8k' in path:row['answer']='work #### 1'
            if 'mmlu' in path:row['choices']=['one','two'];row['answer']=0
            return Data([row])
        bench.ds=dataset
        outer=self.adapter
        class Reference(bench.Adapter):
            def completion_losses(self,ids,completion_start):
                outer.activate(outer.route_key(ids[:completion_start]))
                return super().completion_losses(ids,completion_start)
            def greedy_generate(self,prompt,max_new_tokens=20):
                outer.activate(outer.route_key(self.enc.encode(prompt)))
                return super().greedy_generate(prompt,max_new_tokens)
        reference=Reference(self.model,self.model.config,'cpu','test',outer.enc)
        for task in bench.TASKS:
            with self.subTest(task=task):
                expected=bench.TASKS[task](reference,1)
                actual=evaluate(bench,task,outer,1,4,128)
                for key in expected:
                    self.assertTrue(math.isclose(actual[key],expected[key],rel_tol=1e-5,abs_tol=1e-6),(key,actual,expected))


if __name__=='__main__':unittest.main()
