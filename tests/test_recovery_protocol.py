import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from asi.data.recovery_corpus import convert, signature
from asi.experiments.recovery import read_examples, objective
from asi.experiments.recovery_protocol import TeacherRow, teacher_identity, training_order, task_metrics, padded_batch
from asi.experiments.recovery_study import make_plan, export_review, summarize


class RecoveryProtocolTests(unittest.TestCase):
    def test_padding_preserves_valid_token_predictions(self):
        from test_public_moe import tiny
        model=tiny().eval()
        rows=[{'tokens':[1,2,3,4]},{'tokens':[5,6]}]
        tokens,attention=padded_batch(rows,0,'cpu')
        self.assertEqual(attention.sum().item(),6)
        with torch.no_grad():
            together=model(tokens,attention_mask=attention,use_cache=False).logits
            for i,row in enumerate(rows):
                separate=model(torch.tensor([row['tokens']]),use_cache=False).logits[0]
                torch.testing.assert_close(together[i,:len(row['tokens'])],separate,atol=1e-6,rtol=1e-5)

    def test_lazy_teacher_matches_dense_loss_and_gradients(self):
        from test_public_moe import tiny
        model=tiny()
        dense={'tokens':[1,2,3,4], 'prompt_length':2,
               'teacher_logp':torch.randn(2,model.config.vocab_size).log_softmax(-1)}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'row.pt'
            torch.save(dense,path)
            lazy=TeacherRow({k:v for k,v in dense.items() if k!='teacher_logp'},path)
            self.assertIsNone(dict.__getitem__(lazy,'teacher_logp'))
            first=objective(model,dense,torch.device('cpu'))[0]
            first.backward()
            gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            model.zero_grad()
            second=objective(model,lazy,torch.device('cpu'))[0]
            second.backward()
            torch.testing.assert_close(first,second,rtol=0,atol=0)
            for n,p in model.named_parameters():
                if n in gradients: torch.testing.assert_close(p.grad,gradients[n],rtol=0,atol=0)

    def test_teacher_shared_across_masks_but_not_source_or_corpus(self):
        first={'source':'model','corpus':'corpus','mapping':'32'}
        second=dict(first,mapping='16')
        self.assertEqual(teacher_identity(first,True),teacher_identity(second,True))
        self.assertNotEqual(teacher_identity(first,True),teacher_identity(dict(first,corpus='other'),True))
        self.assertEqual(teacher_identity(first,False),first)

    def test_curriculum_and_reproducible_exposure(self):
        order=training_order(3,['32','16'],9,42,3)
        self.assertTrue(all(mask=='32' for _,mask in order[:3]))
        self.assertEqual(set(order[3:]),{(i,m) for i in range(3) for m in ('32','16')})
        self.assertEqual(order,training_order(3,['32','16'],9,42,3))
        with self.assertRaises(ValueError): training_order(3,['32'],9,42,3)

    def test_task_metrics_do_not_equate_syntax_or_keywords_with_correctness(self):
        score=task_metrics('Calculation.\nFinal answer: 1,234',{'expected_number':'1234'})
        self.assertTrue(score['task_correct']); self.assertTrue(score['format_pass'])
        bare=task_metrics('1234',{'expected_number':'1234'})
        self.assertTrue(bare['task_correct']); self.assertFalse(bare['format_pass'])
        python=task_metrics('```python\ndef answer():\n    return 0\n```',{'evaluation':'python_syntax_and_manual_tests'})
        self.assertTrue(python['python_syntax_pass']); self.assertIsNone(python['task_correct'])
        bio=task_metrics('DNA is not involved.',{'expected_phrase':'DNA'})
        self.assertTrue(bio['answer_phrase_present']); self.assertIsNone(bio['task_correct'])

    def test_public_rows_and_holdouts(self):
        row=convert('gsm8k',{'question':'A rectangle has area 12. Find its perimeter.',
                            'answer':'Add <<2+2=4>>4.\n#### 4'},'train',0)
        self.assertEqual(row['split'],'dev'); self.assertNotIn('<<',row['answer'])
        row=convert('mbpp',{'text':'Sort a list.','code':'def f(x): return sorted(x)'},'train',0)
        self.assertEqual(row['split'],'dev')
        self.assertIsNone(convert('sciq',{'question':'What is voltage?','support':'Electric potential.'},'train',0))
        self.assertEqual(signature('There are 12 cats'),signature('There are 24 cats'))

    def test_cross_split_topic_rejected(self):
        rows=[{'id':s,'split':s,'prompt':s,'answer':'a','topic':'same'} for s in ('train','dev','test')]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'corpus.json'; path.write_text(json.dumps({'examples':rows}))
            with self.assertRaisesRegex(ValueError,'topic'): read_examples(path)

    def test_plan_has_three_equal_exposures_per_seed(self):
        corpus={'examples':[{'split':'train'}]*4,'teacher_dense_fp32_bytes':100}
        plan=make_plan(corpus,Path('corpus'),Path('study'),Path('out'),[1,2],2,Path('model'))
        self.assertEqual(len(plan['jobs']),8)
        for job in plan['jobs'][2:]:
            args=job['argv']; self.assertEqual(args[args.index('--steps')+1],'8')
            self.assertEqual(args[args.index('--mask-sizes')+1],'32')
        self.assertEqual(plan['jobs'][0]['argv'][-1],str(Path('out/teacher_report')))

    def test_review_is_blinded_and_keeps_unknown_scores_empty(self):
        from asi.taxonomy import digest
        corpus={'examples':[{'id':'x','split':'test','prompt':'Question','answer':'Reference'}]}
        report={'completed':True,'identity':{'corpus':digest(corpus)},'generations':[
            {'id':'x','stage':'after','mask':'32','text':'Response'}]}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json'; path.write_text(json.dumps(report))
            output=Path(folder)/'review'; export_review(corpus,[path],output)
            row=json.loads((output/'review.jsonl').read_text())
            self.assertNotIn('stage',row); self.assertNotIn('mask',row)
            self.assertIsNone(row['correctness_0_2'])

    def test_summary_requires_all_variants_and_matching_native_questions(self):
        settings={key:1 for key in ('supervised','steps','seed','lr','alpha','active','experts','mask_sizes',
                                    'max_length','new_tokens','generation_per_category')}
        records=[{'id':'x','mask':'32','ce':3.}]
        test={'records':records,'masks':{'32':{'records':records}}}
        generation={'id':'x','mask':'32','exact_match':None,'repeated_trigram_fraction':0.,
                    'cache_before':{'loads':0},'cache_after':{'loads':0}}
        identity={'source':'same','corpus':'same'}
        report={'completed':True,'action':'train','identity':identity,'settings':settings,
                'steps':[{'id':'x','mask':'32'}],'test_before':test,'test_after':test,
                'generations':[dict(generation,stage=s) for s in ('before','after')],
                'selected_step':1,'trainable_parameters':10,'peak_cuda_bytes':1,'training_with_validation_seconds':1.}
        native={'completed':True,'action':'native','identity':identity,'settings':settings,'records':records,
                'generations':[generation]}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); paths=[]
            for mode in ('router','expert-lora','router-expert-lora'):
                report['settings']['adaptation']=mode
                path=root/(mode+'.json'); path.write_text(json.dumps(report)); paths.append(path)
            path=root/'native.json'; path.write_text(json.dumps(native)); paths.append(path)
            result=summarize(paths)
            self.assertEqual(len(result['summary']),3)
            self.assertIsNone(result['summary'][0]['ce_std'])
            with self.assertRaises(ValueError): summarize(paths[1:])


if __name__=='__main__': unittest.main()
