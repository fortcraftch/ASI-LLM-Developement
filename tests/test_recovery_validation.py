import copy
import json
import tempfile
from pathlib import Path
import unittest

from asi.experiments.recovery_validation import choose_ids, CATEGORIES, VARIANTS, fingerprint
from asi.experiments.recovery_generation import select_prompts
from asi.analysis.recovery_validation_report import validate, render, compare_memory


class RecoveryValidationTests(unittest.TestCase):
    def test_selection_excludes_training_and_seen_questions(self):
        corpus={'examples':[dict(id=f'{c}-{i}',category=c,split='test' if i<4 else 'train')
                            for c in CATEGORIES for i in range(5)]}
        excluded={f'{c}-0' for c in CATEGORIES}
        selected=choose_ids(corpus,excluded,3)
        self.assertEqual(selected,choose_ids(corpus,excluded,3))
        self.assertEqual(len(set(selected)),9)
        self.assertFalse(set(selected)&excluded)
        self.assertFalse(any(x.endswith('-4') for x in selected))
        with self.assertRaises(ValueError):choose_ids(corpus,excluded,4)

    def test_external_selection_still_requires_test_split(self):
        corpus={'examples':[dict(id='train',split='train'),dict(id='new',split='test')]}
        self.assertEqual(select_prompts(corpus,['new'],[],True)[0]['id'],'new')
        for ids in (['train'],['new','new'],[]):
            with self.assertRaises(ValueError):select_prompts(corpus,ids,[],True)
        with self.assertRaises(ValueError):select_prompts(corpus,['new'],[],False)

    def fixtures(self):
        plan={'prompt_ids':['x'],'hashes':{'checkpoint32':'32','checkpoint16':'16'}}
        base=dict(completed=True,validation_plan_sha256='plan',source_identity='s',corpus_identity='c',
                  mapping_identity='m',protocol={},gpu='g',torch='t',records=[dict(id='x',prompt='p',
                  category='c',input_ids=[1],reference_answer='r',generation_transfers={'loads':0})])
        reports={}
        for name in VARIANTS:
            r=copy.deepcopy(base)
            r.update(variant='original' if name=='original' else name[-2:],adapted=name.startswith('recovered'),
                     mask_identity=None if name=='original' else name[-2:],checkpoint_sha256=name[-2:])
            reports[name]=r
        return reports,plan

    def write_complete_fixture(self, directory):
        reports,plan=self.fixtures()
        plan.update(variants=list(VARIANTS),limitations=['Synthetic fixture'])
        directory.mkdir()
        (directory/'plan.json').write_text(json.dumps(plan),encoding='utf8')
        for name,report in reports.items():
            report['validation_plan_sha256']=fingerprint(directory/'plan.json')
            row=report['records'][0]
            row.update(token_ids=[2,3],text='example',stop_reason='EOS',peak_cuda_allocated_bytes=1024,
                       sampled_peak_rss_bytes=2048,prepare_seconds=1,wall_generation_seconds=2,
                       prepare_transfers={'host_to_device_bytes':0},category_changed=False,
                       metrics=dict(task_correct=None,format_pass=None,python_syntax_pass=None,
                                    answer_phrase_present=None,repeated_trigram_fraction=0))
            row['generation_transfers']['host_to_device_bytes']=0
            (directory/name).mkdir()
            (directory/name/'report.json').write_text(json.dumps(report),encoding='utf8')

    def test_render_and_memory_comparison_preserve_human_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);first=root/'first';second=root/'second'
            self.write_complete_fixture(first);self.write_complete_fixture(second)
            result=render(first)
            self.assertTrue(result['completed'])
            blind=first/'blind_review.json'
            blind.write_text('[{"human_score": 2}]',encoding='utf8')
            render(first)
            self.assertEqual(json.loads(blind.read_text()),[{'human_score':2}])
            compare_memory(first,second,root/'memory.json')
            altered=second/'unadapted16'/'report.json'
            r=json.loads(altered.read_text());r['records'][0]['token_ids']=[99]
            altered.write_text(json.dumps(r),encoding='utf8')
            with self.assertRaises(ValueError):compare_memory(first,second,root/'bad.json')
            self.assertFalse((root/'bad.json').exists())

    def test_refuses_mask_mismatch_and_unexpected_swapping(self):
        reports,plan=self.fixtures()
        validate(reports,plan,'plan')
        for mutation in ('mask','swapping','checkpoint','incomplete','input'):
            with self.subTest(mutation=mutation):
                altered=copy.deepcopy(reports);r=altered['recovered16']
                if mutation=='mask':r['mask_identity']='different'
                elif mutation=='swapping':r['records'][0]['generation_transfers']['loads']=1
                elif mutation=='checkpoint':r['checkpoint_sha256']='wrong'
                elif mutation=='incomplete':r['completed']=False
                else:r['records'][0]['input_ids']=[2]
                with self.assertRaises(ValueError):validate(altered,plan,'plan')


if __name__=='__main__':unittest.main()
