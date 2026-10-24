import unittest
from asi.data.recovery_corpus import exposure_conflicts

class ExposureTests(unittest.TestCase):
    def test_revisions_and_relabeling_remain_exposed(self):
        old={'id':'a','prompt':'Calculate 12 plus 7','source':'set','source_split':'test','source_id':'3'}
        candidates=[{'id':'b','parent_id':'a','prompt':'Different wording'},
                    {'id':'c','prompt':'Calculate 99 plus 6'},
                    {'id':'d','prompt':'Unrelated','source':'set','source_split':'test','source_id':'3'},
                    {'id':'e','prompt':'New','train_only':True,'split':'test'}]
        self.assertEqual({r['id'] for r in exposure_conflicts(candidates,[old])},{'b','c','d','e'})
    def test_unseen_row_is_not_automatically_exposed(self):
        self.assertEqual(exposure_conflicts([{'id':'new','prompt':'Explain a tree'}],[{'id':'old','prompt':'Sort a list'}]),[])


class BuilderExposureTests(unittest.TestCase):
    def test_filter_precedes_split_quotas(self):
        from unittest.mock import patch
        from asi.data.recovery_corpus import build
        class Tokens:
            eos_token_id=0
            def encode(self,*args,**kwargs):return [1]
        class RowTokens:
            def tolist(self):return [1]
        rows=[dict(id='old',split='train',category='c',template_group='old',prompt='old',answer='x')]
        rows += [dict(id=s,split=s,category='c',template_group=s,prompt=s,answer='x') for s in ['train','dev','test']]
        with patch('asi.data.recovery_corpus.CATEGORIES',{'x':'c'}), patch('asi.data.recovery_corpus.convert',side_effect=rows), patch('asi.experiments.public_moe.input_ids',return_value=[RowTokens()]):
            result=build([('x',{},'train',i) for i in range(4)],Tokens(),train_cap=1,eval_cap=1,historical_rows=[rows[0]])
        self.assertEqual({r['id'] for r in result['examples']},{'train','dev','test'})
        self.assertEqual(result['rejected']['historical_exposure'],1)

if __name__=='__main__':unittest.main()
