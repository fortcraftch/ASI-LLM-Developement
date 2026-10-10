import unittest
import tempfile
from pathlib import Path
from asi.experiments.recovery_selection import selection_key
from asi.experiments.recovery_200 import screen
from asi.experiments.recovery_200 import resume_teacher_command


class SelectionTests(unittest.TestCase):
    def test_teacher_resume_preserves_previous_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            argv=['prepare','--output',str(root/'prepare')]
            with self.assertRaises(ValueError):resume_teacher_command(root,argv)
            (root/'teacher').mkdir();(root/'teacher/manifest.json').write_text('{}')
            (root/'teacher_resume_001.log').write_text('old log')
            command,log=resume_teacher_command(root,argv)
            self.assertEqual(log.name,'teacher_resume_002.log')
            self.assertIn('--resume-teacher',command)
            self.assertEqual(argv[-1],str(root/'prepare'))
            self.assertEqual((root/'teacher_resume_001.log').read_text(),'old log')

    def fixture(self):
        g=[dict(split='dev',mask='32',category=c,task_correct=True if c=='math' else None,
                format_pass=True,repeated_trigram_fraction=.1) for c in ('math','code','bio')]
        e={'records':[dict(mask='32',category=c,ce=2.,loss=999.) for c in ('math','code','bio')]}
        return g,e

    def test_selection_uses_ce_not_treatment_objective(self):
        g,e=self.fixture();a=selection_key(g,e,e)
        for r in e['records']:r['loss']=.01
        self.assertEqual(selection_key(g,e,e),a)
        g[0]['task_correct']=False
        self.assertGreater(selection_key(g,e,e),a)

    def test_rejects_test_selection_and_missing_groups(self):
        g,e=self.fixture();g[0]['split']='test'
        with self.assertRaises(ValueError):selection_key(g,e,e)
        g,e=self.fixture()
        with self.assertRaises(ValueError):selection_key(g[:-1],e,e)

    def test_reference_screen_preserves_known_errors_as_rejections(self):
        base=dict(id='example',answer='Final answer: 5',prompt='question',answer_tokens=4,expected_number='5')
        self.assertIsNone(screen(base))
        self.assertIsNotNone(screen(dict(base,answer='Final answer: 6')))
        self.assertIsNotNone(screen(dict(base,id='gsm8k_train_4105')))
        self.assertIsNotNone(screen(dict(base,id='mbpp_test_101')))


if __name__=='__main__':unittest.main()
