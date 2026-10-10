import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from asi.experiments.teacher_audit import validate_shard
from asi.experiments.recovery_protocol import durable_save


class TeacherIntegrityTest(unittest.TestCase):
    def test_resume_archives_only_incomplete_teacher_reports(self):
        from asi.experiments.recovery_study import archive_partial_teacher_report
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'teacher_report'; out.mkdir()
            payload={'action':'prepare','completed':False}
            (out/'report.json').write_text(json.dumps(payload))
            archive_partial_teacher_report(out)
            self.assertEqual(json.loads((Path(d)/'teacher_report.attempt0001/report.json').read_text()),payload)
            out.mkdir(); (out/'report.json').write_text(json.dumps({**payload,'completed':True}))
            with self.assertRaises(ValueError): archive_partial_teacher_report(out)
            self.assertTrue(out.exists())

    def test_rejects_corruption_and_wrong_tokens(self):
        expected={'id':'a','tokens':[1,2,3],'prompt_length':1}
        row={**expected,'teacher_logp':torch.zeros(2,5).log_softmax(-1)}
        validate_shard(row,expected,5)
        with self.assertRaises(ValueError):
            validate_shard({**row,'tokens':[1,2,4]},expected,5)
        with self.assertRaises(ValueError):
            validate_shard({**row,'teacher_logp':torch.zeros(2,5)},expected,5)
        with self.assertRaises(ValueError):
            validate_shard({**row,'teacher_logp':torch.full((2,5),float('nan'))},expected,5)

    def test_interrupted_save_preserves_previous_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'manifest.json'
            durable_save(path,{'entries':[1]})
            with patch('asi.experiments.recovery_protocol.os.replace',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    durable_save(path,{'entries':[1,2]})
            self.assertEqual(json.loads(path.read_text()),{'entries':[1]})
            self.assertFalse(path.with_suffix('.json.tmp').exists())
            durable_save(path.with_suffix('.pt'),{'a':torch.tensor([2])},tensor=True)
            self.assertEqual(torch.load(path.with_suffix('.pt'),weights_only=True)['a'].item(),2)


if __name__=='__main__': unittest.main()
