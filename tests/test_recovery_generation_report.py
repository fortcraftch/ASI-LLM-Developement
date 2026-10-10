import copy
import json
import tempfile
import unittest
from pathlib import Path

from asi.analysis.recovery_generation_report import read_reports


class GenerationComparisonTest(unittest.TestCase):
    def reports(self, directory):
        base = dict(completed=True, source_identity='source', corpus_identity='corpus',
                    protocol={'new_tokens': 256}, gpu='3060', torch='test',
                    records=[dict(id='test', prompt='question', input_ids=[1,2], token_ids=[3,4])])
        for variant in ('original','32','16'):
            report = copy.deepcopy(base)
            # Unequal response lengths are expected with EOS stopping.
            if variant == '16':
                report['records'][0]['token_ids'] = [3]
            (directory/variant).mkdir()
            (directory/variant/'report.json').write_text(json.dumps(report),encoding='utf8')

    def test_allows_eos_length_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            self.reports(directory)
            self.assertEqual(len(read_reports(directory)),3)

    def test_rejects_changed_prompt_and_incomplete_run(self):
        for mutation in ('prompt', 'completed', 'protocol'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary)
                self.reports(directory)
                path=directory/'16'/'report.json'
                report=json.loads(path.read_text())
                if mutation=='prompt':
                    report['records'][0]['input_ids']=[99]
                elif mutation=='completed':
                    report['completed']=False
                else:
                    report['protocol']['new_tokens']=32
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    read_reports(directory)


if __name__=='__main__':
    unittest.main()
