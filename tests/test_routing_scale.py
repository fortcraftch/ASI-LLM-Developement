import copy
import json
from pathlib import Path
import tempfile
import unittest

from asi.experiments.routing_scale import ARMS, validate_plan
from asi.experiments.public_moe import sha256


class ScaleProtocolTests(unittest.TestCase):
    def test_exposure_reference_and_source_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = dict(id='new', split='dev', category='math', prompt='Q', answer='A')
            data = {'corpus': {'examples': [row]},
                    'audit': {'records': [dict(id='new', candidate_only=True)]},
                    'reconciliation': {'files': []}, 'mapping': {}}
            plan = dict(purpose='development_scale_pilot', test_frozen=False,
                        arms=list(ARMS), examples=[row], reference_review={'new': 'checked'},
                        protocol=dict(active=8, resident_per_layer=32, new_tokens=2, max_length=32),
                        files_sha256={})
            for key, value in data.items():
                path = root / (key + '.json')
                path.write_text(json.dumps(value), encoding='utf-8')
                plan[key + '_source'] = str(path)
                plan['files_sha256'][str(path)] = sha256(path)
            validate_plan(plan)
            altered = copy.deepcopy(plan)
            altered['examples'][0]['answer'] = 'Different reference'
            with self.assertRaisesRegex(ValueError, 'differs from frozen corpus'):
                validate_plan(altered)
            altered = copy.deepcopy(plan)
            altered['examples'][0]['split'] = 'test'
            with self.assertRaisesRegex(ValueError, 'test examples'):
                validate_plan(altered)
            path = Path(plan['reconciliation_source'])
            path.write_text(json.dumps({'files': [{'id': 'new'}]}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                validate_plan(plan)
            plan['files_sha256'][str(path)] = sha256(path)
            with self.assertRaisesRegex(ValueError, 'historical exclusions'):
                validate_plan(plan)


if __name__ == '__main__':
    unittest.main()
