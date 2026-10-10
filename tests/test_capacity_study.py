import json
import unittest
import tempfile
from pathlib import Path
from asi.experiments.capacity_study import variants, freeze_recipes


class CapacityStudyTest(unittest.TestCase):
    def test_frozen_recipes_have_real_unrestricted_controls(self):
        source=json.loads(Path('configs/experiment_124m_v1/domain_16_uniform_pool.json').read_text())
        manifest=Path('configs/experiment_124m_v1/expert_pools.json')
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'recipes'
            rows=freeze_recipes(source,variants(source),manifest,out)
            self.assertEqual(len(rows),10)
            self.assertEqual((out/'expert_pools.json').read_bytes(),manifest.read_bytes())
            for row in rows:
                recipe=json.loads((out/(row['name']+'.json')).read_text())
                self.assertEqual(recipe['parameter_budget'],recipe['parameter_counts']['total'])
                self.assertEqual(recipe['family'],'base' if recipe['model']['routing_mode']=='learned' else 'domain')
            self.assertTrue(all(line.startswith('#') for line in (out/'commands.sh').read_text().splitlines() if 'python -m' in line))

    def test_matched_granularity_and_shared_capacity(self):
        recipe=json.loads(Path('configs/experiment_124m_v1/domain_16_uniform_pool.json').read_text())
        rows=variants(recipe)
        self.assertEqual(len(rows),12)
        self.assertEqual(len({r['shared_per_moe_layer'] for r in rows}),1)
        self.assertEqual(len({r['routed_active_per_moe_layer'] for r in rows if r['axis']=='granularity'}),1)
        base=next(r for r in rows if r['name']=='capacity_h120_k2_uniform_pool')
        self.assertEqual(base['total'],recipe['parameter_counts']['total'])
        self.assertEqual(base['parameters_per_expert'],184320)


if __name__=='__main__':unittest.main()
