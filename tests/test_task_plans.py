import json
from pathlib import Path
import unittest
from unittest.mock import patch

from asi import ROOT
from asi.data.classifier_examples import generate, audit
from asi.taxonomy import Taxonomy
from asi.runtime.tasks import propose, schedule, execute
from asi.runtime.cache import ExpertCacheManager
from test_expert_runtime import tiny


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = Taxonomy.read(ROOT / 'configs/knowledge_v1.json')
        self.profile = json.loads((ROOT / 'configs/model_categories_124m_v1.json').read_text())

    def case(self, name):
        return json.loads((ROOT / 'data/classifier_seed_v1/tasks' / (name + '.json')).read_text(encoding='utf-8'))

    def test_seed_coverage_and_split_integrity(self):
        rows = generate(self.taxonomy)
        report = audit(rows, self.taxonomy)
        self.assertEqual(report['split_counts'], {'train': 456, 'val': 36, 'test': 36})
        self.assertEqual(set(report['direct_positive_counts']['train']), set(self.taxonomy.labels))
        duplicate = dict(rows[0], split='val')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            audit(rows + [duplicate], self.taxonomy)

    def test_related_question_requires_joint_residency(self):
        case = self.case('neural_math')
        self.assertEqual(schedule(case, self.taxonomy, self.profile)['status'], 'blocked')
        plan = schedule(case, self.taxonomy, self.profile, max_hot_pools=2)
        self.assertEqual(plan['context_switches'], 0)
        self.assertEqual(len(plan['order']), 1)

    def test_action_count_is_not_domain_count(self):
        plan = schedule(self.case('compare_python'), self.taxonomy, self.profile)
        self.assertEqual(plan['context_switches'], 0)
        self.assertEqual(plan['order'], ['compare', 'implement'])
        self.assertEqual(plan['task_pools']['compare'], ['technology'])

    def test_dependency_switch_budget_and_grouping(self):
        plan = schedule(self.case('derive_then_code'), self.taxonomy, self.profile)
        self.assertEqual(plan['status'], 'ready')
        self.assertEqual(plan['context_switches'], 1)
        plan = schedule(self.case('needs_two_switches'), self.taxonomy, self.profile)
        self.assertEqual(plan['status'], 'blocked')
        self.assertEqual(plan['context_switches'], 2)
        case = self.case('needs_two_switches')
        case['relation'] = 'independent'
        for task in case['tasks']:
            task['depends_on'] = []
        plan = schedule(case, self.taxonomy, self.profile)
        self.assertEqual(plan['context_switches'], 1)

    def test_cycle_and_proposal_conservatism(self):
        case = self.case('derive_then_code')
        case['tasks'][0]['depends_on'] = ['code']
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            schedule(case, self.taxonomy, self.profile)
        proposal = propose('Qué código Python es más eficiente entre A y B y el que lo sea prográmalo')
        self.assertEqual(len(proposal['tasks']), 2)
        self.assertFalse(proposal['reviewed'])
        self.assertEqual(len(propose('Explícame las matemáticas de una red neuronal')['tasks']), 1)
        self.assertEqual(len(propose('Compara estas cadenas: ```python\nx = "y luego implementa"\n```')['tasks']), 1)

    def test_dependency_answers_are_passed_without_extra_synthesis(self):
        case = self.case('derive_then_code')
        plan = schedule(case, self.taxonomy, self.profile)
        model = tiny().eval()
        model.config.block_size = 1024
        cache = ExpertCacheManager('cpu', 1)
        cache.initialize(model)
        class Encoder:
            def encode(self, text):
                return text.split()
        seen = []
        def generate(model, enc, prompt, *args):
            seen.append(prompt)
            return prompt + ('6*x+2' if len(seen) == 1 else 'def derivative(x): return 6*x+2')
        try:
            with patch('asi.runtime.generation.generate', side_effect=generate):
                result = execute(case, plan, model, cache, ['math_physics', 'technology'], Encoder(), 'cpu', 4)
            self.assertEqual(len(seen), 2)
            self.assertIn('6*x+2', seen[1])
            self.assertNotIn('SOLICITUD ORIGINAL', result['answers']['derive'])
            self.assertEqual(cache.stats['misses'], 8)
            self.assertEqual(cache.stats['evictions'], 4)
            self.assertEqual(result['steps'][0]['before'], result['steps'][0]['after'])
            with self.assertRaisesRegex(ValueError, 'silent truncation'):
                model.config.block_size = 2
                execute(case, plan, model, cache, ['math_physics', 'technology'], Encoder(), 'cpu', 4)
        finally:
            cache.offload_all(model)
