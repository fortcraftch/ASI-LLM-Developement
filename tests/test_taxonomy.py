import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from asi import ROOT
from asi.taxonomy import Taxonomy, build_profile, resolve, validate_profile
from asi.runtime.classifier import load_head, read_examples
from test_expert_runtime import tiny


def vocabulary():
    return Taxonomy({'schema': 1, 'id': 'test', 'version': '1', 'nodes': [
        {'id': key, 'parent': parent, 'name': key} for key, parent in [
            ('knowledge', None), ('math', 'knowledge'), ('algebra', 'math'),
            ('code', 'knowledge'), ('python', 'code'), ('cooking', 'knowledge'), ('unknown', 'knowledge')]]})


class TaxonomyTests(unittest.TestCase):
    def test_shared_catalog_covers_existing_dataset_without_claiming_new_domains(self):
        taxonomy = Taxonomy.read(ROOT / 'configs/knowledge_v1.json')
        manifest = json.loads((ROOT / 'configs/experiment_124m_v1/expert_pools.json').read_text())
        profile = build_profile(taxonomy, manifest, 'test')
        categories = [c for p in manifest['pools'].values() for c in p['categories']]
        self.assertEqual(len(categories), 137)
        for category in categories:
            self.assertEqual(resolve(taxonomy, profile, {category: .8})['status'], 'ready')
        self.assertEqual(resolve(taxonomy, profile, {'culinary__recipes': .9})['status'], 'unsupported')
        self.assertEqual(profile['status'], 'planned_unbound')

    def test_multilabel_parent_projection_unknown_and_refinement(self):
        taxonomy = vocabulary()
        profile = build_profile(taxonomy, {'pools': {'a': {'broad_domains': ['math']}, 'b': {'broad_domains': ['code']}}}, 'm')
        result = resolve(taxonomy, profile, {'math': .95, 'algebra': .8, 'python': .7})
        self.assertEqual(result['pools'], ['a', 'b'])
        self.assertNotIn('math', result['detected_categories'])
        self.assertEqual(resolve(taxonomy, profile, {'algebra': .9}, max_depth=1)['detected_categories'], {'math': .9})
        self.assertEqual(resolve(taxonomy, profile, {'algebra': .1})['status'], 'uncertain')
        self.assertEqual(resolve(taxonomy, profile, {'unknown': .9})['pools'], [])
        self.assertEqual(resolve(taxonomy, profile, {'algebra': .9, 'cooking': .8})['status'], 'partial')
        fine = build_profile(taxonomy, {'pools': {'a': {'broad_domains': ['algebra']}}}, 'fine')
        self.assertEqual(resolve(taxonomy, fine, {'math': .9})['status'], 'needs_refinement')

    def test_invalid_identity_cycle_and_scores(self):
        taxonomy = vocabulary()
        profile = build_profile(taxonomy, {'pools': {'a': {'broad_domains': ['math']}}}, 'm')
        for scores in ({'not_a_node': .8}, {'math': float('nan')}, {'math': 1.1}):
            with self.assertRaises(ValueError):
                resolve(taxonomy, profile, scores)
        profile['taxonomy'] = {'id': 'wrong'}
        with self.assertRaises(ValueError):
            validate_profile(taxonomy, profile)
        payload = taxonomy.payload
        payload['nodes'][1]['parent'] = 'algebra'
        with self.assertRaisesRegex(ValueError, 'cycle'):
            Taxonomy(payload)

    def test_semantic_head_training_and_identity(self):
        from asi.runtime.classifier import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = tiny()
            torch.save({'config': asdict(model.config), 'model': model.state_dict(), 'pool_names': ['a', 'b']}, root / 'model.pt')
            (root / 'pools.json').write_text(json.dumps({'pools': {'a': {}, 'b': {}}}))
            taxonomy = vocabulary()
            (root / 'taxonomy.json').write_text(json.dumps(taxonomy.payload))
            rows = [{'tokens': [1, 2], 'labels': ['algebra', 'python'], 'split': 'train'},
                    {'tokens': [3, 4], 'labels': ['unknown'], 'split': 'train'},
                    {'tokens': [5, 6], 'labels': ['algebra'], 'split': 'val'}]
            (root / 'examples.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
            examples = read_examples(root / 'examples.jsonl', taxonomy.labels, taxonomy)
            self.assertEqual(examples['train'][0][1][taxonomy.labels.index('math')], 1.)
            args = ['head', '--checkpoint', str(root / 'model.pt'), '--pool-manifest', str(root / 'pools.json'),
                    '--taxonomy', str(root / 'taxonomy.json'), '--examples', str(root / 'examples.jsonl'),
                    '--output', str(root / 'head'), '--steps', '2', '--batch-size', '2']
            with patch('sys.argv', args), contextlib.redirect_stdout(io.StringIO()):
                main()
            head = load_head(root / 'head/head.pt', model, taxonomy.labels, taxonomy_identity=taxonomy.identity)
            self.assertNotIn('cooking', head.trained_labels)
            self.assertIn('math', head.trained_labels)
            with self.assertRaises(ValueError):
                load_head(root / 'head/head.pt', model, taxonomy.labels)
            # An uncertain extra topic must block execution, not be discarded.
            from asi.runtime.adaptive import main as run, fingerprint
            manifest = json.loads((root / 'pools.json').read_text())
            profile = build_profile(taxonomy, {'pools': {'a': {'broad_domains': ['math']}, 'b': {'broad_domains': ['code']}}}, 'test', fingerprint(root / 'model.pt'))
            from asi.taxonomy import digest
            profile['pool_manifest_sha256'] = digest(manifest)
            (root / 'profile.json').write_text(json.dumps(profile))
            saved = torch.load(root / 'head/head.pt', weights_only=False)
            for value in saved['head'].values():
                value.zero_()
            saved['head']['network.3.bias'].fill_(-5)
            for name in ['algebra', 'unknown']:
                saved['head']['network.3.bias'][taxonomy.labels.index(name)] = 5
            torch.save(saved, root / 'head/head.pt')
            args = ['run', '--checkpoint', str(root / 'model.pt'), '--pool-manifest', str(root / 'pools.json'),
                    '--classifier', str(root / 'head/head.pt'), '--taxonomy', str(root / 'taxonomy.json'),
                    '--model-profile', str(root / 'profile.json'), '--prompt', 'test', '--device', 'cpu',
                    '--memory-mib', '100', '--reserve-mib', '0', '--execute', '--output', str(root / 'run.json')]
            with patch('sys.argv', args), patch('asi.runtime.classifier.features', return_value=torch.zeros(1, 16)), contextlib.redirect_stdout(io.StringIO()):
                run()
            report = json.loads((root / 'run.json').read_text())
            self.assertEqual(report['semantic_routing']['status'], 'partial')
            self.assertEqual(report['plan']['status'], 'blocked')
            self.assertFalse(report['executed'])
