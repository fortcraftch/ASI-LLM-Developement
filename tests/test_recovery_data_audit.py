import unittest
from asi.analysis.recovery_data_audit import mentions, inventory, normalize


class ExposureTests(unittest.TestCase):
    def test_revision_keeps_exposure_and_rejects_stale_proposals(self):
        import hashlib
        import json
        from pathlib import Path
        import tempfile
        from asi.data.recovery_corpus import reviewed_draft
        with tempfile.TemporaryDirectory() as tmp:
            c, r, p = [Path(tmp)/name for name in ('c.json','r.json','p.json')]
            row = dict(id='old', split='train', prompt='question', answer='old answer',
                       reference_tests=['assert old() == 1'], expected_number='1',
                       category='math', token_count=20, answer_tokens=10, template_group='question')
            c.write_text(json.dumps({'examples':[row]}))
            review = dict(sources_sha256={str(c):hashlib.sha256(c.read_bytes()).hexdigest()},
                          records=[dict(id='old',status='clarify_or_replace',row_sha256=
                            hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest())])
            r.write_text(json.dumps(review))
            proposal = dict(id='old',old_answer_sha256=hashlib.sha256(b'old answer').hexdigest(),new_answer='new answer')
            p.write_text(json.dumps({'replacements':[proposal]}))
            draft=reviewed_draft(c,r,[p]); changed=draft['examples'][0]
            self.assertFalse(draft['ready_for_training'])
            self.assertEqual(changed['parent_id'],'old')
            self.assertEqual(changed['split'],'train')
            self.assertTrue(changed['train_only'])
            self.assertNotIn('token_count',changed)
            self.assertNotIn('answer_tokens',changed)
            self.assertNotIn('reference_tests',changed)
            self.assertNotIn('expected_number',changed)
            self.assertEqual(changed['parent_reference_tests'],row['reference_tests'])
            self.assertTrue(changed['evaluation_targets_require_review'])
            proposal['new_reference_tests'] = ['assert revised() == 2']
            proposal['new_expected_number'] = '2'
            p.write_text(json.dumps({'replacements':[proposal]}))
            revised=reviewed_draft(c,r,[p])['examples'][0]
            self.assertEqual(revised['reference_tests'],proposal['new_reference_tests'])
            self.assertEqual(revised['expected_number'],'2')
            with self.assertRaisesRegex(ValueError,'multiply replaced'): reviewed_draft(c,r,[p,p])
            proposal['old_answer_sha256']='stale';p.write_text(json.dumps({'replacements':[proposal]}))
            with self.assertRaisesRegex(ValueError,'Old answer hash'): reviewed_draft(c,r,[p])

    def test_missing_teacher_examples_are_not_execution_evidence(self):
        import json
        import tempfile
        from pathlib import Path
        from asi.analysis.recovery_data_audit import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root/'corpus.json'
            corpus.write_text(json.dumps({'examples': [dict(id=i,split='test',category='x',prompt=i)
                                                       for i in ['a','b']]}))
            quarantine = root/'quarantine.json'
            quarantine.write_text('{"excluded":{}}')
            source = root/'results/recovery_old'
            source.mkdir(parents=True)
            (source/'audit.json').write_text(json.dumps({'manifest_valid_json':False,
                    'files':[{'id':'a'}], 'missing_examples':['b']}))
            (source/'manifest.json').write_bytes(b'\x00\x00')
            result = build(corpus,root/'results',root/'output',quarantine)
            self.assertFalse(result['audit_complete'])
            self.assertFalse(result['test_frozen'])
            self.assertEqual([x['id'] for x in result['records'] if x['candidate_only']], ['b'])
            before = (root/'output/report.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'fresh audit'):
                build(corpus, root/'results', root/'output', quarantine)
            self.assertEqual(before, (root/'output/report.json').read_bytes())

    def test_mentions_keys_values_and_prompt(self):
        prompts = {normalize('Some question'): {'c'}}
        self.assertEqual(mentions({'a': ['b', ' SOME  question ']}, {'a','b','c'}, prompts), {'a','b','c'})

    def test_duplicates_and_templates_propagate(self):
        rows = [dict(id='a', split='train', category='x', prompt='Q', template_group='t'),
                dict(id='b', split='dev', category='x', prompt=' q ', template_group='u'),
                dict(id='c', split='test', category='x', prompt='different', template_group='u'),
                dict(id='d', split='test', category='x', prompt='fresh')]
        records, groups = inventory(rows, {'a':['old.json']}, {})
        self.assertEqual([r['id'] for r in records if r['candidate_only']], ['d'])
        self.assertEqual(len(groups), 2)
        self.assertEqual(records[2]['reasons'], ['cross_split_prompt_or_template'])

    def test_unseen_cross_split_duplicates_are_not_clean_candidates(self):
        rows = [dict(id='a', split='train', category='x', prompt='Same question'),
                dict(id='b', split='test', category='x', prompt=' same QUESTION '),
                dict(id='c', split='dev', category='x', prompt='different')]
        records, _ = inventory(rows, {}, {})
        self.assertEqual([r['id'] for r in records if r['candidate_only']], ['c'])
        self.assertIn('cross_split_prompt_or_template', records[1]['reasons'])

    def test_quarantine_and_duplicate_ids(self):
        r=dict(id='a',split='test',category='x',prompt='q')
        records,_=inventory([r], {}, {'a':'bad reference'})
        self.assertFalse(records[0]['candidate_only'])
        with self.assertRaises(ValueError): inventory([r,r],{}, {})
