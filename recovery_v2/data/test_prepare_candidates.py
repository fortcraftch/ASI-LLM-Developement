import json
from pathlib import Path
import tempfile
import unittest
from recovery_v2.data.prepare_candidates import texts, grams, normalize, read_documents


class DataTests(unittest.TestCase):
    def test_nested_fields(self):
        self.assertEqual(list(texts({'rows':[{'question':'hello', 'answer':'yes'}]})), ['hello','yes'])

    def test_normalization_and_overlap(self):
        self.assertEqual(normalize('Hello,  WORLD!'), normalize('hello, world!'))
        text = ' '.join(str(i) for i in range(20))
        self.assertTrue(grams(text) & grams('prefix '+text+' suffix'))
        self.assertFalse(grams('one two'))

    def test_jsonl_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'rows.jsonl'
            p.write_text(json.dumps({'prompt':'one'})+'\n'+json.dumps({'question':'two'}), encoding='utf-8')
            self.assertEqual(len(list(read_documents(p))), 2)
            p = Path(tmp)/'rows.csv'
            p.write_text('question,answer\nhello,world\n', encoding='utf-8')
            self.assertEqual(list(texts(list(read_documents(p)))), ['hello','world'])

    def test_malformed_is_not_silently_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'bad.json'
            p.write_bytes(b'\x00\x00')
            with self.assertRaises(ValueError): list(read_documents(p))


if __name__ == '__main__': unittest.main()
