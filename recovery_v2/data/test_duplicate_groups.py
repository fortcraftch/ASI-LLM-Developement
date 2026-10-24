import unittest
from recovery_v2.data.duplicate_groups import components


class DuplicateTests(unittest.TestCase):
    def test_numeric_templates_and_transitive_overlap(self):
        rows = [{'id':'a','row':{'question':'What is 12 plus 3?'}},
                {'id':'b','row':{'question':'What is 15 plus 9?'}},
                {'id':'c','row':{'question':'Unrelated concept'}}]
        self.assertEqual(components(rows), [['a','b'],['c']])
        prefix = ' '.join(chr(97+i) for i in range(13))
        rows = [{'id':str(i),'row':{'question':q}} for i,q in enumerate(
            [prefix, prefix+' tail', 'extra '+prefix+' tail'])]
        self.assertEqual(components(rows), [['0','1','2']])

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError): components([{'id':'x','row':{}}])


if __name__ == '__main__': unittest.main()
