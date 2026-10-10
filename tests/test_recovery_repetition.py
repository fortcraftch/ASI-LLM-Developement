import unittest
from asi.analysis.recovery_repetition import longest_loop, audit


class RepetitionTest(unittest.TestCase):
    def test_repeated_phrase_positions(self):
        result=longest_loop(['start']+['one','hour','later']*7+['end'])
        self.assertEqual((result['start'],result['end'],result['copies'],result['period']),(1,22,7,3))

    def test_single_word_and_no_loop(self):
        self.assertEqual(longest_loop(['which']*25)['copies'],25)
        self.assertEqual(longest_loop([1,2,3,1,2,3])['span'],0)
        self.assertEqual(longest_loop([])['span'],0)

    def test_prefix_lengths_and_legitimate_repetition_not_scored_as_correctness(self):
        result=audit(list(range(64)),'a sentence and another sentence')
        self.assertEqual([r['tokens'] for r in result['prefixes']],[32,64])
        self.assertTrue(all(r['repeated_trigram_fraction']==0 for r in result['prefixes']))
        self.assertNotIn('correct',result)


if __name__=='__main__':
    unittest.main()
