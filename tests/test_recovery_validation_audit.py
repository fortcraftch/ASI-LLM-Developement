import unittest

from asi.analysis.recovery_validation_audit import code_diagnostics


class CodeDiagnosticTests(unittest.TestCase):
    def test_prose_does_not_invalidate_python_block(self):
        text='Explanation\n```python\ndef answer():\n    return 42\n```\nMore explanation.'
        result=code_diagnostics(text)
        self.assertTrue(result['first_block_syntax_pass'])
        self.assertEqual(result['complete_python_blocks'],1)

    def test_missing_truncated_empty_and_invalid_blocks(self):
        self.assertIsNone(code_diagnostics('```python\nx = 1')['first_block_syntax_pass'])
        self.assertIsNone(code_diagnostics('x = 1')['first_block_syntax_pass'])
        self.assertFalse(code_diagnostics('```python\n\n```')['first_block_syntax_pass'])
        self.assertFalse(code_diagnostics('```python\ndef broken(\n```')['first_block_syntax_pass'])

    def test_first_block_not_cherry_picked_and_never_executed(self):
        text='```python\nraise RuntimeError("must never run")\n```\n```python\ninvalid (\n```'
        result=code_diagnostics(text)
        self.assertEqual(result['complete_python_blocks'],2)
        self.assertTrue(result['first_block_syntax_pass'])


if __name__=='__main__':unittest.main()
