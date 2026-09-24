import contextlib
import importlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from asi import ROOT
from asi.__main__ import COMMANDS, main
from asi.plan import load_plan


class StructureTests(unittest.TestCase):
    def test_plan_references_existing_files_and_all_steps(self):
        steps = load_plan()
        expected = {'1.1', '1.2', '1.3', '1.4', '1.5', '2.1', '2.2', '2.3', '3.1', '3.2', '3.3'}
        self.assertEqual({s['id'] for s in steps}, expected)
        self.assertEqual(len(steps), len(expected))
        for step in steps:
            self.assertTrue(step['pending'])
            for file in step['files']:
                self.assertTrue((ROOT / file).is_file(), file)

    def test_plan_and_help_do_not_load_ml_dependencies(self):
        script = "from asi.__main__ import main; main(['plan', '2.1']); import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(['--help'])
        self.assertEqual(error.exception.code, 0)

    def test_data_import_does_not_parse_arguments_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            script = f"import sys; sys.path.insert(0, {str(ROOT)!r}); import asi.data.pools; import asi.data.inspect"
            result = subprocess.run([sys.executable, '-c', script, '--invalid-flag'], cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_cli_targets_resolve_and_restore_argv(self):
        def leaves(commands, path=()):
            for name, value in commands.items():
                if isinstance(value, dict):
                    yield from leaves(value, (*path, name))
                else:
                    yield (*path, name), value
        for path, (module, function, _) in leaves(COMMANDS):
            self.assertTrue(callable(getattr(importlib.import_module(module), function)))
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main([*path, '--help'])
            self.assertEqual(error.exception.code, 0, path)
        original = sys.argv
        with contextlib.redirect_stdout(io.StringIO()):
            main(['plan', '1.1'])
        self.assertIs(sys.argv, original)


if __name__ == '__main__':
    unittest.main()
