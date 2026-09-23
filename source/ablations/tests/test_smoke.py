"""CLI fences only: these tests never execute CUDA or read official files."""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from ablations.smoke import parse_args, _preflight_official_files
from ablations.public import PublicError
from ablations import runtime


class SmokeCommandTests(unittest.TestCase):
    def arguments(self):
        return ['--dataset', 'darcy', '--model-variant', 'dense', '--condition', 'full',
                '--seed', '42', '--report', 'results/smoke.json']

    def test_cuda_requires_explicit_cuda_and_refuses_data_root(self):
        for tail in (['--device', 'cpu'], ['--device', 'cuda', '--data-root', 'data']):
            with self.subTest(tail=tail), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(self.arguments() + ['--kind', 'cuda'] + tail)
        self.assertEqual(parse_args(self.arguments() + ['--kind', 'cuda', '--device', 'cuda']).device, 'cuda')

    def test_official_loader_scope_is_explicit_and_requires_data_root(self):
        for dataset, tail in [('darcy', []), ('ns2d', ['--data-root', 'data'])]:
            args = self.arguments()
            args[1] = dataset
            with self.subTest(dataset=dataset), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(args + ['--kind', 'official-data', '--device', 'cpu'] + tail)
        args = parse_args(self.arguments() + ['--kind', 'official-data', '--device', 'cpu', '--data-root', 'data'])
        self.assertEqual(args.dataset, 'darcy')

    def test_smoke_cache_files_stay_in_release_allowed_results(self):
        """Exercise real cache configuration; no torch or official files."""
        policy, tempdir = runtime.get_runtime_policy(), tempfile.tempdir
        with tempfile.TemporaryDirectory(prefix='smoke-cache-') as directory:
            root = Path(directory)
            try:
                with patch.dict(os.environ, {
                    'CAFE_ABLATION_WORKSPACE_ROOT': str(root),
                    'CAFE_ABLATION_REPOSITORY_ROOT': str(root),
                }):
                    selected = runtime.configure_runtime(cpu=True, profile='diagnostic', threads=1,
                                                         cache_area='results')
                    self.assertEqual(selected, root / 'results' / 'ablations_runtime')
                    for name in ('TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'TORCH_HOME'):
                        destination = Path(os.environ[name]) / 'synthetic-cache.txt'
                        destination.write_text('TEST_CACHE', encoding='utf-8')
                        self.assertTrue(destination.is_relative_to(root / 'results'))
                    self.assertFalse((root / 'tmp').exists())
                    self.assertEqual(runtime.get_runtime_policy()['cache_area'], 'results')
                    self.assertEqual(runtime.get_runtime_policy()['numerical_policy'], 'inherited_without_override')
                    default = runtime.configure_runtime(cpu=True, profile='diagnostic', threads=1)
                    self.assertEqual(default, root / 'tmp' / 'ablations_runtime')
                    self.assertNotIn('cache_area', runtime.get_runtime_policy())
                    formal = runtime.configure_runtime(cpu=True, profile='formal')
                    self.assertEqual(formal, root / 'results' / 'ablations_runtime')
                    for invalid in ('../outside', 'cache', True):
                        with self.assertRaises(ValueError):
                            runtime.configure_runtime(cpu=True, cache_area=invalid)
            finally:
                runtime._RUNTIME_POLICY = policy
                tempfile.tempdir = tempdir

    def test_official_preflight_rejects_shared_hardlink_before_read(self):
        with tempfile.TemporaryDirectory(prefix='smoke-file-') as directory:
            root = Path(directory)
            original, linked = root / 'synthetic.pt', root / 'synthetic_shared.pt'
            original.write_bytes(b'TEST_SYNTHETIC_NOT_A_DATASET')
            _preflight_official_files([original], root)
            os.link(original, linked)
            for path in (original, linked):
                with self.subTest(path=path.name), self.assertRaises(PublicError) as caught:
                    _preflight_official_files([path], root)
                self.assertEqual(caught.exception.reason_code, 'OFFICIAL_SMOKE_SHARED_OR_NONREGULAR_FILE')
