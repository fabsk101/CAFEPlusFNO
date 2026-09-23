from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import stat
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from ablations.contracts import DEFAULT_SEEDS
from ablations.isolated import audit_site
from ablations.repository import PACKAGE_ROOT, RepositoryImportGuard, baseline_datasets, resolve_repository_root, source_state, _git
from ablations.run import main as run_main, parse_args as parse_run
from ablations.train import _prepare_run_paths, parse_args as parse_train
from ablations.runtime import contained, output_path, workspace_root


class IsolationTests(unittest.TestCase):
    def test_paths_are_workspace_local_and_separate_variants(self):
        root = workspace_root()
        with self.assertRaises(ValueError):
            contained(root.parent / 'outside-test-target', root)
        with self.assertRaises(ValueError):
            output_path(resolve_repository_root() / 'test-output') if resolve_repository_root() != root else contained(root.parent, root)
        with tempfile.TemporaryDirectory() as directory:
            args = dict(dataset='darcy', model_variant='dense', seed=0, diagnostic=True, overwrite=False)
            a = _prepare_run_paths(Path(directory), condition='full', **args)
            b = _prepare_run_paths(Path(directory), condition='fourier_only', **args)
            self.assertNotEqual(a[0], b[0])
            a[2].write_text('{}')
            with self.assertRaises(FileExistsError):
                _prepare_run_paths(Path(directory), condition='full', **args)
        self.assertTrue(Path(tempfile.gettempdir()).is_relative_to(root))
        for key in ('TMP', 'TEMP', 'TMPDIR', 'TORCH_HOME', 'XDG_CACHE_HOME', 'MPLCONFIGDIR'):
            self.assertTrue(Path(os.environ[key]).is_relative_to(root), key)

    def test_external_editable_mapping_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            external = str(workspace_root().parent / 'unopened-external-project' / 'models')
            (site / '__editable__demo_finder.py').write_text(f'MAPPING: dict = {{"models": {external!r}}}\n')
            with self.assertRaisesRegex(RuntimeError, 'editable mapping'):
                audit_site(site, workspace_root())

    def test_run_files_are_link_checked_before_existence_probe(self):
        from ablations.train import run_directory, skip_completed_run
        with tempfile.TemporaryDirectory() as directory:
            args = dict(dataset='darcy', model_variant='dense', condition='full', seed=0, diagnostic=True)
            run_dir = run_directory(Path(directory), **args)
            run_dir.mkdir(parents=True)
            guarded = run_dir / 'summary.json'
            original_lstat, original_exists = Path.lstat, Path.exists
            def lstat(path, *a, **kw):
                if path == guarded:
                    return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
                return original_lstat(path, *a, **kw)
            def exists(path):
                if path == guarded:
                    self.fail('A link target was probed before the no-follow guard')
                return original_exists(path)
            # Synthetic link metadata never points to or probes another path.
            with patch.object(Path, 'lstat', lstat), patch.object(Path, 'exists', exists):
                with self.assertRaises(ValueError):
                    _prepare_run_paths(Path(directory), overwrite=False, **args)
                with self.assertRaises(ValueError):
                    skip_completed_run(run_dir, expected_context={}, overwrite=False, **args)

    def test_import_guard_blocks_external_fallback_and_namespace_paths(self):
        guard = RepositoryImportGuard(resolve_repository_root())
        with self.assertRaises(ImportError):
            guard.find_spec('models.this_module_does_not_exist', [str(resolve_repository_root() / 'models')])
        with self.assertRaises(ValueError):
            guard.find_spec('models.anything', [str(workspace_root().parent / 'unopened-project')])

    def test_cli_five_defaults_arbitrary_seeds_and_duplicate_rejection(self):
        args = ['--dataset', baseline_datasets()[0], '--model-variant', 'dense', '--condition', 'full', '--data-root', str(workspace_root() / 'tmp/fake-data')]
        self.assertEqual(parse_run(args).seeds, [0, 42, 73, 108, 202])
        self.assertEqual(parse_run(args + ['--seeds', '11', '23']).seeds, [11, 23])
        self.assertEqual(parse_train(args + ['--seed', '11']).seed, 11)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_run(args + ['--seeds', '42', '42'])
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            run_main(args + ['--dry-run'])
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan['requested_seeds'], list(DEFAULT_SEEDS))
        self.assertEqual(plan['process_count'], len(DEFAULT_SEEDS))
        self.assertEqual(plan['device'], 'cpu')
        self.assertEqual(plan['baseline_profile'], 'C')
        self.assertEqual(plan['base_source_commit'], '5d1ec68dfe2774b74eec45f402b6f7f028a47b3e')
        self.assertNotIn(str(workspace_root()), stream.getvalue())

    def test_formal_source_rejects_unintegrated_or_uncommitted_package(self):
        root = resolve_repository_root()
        if PACKAGE_ROOT.parent != root or _git(root, 'status', '--porcelain=v1', '--untracked-files=all'):
            with self.assertRaisesRegex(RuntimeError, 'copied directly|clean committed'):
                source_state(root, formal=True)
        else:
            self.assertEqual(source_state(root, formal=True)['source_mode'], 'formal')

    def test_existing_pinned_backend_verifier_on_complete_snapshot(self):
        from experiments.common.sirenfno_backend import verify_sirenfno_checkout
        record = verify_sirenfno_checkout()
        self.assertEqual(record['commit'], '81918ecce323a2fd5c5a54db917598bda088574b')
