"""Role-based tests consolidated from 7 legacy modules."""

from __future__ import annotations

# Migrated from tests/test_airfoil_import_isolation.py
import os
import subprocess

import sys

import tempfile

import unittest

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def assert_dummy_external_neuralop_is_rejected(
    test_case: unittest.TestCase,
    training_module: str,
) -> None:
    """Preload an importable external package and assert the fail-closed path.

    ``-S`` and a replacement ``PYTHONPATH`` make the fixture independent of an
    installed PyPI neuralop.  All mutation is confined to a child process and a
    temporary directory, so neither ``sys.modules`` nor the parent environment
    can leak between tests.
    """

    with tempfile.TemporaryDirectory(prefix="dummy-external-neuralop-") as temp:
        site_root = Path(temp) / "external-site"
        package = site_root / "neuralop"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            "EXTERNAL_TEST_SENTINEL = True\n", encoding="utf-8"
        )
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment["PYTHONPATH"] = str(site_root)
        environment["PYTHONNOUSERSITE"] = "1"
        probe = (
            "import neuralop; "
            "assert neuralop.EXTERNAL_TEST_SENTINEL is True; "
            "print('DUMMY_EXTERNAL_NEURALOP_PRELOADED', flush=True); "
            f"from experiments import {training_module}"
        )
        completed = subprocess.run(
            [sys.executable, "-S", "-B", "-c", probe],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    test_case.assertNotEqual(completed.returncode, 0)
    test_case.assertIn("DUMMY_EXTERNAL_NEURALOP_PRELOADED", completed.stdout)
    test_case.assertIn(
        "preloaded outside third_party/SirenFNO/neuralop", completed.stderr
    )
    test_case.assertNotIn("ModuleNotFoundError", completed.stderr)

class AirfoilImportIsolationTests(unittest.TestCase):

    def test_airfoil_import_resolves_to_pinned_sources(self) -> None:
        command = "from experiments import train_airfoil as t; assert 'UFNO' not in t.AUTHOR_LOCAL_IMPORT_PATHS; assert t.AUTHOR_LOCAL_IMPORT_PATHS['neuralop'].startswith('third_party/SirenFNO/neuralop/'); assert 'site-packages' not in t.AUTHOR_LOCAL_IMPORT_PATHS['neuralop'].lower()"
        completed = subprocess.run([sys.executable, '-B', '-c', command], cwd=REPOSITORY_ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)

# Migrated from tests/test_burgers_import_isolation.py
import json





class BurgersImportIsolationTests(unittest.TestCase):

    def test_fresh_process_uses_repository_local_implementations(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        probe = '\nimport json\nfrom experiments import train_burgers\nprint(json.dumps(train_burgers.author_local_import_audit(), sort_keys=True))\n'
        completed = subprocess.run([sys.executable, '-B', '-c', probe], cwd=repository_root, check=True, capture_output=True, text=True)
        audited = json.loads(completed.stdout.strip().splitlines()[-1])
        for label, path in audited.items():
            if label == 'CAFEPlusFNO1D':
                self.assertEqual(path, 'models/Cafe_Plus_FNO1D.py')
            else:
                self.assertTrue(path.startswith('third_party/SirenFNO/'))
                self.assertNotIn('site-packages', path.lower())

    def test_preloaded_site_package_is_rejected(self) -> None:
        assert_dummy_external_neuralop_is_rejected(self, "train_burgers")

# Migrated from tests/test_cfd1d_import_isolation.py





class CFD1DImportIsolationTests(unittest.TestCase):

    def test_fresh_process_uses_repository_local_implementations(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        probe = '\nimport json\nfrom experiments import train_cfd1d\nprint(json.dumps(train_cfd1d.author_local_import_audit(), sort_keys=True))\n'
        completed = subprocess.run([sys.executable, '-B', '-c', probe], cwd=repository_root, check=True, capture_output=True, text=True)
        audited = json.loads(completed.stdout.strip().splitlines()[-1])
        for label, path in audited.items():
            if label == 'CAFEPlusFNO1D':
                self.assertEqual(path, 'models/Cafe_Plus_FNO1D.py')
            else:
                self.assertTrue(path.startswith('third_party/SirenFNO/'))
                self.assertNotIn('site-packages', path.lower())

    def test_preloaded_site_package_is_rejected(self) -> None:
        assert_dummy_external_neuralop_is_rejected(self, "train_cfd1d")

# Migrated from tests/test_cfd2d_import_isolation.py





class CFD2DImportIsolationTests(unittest.TestCase):

    def test_fresh_process_uses_repository_local_implementations(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        probe = '\nimport json\nfrom experiments import train_cfd2d\nprint(json.dumps(train_cfd2d.author_local_import_audit(), sort_keys=True))\n'
        completed = subprocess.run([sys.executable, '-B', '-c', probe], cwd=repository_root, check=True, capture_output=True, text=True)
        audited = json.loads(completed.stdout.strip().splitlines()[-1])
        for label, path in audited.items():
            if label == 'CAFEPlusFNO2D':
                self.assertEqual(path, 'models/Cafe_Plus_FNO2D.py')
            else:
                self.assertTrue(path.startswith('third_party/SirenFNO/'))
                self.assertNotIn('site-packages', path.lower())

    def test_preloaded_site_package_is_rejected(self) -> None:
        assert_dummy_external_neuralop_is_rejected(self, "train_cfd2d")

# Migrated from tests/test_darcy_import_isolation.py





class DarcyImportIsolationTests(unittest.TestCase):

    def test_fresh_process_uses_only_repository_local_neuralop(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        probe = '\nimport json\nfrom experiments import train_darcy\nprint(json.dumps(train_darcy.author_local_import_audit(), sort_keys=True))\n'
        completed = subprocess.run([sys.executable, '-B', '-c', probe], cwd=repository_root, check=True, capture_output=True, text=True)
        audited = json.loads(completed.stdout.strip().splitlines()[-1])
        for name in ('neuralop', 'FNO', 'Trainer', 'LpLoss', 'H1Loss', 'load_darcy_flow_small', 'AdamW', 'count_model_params'):
            self.assertTrue(audited[name].startswith('third_party/SirenFNO/neuralop/'))
        for name in ('SirenFNO2d', 'FNO2dMLP', 'UFNO'):
            self.assertTrue(audited[name].startswith('third_party/SirenFNO/'))
        self.assertEqual(audited['CAFEPlusFNO2D'], 'models/Cafe_Plus_FNO2D.py')
        self.assertNotIn('site-packages', audited['neuralop'].lower())

    def test_preloaded_site_package_is_rejected_not_silently_reused(self) -> None:
        assert_dummy_external_neuralop_is_rejected(self, "train_darcy")

# Migrated from tests/test_reacdiff_import_isolation.py





class ReacDiffImportIsolationTests(unittest.TestCase):

    def test_fresh_process_uses_repository_local_implementations(self) -> None:
        root = Path(__file__).resolve().parents[1]
        probe = 'import json; from experiments import train_reacdiff; print(json.dumps(train_reacdiff.author_local_import_audit(), sort_keys=True))'
        completed = subprocess.run([sys.executable, '-B', '-c', probe], cwd=root, check=True, capture_output=True, text=True)
        audited = json.loads(completed.stdout.strip().splitlines()[-1])
        for label, path in audited.items():
            if label == 'CAFEPlusFNO1D':
                self.assertEqual(path, 'models/Cafe_Plus_FNO1D.py')
            else:
                self.assertTrue(path.startswith('third_party/SirenFNO/'))
                self.assertNotIn('site-packages', path.lower())

    def test_preloaded_site_package_is_rejected(self) -> None:
        assert_dummy_external_neuralop_is_rejected(self, "train_reacdiff")

# Migrated from tests/test_imports.py

class LocalModelImportTests(unittest.TestCase):

    def test_cafe_models(self) -> None:
        from models.Cafe_Plus_FNO1D import CAFEPlusFNO1D
        from models.Cafe_Plus_FNO2D import CAFEPlusFNO2D
        self.assertTrue(callable(CAFEPlusFNO1D))
        self.assertTrue(callable(CAFEPlusFNO2D))

    def test_sirenfno_author_models(self) -> None:
        from experiments.common.sirenfno_backend import bootstrap_sirenfno_backend
        bootstrap_sirenfno_backend()
        from baseline.AMFNO import FNO1dMLP, FNO2dMLP
        from baseline.UFNO import UFNO, UFNO1d
        from SirenFNO1D import SirenFNO1d
        from SirenFNO2D import SirenFNO2d
        for model_class in (FNO1dMLP, FNO2dMLP, SirenFNO1d, SirenFNO2d, UFNO, UFNO1d):
            self.assertTrue(callable(model_class))

class NeuralOperatorImportTests(unittest.TestCase):

    def test_pinned_bundled_neuraloperator(self) -> None:
        from experiments import train_darcy
        audited = train_darcy.author_local_import_audit()
        for name in ('neuralop', 'FNO', 'Trainer', 'LpLoss', 'H1Loss', 'load_darcy_flow_small', 'AdamW'):
            self.assertTrue(audited[name].startswith('third_party/SirenFNO/neuralop/'))
        self.assertEqual(str(train_darcy.neuralop.__version__), '1.0.2')
