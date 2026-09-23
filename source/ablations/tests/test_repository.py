from __future__ import annotations

import unittest
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

from ablations.repository import (
    PACKAGE_ROOT,
    activate_repository,
    baseline_datasets,
    baseline_profile,
    import_audit,
    load_source_manifest,
    load_source_compatibility,
    require_dataset_baseline,
    resolve_repository_root,
    source_compatibility_path,
    source_manifest_path,
    source_state,
    validate_source_compatibility,
    verify_base_sources,
)
from ablations.contracts import DATASETS, MODEL_VARIANTS
from ablations.public import PublicError
from ablations.runtime import contained, output_path, workspace_root


class RepositoryIntegrationTests(unittest.TestCase):
    def test_parent_repository_and_import_locations(self) -> None:
        root = resolve_repository_root()
        activate_repository(root)
        from models import Cafe_Plus_FNO1D, Cafe_Plus_FNO2D

        audit = import_audit(root, (Cafe_Plus_FNO1D, Cafe_Plus_FNO2D))
        self.assertEqual(audit["models.Cafe_Plus_FNO1D"], "models/Cafe_Plus_FNO1D.py")
        self.assertEqual(audit["models.Cafe_Plus_FNO2D"], "models/Cafe_Plus_FNO2D.py")
        state = source_state(root, formal=False)
        self.assertEqual(state["source_mode"], "diagnostic")
        self.assertEqual(state["ablation_package_integrated"], PACKAGE_ROOT.parent == root)

    def test_single_b_manifest_and_complete_seven_dataset_coverage(self) -> None:
        self.assertEqual(baseline_profile(), "C")
        self.assertEqual(set(baseline_datasets()), set(DATASETS))
        self.assertEqual(len(baseline_datasets()), 7)
        self.assertEqual(source_manifest_path(), PACKAGE_ROOT / "source_manifest_baseline_c.json")
        manifest = load_source_manifest()
        self.assertEqual(manifest["base_source_commit"], "5d1ec68dfe2774b74eec45f402b6f7f028a47b3e")
        self.assertEqual(manifest["sirenfno_commit"], "81918ecce323a2fd5c5a54db917598bda088574b")
        self.assertGreaterEqual(len(manifest["files"]), 189)
        for name in ("models/Cafe_Plus_FNO1D.py", "models/Cafe_Plus_FNO2D.py",
                     "experiments/configs/seeds.py", "experiments/common/evaluation.py"):
            self.assertIn(name, manifest["files"])
        for dataset in DATASETS:
            require_dataset_baseline(dataset)
            self.assertIn("experiments/configs/" + dataset + ".py", manifest["files"])
        historical_b = json.loads((PACKAGE_ROOT / "source_manifest_baseline_b.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["files"]), set(historical_b["files"]))
        expected_changed = {
            "experiments/configs/airfoil.py", "experiments/configs/burgers1d.py",
            "experiments/configs/cfd1d.py", "experiments/configs/cfd2d.py",
            "experiments/configs/darcy.py", "experiments/configs/ns2d.py",
            "experiments/configs/reacdiff1d.py", "scripts/audit_reproducibility.py",
            "tests/test_cafe_compact.py", "tests/test_configs.py",
            "tests/test_import_isolation.py",
        }
        changed = {name for name in manifest["files"]
                   if manifest["files"][name] != historical_b["files"][name]}
        self.assertEqual(changed, expected_changed)
        self.assertEqual(len(manifest["files"]), 189)
        self.assertEqual(manifest["baseline_profile"], "C")
        for name, expected_hash in manifest["historical_manifests_canonical_sha256"].items():
            self.assertEqual(hashlib.sha256((PACKAGE_ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), expected_hash)

    def test_single_baseline_default_and_stale_environment_are_rejected(self) -> None:
        with patch.dict(os.environ):
            os.environ.pop("CAFE_ABLATION_BASELINE_PROFILE", None)
            self.assertEqual(baseline_profile(), "C")
            self.assertEqual(set(baseline_datasets()), set(DATASETS))
        with patch.dict(os.environ, {"CAFE_ABLATION_BASELINE_PROFILE": "C"}):
            self.assertEqual(baseline_profile(), "C")
        for value in ("A", "B", "", "a", "b", "c", "D", "A,B", " A "):
            with self.subTest(value=value), patch.dict(os.environ, {"CAFE_ABLATION_BASELINE_PROFILE": value}):
                with self.assertRaises(PublicError) as caught:
                    load_source_manifest()
                self.assertEqual(caught.exception.reason_code, "BASELINE_PROFILE_INVALID")

    def test_deployment_compatibility_is_exact_atomic_and_not_training_provenance(self) -> None:
        manifest = load_source_manifest()
        record = load_source_compatibility()
        self.assertEqual(
            source_compatibility_path().name,
            "source_compatibility_anonymity_release_v1.json",
        )
        self.assertEqual(record["base_source_commit"], manifest["base_source_commit"])
        self.assertEqual(
            {change["path"] for change in record["approved_changes"]},
            {"scripts/audit_anonymity.py", "tests/test_release.py"},
        )
        for change in record["approved_changes"]:
            self.assertEqual(
                manifest["files"][change["path"]], change["baseline_sha256"]
            )
            self.assertFalse(change["training_or_evaluation_effect"])

        state = source_state(resolve_repository_root(), formal=False)
        compatibility = state["deployment_source_compatibility"]
        self.assertEqual(compatibility["status"], "APPLIED")
        self.assertEqual(
            set(compatibility["applied_paths"]),
            {"scripts/audit_anonymity.py", "tests/test_release.py"},
        )
        self.assertFalse(compatibility["training_source_equivalence_claim"])
        self.assertEqual(state["base_source_commit"], manifest["base_source_commit"])
        self.assertEqual(state["base_source_sha256"], manifest["files"])

        for field, replacement in (
            ("base_source_commit", "0" * 40),
            ("baseline_manifest_sha256", "0" * 64),
        ):
            with self.subTest(field=field):
                changed = copy.deepcopy(record)
                changed[field] = replacement
                with self.assertRaisesRegex(
                    RuntimeError, "SOURCE_COMPATIBILITY_RECORD_INVALID"
                ):
                    validate_source_compatibility(changed, manifest)
        changed = copy.deepcopy(record)
        changed["approved_changes"][0]["path"] = "models/Cafe_Plus_FNO1D.py"
        with self.assertRaisesRegex(
            RuntimeError, "SOURCE_COMPATIBILITY_RECORD_INVALID"
        ):
            validate_source_compatibility(changed, manifest)

    def test_selected_b_source_passes_but_stale_a_environment_is_rejected(self) -> None:
        root = resolve_repository_root()
        actual = verify_base_sources(root)
        self.assertEqual(actual, load_source_manifest()["files"])
        with patch.dict(os.environ, {"CAFE_ABLATION_BASELINE_PROFILE": "A"}):
            with self.assertRaises(PublicError) as caught:
                verify_base_sources(root)
            self.assertEqual(caught.exception.reason_code, "BASELINE_PROFILE_INVALID")
        self.assertEqual(verify_base_sources(root), actual)

    def test_independent_temporary_source_tamper_and_missing_are_rejected(self) -> None:
        source = resolve_repository_root()
        expected = load_source_manifest()["files"]
        temporary_root = output_path(workspace_root() / "tmp" / "baseline_manifest_regression")
        temporary_root.mkdir(parents=True, exist_ok=True)
        # Copy only covered source files. No Git links, datasets, result files,
        # or imports from this deliberately damaged temporary tree are needed.
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            target = contained(directory, temporary_root)
            for relative in expected:
                origin = contained(source / relative, source)
                destination = contained(target / relative, target)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, destination)
            self.assertEqual(verify_base_sources(target), expected)
            historical_a = json.loads((PACKAGE_ROOT / "source_manifest_verified.json").read_text(encoding="utf-8"))
            for dataset in DATASETS:
                with self.subTest(dataset=dataset):
                    relative = "experiments/configs/" + dataset + ".py"
                    changed = contained(target / relative, target)
                    before = changed.read_bytes()
                    changed.write_bytes(before + b"\n# TEST/DIAGNOSTIC tamper fixture\n")
                    with self.assertRaisesRegex(RuntimeError, "Base source verification failed"):
                        verify_base_sources(target)
                    changed.write_bytes(before)
                    self.assertEqual(verify_base_sources(target), expected)
                    changed.unlink()
                    with self.assertRaisesRegex(RuntimeError, "Base source verification failed"):
                        verify_base_sources(target)
                    changed.write_bytes(before)
                    if dataset in {"darcy", "ns2d"}:
                        # Reconstruct only the exact historical dense-entry
                        # spelling; its preserved A hash proves the fixture is
                        # stale A text. No historical A source tree is read.
                        text = before.decode("utf-8").replace("\r\n", "\n")
                        declaration = next(node for node in ast.parse(text).body
                                           if isinstance(node, ast.AnnAssign)
                                           and isinstance(node.target, ast.Name)
                                           and node.target.id == "CAFEPLUSFNO_FACTORIZATIONS")
                        entry = next(value for key, value in zip(declaration.value.keys, declaration.value.values)
                                     if isinstance(key, ast.Constant) and key.value == "cafe_plus_fno")
                        fields = ast.literal_eval(entry)
                        fields.pop("cafe_branch_dim")
                        fields.pop("kernel_hidden_dim")
                        lines = text.splitlines(keepends=True)
                        prefix = lines[entry.lineno - 1].split('"cafe_plus_fno"', 1)[0]
                        lines[entry.lineno - 1:entry.end_lineno] = [prefix + '"cafe_plus_fno": ' + json.dumps(fields) + ',\n']
                        stale = "".join(lines).encode("utf-8")
                        self.assertEqual(hashlib.sha256(stale).hexdigest(), historical_a["files"][relative])
                        changed.write_bytes(stale)
                        with self.assertRaisesRegex(RuntimeError, "Base source verification failed"):
                            verify_base_sources(target)
                        changed.write_bytes(before)
            for relative in ("scripts/audit_anonymity.py", "tests/test_release.py"):
                with self.subTest(unapproved_deployment_mutation=relative):
                    changed = contained(target / relative, target)
                    before = changed.read_bytes()
                    changed.write_bytes(before + b"\n# TEST/DIAGNOSTIC unapproved mutation\n")
                    with self.assertRaisesRegex(
                        RuntimeError, "Base source verification failed"
                    ):
                        verify_base_sources(target)
                    changed.write_bytes(before)
            from ablations import repository as repository_module
            real_digest = repository_module._canonical_source_sha256
            baseline_only = contained(target / "tests/test_release.py", target)

            def mixed_transition_digest(path: Path) -> str:
                if path == baseline_only:
                    return expected["tests/test_release.py"]
                return real_digest(path)

            with patch.object(
                repository_module,
                "_canonical_source_sha256",
                side_effect=mixed_transition_digest,
            ), self.assertRaisesRegex(RuntimeError, "compatibility-partial"):
                verify_base_sources(target)
            self.assertEqual(verify_base_sources(target), expected)
        self.assertEqual(verify_base_sources(source), expected)

    def test_all_seven_dataset_factories_use_actual_b_constructor_configuration(self) -> None:
        from ablations.configs import model_configuration
        for dataset, spec in DATASETS.items():
            module = importlib.import_module(spec.config_module)
            for variant, native_name in MODEL_VARIANTS.items():
                with self.subTest(dataset=dataset, model_variant=variant):
                    self.assertEqual(model_configuration(dataset, variant), module.model_constructor_kwargs(native_name))
        with self.assertRaises((PublicError, KeyError)):
            model_configuration("unknown_dataset", "dense")


if __name__ == "__main__":
    unittest.main()
