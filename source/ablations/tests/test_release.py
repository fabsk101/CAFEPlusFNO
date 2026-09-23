"""Real temporary Git/release checks, without training or touching the source copy.

Temporary commits identify test fixtures only. These tests use the unchanged
parent release writer/verifier; they do not claim the full official builder's
dataset/environment gates ran.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from ablations.repository import (
    PACKAGE_ROOT,
    load_source_compatibility,
    load_source_manifest,
    resolve_repository_root,
)
from ablations.runtime import contained, output_path, workspace_root


_CHECK = r'''
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
os.environ['CAFE_ABLATION_WORKSPACE_ROOT'] = str(root)
os.environ['CAFE_ABLATION_REPOSITORY_ROOT'] = str(root)
from ablations.repository import source_state, resolve_repository_root
try:
    state = source_state(resolve_repository_root(), formal=True)
except Exception as exc:
    reasons = {'SOURCE_GIT_VERIFICATION_FAILED', 'SOURCE_PINNED_UPSTREAM_VERIFICATION_FAILED',
               'SOURCE_RELEASE_VERIFICATION_FAILED', 'SOURCE_RELEASE_MANIFEST_MISSING'}
    reason = str(exc) if str(exc) in reasons else 'SOURCE_CHECK_REJECTED'
    print(json.dumps({'status': 'REJECTED', 'error_type': type(exc).__name__, 'reason_code': reason}))
else:
    print(json.dumps({'status': 'PASS', 'method': state['source_verification_method'],
        'ancestry': state['git_ancestry_verified'],
        'dirty': state['source_repository_dirty'],
        'base_commit': state['base_source_commit'],
        'actual_commit': state['actual_ablation_source_commit'],
        'compatibility_status': state['deployment_source_compatibility']['status'],
        'compatibility_applied': state['deployment_source_compatibility']['applied_paths'],
        'training_source_equivalence_claim': state['deployment_source_compatibility']['training_source_equivalence_claim'],
        'addon_file_count': len(state['ablation_source_files_sha256'])}))
'''


_BUILD_TEST_RELEASE = r'''
import io, json, os, subprocess, sys, zipfile
from pathlib import Path
root, target, temporary = map(Path, sys.argv[1:])
sys.path.insert(0, str(root))
from scripts.build_anonymous_release import add_release_provenance, append_pinned_upstream_archive
def git(*args, cwd=root):
    p = subprocess.run(['git', *args], cwd=cwd, capture_output=True, check=False)
    if p.returncode:
        raise RuntimeError('TEST_GIT_COMMAND_FAILED')
    return p.stdout
archive, upstream = temporary / 'fixture.zip', temporary / 'upstream.zip'
archive.write_bytes(git('archive', '--format=zip', 'HEAD'))
upstream.write_bytes(git('archive', '--format=zip', 'HEAD', cwd=root/'third_party/SirenFNO'))
append_pinned_upstream_archive(archive, upstream)
add_release_provenance(archive,
    source_repository_commit=git('rev-parse', 'HEAD').decode().strip(),
    source_tree_sha1=git('rev-parse', 'HEAD^{tree}').decode().strip())
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(target)
print('TEST_RELEASE_CREATED')
'''


_DRY_RUN = r'''
import os, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
os.environ['CAFE_ABLATION_WORKSPACE_ROOT'] = str(root)
os.environ['CAFE_ABLATION_REPOSITORY_ROOT'] = str(root)
from ablations.run import main
main(['--repository-root', str(root), '--dataset', 'darcy',
      '--model-variant', 'dense', '--condition', 'full', '--seeds', '0',
      '--data-root', str(root / 'data'), '--device', 'cpu', '--dry-run'])
'''


class FormalReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("git") is None:
            raise unittest.SkipTest("NOT RUN: Git executable is unavailable")
        source = resolve_repository_root()
        if not (source / ".git").exists() or not (source / "third_party/SirenFNO/.git").exists():
            raise unittest.SkipTest("NOT RUN: independent source Git objects are required for fixture commits")
        temporary_root = output_path(workspace_root() / "tmp" / "release_regression")
        temporary_root.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(prefix="fixture-", dir=temporary_root)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = contained(cls.temp.name, temporary_root)
        cls.checkout = cls.work / "checkout"
        cls.release = cls.work / "release"
        cls.environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        cls.environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_OPTIONAL_LOCKS="0",
                               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        cls.environment.pop("PYTHONPATH", None)
        cls.environment.pop("PYTHONHOME", None)
        for index, candidate in enumerate((source, source / "third_party/SirenFNO")):
            cls.environment[f"GIT_CONFIG_KEY_{index}"] = "safe.directory"
            cls.environment[f"GIT_CONFIG_VALUE_{index}"] = str(candidate)
        cls.environment["GIT_CONFIG_COUNT"] = "2"
        cls._execute(["git", "clone", "--no-hardlinks", "--quiet", str(source), str(cls.checkout)], cwd=cls.work)
        cls._execute(["git", "clone", "--no-hardlinks", "--quiet", str(source / "third_party/SirenFNO"),
                      str(cls.checkout / "third_party/SirenFNO")], cwd=cls.work)
        public_upstream = json.loads((source / "third_party/UPSTREAM_VERSIONS.json").read_text(encoding="utf-8"))["SirenFNO"]["repository"]
        # The unchanged parent gate checks this public origin string. Changing
        # only this temporary fixture's remote never fetches from the network.
        cls._execute(["git", "-c", f"safe.directory={cls.checkout / 'third_party/SirenFNO'}",
                      "remote", "set-url", "origin", public_upstream], cwd=cls.checkout / "third_party/SirenFNO")
        cls._git("config", "core.autocrlf", "false")
        # Only this freshly created tmp fixture may receive commits.
        # The source may already contain a committed ablations/ directory.
        # Replace ONLY that directory in this new, independent tmp clone with
        # the exact package currently under validation (including local edits).
        owned_root = contained(cls.temp.name, temporary_root)
        checkout = contained(cls.checkout, owned_root)
        if checkout != owned_root / "checkout" or not (checkout / ".git").is_dir():
            raise RuntimeError("TEST_FIXTURE_CHECKOUT_OWNERSHIP_INVALID")
        package = contained(PACKAGE_ROOT, workspace_root())
        if package == checkout or package.is_relative_to(checkout):
            raise RuntimeError("TEST_FIXTURE_MUST_NOT_REPLACE_ACTIVE_PACKAGE")
        target = contained(checkout / "ablations", checkout)
        if target.exists():
            if not target.is_dir():
                raise RuntimeError("TEST_FIXTURE_ADDON_TARGET_NOT_DIRECTORY")
            # Reject indirection before touching the copied tree.
            pending = [target]
            while pending:
                directory = pending.pop()
                for child in directory.iterdir():
                    child = contained(child, checkout)
                    if child.name == ".git":
                        raise RuntimeError("TEST_FIXTURE_ADDON_HAS_GIT_METADATA")
                    if child.is_dir():
                        pending.append(child)
                    elif not child.is_file() or child.stat().st_nlink > 1:
                        raise RuntimeError("TEST_FIXTURE_ADDON_NOT_INDEPENDENT")
            shutil.rmtree(target)
        shutil.copytree(package, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
        compatibility_paths = [
            change["path"] for change in load_source_compatibility()["approved_changes"]
        ]
        for relative in compatibility_paths:
            origin = contained(source / relative, source)
            destination = contained(checkout / relative, checkout)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
        cls._git("add", "--", "ablations", *compatibility_paths)
        # When an already committed package is copied unchanged, the fixture
        # still needs its own child commit for the existing ancestry assertion.
        # This commit is made ONLY inside cls.checkout, never in the source.
        cls._git("-c", "user.name=Regression Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "--allow-empty", "--quiet", "-m", "Test-only integrated ablation fixture")
        # The command environment may list only these controlled temporary roots.
        for index, candidate in enumerate((cls.checkout, cls.checkout / "third_party/SirenFNO")):
            cls.environment[f"GIT_CONFIG_KEY_{index}"] = "safe.directory"
            cls.environment[f"GIT_CONFIG_VALUE_{index}"] = str(candidate)
        cls.environment["GIT_CONFIG_COUNT"] = "2"
        cls._execute([sys.executable, "-I", "-S", "-B", "-c", _BUILD_TEST_RELEASE,
                      str(cls.checkout), str(cls.release), str(cls.work)], cwd=cls.work)

    @classmethod
    def _execute(cls, command, *, cwd):
        completed = subprocess.run(command, cwd=cwd, env=cls.environment,
                                   capture_output=True, text=True, encoding="utf-8", check=False)
        if completed.returncode:
            # A path-bearing traceback/stderr must not leak into public tests.
            raise RuntimeError("TEST_FIXTURE_SUBPROCESS_FAILED")
        return completed.stdout.strip()

    @classmethod
    def _git(cls, *args):
        return cls._execute(["git", "-c", f"safe.directory={cls.checkout}", *args], cwd=cls.checkout)

    def _check(self, root):
        output = self._execute([sys.executable, "-I", "-S", "-B", "-c", _CHECK, str(root)], cwd=self.work)
        return json.loads(output)

    def _dry_run(self, root):
        output = self._execute(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                _DRY_RUN,
                str(root),
            ],
            cwd=root,
        )
        return json.loads(output)

    def test_git_clean_commit_and_actual_addon_included(self):
        actual = self._check(self.checkout)
        self.assertEqual(actual["status"], "PASS", actual)
        self.assertEqual(actual["method"], "git_clean_checkout")
        self.assertIs(actual["ancestry"], True)
        self.assertIs(actual["dirty"], False)
        self.assertEqual(actual["base_commit"], load_source_manifest()["base_source_commit"])
        self.assertNotEqual(actual["base_commit"], actual["actual_commit"])
        self.assertEqual(actual["compatibility_status"], "APPLIED")
        self.assertEqual(
            set(actual["compatibility_applied"]),
            {"scripts/audit_anonymity.py", "tests/test_release.py"},
        )
        self.assertIs(actual["training_source_equivalence_claim"], False)
        self.assertGreater(actual["addon_file_count"], 10)

    def test_git_dirty_addon_rejected(self):
        path = self.checkout / "ablations/__init__.py"
        before = path.read_bytes()
        try:
            path.write_bytes(before + b"\n# test-only mutation\n")
            self.assertEqual(self._check(self.checkout)["status"], "REJECTED")
        finally:
            path.write_bytes(before)

    def test_git_runtime_results_do_not_dirty_sources(self):
        result = self.checkout / "results/test_fixture/summary.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text('{"test_fixture": true}\n', encoding="utf-8")
        self.assertEqual(self._check(self.checkout)["status"], "PASS")

    def test_git_skip_worktree_does_not_hide_changed_addon(self):
        path = self.checkout / "ablations/__init__.py"
        before = path.read_bytes()
        self._git("update-index", "--skip-worktree", "--", "ablations/__init__.py")
        try:
            path.write_bytes(before + b"\n# test-only hidden mutation\n")
            self.assertEqual(self._check(self.checkout)["status"], "REJECTED")
        finally:
            path.write_bytes(before)
            self._git("update-index", "--no-skip-worktree", "--", "ablations/__init__.py")

    def test_release_hashes_include_addon_without_ancestry_claim(self):
        self.assertFalse((self.release / ".git").exists())
        actual = self._check(self.release)
        self.assertEqual(actual["status"], "PASS")
        self.assertEqual(actual["method"], "release_manifest_hashes")
        self.assertIs(actual["ancestry"], False)
        self.assertIs(actual["dirty"], False)
        self.assertEqual(actual["compatibility_status"], "APPLIED")
        self.assertEqual(
            set(actual["compatibility_applied"]),
            {"scripts/audit_anonymity.py", "tests/test_release.py"},
        )
        self.assertIs(actual["training_source_equivalence_claim"], False)

    def test_release_dry_run_verifies_source_without_starting_training(self):
        actual = self._dry_run(self.release)
        self.assertEqual(actual["status"], "DRY_RUN")
        self.assertEqual(actual["requested_seeds"], [0])
        self.assertEqual(actual["process_count"], 1)
        results = self.release / "results"
        self.assertFalse(list(results.rglob("training_log.csv")))
        self.assertFalse(list(results.rglob("final_checkpoint.pt")))
        self.assertFalse(list(results.rglob("summary.json")))

    def test_release_addon_tampering_rejected(self):
        path = self.release / "ablations/__init__.py"
        before = path.read_bytes()
        try:
            path.write_bytes(before + b"\n# test-only mutation\n")
            self.assertEqual(self._check(self.release)["status"], "REJECTED")
        finally:
            path.write_bytes(before)

    def test_release_base_model_and_config_tampering_rejected(self):
        for relative in (
            "models/Cafe_Plus_FNO1D.py",
            "experiments/configs/seeds.py",
        ):
            with self.subTest(relative=relative):
                path = self.release / relative
                before = path.read_bytes()
                try:
                    path.write_bytes(before + b"\n# test-only mutation\n")
                    self.assertEqual(self._check(self.release)["status"], "REJECTED")
                finally:
                    path.write_bytes(before)

    def test_release_unapproved_audit_change_is_rejected(self):
        path = self.release / "scripts/audit_anonymity.py"
        before = path.read_bytes()
        try:
            path.write_bytes(before + b"\n# test-only mutation\n")
            self.assertEqual(self._check(self.release)["status"], "REJECTED")
        finally:
            path.write_bytes(before)

    def test_release_manifest_missing_or_addon_uncovered_rejected(self):
        path = self.release / "RELEASE_PROVENANCE.json"
        before = path.read_bytes()
        try:
            manifest = json.loads(before)
            del manifest["files"]["ablations/__init__.py"]
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(self._check(self.release)["status"], "REJECTED")
            path.unlink()
            self.assertEqual(self._check(self.release)["status"], "REJECTED")
        finally:
            path.write_bytes(before)

    def test_release_runtime_results_do_not_dirty_sources(self):
        result = self.release / "results/test_fixture/summary.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text('{"test_fixture": true}\n', encoding="utf-8")
        self.assertEqual(self._check(self.release)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
