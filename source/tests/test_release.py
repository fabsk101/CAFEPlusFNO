"""Role-based tests consolidated from 1 legacy modules."""

from __future__ import annotations

# Migrated from tests/test_release_engineering.py
import json

import os

import re

import subprocess

import tempfile

import unittest

import zipfile

from pathlib import Path

from unittest.mock import patch

from experiments.common import sirenfno_backend

from experiments.common.sirenfno_backend import BackendVerificationError, git_tree_sha1, source_repository_state, verify_git_checkout_cleanliness, verify_git_repository_state, verify_pinned_source_blobs, verify_release_provenance, verify_sirenfno_checkout

from scripts.build_anonymous_release import RELEASE_PROVENANCE_FILENAME, add_release_provenance, append_pinned_upstream_archive, verify_archive, verify_archive_anonymity, verify_extracted_release

from scripts.audit_anonymity import archive_anonymity_findings, content_anonymity_findings, git_history_anonymity_findings, private_denylists_from_environment, release_candidate_files, release_content_anonymity_findings

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_COMMIT = '81918ecce323a2fd5c5a54db917598bda088574b'

EXPECTED_NEURALOP_TREE = 'fbc6aa738d6ffc2e2ca6808e724b0083c3a688eb'

EXPECTED_SIRENFNO_TREE = '57e60feff96fa38a48227dd5217b6a99fe0e8402'

EXPECTED_BLOBS = {'SirenFNO2D.py': 'b0a987e42cd00547dfa64d0e4dfa619ee7106730', 'baseline/AMFNO.py': '70b2affe7237e0123ef0baa0bd970b62ec87a020', 'baseline/UFNO.py': '594dfbc7147ffe737930eaa12e12330c085448d3'}

def run_git(root: Path, *args: str) -> str:
    completed = subprocess.run(['git', *args], cwd=root, check=True, capture_output=True, text=True)
    return completed.stdout.strip()

def run_git_with_environment(root: Path, environment: dict[str, str], *args: str) -> str:
    merged = os.environ.copy()
    merged.update(environment)
    completed = subprocess.run(
        ['git', *args],
        cwd=root,
        env=merged,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()

def initialize_repository(root: Path) -> None:
    run_git(root, 'init', '--initial-branch=main')
    run_git(root, 'config', 'user.name', 'Anonymous Author')
    run_git(root, 'config', 'user.email', 'anonymous@users.noreply.github.com')
    run_git(root, 'config', 'core.autocrlf', 'false')

def commit_all(root: Path, message: str='fixture') -> None:
    run_git(root, 'add', '-A')
    run_git(root, 'commit', '-m', message)

def write_fixture_archive(
    path: Path,
    *,
    include_git_metadata: bool = False,
    extra_members: dict[str, str | bytes] | None = None,
) -> None:
    members: dict[str, str | bytes] = {
        '.gitmodules': '[submodule "third_party/SirenFNO"]\n\tpath = third_party/SirenFNO\n\turl = https://github.com/pengqingshi/SirenFNO.git\n',
        'LICENSE': 'Fixture license.\n',
        'README.md': 'Anonymous release fixture.\n',
        'THIRD_PARTY_NOTICES.md': 'SirenFNO is MIT licensed.\n',
        'models/example.py': 'VALUE = 1\n',
        'scripts/setup_sirenfno.py': '# fixture\n',
        'third_party/UPSTREAM_VERSIONS.json': (REPOSITORY_ROOT / 'third_party/UPSTREAM_VERSIONS.json').read_bytes(),
    }
    for relative in (
        'experiments/__init__.py',
        'experiments/common/__init__.py',
        'experiments/common/sirenfno_backend.py',
        'experiments/configs/__init__.py',
        'experiments/configs/airfoil.py',
        'experiments/configs/burgers1d.py',
        'experiments/configs/cfd1d.py',
        'experiments/configs/cfd2d.py',
        'experiments/configs/darcy.py',
        'experiments/configs/ns2d.py',
        'experiments/configs/reacdiff1d.py',
    ):
        members[relative] = (REPOSITORY_ROOT / relative).read_bytes()
    if include_git_metadata:
        members['nested/submodule/.git'] = 'gitdir: C:' + '\\Users\\private-user\\worktree\n'
    if extra_members:
        members.update(extra_members)
    with zipfile.ZipFile(path, mode='w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, payload in members.items():
            bundle.writestr(name, payload)
    upstream_archive = path.with_name('fixture-upstream.zip')
    subprocess.run(
        ['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip', f'--output={upstream_archive}', EXPECTED_COMMIT],
        cwd=REPOSITORY_ROOT / 'third_party/SirenFNO',
        check=True,
        capture_output=True,
    )
    try:
        append_pinned_upstream_archive(path, upstream_archive)
    finally:
        upstream_archive.unlink()
    add_release_provenance(path, source_repository_commit='a' * 40, source_tree_sha1='b' * 40)

class DatasetArtifactReleaseTests(unittest.TestCase):

    def test_hdf5_dataset_binaries_are_not_tracked(self) -> None:
        completed = subprocess.run(
            ['git', 'ls-files', '--', '*.hdf5'],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), '')

class AnonymityScannerRegressionTests(unittest.TestCase):

    REVIEWED_SYNTHETIC_FIXTURES = (
        'ablations/tests/test_references.py',
        'ablations/tests/test_runtime_privacy.py',
        'ablations/tests/test_validation_accounting.py',
    )

    @staticmethod
    def empty_private_denylists() -> dict[str, set[str]]:
        return {key: set() for key in ('names', 'emails', 'affiliations', 'githubs')}

    @staticmethod
    def fixture_text(relative: str) -> str:
        return (REPOSITORY_ROOT / relative).read_bytes().decode('utf-8')

    def test_generated_review_zip_is_not_a_source_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initialize_repository(root)
            (root / 'source.py').write_text('VALUE = 1\n', encoding='utf-8')
            commit_all(root)
            (root / 'CAFEPlusFNO_full_review.zip').write_bytes(b'generated review')
            relative = {
                path.relative_to(root).as_posix()
                for path in release_candidate_files(root)
            }
            self.assertEqual(relative, {'source.py'})

    @staticmethod
    def scan(text: str, private: dict[str, set[str]] | None=None):
        empty = {key: set() for key in ('names', 'emails', 'affiliations', 'githubs')}
        return content_anonymity_findings([('fixture.txt', text)], private_denylists=empty if private is None else private, hostname='fixture-machine-id')

    def test_generic_os_account_words_are_not_personal_names(self) -> None:
        for text in ('repository root', 'repository runner is not used', 'ubuntu build user and admin documentation'):
            with self.subTest(text=text):
                findings = self.scan(text)
                self.assertFalse(any(findings.values()))

    def test_account_paths_are_detected_by_structure(self) -> None:
        paths = ('/home/' + 'root/private_project/', 'C:' + '\\Users\\alice\\Desktop\\project', '/mnt/c/' + 'Users/alice/project/')
        for text in paths:
            with self.subTest(style=text.split('/')[0]):
                findings = self.scan(text)
                self.assertEqual(findings['local absolute paths'], {'fixture.txt'})

    def test_reviewed_synthetic_paths_are_release_only_exemptions(self) -> None:
        for relative in self.REVIEWED_SYNTHETIC_FIXTURES:
            with self.subTest(relative=relative):
                text = self.fixture_text(relative)
                low_level = content_anonymity_findings(
                    [(relative, text)],
                    private_denylists=self.empty_private_denylists(),
                    hostname='fixture-machine-id',
                )
                self.assertEqual(low_level['local absolute paths'], {relative})
                applied: set[str] = set()
                release = release_content_anonymity_findings(
                    [(relative, text)],
                    private_denylists=self.empty_private_denylists(),
                    hostname='fixture-machine-id',
                    applied_exemptions=applied,
                )
                self.assertFalse(release['local absolute paths'])
                self.assertEqual(applied, {relative})

    def test_reviewed_content_in_readme_or_code_is_still_detected(self) -> None:
        text = self.fixture_text(self.REVIEWED_SYNTHETIC_FIXTURES[0])
        for relative in ('README.md', 'runner.py'):
            with self.subTest(relative=relative):
                findings = release_content_anonymity_findings(
                    [(relative, text)],
                    private_denylists=self.empty_private_denylists(),
                    hostname='fixture-machine-id',
                )
                self.assertEqual(findings['local absolute paths'], {relative})

    def test_unreviewed_path_added_to_reviewed_fixture_is_detected(self) -> None:
        relative = self.REVIEWED_SYNTHETIC_FIXTURES[0]
        text = self.fixture_text(relative)
        changed = text + text.replace('SyntheticOwner', 'UnreviewedIdentity', 1)
        findings = release_content_anonymity_findings(
            [(relative, changed)],
            private_denylists=self.empty_private_denylists(),
            hostname='fixture-machine-id',
        )
        self.assertEqual(findings['local absolute paths'], {relative})

    def test_other_identity_categories_remain_active_for_exempt_fixture(self) -> None:
        relative = self.REVIEWED_SYNTHETIC_FIXTURES[0]
        private = self.empty_private_denylists()
        private['names'] = {'synthetic_private_key'}
        private['emails'] = {'synthetic_secret_value'}
        private['affiliations'] = {'summary.model_hyperparameters'}
        private['githubs'] = {'syntheticowner'}
        applied: set[str] = set()
        findings = release_content_anonymity_findings(
            [(relative, self.fixture_text(relative))],
            private_denylists=private,
            hostname='fixture-machine-id',
            applied_exemptions=applied,
        )
        self.assertFalse(findings['local absolute paths'])
        self.assertEqual(findings['tracked personal name'], {relative})
        self.assertEqual(findings['tracked personal email'], {relative})
        self.assertEqual(findings['institutional affiliation'], {relative})
        self.assertEqual(findings['personal GitHub references'], {relative})
        self.assertEqual(applied, {relative})

    def test_reviewed_fixture_policy_is_identical_for_zip_contents(self) -> None:
        relative = self.REVIEWED_SYNTHETIC_FIXTURES[0]
        original = (REPOSITORY_ROOT / relative).read_bytes()
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'reviewed.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr(relative, original)
            applied: set[str] = set()
            findings = archive_anonymity_findings(
                archive,
                applied_exemptions=applied,
            )
            self.assertFalse(findings['local absolute paths'])
            self.assertEqual(applied, {relative})
            verify_archive_anonymity(archive)

            changed_archive = Path(temp) / 'changed.zip'
            changed = original.replace(b'SyntheticOwner', b'UnreviewedIdentity', 1)
            with zipfile.ZipFile(changed_archive, 'w') as bundle:
                bundle.writestr(relative, changed)
            changed_findings = archive_anonymity_findings(changed_archive)
            self.assertEqual(changed_findings['local absolute paths'], {relative})
            with self.assertRaises(RuntimeError):
                verify_archive_anonymity(changed_archive)

    def test_exact_reviewed_history_blobs_are_the_only_history_exemptions(self) -> None:
        applied: set[str] = set()
        findings = git_history_anonymity_findings(
            REPOSITORY_ROOT,
            'HEAD',
            applied_exemptions=applied,
        )
        self.assertFalse(findings['Git reachable history contents'])
        self.assertEqual(len(applied), 4)

    def test_explicit_private_name_and_email_denylists_are_enforced(self) -> None:
        private_name = 'private ' + 'researcher'
        private_email = 'private' + '@example.invalid'
        private = {'names': {private_name}, 'emails': {private_email}, 'affiliations': set(), 'githubs': set()}
        findings = self.scan(f'Contact {private_name.title()} at {private_email}.', private=private)
        self.assertEqual(findings['tracked personal name'], {'fixture.txt'})
        self.assertEqual(findings['tracked personal email'], {'fixture.txt'})

    def test_private_denylists_are_json_environment_arrays(self) -> None:
        values = {'ANONYMITY_PRIVATE_NAMES': json.dumps(['private ' + 'researcher']), 'ANONYMITY_PRIVATE_EMAILS': json.dumps(['private' + '@example.invalid']), 'ANONYMITY_PRIVATE_AFFILIATIONS': json.dumps(['private ' + 'institute']), 'ANONYMITY_PRIVATE_GITHUBS': json.dumps(['private-' + 'account'])}
        with patch.dict(os.environ, values, clear=True):
            parsed = private_denylists_from_environment()
        self.assertEqual(parsed['names'], {'private researcher'})
        self.assertEqual(parsed['emails'], {'private' + '@example.invalid'})
        self.assertEqual(parsed['affiliations'], {'private institute'})
        self.assertEqual(parsed['githubs'], {'private-account'})

    def test_official_sirenfno_attribution_is_allowed(self) -> None:
        findings = self.scan('Pengqing Shi; https://github.com/pengqingshi/SirenFNO.git')
        self.assertFalse(any(findings.values()))

    def test_archive_error_redacts_private_content_and_member_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            private_fragment = 'private-' + 'researcher'
            member = f'notes/{private_fragment}.txt'
            text = 'C:' + f'\\Users\\{private_fragment}\\project\\result.txt'
            archive = Path(temp) / 'release.zip'
            write_fixture_archive(archive, extra_members={member: text})
            with self.assertRaises(RuntimeError) as raised:
                verify_archive_anonymity(archive)
            rendered = str(raised.exception)
            self.assertNotIn(private_fragment, rendered)
            self.assertIn('location:', rendered)

    def test_binary_member_is_not_silently_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'release.zip'
            write_fixture_archive(archive, extra_members={'opaque.bin': b'\x00\xff'})
            findings = archive_anonymity_findings(archive)
            self.assertEqual(findings['archive uninspectable files'], {'opaque.bin'})
            with self.assertRaises(RuntimeError):
                verify_archive_anonymity(archive)

    def test_cache_member_is_rejected_independent_of_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'release.zip'
            write_fixture_archive(
                archive,
                extra_members={'package/__pycache__/apparently_source.py': 'VALUE = 1\n'},
            )
            with self.assertRaises(RuntimeError):
                verify_archive(archive)


class GitHistoryAnonymityTests(unittest.TestCase):

    def test_committer_is_checked_independently_of_author(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initialize_repository(root)
            (root / 'source.py').write_text('VALUE = 1\n', encoding='utf-8')
            run_git(root, 'add', 'source.py')
            private_email = 'private-committer' + '@example.invalid'
            run_git_with_environment(
                root,
                {
                    'GIT_AUTHOR_NAME': 'Anonymous Author',
                    'GIT_AUTHOR_EMAIL': 'anonymous@users.noreply.github.com',
                    'GIT_COMMITTER_NAME': 'Private Committer',
                    'GIT_COMMITTER_EMAIL': private_email,
                },
                'commit',
                '-m',
                'fixture',
            )
            findings = git_history_anonymity_findings(root)
            self.assertFalse(findings['Git commit author metadata'])
            self.assertTrue(findings['Git commit committer metadata'])

    def test_annotated_tag_tagger_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initialize_repository(root)
            (root / 'source.py').write_text('VALUE = 1\n', encoding='utf-8')
            commit_all(root)
            private_email = 'private-tagger' + '@example.invalid'
            run_git(
                root,
                '-c',
                'user.name=Private Tagger',
                '-c',
                f'user.email={private_email}',
                'tag',
                '-a',
                'review-tag',
                '-m',
                'fixture tag',
            )
            findings = git_history_anonymity_findings(root)
            self.assertTrue(findings['Git annotated tag metadata'])

    def test_deleted_sensitive_content_remains_a_history_finding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initialize_repository(root)
            sensitive = 'C:' + '\\Users\\historical-user\\private\\result.txt'
            source = root / 'old.txt'
            source.write_text(sensitive, encoding='utf-8')
            commit_all(root, 'add old fixture')
            source.unlink()
            commit_all(root, 'remove old fixture')
            findings = git_history_anonymity_findings(root)
            self.assertTrue(findings['Git reachable history contents'])

class NSRenameTests(unittest.TestCase):

    def test_no_stale_ns_config_reference_remains(self) -> None:
        old_module = 'experiments.configs.' + 'ns'
        module_pattern = re.compile(re.escape(old_module) + '\\b')
        aggregate_pattern = re.compile('from\\s+experiments\\.configs\\s+import[^\\n]*\\bns\\b')
        path_reference = 'experiments/configs/' + 'ns.py'
        findings: list[str] = []
        for path in REPOSITORY_ROOT.rglob('*.py'):
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            if relative.startswith(('third_party/', 'dist/')):
                continue
            text = path.read_text(encoding='utf-8')
            if module_pattern.search(text) or aggregate_pattern.search(text) or path_reference in text:
                findings.append(relative)
        self.assertEqual(findings, [])
        old_config_path = 'experiments/configs/' + 'ns.py'
        self.assertFalse((REPOSITORY_ROOT / old_config_path).exists())

    def test_ns2d_configuration_values_remain_fixed(self) -> None:
        from experiments.configs import ns2d
        self.assertEqual(ns2d.FNO_CONFIG['n_modes'], (32, 32))
        self.assertEqual(ns2d.FNO_CONFIG['hidden_channels'], 32)
        self.assertEqual((ns2d.N_TRAIN, ns2d.N_TEST), (1000, 200))
        self.assertEqual((ns2d.BATCH_SIZE, ns2d.EPOCHS), (32, 500))
        self.assertTrue(all((ns2d.CAFEPLUSFNO_FACTORIZATIONS[name]['rank'] == 16 for name in ('cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'))))
        self.assertFalse(ns2d.CAFEPLUSFNO_COMMON_CONFIG['learnable_sigma'])

class SirenFNOCrossPlatformVerificationTests(unittest.TestCase):

    def test_pinned_commit_tree_and_canonical_blobs(self) -> None:
        manifest = json.loads((REPOSITORY_ROOT / 'third_party/UPSTREAM_VERSIONS.json').read_text(encoding='utf-8'))['SirenFNO']
        self.assertEqual(manifest['commit'], EXPECTED_COMMIT)
        self.assertEqual(manifest['git_tree_sha1'], EXPECTED_SIRENFNO_TREE)
        self.assertEqual(manifest['bundled_neuraloperator']['git_tree_sha1'], EXPECTED_NEURALOP_TREE)
        self.assertEqual({name: identity['git_blob_sha1'] for name, identity in manifest['source_files'].items()}, EXPECTED_BLOBS)
        self.assertTrue(all(('sha256' not in identity for identity in manifest['source_files'].values())))
        verified = verify_sirenfno_checkout()
        self.assertEqual(verified['commit'], EXPECTED_COMMIT)
        self.assertEqual(verified['neuraloperator_git_tree_sha1'], EXPECTED_NEURALOP_TREE)

    def test_eol_only_change_passes_but_content_change_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initialize_repository(root)
            source = root / 'source.py'
            source.write_bytes(b'answer = 42\n')
            commit_all(root)
            source.write_bytes(b'answer = 42\r\n')
            verify_git_checkout_cleanliness(root)
            source.write_bytes(b'answer = 43\r\n')
            with self.assertRaises(BackendVerificationError):
                verify_git_checkout_cleanliness(root)

class RepositoryStateTests(unittest.TestCase):

    def test_normal_git_checkout_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initialize_repository(root)
            (root / 'source.py').write_text('VALUE = 1\n', encoding='utf-8')
            commit_all(root)
            with patch.object(sirenfno_backend, 'REPOSITORY_ROOT', root):
                clean = source_repository_state(allow_dirty=False)
            self.assertFalse(clean['source_repository_dirty'])
            (root / 'extra.py').write_text('VALUE = 2\n', encoding='utf-8')
            dirty = verify_git_repository_state(root, allow_dirty=True)
            self.assertTrue(dirty['source_repository_dirty'])
            with self.assertRaises(BackendVerificationError):
                verify_git_repository_state(root, allow_dirty=False)

    def test_submodule_worktree_dirtiness_is_ignored_but_pointer_change_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_root = Path(temp)
            upstream = temp_root / 'upstream'
            superproject = temp_root / 'superproject'
            upstream.mkdir()
            superproject.mkdir()
            initialize_repository(upstream)
            (upstream / 'source.py').write_text('VALUE = 1\n', encoding='utf-8')
            commit_all(upstream)
            initialize_repository(superproject)
            run_git(superproject, '-c', 'protocol.file.allow=always', 'submodule', 'add', str(upstream), 'third_party/SirenFNO')
            commit_all(superproject)
            checkout = superproject / 'third_party/SirenFNO'
            (checkout / 'source.py').write_text('VALUE = 2\n', encoding='utf-8')
            state = verify_git_repository_state(superproject, allow_dirty=False)
            self.assertFalse(state['source_repository_dirty'])
            run_git(checkout, 'config', 'user.name', 'Anonymous Author')
            run_git(checkout, 'config', 'user.email', 'anonymous@users.noreply.github.com')
            commit_all(checkout, 'new upstream fixture')
            state = verify_git_repository_state(superproject, allow_dirty=True)
            self.assertTrue(state['source_repository_dirty'])

class GitlessReleaseTests(unittest.TestCase):

    def test_config_blob_provenance_uses_actual_gitless_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'release.zip'
            extracted = root / 'extracted'
            write_fixture_archive(archive)
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extracted)
            upstream_root = extracted / 'third_party/SirenFNO'
            with patch.multiple(
                sirenfno_backend,
                REPOSITORY_ROOT=extracted,
                THIRD_PARTY_ROOT=extracted / 'third_party',
                SIRENFNO_ROOT=upstream_root,
                NEURALOP_ROOT=upstream_root / 'neuralop',
                MANIFEST_PATH=extracted / 'third_party/UPSTREAM_VERSIONS.json',
            ):
                expected = {'SirenFNO2D.py': EXPECTED_BLOBS['SirenFNO2D.py']}
                self.assertEqual(verify_pinned_source_blobs(expected), expected)
                with self.assertRaisesRegex(
                    BackendVerificationError,
                    'canonical Git blob mismatch',
                ):
                    verify_pinned_source_blobs({'SirenFNO2D.py': '0' * 40})

    def test_archive_and_gitless_provenance_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'release.zip'
            write_fixture_archive(archive)
            verify_archive(archive)
            verify_archive_anonymity(archive)
            verify_extracted_release(archive, runtime_imports=False)
            with zipfile.ZipFile(archive) as bundle:
                members = set(bundle.namelist())
                self.assertIn(RELEASE_PROVENANCE_FILENAME, members)
                self.assertFalse(any(('.git' in Path(name).parts for name in members)))
                self.assertIn('third_party/SirenFNO/LICENSE', members)
                self.assertIn('third_party/SirenFNO/neuralop/__init__.py', members)
                extracted = root / 'extracted'
                bundle.extractall(extracted)
            state = verify_release_provenance(extracted, allow_dirty=False)
            self.assertEqual(state['source_repository_commit'], 'a' * 40)
            self.assertFalse(state['source_repository_dirty'])
            self.assertEqual(state['bundled_upstream']['commit'], EXPECTED_COMMIT)
            self.assertEqual(
                git_tree_sha1(extracted / 'third_party/SirenFNO'),
                EXPECTED_SIRENFNO_TREE,
            )
            with patch.object(sirenfno_backend, 'REPOSITORY_ROOT', extracted):
                state = source_repository_state(allow_dirty=False)
            self.assertFalse(state['source_repository_dirty'])
            (extracted / 'data').mkdir()
            (extracted / 'data/runtime.pt').write_bytes(b'runtime data')
            state = verify_release_provenance(extracted, allow_dirty=False)
            self.assertFalse(state['source_repository_dirty'])
            (extracted / 'third_party/SirenFNO/source.py').write_text('# modified upstream\n', encoding='utf-8')
            with self.assertRaises(BackendVerificationError):
                verify_release_provenance(extracted, allow_dirty=False)
            (extracted / 'third_party/SirenFNO/source.py').unlink()
            (extracted / 'README.md').write_text('modified\n', encoding='utf-8')
            with self.assertRaises(BackendVerificationError):
                verify_release_provenance(extracted, allow_dirty=False)
            dirty = verify_release_provenance(extracted, allow_dirty=True)
            self.assertTrue(dirty['source_repository_dirty'])

    def test_archive_rejects_git_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'unsafe.zip'
            write_fixture_archive(archive, include_git_metadata=True)
            with self.assertRaises(RuntimeError) as raised:
                verify_archive(archive)
            self.assertIn('Git metadata present', str(raised.exception))
            self.assertNotIn('private-user', str(raised.exception))

    def test_archive_provenance_hashes_the_bundled_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'release.zip'
            write_fixture_archive(archive)
            with zipfile.ZipFile(archive) as bundle:
                provenance = json.loads(bundle.read(RELEASE_PROVENANCE_FILENAME))
            self.assertEqual(provenance['release_format_version'], 2)
            self.assertEqual(provenance['bundled_upstream']['commit'], EXPECTED_COMMIT)
            self.assertIn('third_party/SirenFNO/LICENSE', provenance['files'])
            self.assertNotIn(RELEASE_PROVENANCE_FILENAME, provenance['files'])

    def test_archive_rejects_tampered_pinned_upstream_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'release.zip'
            rewritten = root / 'rewritten.zip'
            write_fixture_archive(archive)
            target = 'third_party/SirenFNO/SirenFNO2D.py'
            with zipfile.ZipFile(archive) as source, zipfile.ZipFile(
                rewritten, mode='w', compression=zipfile.ZIP_DEFLATED
            ) as destination:
                for info in source.infolist():
                    payload = source.read(info)
                    if info.filename == target:
                        payload += b'\n# tampered\n'
                    destination.writestr(info, payload)
            os.replace(rewritten, archive)
            with self.assertRaises(RuntimeError) as raised:
                verify_archive(archive)
            self.assertIn('bundled SirenFNO Git tree', str(raised.exception))

    def test_gitless_manifest_rejects_unrecorded_source_but_allows_runtime_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'release.zip'
            write_fixture_archive(archive)
            extracted = root / 'extracted'
            with zipfile.ZipFile(archive) as bundle:
                bundle.extractall(extracted)
            (extracted / 'results/paper').mkdir(parents=True)
            (extracted / 'results/paper/metric.json').write_text('{}\n', encoding='utf-8')
            self.assertFalse(
                verify_release_provenance(extracted, allow_dirty=False)[
                    'source_repository_dirty'
                ]
            )
            (extracted / 'scripts/unrecorded.py').write_text('VALUE = 1\n', encoding='utf-8')
            with self.assertRaises(BackendVerificationError):
                verify_release_provenance(extracted, allow_dirty=False)
