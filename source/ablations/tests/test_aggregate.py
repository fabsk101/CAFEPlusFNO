from __future__ import annotations

import copy
import csv
import json
import os
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ablations.aggregate import aggregate, parse_args, _export
from ablations.artifacts import canonical_sha256
from ablations.contracts import DEFAULT_SEEDS
from ablations.repository import sha256_file
from ablations.tests.artifact_fixtures import make_artifact
from ablations.output_bundle import (OWNED_OUTPUT_NAMES, LOCK_NAME, TRANSACTION_NAME,
                                     assert_publication_idle, publish_output_bundle)
from ablations.public import PublicError


class AggregateContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.paths = {}
        self.contract = None

    def make(self, seeds):
        for seed in seeds:
            self.paths[seed], self.contract = make_artifact(self.root, seed=seed, condition="fourier_only")
        return self.paths

    def run_aggregate(self, seeds=None, **kwargs):
        return aggregate(results_root=self.root, dataset="burgers1d", model_variant="dense",
                         condition="fourier_only", seeds=seeds, test_contract=self.contract, **kwargs)

    def mutate(self, seed, change):
        from experiments.common.checkpoints import load_weights_only_checkpoint
        path = self.paths[seed]
        summary = json.loads(path.read_text())
        checkpoint = path.parent / "final_checkpoint.pt"
        payload = load_weights_only_checkpoint(checkpoint).checkpoint
        change(summary)
        change(payload)
        torch.save(payload, checkpoint)
        summary["checkpoint_sha256"] = sha256_file(checkpoint)
        path.write_text(json.dumps(summary), encoding="utf-8")

    def test_default_five_one_three_and_extra_selected_policy(self):
        self.make(DEFAULT_SEEDS)
        report = self.run_aggregate()
        self.assertEqual(report["selected_seeds"], list(DEFAULT_SEEDS))
        self.assertEqual(report["seed_count"], 5)
        self.assertEqual(set(report["metrics"]), {"step_relative_l2", "trajectory_relative_l2"})
        for index, chosen in enumerate(([42], [73, 0, 42])):
            result = self.run_aggregate(chosen, output_dir=self.root / f"selection_{index}")
            self.assertEqual([run["seed"] for run in result["runs"]], chosen)
            self.assertEqual(result["unselected_count"], 5 - len(chosen))
            self.assertEqual(result["missing_seeds"], [])
        one = self.run_aggregate([42], output_dir=self.root / "one_seed")
        self.assertIsNone(one["metrics"]["step_relative_l2"]["sample_std"])
        three = self.run_aggregate([0, 42, 73], output_dir=self.root / "three_seeds")
        import statistics
        values = [0.1 + seed / 10000 for seed in (0, 42, 73)]
        self.assertAlmostEqual(three["metrics"]["step_relative_l2"]["mean"], statistics.mean(values))
        self.assertAlmostEqual(three["metrics"]["step_relative_l2"]["sample_std"], statistics.stdev(values))
        target = self.paths[0].parent.parent
        for name in ("aggregate.json", "aggregate.csv", "seed_values.csv", "table.md", "table.tex"):
            self.assertTrue((target / name).is_file())
            self.assertNotIn(str(self.root), (target / name).read_text(encoding="utf-8"))

    def test_arbitrary_more_than_five_and_order(self):
        chosen = [7, 19, 101, 503, 303, 404, 2**32-1]
        self.make(chosen)
        result = self.run_aggregate(chosen)
        self.assertEqual(result["present_seeds"], chosen)
        self.assertEqual(result["seed_count"], 7)

    def test_three_requires_explicit_selection_default_five_missing(self):
        self.make([0, 42, 73])
        with self.assertRaisesRegex(RuntimeError, "SELECTED_SEEDS_MISSING"):
            self.run_aggregate()
        self.assertEqual(self.run_aggregate([0, 42, 73])["status"], "COMPLETE")
        partial = self.run_aggregate(allow_partial=True, output_dir=self.root / "partial")
        self.assertEqual(partial["missing_seeds"], [108, 202])
        self.assertEqual(partial["expected_seeds"], list(DEFAULT_SEEDS))

    def test_partial_never_allows_incomplete_or_corrupt_selected_run(self):
        self.make([0])
        directory = self.paths[0].parent.parent / "seed_42"
        directory.mkdir()
        with self.assertRaisesRegex(RuntimeError, "INCOMPLETE"):
            self.run_aggregate([0, 42], allow_partial=True)
        self.paths[0].parent.joinpath("training_log.csv").write_text("epoch\n1\n")
        with self.assertRaises(ValueError):
            self.run_aggregate([0], allow_partial=True)

    def test_required_metrics_reject_negative_text_bool_missing_and_nonfinite(self):
        self.make([0])
        summary = self.paths[0].read_bytes()
        checkpoint = self.paths[0].parent / "final_checkpoint.pt"
        checkpoint_bytes = checkpoint.read_bytes()
        for value in (-1, "0.2", True, float("nan"), float("inf"), None):
            with self.subTest(value=repr(value)):
                self.paths[0].write_bytes(summary)
                checkpoint.write_bytes(checkpoint_bytes)
                self.mutate(0, lambda s: s["final_metrics"].update(test_step_relative_l2=value))
                with self.assertRaises(ValueError):
                    self.run_aggregate([0])
        self.paths[0].write_bytes(summary)
        checkpoint.write_bytes(checkpoint_bytes)
        self.mutate(0, lambda s: s.update(final_metrics={"epoch": 1, "epoch_time_seconds": 0.1, "learning_rate": .001}))
        with self.assertRaises(ValueError):
            self.run_aggregate([0])

    def test_real_checkpoint_copy_and_mismatched_config_runtime_rejected(self):
        self.make([0, 42])
        first = self.paths[0].parent / "final_checkpoint.pt"
        second = self.paths[42].parent / "final_checkpoint.pt"
        original = second.read_bytes()
        summary_original = self.paths[42].read_bytes()
        second.write_bytes(first.read_bytes())
        summary = json.loads(self.paths[42].read_text())
        summary["checkpoint_sha256"] = sha256_file(second)
        self.paths[42].write_text(json.dumps(summary))
        with self.assertRaises(ValueError):
            self.run_aggregate([0, 42])
        second.write_bytes(original)
        self.paths[42].write_bytes(summary_original)
        self.mutate(42, lambda s: s["runtime"]["operations"].update(tf32_matmul=not s["runtime"]["operations"]["tf32_matmul"]))
        with self.assertRaisesRegex(RuntimeError, "CONTEXT"):
            self.run_aggregate([0, 42])

    def test_diagnostic_is_never_formal_and_selected_duplicate_rejected(self):
        self.make([0])
        with self.assertRaises(ValueError):
            self.run_aggregate([0, 0])
        with self.assertRaises(RuntimeError):
            aggregate(results_root=self.root, dataset="burgers1d", model_variant="dense",
                      condition="fourier_only", seeds=[0])
        path = self.paths[0]
        duplicate = path.parent / "nested" / "summary.json"
        duplicate.parent.mkdir()
        duplicate.write_bytes(path.read_bytes())
        # A selected run must contain exactly its three owned artifacts; nested
        # duplicate summaries cannot create a second independent observation.
        with self.assertRaises(ValueError):
            self.run_aggregate([0])

    def input_hashes(self):
        return {(seed, name): sha256_file(path.parent / name)
                for seed, path in self.paths.items()
                for name in ("summary.json", "final_checkpoint.pt", "training_log.csv")}

    def test_default_refusal_five_to_three_and_explicit_overwrite_preserve_inputs(self):
        self.make(DEFAULT_SEEDS)
        inputs = self.input_hashes()
        destination = self.root / "export"
        first = self.run_aggregate(output_dir=destination)
        before = bundle_hashes(destination)
        for seeds in (list(DEFAULT_SEEDS), [0, 42, 73]):
            with self.assertRaisesRegex(PublicError, "AGGREGATE_OUTPUT_EXISTS"):
                self.run_aggregate(seeds, output_dir=destination)
            self.assertEqual(bundle_hashes(destination), before)
        note = destination / "unowned-note.txt"
        note.write_bytes(b"keep unknown file")
        nested = destination / "unowned-directory"
        nested.mkdir()
        (nested / "keep.txt").write_bytes(b"keep unknown directory")
        second = self.run_aggregate([0, 42, 73], output_dir=destination, overwrite_aggregate=True)
        self.assertNotEqual(second["aggregate_id"], first["aggregate_id"])
        self.assertEqual(second["seed_count"], 3)
        self.assertTrue(all(before[name] != bundle_hashes(destination)[name] for name in OWNED_OUTPUT_NAMES))
        self.assertEqual(self.input_hashes(), inputs)
        self.assertEqual(note.read_bytes(), b"keep unknown file")
        self.assertEqual((nested / "keep.txt").read_bytes(), b"keep unknown directory")

    def test_separate_output_directory_keeps_five_and_three_bundles(self):
        self.make(DEFAULT_SEEDS)
        self.run_aggregate()
        target = self.paths[0].parent.parent
        before = bundle_hashes(target)
        separate = self.root / "selected_three"
        self.run_aggregate([0, 42, 73], output_dir=separate)
        self.assertEqual(bundle_hashes(target), before)
        self.assertEqual(json.loads((target / "aggregate.json").read_text())["seed_count"], 5)
        self.assertEqual(json.loads((separate / "aggregate.json").read_text())["seed_count"], 3)

    def test_partial_existing_output_and_serialization_failure_preserve_every_output(self):
        self.make([42])
        partial = self.root / "partial_existing"
        partial.mkdir()
        (partial / "table.tex").write_bytes(b"existing partial aggregate")
        with self.assertRaisesRegex(PublicError, "AGGREGATE_OUTPUT_EXISTS"):
            self.run_aggregate([42], output_dir=partial)
        self.assertEqual(list(partial.iterdir()), [partial / "table.tex"])
        self.assertEqual((partial / "table.tex").read_bytes(), b"existing partial aggregate")
        destination = self.root / "valid"
        result = self.run_aggregate([42], output_dir=destination)
        before, inputs = bundle_hashes(destination), self.input_hashes()
        invalid = copy.deepcopy(result)
        invalid["metrics"]["step_relative_l2"]["mean"] = float("nan")
        with self.assertRaises(ValueError):
            _export(invalid, destination, overwrite=True)
        self.assertEqual(bundle_hashes(destination), before)
        self.assertEqual(self.input_hashes(), inputs)
        assert_publication_idle(destination)

    def assert_displayed_state(self, destination, result, *, status, selected, present, missing):
        stored = json.loads((destination / "aggregate.json").read_text(encoding="utf-8"))
        self.assertEqual(stored, result)
        self.assertEqual(result["status"], status)
        self.assertEqual(result["seed_count"], len(present))
        self.assertEqual(result["ddof"], 1)
        self.assertEqual(result["result_label"], "TEST/DIAGNOSTIC")
        expected_fields = {"expected_seeds": selected, "selected_seeds": selected,
                           "present_seeds": present, "missing_seeds": missing}
        for key, value in expected_fields.items():
            self.assertEqual(result[key], value)
        for filename in ("aggregate.csv", "seed_values.csv"):
            with (destination / filename).open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(row["status"], status)
                self.assertEqual(row["result_kind"], "diagnostic_test")
                self.assertEqual(row["result_label"], "TEST/DIAGNOSTIC")
                self.assertEqual(int(row["seed_count"]), len(present))
                self.assertEqual(int(row["ddof"]), 1)
                self.assertEqual(row["aggregate_id"], result["aggregate_id"])
                for key, value in expected_fields.items():
                    self.assertEqual(json.loads(row[key]), value)
                if len(present) == 1:
                    self.assertEqual(row["standard_deviation_status"], "undefined")
                    if filename == "aggregate.csv":
                        self.assertEqual(row["sample_std"], "")
        for filename in ("table.md", "table.tex"):
            text = (destination / filename).read_text(encoding="utf-8")
            visible = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("%"))
            self.assertIn("TEST/DIAGNOSTIC", visible)
            self.assertIn(status, visible)
            self.assertIn(f"actual n={len(present)}", visible)
            self.assertIn("ddof=1", visible)
            self.assertIn(str(selected), visible)
            self.assertIn(str(present), visible)
            self.assertIn(str(missing), visible)
            self.assertIn(result["aggregate_id"], visible)
            if len(present) == 1:
                self.assertIn("undefined", visible)
                self.assertTrue(all(item["sample_std"] is None for item in result["metrics"].values()))
            if filename == "table.tex":
                self.assertIn(r"\par\noindent\textbf{", visible)

    def test_four_selected_partial_cases_in_json_csv_markdown_and_visible_latex(self):
        self.make([0, 42, 73])
        complete_dir = self.root / "complete_three"
        complete = self.run_aggregate([0, 42, 73], output_dir=complete_dir)
        self.assert_displayed_state(complete_dir, complete, status="COMPLETE",
            selected=[0, 42, 73], present=[0, 42, 73], missing=[])
        rejected_dir = self.root / "missing_default_rejected"
        with self.assertRaisesRegex(PublicError, "SELECTED_SEEDS_MISSING"):
            self.run_aggregate(output_dir=rejected_dir)
        self.assertFalse(any((rejected_dir / name).exists() for name in OWNED_OUTPUT_NAMES))
        partial_dir = self.root / "partial_default"
        partial = self.run_aggregate(output_dir=partial_dir, allow_partial=True)
        self.assert_displayed_state(partial_dir, partial, status="PARTIAL",
            selected=list(DEFAULT_SEEDS), present=[0, 42, 73], missing=[108, 202])
        one_dir = self.root / "complete_one"
        one = self.run_aggregate([42], output_dir=one_dir)
        self.assert_displayed_state(one_dir, one, status="COMPLETE", selected=[42], present=[42], missing=[])

    def test_pending_publication_refused_before_consuming_or_overwriting_target(self):
        self.make([42])
        target = self.paths[42].parent.parent
        (target / LOCK_NAME).write_bytes(b"pending synthetic writer")
        with self.assertRaisesRegex(PublicError, "AGGREGATE_PUBLICATION_PENDING"):
            self.run_aggregate([42], overwrite_aggregate=True)

    def test_overwrite_cli_and_latex_escape_are_explicit(self):
        import contextlib
        import io
        argv = ["--results-root", str(self.root), "--dataset", "burgers1d", "--model-variant", "dense",
                "--condition", "fourier_only", "--seeds", "42"]
        self.assertFalse(parse_args(argv).overwrite_aggregate)
        self.assertTrue(parse_args([*argv, "--overwrite-aggregate"]).overwrite_aggregate)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
            parse_args([*argv, "--overwrite"])
        self.assertEqual(rejected.exception.code, 2)
        self.make([42])
        result = self.run_aggregate([42])
        result["condition_label"] = r"w/o Linear branches & Hadamard_product {TEST} 50%"
        destination = self.root / "escaped_label"
        _export(result, destination)
        tex = (destination / "table.tex").read_text(encoding="utf-8")
        self.assertIn(r"branches \& Hadamard\_product \{TEST\} 50\%", tex)


def bundle_hashes(directory):
    return {name: sha256_file(directory / name) for name in OWNED_OUTPUT_NAMES}


class AggregatePublicationTests(unittest.TestCase):
    """Failure injection targets only output I/O, never artifact/source gates."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.old = {name: ("OLD " + name).encode() for name in OWNED_OUTPUT_NAMES}
        self.new = {name: ("NEW " + name).encode() for name in OWNED_OUTPUT_NAMES}

    def test_staging_write_failure_leaves_old_bundle_and_no_marker(self):
        from ablations import output_bundle as output
        publish_output_bundle(self.root, self.old)
        before = bundle_hashes(self.root)
        original = output._write_new
        def fail(path, payload):
            if path.name == "new-seed_values.csv":
                with path.open("xb") as handle:
                    handle.write(payload[:2])
                raise OSError("synthetic partial staging failure")
            return original(path, payload)
        with patch.object(output, "_write_new", fail):
            with self.assertRaisesRegex(PublicError, "AGGREGATE_PUBLICATION_STAGING_FAILED"):
                publish_output_bundle(self.root, self.new, overwrite=True)
        self.assertEqual(bundle_hashes(self.root), before)
        assert_publication_idle(self.root)

    def test_replace_failure_rolls_back_every_owned_file(self):
        from ablations import output_bundle as output
        publish_output_bundle(self.root, self.old)
        before = bundle_hashes(self.root)
        original = output.os.replace
        def fail(source, destination):
            if Path(source).name == "new-seed_values.csv":
                raise PermissionError("synthetic publication failure")
            return original(source, destination)
        with patch.object(output.os, "replace", fail):
            with self.assertRaisesRegex(PublicError, "AGGREGATE_PUBLICATION_REPLACEMENT_FAILED"):
                publish_output_bundle(self.root, self.new, overwrite=True)
        self.assertEqual(bundle_hashes(self.root), before)
        assert_publication_idle(self.root)

    def test_rollback_failure_leaves_journal_and_blocks_every_future_write(self):
        from ablations import output_bundle as output
        publish_output_bundle(self.root, self.old)
        original = output.os.replace
        def fail(source, destination):
            if Path(source).name == "new-seed_values.csv" or Path(source).name.startswith("old-"):
                raise PermissionError("synthetic publication and rollback failure")
            return original(source, destination)
        with patch.object(output.os, "replace", fail):
            with self.assertRaisesRegex(PublicError, "AGGREGATE_PUBLICATION_RECOVERY_REQUIRED"):
                publish_output_bundle(self.root, self.new, overwrite=True)
        self.assertTrue((self.root / LOCK_NAME).is_file())
        journal = json.loads((self.root / TRANSACTION_NAME / "journal.json").read_text())
        self.assertEqual(journal["status"], "RECOVERY_REQUIRED")
        self.assertTrue(journal["rollback_failed_files"])
        before = bundle_hashes(self.root)
        for overwrite in (False, True):
            with self.assertRaisesRegex(PublicError, "AGGREGATE_PUBLICATION_PENDING"):
                publish_output_bundle(self.root, self.new, overwrite=overwrite)
            self.assertEqual(bundle_hashes(self.root), before)

    def test_post_commit_cleanup_failure_keeps_marker_and_complete_new_bundle(self):
        from ablations import output_bundle as output
        publish_output_bundle(self.root, self.old)
        with patch.object(output, "_clean_transaction", side_effect=OSError("synthetic cleanup failure")):
            with self.assertRaisesRegex(PublicError, "AGGREGATE_PUBLICATION_CLEANUP_PENDING"):
                publish_output_bundle(self.root, self.new, overwrite=True)
        self.assertEqual({name: (self.root / name).read_bytes() for name in OWNED_OUTPUT_NAMES}, self.new)
        self.assertEqual(json.loads((self.root / TRANSACTION_NAME / "journal.json").read_text())["status"], "COMMITTED")
        with self.assertRaisesRegex(PublicError, "AGGREGATE_PUBLICATION_PENDING"):
            assert_publication_idle(self.root)

    def test_failed_first_publication_rolls_back_to_no_owned_files(self):
        from ablations import output_bundle as output
        original = output.os.replace
        def fail(source, destination):
            if Path(source).name == "new-seed_values.csv":
                raise OSError("synthetic first-publication failure")
            return original(source, destination)
        with patch.object(output.os, "replace", fail):
            with self.assertRaisesRegex(PublicError, "AGGREGATE_PUBLICATION_REPLACEMENT_FAILED"):
                publish_output_bundle(self.root, self.new)
        self.assertFalse(any((self.root / name).exists() for name in OWNED_OUTPUT_NAMES))
        assert_publication_idle(self.root)

    def test_concurrent_writer_is_excluded_by_real_exclusive_lock(self):
        from ablations import output_bundle as output
        staged, release = threading.Event(), threading.Event()
        original = output._write_new
        def pause(path, payload):
            if path.name == "new-aggregate.json":
                staged.set()
                if not release.wait(10):
                    raise OSError("synthetic lock test timed out")
            return original(path, payload)
        with patch.object(output, "_write_new", pause), ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(publish_output_bundle, self.root, self.old)
            try:
                self.assertTrue(staged.wait(10))
                with self.assertRaisesRegex(PublicError, "AGGREGATE_PUBLICATION_PENDING"):
                    publish_output_bundle(self.root, self.new, overwrite=True)
            finally:
                release.set()
            future.result(timeout=10)
        self.assertEqual({name: (self.root / name).read_bytes() for name in OWNED_OUTPUT_NAMES}, self.old)
        assert_publication_idle(self.root)

    def test_hardlink_and_each_synthetic_link_target_are_rejected_before_read(self):
        external = self.root / "independent_workspace_file.txt"
        external.write_bytes(b"preserve independent file")
        destination = self.root / "hardlink_output"
        destination.mkdir()
        os.link(external, destination / "table.tex")
        with self.assertRaisesRegex(PublicError, "AGGREGATE_OUTPUT_UNSAFE"):
            publish_output_bundle(destination, self.new, overwrite=True)
        self.assertEqual(external.read_bytes(), b"preserve independent file")
        self.assertFalse((destination / "aggregate.json").exists())
        original_lstat, original_read = Path.lstat, Path.read_bytes
        for name in OWNED_OUTPUT_NAMES:
            guarded = self.root / "synthetic_link_output" / name
            def lstat(path, *args, **kwargs):
                if path == guarded:
                    return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
                return original_lstat(path, *args, **kwargs)
            def read(path):
                if path == guarded:
                    self.fail("Linked destination was read before the no-follow guard")
                return original_read(path)
            with self.subTest(name=name), patch.object(Path, "lstat", lstat), patch.object(Path, "read_bytes", read):
                with self.assertRaises(ValueError):
                    publish_output_bundle(guarded.parent, self.new, overwrite=True)

    def test_all_formats_are_staged_with_backups_before_first_replacement(self):
        from ablations import output_bundle as output
        publish_output_bundle(self.root, self.old)
        original = output.os.replace
        observed = []
        def inspect(source, destination):
            if Path(source).name.startswith("new-") and not observed:
                transaction = self.root / TRANSACTION_NAME
                for name in OWNED_OUTPUT_NAMES:
                    self.assertEqual((transaction / f"new-{name}").read_bytes(), self.new[name])
                    self.assertEqual((transaction / f"old-{name}").read_bytes(), self.old[name])
                self.assertEqual({name: (self.root / name).read_bytes() for name in OWNED_OUTPUT_NAMES}, self.old)
                observed.append(True)
            return original(source, destination)
        with patch.object(output.os, "replace", inspect):
            publish_output_bundle(self.root, self.new, overwrite=True)
        self.assertEqual(observed, [True])
