from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ablations.artifacts import (validate_artifact, validate_csv, canonical_sha256,
                                 build_artifact_contract)
from ablations.checkpoints import restore_model_checkpoint, validate_checkpoint_payload
from ablations.protocols import pinned_contract
from ablations.repository import sha256_file
from ablations.tests.artifact_fixtures import make_artifact, tiny_contract


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path, self.contract = make_artifact(Path(self.directory.name))

    def validate(self):
        return validate_artifact(self.path, dataset="burgers1d", model_variant="dense",
            condition="full", seed=0, allow_diagnostic=True, test_contract=self.contract)

    def mutate(self, *, summary=None, checkpoint=None):
        metadata = json.loads(self.path.read_text())
        pt = self.path.parent / "final_checkpoint.pt"
        if checkpoint:
            from experiments.common.checkpoints import load_weights_only_checkpoint
            payload = load_weights_only_checkpoint(pt).checkpoint
            checkpoint(payload)
            torch.save(payload, pt)
            metadata["checkpoint_sha256"] = sha256_file(pt)
        if summary:
            summary(metadata)
        self.path.write_text(json.dumps(metadata), encoding="utf-8")

    def test_real_checkpoint_csv_validates_without_rng_consumption(self):
        before = torch.get_rng_state().clone()
        result = self.validate()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(set(result.metrics), {"step_relative_l2", "trajectory_relative_l2"})
        self.assertEqual(result.training_log_identity["row_count"], 1)
        self.assertEqual(result.summary["result_kind"], "diagnostic")
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            validate_artifact(self.path, dataset="burgers1d", model_variant="dense", condition="full", seed=0)

    def test_actual_checkpoint_reader_rejects_text_and_missing_csv(self):
        self.mutate(checkpoint=lambda p: None)
        pt = self.path.parent / "final_checkpoint.pt"
        pt.write_text("checkpoint-0")
        self.mutate(summary=lambda s: s.update(checkpoint_sha256=sha256_file(pt)))
        with self.assertRaisesRegex(ValueError, "weights_only"):
            self.validate()
        (self.path.parent / "training_log.csv").unlink()
        with self.assertRaises(ValueError):
            self.validate()

    def test_nonfinite_weights_buffers_empty_keys_shapes_and_optimizer_rejected(self):
        from experiments.common.checkpoints import load_weights_only_checkpoint
        original = load_weights_only_checkpoint(self.path.parent / "final_checkpoint.pt").checkpoint
        tensor_names = list(original["model_state_dict"])
        buffer_name = next(name for name in tensor_names if name.endswith(".G"))
        def assign_nan(p, key, value):
            p["model_state_dict"][key] = torch.full_like(p["model_state_dict"][key], value)
        changes = [lambda p: p.update(model_state_dict={}),
                   lambda p: p["model_state_dict"].pop(tensor_names[0]),
                   lambda p: p["model_state_dict"].update({tensor_names[0]: torch.zeros(1)}),
                   lambda p: assign_nan(p, tensor_names[0], float("nan")),
                   lambda p: assign_nan(p, buffer_name, float("inf")),
                   lambda p: assign_nan(p, buffer_name, float("-inf")),
                   lambda p: p.update(optimizer_state_dict={}),
                   lambda p: p["scheduler_state_dict"].update(last_epoch=500)]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                payload = copy.deepcopy(original)
                change(payload)
                pt = self.path.parent / "final_checkpoint.pt"
                torch.save(payload, pt)
                self.mutate(summary=lambda s: s.update(checkpoint_sha256=sha256_file(pt)))
                with self.assertRaises(ValueError):
                    self.validate()

    def test_restore_model_alone_rejects_nan_and_preserves_rng(self):
        path = self.path.parent / "final_checkpoint.pt"
        before = torch.get_rng_state().clone()
        restore_model_checkpoint(path, expected_dataset="burgers1d", expected_condition="full",
                                 expected_model_variant="dense", expected_seed=0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        def poison(payload):
            key = next(iter(payload["model_state_dict"]))
            payload["model_state_dict"][key].fill_(float("nan"))
        self.mutate(checkpoint=poison)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            restore_model_checkpoint(path, expected_dataset="burgers1d", expected_condition="full",
                                     expected_model_variant="dense", expected_seed=0)

    def test_metrics_missing_negative_string_bool_nan_inf_and_timing_only_rejected(self):
        initial = json.loads(self.path.read_text())
        for value in (-1, "0.1", True, float("nan"), float("inf"), None):
            with self.subTest(value=str(value)):
                metadata = copy.deepcopy(initial)
                metadata["final_metrics"]["test_full_relative_l2"] = value
                self.path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.validate()
        metadata = copy.deepcopy(initial)
        metadata["final_metrics"] = {"epoch": 1, "epoch_time_seconds": .1, "learning_rate": .001}
        self.path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.validate()

    def test_config_source_data_evaluation_and_protocol_tampering_rejected(self):
        original = json.loads(self.path.read_text())
        for field in ("dataset_id", "data_configuration", "evaluation_definition", "source", "runtime"):
            with self.subTest(field=field):
                metadata = copy.deepcopy(original)
                metadata[field] = {} if isinstance(metadata[field], dict) else "wrong"
                self.path.write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.validate()
        metadata = copy.deepcopy(original)
        metadata["final_epoch"] = 500
        self.path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.validate()

    def test_csv_requires_contiguous_history_and_rejects_header_only(self):
        protocol = pinned_contract("burgers1d", test_contract=tiny_contract(epochs=500))
        columns = protocol["csv_columns"]
        rows = [{name: (epoch if name == "epoch" else .1) for name in columns} for epoch in range(1, 501)]
        path = self.path.parent / "synthetic_500_history.csv"
        def write(values):
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(values)
        write(rows)
        # Synthetic completeness only; no model run is represented as formal.
        self.assertEqual(validate_csv(path, protocol=protocol, final_metrics=rows[-1])["row_count"], 500)
        for values in ([], rows[-1:], rows[:-1], rows[:3] + rows[4:], rows[:3] + [rows[2]] + rows[4:]):
            write(values)
            with self.assertRaises(ValueError):
                validate_csv(path, protocol=protocol, final_metrics=rows[-1])

    def test_csv_hash_and_reference_escape_rejected(self):
        self.mutate(summary=lambda s: s.update(training_log="../training_log.csv"))
        with self.assertRaises(ValueError):
            self.validate()

    def test_jointly_forged_summary_checkpoint_and_fingerprint_still_rejected(self):
        from experiments.common.checkpoints import load_weights_only_checkpoint
        summary = json.loads(self.path.read_text())
        pt = self.path.parent / "final_checkpoint.pt"
        payload = load_weights_only_checkpoint(pt).checkpoint
        for record in (summary, payload):
            record["comparison_contract"]["training_configuration"]["epochs"] = 500
            record["training_configuration"]["epochs"] = 500
            record["comparison_fingerprint"] = canonical_sha256(record["comparison_contract"])
            record["final_epoch"] = 500
        torch.save(payload, pt)
        summary["checkpoint_sha256"] = sha256_file(pt)
        self.path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pinned_configuration"):
            self.validate()

    def test_jointly_recorded_invalid_paper_metrics_are_not_ignored(self):
        from experiments.common.checkpoints import load_weights_only_checkpoint
        initial_summary = json.loads(self.path.read_text())
        pt = self.path.parent / "final_checkpoint.pt"
        initial_checkpoint = load_weights_only_checkpoint(pt).checkpoint
        for bad in (-1, "0.1", True):
            with self.subTest(value=repr(bad)):
                summary, payload = copy.deepcopy(initial_summary), copy.deepcopy(initial_checkpoint)
                for record in (summary, payload):
                    record["final_metrics"]["test_full_relative_l2"] = bad
                torch.save(payload, pt)
                summary["checkpoint_sha256"] = sha256_file(pt)
                self.path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "invalid_numeric_value"):
                    self.validate()

    def test_integer_bool_buffers_finite_and_invalid_optimizer_states_rejected(self):
        from ablations.checkpoints import validate_finite_tree
        validate_finite_tree({"integer": torch.tensor([-2, 0, 3]), "bool": torch.tensor([True, False])}, state=True)
        def invalid_state(payload):
            entry = next(iter(payload["optimizer_state_dict"]["state"].values()))
            entry["exp_avg"].fill_(float("inf"))
        self.mutate(checkpoint=invalid_state)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.validate()

    def test_all_dataset_protocols_preserve_approved_metrics(self):
        from ablations.contracts import DATASETS
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                contract = pinned_contract(dataset)
                self.assertEqual(contract["training_configuration"]["epochs"], 500)
                self.assertEqual(contract["evaluation_definition"]["evaluation_sample_count"]["test"], 200)
                self.assertTrue(contract["paper_metrics"])
        airfoil = pinned_contract("airfoil")["training_configuration"]
        self.assertEqual(airfoil["scheduler_t_max"], 62500)
        self.assertEqual(airfoil["scheduler_step_policy"],
            "after_every_optimizer_step_plus_one_additional_step_at_epoch_end")

    def test_learnable_sigma_metadata_recomputed_from_actual_optimizer(self):
        path, contract = make_artifact(Path(self.directory.name), seed=19, condition="learnable_sigma")
        kwargs = dict(dataset="burgers1d", model_variant="dense", condition="learnable_sigma",
                      seed=19, allow_diagnostic=True, test_contract=contract)
        result = validate_artifact(path, **kwargs)
        self.assertGreater(result.summary["sigma_optimizer"]["trainable_scalar_count"], 0)
        self.assertFalse(result.summary["sigma_optimizer"]["gaussian_G_trainable"])
        for item in result.summary["sigma_optimizer"]["encoders"]:
            self.assertTrue(item["optimizer_included"])
            self.assertEqual(item["initial_learning_rate"], .001)
            self.assertEqual(item["weight_decay"], .0001)
        from experiments.common.checkpoints import load_weights_only_checkpoint
        checkpoint_path = path.parent / "final_checkpoint.pt"
        checkpoint = load_weights_only_checkpoint(checkpoint_path).checkpoint
        summary = json.loads(path.read_text())
        for record in (summary, checkpoint):
            record["sigma_optimizer"]["encoders"][0]["weight_decay"] = .1
        torch.save(checkpoint, checkpoint_path)
        summary["checkpoint_sha256"] = sha256_file(checkpoint_path)
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sigma_optimizer"):
            validate_artifact(path, **kwargs)
