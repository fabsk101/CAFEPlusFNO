"""Actual tiny native Full bundles; synthetic metrics never represent training."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from ablations.artifacts import canonical_json
from ablations.references import reference_spec, reference_test_source_id, validate_full_reference
from ablations.repository import sha256_file
from ablations.runtime import output_path, workspace_root
from ablations.tests.artifact_fixtures import tiny_contract


def make_native_reference(root, *, seed=42, test_contract=None):
    """One real CAFE update plus a labelled native-schema synthetic fixture.

    The dataset identity is copied expected public metadata required by the
    native parser. No dataset bytes were acquired or validated in this fixture.
    """
    import torch
    from scripts import run_paper_sweep as sweep
    from experiments.common.dataset_provenance import dataset_manifest
    from experiments.common.checkpoints import prepare_weights_only_checkpoint
    from ablations.models import preserved_rng_state
    from ablations.runtime_metadata import capture_runtime_metadata
    from ablations.contracts import MODEL_VARIANTS
    from ablations.repository import load_source_manifest
    test_contract = test_contract or tiny_contract()
    spec, protocol = reference_spec("burgers1d", "dense", test_contract=test_contract)
    model_name = MODEL_VARIANTS["dense"]
    directory = Path(root) / model_name / f"seed_{seed}"
    directory.mkdir(parents=True)
    with preserved_rng_state():
        torch.manual_seed(seed)
        built = spec["model_builder"](model_name, "cpu")
        cfg = spec["training_configuration"]
        optimizer = torch.optim.AdamW(built.model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler_t_max"])
        for _ in range(test_contract.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = built.model(torch.randn(2, 1, 7)).square().mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
    identity = copy.deepcopy(dataset_manifest()[spec["dataset_id"]])
    identity["dataset_id"] = spec["dataset_id"]
    identity["test_fixture_expected_metadata_only"] = True
    source_id = reference_test_source_id(test_contract)
    source = {"training_source_commit": source_id, "source_repository_commit": source_id,
              "evaluation_source_commit": source_id, "training_source_dirty": False,
              "source_repository_dirty": False, "evaluation_source_dirty": False,
              "sirenfno_upstream_commit": load_source_manifest()["sirenfno_commit"]}
    runtime = capture_runtime_metadata(profile="diagnostic", device="cpu")
    shared = {"completion_schema_version": sweep.COMPLETION_SCHEMA_VERSION,
              "experiment": spec["experiment"], "dataset_id": spec["dataset_id"],
              "dataset_identity": identity, "model": model_name,
              "model_class": f"{built.model.__class__.__module__}.{built.model.__class__.__qualname__}",
              "factorization": built.factorization, "rank": built.rank,
              "parameter_count": sum(p.numel() for p in built.model.parameters()),
              "seed": seed, "experiment_seed": seed,
              "training_protocol_version": spec["training_protocol_version"],
              "training_protocol_identifier": sweep.training_protocol_identifier(spec),
              "training_configuration": spec["training_configuration"],
              "data_configuration": spec["data_configuration"],
              **spec["training_configuration"], **source,
              "evaluation_schema_version": sweep.EVALUATION_SCHEMA_VERSION,
              "evaluation_protocol_version": spec["evaluation_protocol_version"],
              **spec["evaluation_metadata"], "runtime": runtime,
              "result_kind": "diagnostic", "test_contract": test_contract.label,
              "fixture_note": "TEST_ONLY: one tiny CPU CAFE update; synthetic metrics and expected dataset metadata"}
    rows = []
    for epoch in range(1, test_contract.epochs + 1):
        row = {name: 0.25 + seed / 10000 for name in protocol["csv_columns"]}
        row.update(epoch=epoch, learning_rate=optimizer.param_groups[0]["lr"], train_time_seconds=0.1, epoch_time_seconds=0.2)
        rows.append(row)
    with (directory / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=protocol["csv_columns"])
        writer.writeheader()
        writer.writerows(rows)
    metrics = {name: rows[-1][column] for name, column in spec["metric_columns"].items()}
    checkpoint = prepare_weights_only_checkpoint({**shared, **metrics,
        "model_state_dict": built.model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(), "model_configuration": built.configuration,
        "epoch": test_contract.epochs, "environment": {**source, "pytorch_version": str(torch.__version__)}})
    torch.save(checkpoint, directory / "final_checkpoint.pt")
    summary = {**shared, **metrics, "model_hyperparameters": built.configuration,
               "final_epoch": test_contract.epochs, "final_checkpoint": "final_checkpoint.pt",
               "training_log": "training_log.csv", "checkpoint_sha256": sha256_file(directory / "final_checkpoint.pt")}
    path = directory / "summary.json"
    path.write_text(json.dumps(summary, allow_nan=False), encoding="utf-8")
    return path, test_contract


class NativeFullReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=output_path(workspace_root() / "tmp"))
        self.addCleanup(self.temp.cleanup)
        self.path, self.contract = make_native_reference(Path(self.temp.name))

    def validate(self, **options):
        return validate_full_reference(self.path, dataset="burgers1d", model_variant="dense",
            test_contract=self.contract, require_comparable=False, **options)

    def test_real_resolved_model_reference_readonly_and_source_preserved(self):
        import torch
        files = list(self.path.parent.iterdir())
        before = {path.name: sha256_file(path) for path in files}
        rng = torch.random.get_rng_state().clone()
        result = self.validate(expected_seed=42)
        self.assertEqual(result["origin"], "main_full_reference")
        self.assertEqual(result["seed"], 42)
        self.assertTrue(result["metrics"])
        self.assertEqual(result["source"]["training_source_commit"], reference_test_source_id(self.contract))
        self.assertEqual(result["reference_provenance"]["training_source_commit"], reference_test_source_id(self.contract))
        self.assertIs(result["reference_provenance"]["reevaluation_performed"], False)
        self.assertIs(result["comparison_eligible"], False)
        self.assertEqual(before, {path.name: sha256_file(path) for path in files})
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_source_data_config_and_metric_counterexamples_rejected(self):
        pristine = self.path.read_bytes()
        for mutate in (
            lambda s: s.pop("dataset_identity"),
            lambda s: s.update(training_source_commit="0" * 40),
            lambda s: s.update(source_repository_commit="0" * 40),
            lambda s: s["model_hyperparameters"].update(width=999),
            lambda s: s["data_configuration"].update(normalization_policy="DIFFERENT"),
            lambda s: s.update(final_test_step_relative_l2=-1),
        ):
            with self.subTest(case=mutate.__code__.co_firstlineno):
                value = json.loads(pristine)
                mutate(value)
                self.path.write_text(json.dumps(value), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.validate()
        self.path.write_bytes(pristine)

    def test_test_fixture_never_promoted_to_formal_reference(self):
        with self.assertRaises(ValueError):
            validate_full_reference(self.path, dataset="burgers1d", model_variant="dense")
        with self.assertRaises(ValueError):
            validate_full_reference(self.path, dataset="burgers1d", model_variant="dense",
                test_contract=self.contract, require_comparable=True)

    def test_missing_runtime_is_unverified_not_current_environment(self):
        import torch
        from experiments.common.checkpoints import load_weights_only_checkpoint, prepare_weights_only_checkpoint
        summary = json.loads(self.path.read_text(encoding="utf-8"))
        checkpoint_path = self.path.parent / "final_checkpoint.pt"
        checkpoint = load_weights_only_checkpoint(checkpoint_path).checkpoint
        summary.pop("runtime")
        checkpoint.pop("runtime")
        torch.save(prepare_weights_only_checkpoint(checkpoint), checkpoint_path)
        summary["checkpoint_sha256"] = sha256_file(checkpoint_path)
        self.path.write_text(json.dumps(summary), encoding="utf-8")
        result = self.validate()
        self.assertIsNone(result["runtime"])
        self.assertEqual(result["runtime_verification"], "UNVERIFIED")
        self.assertIs(result["comparison_eligible"], False)
        self.assertIn("runtime.python.version", result["runtime_diagnostics"]["missing_fields"])
        self.assertIn("runtime.operations.tf32_matmul", result["runtime_diagnostics"]["missing_fields"])
        self.assertIn("runtime.threads.intra_op", result["runtime_diagnostics"]["missing_fields"])
        from ablations.public import PublicFieldError, error_record
        with self.assertRaises(PublicFieldError) as caught:
            validate_full_reference(self.path, dataset="burgers1d", model_variant="dense",
                test_contract=self.contract, require_comparable=True)
        public = error_record(caught.exception)
        self.assertEqual(public["reason_code"], "REFERENCE_RUNTIME_EVIDENCE_INCOMPLETE")
        self.assertIn("runtime.packages.torch", public["missing_fields"])

    def test_safe_diagnostics_name_known_fields_without_private_values_or_keys(self):
        from ablations.public import PublicFieldError, error_record
        pristine = self.path.read_bytes()
        cases = (
            ("summary.model_hyperparameters.width", lambda s: s["model_hyperparameters"].update(width="C:/Users/SyntheticOwner/private")),
            ("summary.model_hyperparameters", lambda s: s["model_hyperparameters"].update(SYNTHETIC_PRIVATE_KEY=1)),
            ("summary.data_configuration.normalization_policy", lambda s: s["data_configuration"].update(normalization_policy="SYNTHETIC_SECRET_VALUE")),
            ("summary.training_source_commit", lambda s: s.update(training_source_commit="SYNTHETIC_SECRET_VALUE")),
        )
        for field, mutate in cases:
            with self.subTest(field=field):
                summary = json.loads(pristine)
                mutate(summary)
                self.path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaises(PublicFieldError) as caught:
                    self.validate()
                record = error_record(caught.exception)
                self.assertIn(field, record["mismatched_fields"])
                text = json.dumps(record)
                for secret in ("SyntheticOwner", "SYNTHETIC_PRIVATE_KEY", "SYNTHETIC_SECRET_VALUE"):
                    self.assertNotIn(secret, text)
        self.path.write_bytes(pristine)
        error = PublicFieldError("REFERENCE_CONFIGURATION_MISMATCH",
            mismatched_fields=["summary.model_hyperparameters.width", "SYNTHETIC_PRIVATE_KEY", "/home/synthetic"],
            field_catalog=["summary.model_hyperparameters.width"])
        self.assertEqual(error_record(error)["mismatched_fields"], ["summary.model_hyperparameters.width"])

    def test_native_checkpoint_config_failure_reports_resolved_field(self):
        import torch
        from experiments.common.checkpoints import load_weights_only_checkpoint, prepare_weights_only_checkpoint
        from ablations.public import PublicFieldError, error_record
        checkpoint_path = self.path.parent / "final_checkpoint.pt"
        checkpoint = load_weights_only_checkpoint(checkpoint_path).checkpoint
        checkpoint["model_configuration"]["width"] = 999
        torch.save(prepare_weights_only_checkpoint(checkpoint), checkpoint_path)
        summary = json.loads(self.path.read_text(encoding="utf-8"))
        summary["checkpoint_sha256"] = sha256_file(checkpoint_path)
        self.path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(PublicFieldError) as caught:
            self.validate()
        self.assertIn("checkpoint.model_configuration.width", error_record(caught.exception)["mismatched_fields"])

    def test_runtime_invalid_observation_is_named_even_when_other_fields_missing(self):
        import torch
        from experiments.common.checkpoints import load_weights_only_checkpoint, prepare_weights_only_checkpoint
        from ablations.public import PublicFieldError, error_record
        checkpoint_path = self.path.parent / "final_checkpoint.pt"
        checkpoint_template = load_weights_only_checkpoint(checkpoint_path).checkpoint
        summary_template = json.loads(self.path.read_text(encoding="utf-8"))
        mutations = (
            ("runtime.operations.tf32_matmul", lambda r: (r["operations"].update(tf32_matmul="SYNTHETIC_PRIVATE_VALUE"), r["packages"].pop("numpy"))),
            ("runtime.device", lambda r: r.update(device=["SYNTHETIC_PRIVATE_VALUE"])),
            ("runtime.schema_version", lambda r: r.update(schema_version=999)),
        )
        for field, mutate in mutations:
            with self.subTest(field=field):
                checkpoint = dict(checkpoint_template)
                checkpoint["runtime"] = copy.deepcopy(checkpoint_template["runtime"])
                summary = copy.deepcopy(summary_template)
                for record in (summary, checkpoint):
                    mutate(record["runtime"])
                torch.save(prepare_weights_only_checkpoint(checkpoint), checkpoint_path)
                summary["checkpoint_sha256"] = sha256_file(checkpoint_path)
                self.path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaises(PublicFieldError) as caught:
                    self.validate()
                record = error_record(caught.exception)
                self.assertEqual(record["reason_code"], "REFERENCE_RUNTIME_EVIDENCE_INVALID")
                self.assertIn(field, record["mismatched_fields"])
                self.assertNotIn("SYNTHETIC_PRIVATE_VALUE", json.dumps(record))

    def test_reference_only_selected_seed_aggregate_and_public_export(self):
        from ablations.aggregate import aggregate
        root = Path(self.temp.name)
        second, _ = make_native_reference(root, seed=7, test_contract=self.contract)
        third, _ = make_native_reference(root, seed=19, test_contract=self.contract)
        report = aggregate(results_root=root / "empty_ablation_results", dataset="burgers1d",
            model_variant="dense", condition="full", seeds=[19, 42, 7],
            reference_full_summaries=[self.path, second, third], test_contract=self.contract,
            output_dir=root / "public_export")
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["result_kind"], "diagnostic_test")
        self.assertEqual(report["present_seeds"], [19, 42, 7])
        self.assertEqual(report["seed_count"], 3)
        self.assertTrue(all(run["origin"] == "main_full_reference" for run in report["runs"]))
        expected = sum(0.25 + seed / 10000 for seed in [19, 42, 7]) / 3
        for value in report["metrics"].values():
            self.assertAlmostEqual(value["mean"], expected)
        exported = (root / "public_export/aggregate.json").read_text(encoding="utf-8")
        self.assertNotIn(str(root), exported)
        self.assertNotIn(root.as_posix(), exported)
        self.assertIn(reference_test_source_id(self.contract), exported)

    def test_same_seed_reference_and_new_full_need_explicit_preference(self):
        from ablations.aggregate import aggregate
        from ablations.tests.artifact_fixtures import make_artifact
        root = Path(self.temp.name)
        make_artifact(root / "new_results", seed=42, test_contract=self.contract)
        args = dict(results_root=root / "new_results", dataset="burgers1d", model_variant="dense",
                    condition="full", seeds=[42], reference_full_summaries=[self.path],
                    test_contract=self.contract, output_dir=root / "selection_export")
        with self.assertRaises(RuntimeError):
            aggregate(**args)
        for preference, origin in (("reference", "main_full_reference"), ("ablation", "ablation")):
            with self.subTest(preference=preference):
                report = aggregate(**args, reference_preference=preference, overwrite_aggregate=True)
                self.assertEqual(report["runs"][0]["origin"], origin)
                self.assertEqual(report["seed_count"], 1)
                self.assertTrue(all(value["sample_std"] is None for value in report["metrics"].values()))
                self.assertEqual(len(report["preference_exclusions"]), 1)


if __name__ == "__main__":
    unittest.main()
