"""Regression tests for checkpoint-evaluation aggregation and output tables."""

from __future__ import annotations

import copy
import functools
import io
import json
import math
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.common.dataset_provenance import dataset_manifest
from experiments.common.evaluation import corrected_relative_l2_metadata
from experiments.configs import cfd1d
from scripts import run_paper_sweep, summarize_results
from scripts.summarize_results import (
    DATASETS,
    PAPER_METRIC_FIELDS,
    ResultValidationError,
    _evaluate_neuraloperator,
    _write_aggregate_outputs,
    aggregate_records,
    canonical_sha256,
    main,
    validate_evaluation_record,
)


@functools.lru_cache(maxsize=1)
def _fixture_contract() -> dict[str, object]:
    spec = run_paper_sweep.DATASETS["cfd1d1024"]
    manifest = dataset_manifest()["cfd1d1024"]
    configuration = cfd1d.model_constructor_kwargs("fno")
    dataset_identity = {
        "dataset_id": "cfd1d1024",
        "source": manifest["source"],
        "resolution": manifest["resolution"],
        "n_train": manifest["n_train"],
        "n_test": manifest["n_test"],
        "files": {
            filename: {
                "path": f"data/{filename}",
                "sha256": metadata["sha256"],
                "size_bytes": metadata["size_bytes"],
            }
            for filename, metadata in manifest["files"].items()
        },
    }
    training_configuration = copy.deepcopy(spec["training_configuration"])
    data_configuration = copy.deepcopy(spec["data_configuration"])
    training_log_columns = [
        "epoch",
        "learning_rate",
        "train_time_seconds",
        "epoch_time_seconds",
        *dict.fromkeys(spec["metric_columns"].values()),
    ]
    return {
        "configuration": configuration,
        "dataset_identity": dataset_identity,
        "training_configuration": training_configuration,
        "data_configuration": data_configuration,
        "training_log_columns": training_log_columns,
        "training_log_metric_columns": copy.deepcopy(spec["metric_columns"]),
        "training_protocol_identifier": run_paper_sweep.training_protocol_identifier(
            spec
        ),
    }


def evaluation_record(seed: int, value: float) -> dict[str, object]:
    fixture = copy.deepcopy(_fixture_contract())
    configuration = fixture["configuration"]
    dataset_identity = fixture["dataset_identity"]
    training_configuration = fixture["training_configuration"]
    training_protocol = copy.deepcopy(training_configuration)
    data_configuration = fixture["data_configuration"]
    evaluation_definition = corrected_relative_l2_metadata(
        sample_count={"test": 200},
        horizon=10,
        evaluation_space="physical_source_values_no_normalization",
    )
    return {
        "record_type": "checkpoint_evaluation",
        "schema_version": 2,
        "evaluation_schema_version": "paper_evaluation_record_v2",
        "dataset": "cfd1d1024",
        "dataset_id": "cfd1d1024",
        "experiment": "cfd1d1024_released_sirenfno_vx",
        "model": "fno",
        "factorization": None,
        "rank": None,
        "seed": seed,
        "fixed_final_epoch": 500,
        "checkpoint_selection_policy": "fixed_final_epoch_no_test_selection",
        "source_checkpoint": f"results/fno/seed_{seed}/final_checkpoint.pt",
        "checkpoint_sha256": f"{seed:064x}",
        "training_source_commit": "b" * 40,
        "training_source_dirty": False,
        "evaluation_source_commit": "c" * 40,
        "evaluation_source_dirty": False,
        "training_artifact_validation_status": (
            "valid_training_current_evaluation"
        ),
        "training_artifact_evaluation_classification": "current",
        "original_training_evaluation_protocol_version": (
            "corrected_relative_l2_v1"
        ),
        "training_log_identity": {
            "sha256": f"{seed + 1:064x}",
            "columns": fixture["training_log_columns"],
            "row_count": 500,
            "first_epoch": 1,
            "final_epoch": 500,
            "epoch_sequence_sha256": canonical_sha256(list(range(1, 501))),
            "metric_columns": fixture["training_log_metric_columns"],
        },
        "checkpoint_load_metadata": {
            "checkpoint_load_mode": "weights_only",
            "legacy_torch_version_compatibility_used": False,
        },
        "diagnostic_only": False,
        "paper_eligible": True,
        "diagnostic_reasons": [],
        "training_protocol": training_protocol,
        "training_protocol_sha256": canonical_sha256(training_protocol),
        "training_protocol_version": "cfd1d1024_training_v1",
        "training_protocol_identifier": fixture["training_protocol_identifier"],
        "training_configuration": training_configuration,
        "data_configuration": data_configuration,
        "model_configuration": configuration,
        "model_configuration_sha256": canonical_sha256(configuration),
        "dataset_identity": dataset_identity,
        "dataset_identity_sha256": canonical_sha256(dataset_identity),
        "evaluation_protocol_version": "corrected_relative_l2_v1",
        "evaluation_space": "physical_source_values_no_normalization",
        "evaluation_tensor_layout": "B,T,C,*spatial",
        "evaluation_zero_reference_policy": "max(target_l2, epsilon)",
        "evaluation_definition": evaluation_definition,
        "evaluation_definition_sha256": canonical_sha256(evaluation_definition),
        "evaluation_epsilon": 1e-12,
        "evaluation_horizon": 10,
        "evaluation_sample_count": {"test": 200},
        "evaluation_batch_size": 32,
        "metrics": {
            "corrected_step_relative_l2": value,
            "corrected_trajectory_relative_l2": value / 2,
        },
        "parameter_count": cfd1d.EXPECTED_PINNED_PARAMETER_COUNTS["fno"],
        "training_time_seconds": None,
        "reevaluation_time_seconds": 0.25,
        "final_test_corrected_step_relative_l2": value,
        "final_test_corrected_trajectory_relative_l2": value / 2,
    }


class EvaluationRecordTests(unittest.TestCase):
    def test_finite_record_and_hashes_are_required(self) -> None:
        record = evaluation_record(42, 1.0)
        self.assertEqual(validate_evaluation_record(record)["seed"], 42)
        for bad in (math.nan, math.inf, -math.inf):
            with self.subTest(value=bad):
                invalid = copy.deepcopy(record)
                invalid["metrics"]["corrected_step_relative_l2"] = bad
                with self.assertRaises(ResultValidationError):
                    validate_evaluation_record(invalid)
        invalid = copy.deepcopy(record)
        invalid["model_configuration"]["width"] = 64
        with self.assertRaisesRegex(ResultValidationError, "hash mismatch"):
            validate_evaluation_record(invalid)

    def test_training_log_identity_requires_production_columns(self) -> None:
        for column in (
            "learning_rate",
            "train_time_seconds",
            "epoch_time_seconds",
        ):
            record = evaluation_record(42, 1.0)
            record["training_log_identity"]["columns"].remove(column)
            with self.subTest(column=column):
                with self.assertRaisesRegex(
                    ResultValidationError, "production columns"
                ):
                    validate_evaluation_record(record)

    def test_full_evaluation_definition_is_code_owned_not_hash_owned(self) -> None:
        mutations = (
            ("evaluation_implementation", "unapproved/private-evaluator.py"),
            ("evaluation_norm_axes", {"corrected_step_relative_l2": "WRONG"}),
            (
                "evaluation_aggregation",
                {"corrected_step_relative_l2": "best_seed_only"},
            ),
        )
        derived_hashes = (
            "training_configuration_sha256",
            "data_configuration_sha256",
            "training_provenance_sha256",
            "evaluation_provenance_sha256",
            "paper_group_contract_sha256",
        )
        for field, value in mutations:
            record = evaluation_record(42, 1.0)
            record["evaluation_definition"][field] = value
            record["evaluation_definition_sha256"] = canonical_sha256(
                record["evaluation_definition"]
            )
            for hash_field in derived_hashes:
                record.pop(hash_field, None)
            summarize_results._add_recomputed_contract_hashes(record, record)
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                    ResultValidationError, "approved evaluator contract"
                ):
                    validate_evaluation_record(record)


class DatasetContractTests(unittest.TestCase):
    def test_reevaluation_adapters_match_completion_contracts(self) -> None:
        from scripts import run_paper_sweep

        self.assertEqual(set(DATASETS), set(run_paper_sweep.DATASETS))
        for dataset, adapter in DATASETS.items():
            with self.subTest(dataset=dataset):
                spec = run_paper_sweep.DATASETS[dataset]
                self.assertEqual(adapter.dataset_id, spec["dataset_id"])
                self.assertEqual(
                    adapter.evaluation_protocol,
                    spec["evaluation_protocol_version"],
                )
                self.assertEqual(
                    PAPER_METRIC_FIELDS[dataset],
                    {
                        field: (
                            "trajectory_relative_l2"
                            if field == "final_test_full_relative_l2"
                            else field.removeprefix("final_test_")
                        )
                        for field in spec["paper_metric_columns"]
                    },
                )

    def test_dirty_diagnostic_record_is_not_paper_eligible(self) -> None:
        record = evaluation_record(42, 1.0)
        record["evaluation_source_dirty"] = True
        with self.assertRaisesRegex(ResultValidationError, "Dirty-source"):
            validate_evaluation_record(record)

    def test_neuraloperator_reevaluation_decodes_in_eval_mode_and_restores_it(self) -> None:
        class Dataset(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 3

            def __getitem__(self, index: int):
                value = torch.full((1, 2, 2), float(index + 1))
                return {"x": value, "y": value * 2}

        class Processor(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.device = torch.device("cpu")

            def to(self, device):
                self.device = torch.device(device)
                return self

            def preprocess(self, sample):
                sample = {
                    key: value.to(self.device) for key, value in sample.items()
                }
                if self.training:
                    sample["y"] = sample["y"] * 3
                return sample

            def postprocess(self, output, sample):
                return (output * 2 if not self.training else output), sample

        class Model(torch.nn.Module):
            def forward(self, x, **_kwargs):
                return x

        class RelativeLoss:
            eps = 1e-8

            def __init__(self, **_kwargs) -> None:
                pass

            def __call__(self, output, target):
                batch = output.shape[0]
                numerator = (output - target).reshape(batch, -1).norm(dim=1)
                denominator = target.reshape(batch, -1).norm(dim=1) + self.eps
                return (numerator / denominator).sum()

        class H1Loss:
            def __init__(self, **_kwargs) -> None:
                pass

            def __call__(self, output, target):
                return (output - target).abs().reshape(output.shape[0], -1).sum()

        loader = torch.utils.data.DataLoader(Dataset(), batch_size=2)
        processor = Processor()
        model = Model()
        module = types.SimpleNamespace(LpLoss=RelativeLoss, H1Loss=H1Loss)
        metrics, count, horizon, epsilon, layout = _evaluate_neuraloperator(
            module,
            model,
            (None, {128: loader}, processor),
            batch_size=2,
            device=torch.device("cpu"),
            torch=torch,
        )
        self.assertEqual(metrics, {"relative_l2": 0.0, "h1": 0.0})
        self.assertEqual((count, horizon, epsilon), (3, 1, 1e-8))
        self.assertEqual(layout, "B,C=1,*spatial")
        self.assertTrue(processor.training)
        self.assertTrue(model.training)


class AggregateTests(unittest.TestCase):
    def assert_paper_rejected_even_with_partial(
        self,
        records: list[dict[str, object]],
    ) -> None:
        seeds = (0, 42, 73, 108, 202)
        for allow_partial in (False, True):
            with self.subTest(allow_partial=allow_partial):
                with self.assertRaises(ResultValidationError):
                    aggregate_records(
                        records,
                        required_seeds=seeds,
                        allow_partial=allow_partial,
                    )

    def test_five_seed_mean_and_sample_standard_deviation(self) -> None:
        seeds = (0, 42, 73, 108, 202)
        records = [evaluation_record(seed, float(index + 1)) for index, seed in enumerate(seeds)]
        report = aggregate_records(records, required_seeds=seeds, allow_partial=False)
        self.assertFalse(report["partial"])
        group = report["aggregates"][0]
        self.assertEqual(group["seeds"], list(seeds))
        self.assertEqual(group["seed_count"], 5)
        metric = group["metrics"]["corrected_step_relative_l2"]
        self.assertAlmostEqual(metric["mean"], 3.0)
        self.assertAlmostEqual(metric["sample_standard_deviation_ddof_1"], math.sqrt(2.5))

    def test_exact_historical_reuse_combines_with_current_source_and_keeps_evidence(
        self,
    ) -> None:
        historical = evaluation_record(0, 1.0)
        historical["checkpoint_load_metadata"] = {
            "checkpoint_load_mode": "weights_only",
            "legacy_torch_version_compatibility_used": False,
            "legacy_neuraloperator_metadata_compatibility_used": True,
        }
        historical["training_source_reuse"] = {
            "policy": "exact_historical_commit_and_checkpoint_sha256",
            "aggregation_source_commit": "c" * 40,
            "historical_training_source_commit": "b" * 40,
            "historical_checkpoint_sha256": historical["checkpoint_sha256"],
            "verified_change_scope": (
                "checkpoint_storage_loading_validation_only"
            ),
            "source_diff_sha256": "d" * 64,
            "changed_paths": [
                "experiments/common/checkpoints.py",
                "scripts/run_paper_sweep.py",
            ],
        }
        current = evaluation_record(42, 2.0)
        current["training_source_commit"] = "c" * 40
        current["checkpoint_load_metadata"] = {
            "checkpoint_load_mode": "weights_only",
            "legacy_torch_version_compatibility_used": False,
            "legacy_neuraloperator_metadata_compatibility_used": False,
        }
        current["training_source_reuse"] = {
            "policy": "current_source_commit",
            "aggregation_source_commit": "c" * 40,
        }

        report = aggregate_records(
            [historical, current],
            required_seeds=(0, 42),
            allow_partial=False,
        )
        self.assertEqual(len(report["aggregates"]), 1)
        group = report["aggregates"][0]
        self.assertEqual(group["training_source_commits"], ["b" * 40, "c" * 40])
        self.assertIsNone(group["training_source_commit"])
        self.assertEqual(
            [item["seed"] for item in group["training_source_reuse_evidence"]],
            [0, 42],
        )
        self.assertTrue(
            group["checkpoint_load_metadata"][
                "legacy_neuraloperator_metadata_compatibility_used"
            ]
        )

        tampered = copy.deepcopy(historical)
        tampered["training_source_reuse"]["historical_checkpoint_sha256"] = (
            "f" * 64
        )
        with self.assertRaisesRegex(ResultValidationError, "exact provenance"):
            validate_evaluation_record(tampered)

    def test_one_field_paper_contract_counterexamples_are_rejected(self) -> None:
        seeds = (0, 42, 73, 108, 202)

        def records() -> list[dict[str, object]]:
            return [
                evaluation_record(seed, float(index + 1))
                for index, seed in enumerate(seeds)
            ]

        cases = []

        wrong_epoch = records()
        wrong_epoch[2]["fixed_final_epoch"] = 1
        cases.append(("fixed_final_epoch", wrong_epoch))

        wrong_split = records()
        wrong_split[2]["data_configuration"]["split_policy"] = "diagnostic_split"
        cases.append(("split_policy", wrong_split))

        wrong_experiment = records()
        wrong_experiment[2]["experiment"] = "diagnostic_smoke"
        cases.append(("experiment", wrong_experiment))

        wrong_internal_id = records()
        wrong_internal_id[2]["dataset_identity"]["dataset_id"] = "burgers1d"
        wrong_internal_id[2]["dataset_identity_sha256"] = canonical_sha256(
            wrong_internal_id[2]["dataset_identity"]
        )
        cases.append(("dataset_identity_internal_id", wrong_internal_id))

        unknown_model = records()
        unknown_model[2]["model"] = "not_a_paper_model"
        cases.append(("unknown_model", unknown_model))

        missing_dirty = records()
        del missing_dirty[2]["training_source_dirty"]
        cases.append(("missing_training_dirty", missing_dirty))

        bad_hash = records()
        bad_hash[2]["model_configuration"]["hidden_channels"] = 999
        cases.append(("configuration_hash", bad_hash))

        for name, invalid_records in cases:
            with self.subTest(case=name):
                self.assert_paper_rejected_even_with_partial(invalid_records)

        for bad_metric in (-1.0, math.nan, math.inf, -math.inf):
            invalid_records = records()
            invalid_records[2]["metrics"]["corrected_step_relative_l2"] = bad_metric
            invalid_records[2]["final_test_corrected_step_relative_l2"] = bad_metric
            with self.subTest(case="invalid_metric", value=bad_metric):
                self.assert_paper_rejected_even_with_partial(invalid_records)

    def test_batch_size_does_not_split_an_identical_metric_contract(self) -> None:
        seeds = (0, 42, 73, 108, 202)
        batch_sizes = (1, 7, 32, 64, 200)
        records = [
            evaluation_record(seed, float(index + 1))
            for index, seed in enumerate(seeds)
        ]
        for record, batch_size in zip(records, batch_sizes):
            record["evaluation_batch_size"] = batch_size

        report = aggregate_records(
            records,
            required_seeds=seeds,
            allow_partial=False,
        )

        self.assertEqual(len(report["aggregates"]), 1)
        self.assertEqual(
            report["aggregates"][0]["evaluation_batch_sizes"],
            list(batch_sizes),
        )
        self.assertEqual(report["aggregates"][0]["seed_count"], 5)

    def test_resolved_model_metadata_is_authoritative(self) -> None:
        mutations = (
            ("factorization", "dense"),
            ("rank", 8),
            ("parameter_count", 1),
        )
        for key, value in mutations:
            record = evaluation_record(42, 1.0)
            record[key] = value
            with self.subTest(field=key):
                with self.assertRaises(ResultValidationError):
                    validate_evaluation_record(record)

        record = evaluation_record(42, 1.0)
        record["model_configuration"]["hidden_channels"] = 999
        record["model_configuration_sha256"] = canonical_sha256(
            record["model_configuration"]
        )
        with self.assertRaisesRegex(ResultValidationError, "resolved paper model"):
            validate_evaluation_record(record)

    def test_missing_seed_requires_explicit_partial_mode(self) -> None:
        record = evaluation_record(42, 1.0)
        with self.assertRaisesRegex(ResultValidationError, "Missing required"):
            aggregate_records(
                [record], required_seeds=(0, 42), allow_partial=False
            )
        report = aggregate_records(
            [record], required_seeds=(0, 42), allow_partial=True
        )
        group = report["aggregates"][0]
        self.assertTrue(group["partial"])
        self.assertEqual(group["missing_required_seeds"], [0])
        self.assertIsNone(
            group["metrics"]["corrected_step_relative_l2"][
                "sample_standard_deviation_ddof_1"
            ]
        )

    def test_duplicate_seed_is_rejected(self) -> None:
        record = evaluation_record(42, 1.0)
        with self.assertRaisesRegex(ResultValidationError, "Duplicate seed"):
            aggregate_records(
                [record, copy.deepcopy(record)],
                required_seeds=(42,),
                allow_partial=False,
            )

    def test_same_checkpoint_cannot_impersonate_two_seed_runs(self) -> None:
        first = evaluation_record(0, 1.0)
        second = evaluation_record(42, 2.0)
        second["checkpoint_sha256"] = first["checkpoint_sha256"]
        with self.assertRaisesRegex(ResultValidationError, "Duplicate checkpoint"):
            aggregate_records(
                [first, second],
                required_seeds=(0, 42),
                allow_partial=False,
            )

    def test_protocols_are_never_mixed(self) -> None:
        first = evaluation_record(0, 1.0)
        second = evaluation_record(42, 2.0)
        second["evaluation_protocol_version"] = "legacy_metric"
        second["evaluation_definition"]["evaluation_protocol_version"] = (
            "legacy_metric"
        )
        second["evaluation_definition_sha256"] = canonical_sha256(
            second["evaluation_definition"]
        )
        with self.assertRaisesRegex(ResultValidationError, "incompatible"):
            aggregate_records(
                [first, second],
                required_seeds=(0, 42),
                allow_partial=False,
            )
        with self.assertRaisesRegex(ResultValidationError, "incompatible"):
            aggregate_records(
                [first, second], required_seeds=(0, 42), allow_partial=True
            )

    def test_outputs_keep_raw_results_and_never_invent_timing(self) -> None:
        report = aggregate_records(
            [evaluation_record(42, 1.0)],
            required_seeds=(42,),
            allow_partial=False,
        )
        self.assertIsNone(report["aggregates"][0]["training_timing"])
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_aggregate_outputs(
                Path(directory), report, overwrite=False
            )
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertIn("undefined", (Path(directory) / "paper_table.md").read_text())
            written = json.loads(
                (Path(directory) / "results.json").read_text(encoding="utf-8")
            )
            self.assertEqual(written["schema_version"], 2)
            self.assertIn(
                "paper_group_contract_sha256", written["aggregates"][0]
            )
            self.assertIn(
                "training_provenance_sha256", written["raw_results"][0]
            )
            with self.assertRaises(FileExistsError):
                _write_aggregate_outputs(Path(directory), report, overwrite=False)


class ResultCLIRedactionTests(unittest.TestCase):
    @staticmethod
    def invoke(arguments: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_foreign_absolute_path_syntax_is_always_external(self) -> None:
        token = "synthetic-person-token"
        foreign_paths = (
            Path(f"Z:\\Profiles\\{token}\\private-result.json"),
            Path(f"/home/{token}/private-result.json"),
            Path(f"/mnt/z/Profiles/{token}/private-result.json"),
            Path(
                "\\\\wsl.localhost\\SyntheticDistro\\home\\"
                f"{token}\\private-result.json"
            ),
        )
        for path in foreign_paths:
            with self.subTest(path_flavour=type(path).__name__, value=str(path)):
                public = summarize_results._public_path(
                    path,
                    dataset="cfd1d1024",
                    model="fno",
                    seed=42,
                    artifact="evaluation-record",
                )
                self.assertEqual(
                    public,
                    "<external-results>/cfd1d1024/fno/seed-42/evaluation-record",
                )
                self.assertNotIn(token, public)

    def test_invalid_path_shaped_arguments_use_only_safe_envelope(self) -> None:
        token = "synthetic-person-token"
        invalid_values = (
            f"Z:\\Profiles\\{token}\\dataset",
            f"/home/{token}/dataset",
            f"/mnt/z/Profiles/{token}/dataset",
            f"\\\\wsl.localhost\\SyntheticDistro\\home\\{token}\\dataset",
        )
        for value in invalid_values:
            with self.subTest(path_flavour=value.split("dataset", 1)[0]):
                status, stdout, stderr = self.invoke(
                    [
                        "reevaluate",
                        "--dataset",
                        value,
                        "--checkpoints",
                        "final_checkpoint.pt",
                    ]
                )
                self.assertEqual(status, 2)
                self.assertEqual(stdout, "")
                self.assertEqual(stderr, "ERROR reason=invalid_arguments\n")
                self.assertNotIn(token, stderr)
                self.assertNotIn(value, stderr)

    def test_noncanonical_checkpoint_name_is_rejected_before_loading(self) -> None:
        token = "synthetic-person-token"
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / token / "fno" / "seed_42"
            run_dir.mkdir(parents=True)
            (run_dir / "final_checkpoint.pt").write_bytes(b"canonical")
            decoy = run_dir / "private-decoy.pt"
            decoy.write_bytes(b"decoy")
            status, stdout, stderr = self.invoke(
                [
                    "reevaluate",
                    "--dataset",
                    "cfd1d1024",
                    "--checkpoints",
                    str(decoy),
                    "--device",
                    "cpu",
                ]
            )
        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertEqual(
            stderr,
            "ERROR reason=checkpoint_path_mismatch "
            "checkpoint=<external-results>/cfd1d1024/checkpoint "
            "dataset=cfd1d1024\n",
        )
        self.assertNotIn(token, stderr)
        self.assertNotIn("private-decoy.pt", stderr)

    def test_successful_reevaluation_prints_verified_external_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret_checkpoint_name = "private-" + "checkpoint-name.pt"
            secret_output_name = "private-" + "result-name.json"
            result_path = root / secret_output_name

            def fake_reevaluate(**_kwargs):
                result_path.write_text(
                    json.dumps(
                        {"dataset": "cfd1d1024", "model": "fno", "seed": 42}
                    ),
                    encoding="utf-8",
                )
                return result_path

            with patch.object(
                summarize_results,
                "reevaluate_checkpoint",
                side_effect=fake_reevaluate,
            ):
                status, stdout, stderr = self.invoke(
                    [
                        "reevaluate",
                        "--dataset",
                        "cfd1d1024",
                        "--checkpoints",
                        str(root / secret_checkpoint_name),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(
                stdout,
                "WROTE <external-results>/cfd1d1024/fno/seed-42/evaluation-record\n",
            )
            self.assertEqual(stderr, "")
            self.assertNotIn(secret_checkpoint_name, stdout + stderr)
            self.assertNotIn(secret_output_name, stdout + stderr)

    def test_successful_aggregate_never_prints_external_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret_input_name = "private-" + "evaluation-input.json"
            secret_output_name = "private-" + "aggregate-directory"
            record_path = root / secret_input_name
            output_dir = root / secret_output_name
            record_path.write_text(
                json.dumps(evaluation_record(42, 1.0)), encoding="utf-8"
            )
            status, stdout, stderr = self.invoke(
                [
                    "aggregate",
                    "--inputs",
                    str(record_path),
                    "--output-dir",
                    str(output_dir),
                    "--seeds",
                    "42",
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(stdout, "WROTE <external-results>/aggregate-output\n")
            self.assertEqual(stderr, "")
            self.assertNotIn(secret_input_name, stdout + stderr)
            self.assertNotIn(secret_output_name, stdout + stderr)

    def test_missing_external_input_uses_reason_code_without_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secret_name = "private-" + "missing-evaluation.json"
            missing = Path(directory) / secret_name
            status, stdout, stderr = self.invoke(
                [
                    "aggregate",
                    "--inputs",
                    str(missing),
                    "--output-dir",
                    str(Path(directory) / "output"),
                    "--seeds",
                    "42",
                ]
            )
            self.assertEqual(status, 2)
            self.assertEqual(stdout, "")
            self.assertEqual(
                stderr,
                "ERROR reason=input_missing input=<external-results>/evaluation-record\n",
            )
            self.assertNotIn(secret_name, stderr)
            self.assertNotIn("Traceback", stderr)

    def test_existing_external_output_is_nonzero_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record_path = root / "input.json"
            secret_output_name = "private-" + "existing-output"
            output_dir = root / secret_output_name
            record_path.write_text(
                json.dumps(evaluation_record(42, 1.0)), encoding="utf-8"
            )
            output_dir.mkdir()
            (output_dir / "seed_results.csv").write_text("existing\n", encoding="utf-8")
            status, stdout, stderr = self.invoke(
                [
                    "aggregate",
                    "--inputs",
                    str(record_path),
                    "--output-dir",
                    str(output_dir),
                    "--seeds",
                    "42",
                ]
            )
            self.assertEqual(status, 2)
            self.assertEqual(stdout, "")
            self.assertEqual(
                stderr,
                "ERROR reason=output_exists output=<external-results>/aggregate-output\n",
            )
            self.assertNotIn(secret_output_name, stderr)
            self.assertNotIn("Traceback", stderr)

    def test_checkpoint_load_failure_does_not_echo_loader_exception(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            secret_path_token = "synthetic-person-token"
            root = Path(directory) / secret_path_token / "fno" / "seed_42"
            root.mkdir(parents=True)
            checkpoint = root / "final_checkpoint.pt"
            checkpoint.write_bytes(b"not a checkpoint")
            (root / "summary.json").write_text("{}\n", encoding="utf-8")

            class FakeCuda:
                @staticmethod
                def is_available() -> bool:
                    return False

            fake_torch = types.SimpleNamespace(
                cuda=FakeCuda(),
                device=lambda value: value,
                load=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("loader leaked " + str(root))
                ),
            )
            class FakeTrainingArtifactValidationError(RuntimeError):
                def __init__(self) -> None:
                    self.audit = types.SimpleNamespace(
                        reasons=("checkpoint.unreadable",)
                    )
                    super().__init__("loader leaked " + str(root))

            def fail_shared_reader(*_args, **_kwargs):
                raise FakeTrainingArtifactValidationError()

            fake_sweep = types.SimpleNamespace(
                DATASETS={"cfd1d1024": {}},
                TrainingArtifactValidationError=(
                    FakeTrainingArtifactValidationError
                ),
                validate_training_artifacts=fail_shared_reader,
            )

            def fake_import(name: str):
                if name == "torch":
                    return fake_torch
                if name == "scripts.run_paper_sweep":
                    return fake_sweep
                if name == DATASETS["cfd1d1024"].module:
                    return types.SimpleNamespace()
                raise AssertionError("unexpected import")

            with patch.object(
                summarize_results.importlib,
                "import_module",
                side_effect=fake_import,
            ):
                status, stdout, stderr = self.invoke(
                    [
                        "reevaluate",
                        "--dataset",
                        "cfd1d1024",
                        "--checkpoints",
                        str(checkpoint),
                        "--device",
                        "cpu",
                    ]
                )
            self.assertEqual(status, 2)
            self.assertEqual(stdout, "")
            self.assertEqual(
                stderr,
                "ERROR reason=checkpoint_load_failed "
                "checkpoint=<external-results>/cfd1d1024/checkpoint "
                "dataset=cfd1d1024\n",
            )
            self.assertNotIn(secret_path_token, stderr)
            self.assertNotIn("loader leaked", stderr)
            self.assertNotIn("Traceback", stderr)


if __name__ == "__main__":
    unittest.main()
