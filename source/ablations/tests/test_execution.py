"""Execution-plan/reuse tests; process dispatch is tested without training."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ablations.contracts import DATASETS, DEFAULT_SEEDS, seed_value, validate_seeds
from ablations.public import PublicError
from ablations.repository import baseline_datasets, baseline_profile
from ablations.run import main as run_main, parse_args as parse_run
from ablations.runtime import workspace_root
from ablations.train import main as train_main, parse_args as parse_train, sigma_optimizer_record, skip_completed_run


class ExecutionTests(unittest.TestCase):
    def cli(self):
        return ["--dataset", baseline_datasets()[0], "--model-variant", "dense", "--condition", "full",
                "--data-root", str(workspace_root() / "tmp" / "test_synthetic_data")]

    def test_default_and_explicit_seeds_dispatch_in_order_separate_processes(self):
        selected = [None, [42], [0, 42, 73], [0, 42, 73, 108, 202], [7, 19, 101, 503],
                    [9, 8, 7, 6, 5, 4, 3], [2**32 - 1]]
        for seeds in selected:
            expected = list(DEFAULT_SEEDS) if seeds is None else seeds
            args = self.cli() + ([] if seeds is None else ["--seeds", *map(str, seeds)])
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch("ablations.run.subprocess.run") as dispatch:
                dispatch.return_value = subprocess.CompletedProcess([], 0)
                run_main(args + ["--threads", "2"])
            self.assertEqual(json.loads(output.getvalue())["requested_seeds"], expected)
            commands = [call.args[0] for call in dispatch.call_args_list]
            self.assertEqual([int(command[command.index("--seed") + 1]) for command in commands], expected)
            for command in commands:
                self.assertEqual(command[1:3], ["-I", "-S"])
                self.assertTrue(command[3].endswith("isolated.py"))
                self.assertEqual(command[command.index("--module") + 1], "train")
                self.assertNotIn("--baseline-profile", command)
                start, end = command.index("--requested-seeds") + 1, command.index("--data-root")
                self.assertEqual([int(item) for item in command[start:end]], expected)
                self.assertEqual(command[command.index("--threads") + 1], "2")

    def test_all_seven_default_seed_plans_and_dispatch_have_one_b_baseline(self):
        for dataset in DATASETS:
            args = self.cli()
            args[args.index("--dataset") + 1] = dataset
            for dry_run in (True, False):
                with self.subTest(dataset=dataset, dry_run=dry_run):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output), patch("ablations.run.subprocess.run") as dispatch:
                        dispatch.return_value = subprocess.CompletedProcess([], 0)
                        run_main(args + (["--dry-run"] if dry_run else []))
                    record = json.loads(output.getvalue())
                    self.assertEqual(record["dataset"], dataset)
                    self.assertEqual(record["baseline_profile"], "C")
                    self.assertEqual(record["base_source_commit"], "5d1ec68dfe2774b74eec45f402b6f7f028a47b3e")
                    self.assertEqual(record["requested_seeds"], list(DEFAULT_SEEDS))
                    self.assertEqual(record["process_count"], len(DEFAULT_SEEDS))
                    self.assertEqual(dispatch.call_count, 0 if dry_run else len(DEFAULT_SEEDS))
                    for call in dispatch.call_args_list:
                        command = call.args[0]
                        self.assertNotIn("--baseline-profile", command)
                        self.assertEqual(command[command.index("--dataset") + 1], dataset)

    def test_public_run_and_train_reject_unknown_dataset_before_dispatch(self):
        args = self.cli()
        args[args.index("--dataset") + 1] = "unknown_dataset"
        for diagnostic_flags in ([], ["--diagnostic-source"]):
            with self.subTest(diagnostic=bool(diagnostic_flags)):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), patch("ablations.run.subprocess.run") as dispatch:
                    with self.assertRaises(SystemExit):
                        run_main(args + diagnostic_flags)
                    dispatch.assert_not_called()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train_main(args + ["--seed", "42", *diagnostic_flags])

    def test_stale_a_environment_and_removed_profile_flag_are_rejected(self):
        args = self.cli()
        with patch.dict(os.environ, {"CAFE_ABLATION_BASELINE_PROFILE": "A"}):
            for entrypoint, suffix in ((run_main, ["--dry-run"]), (train_main, ["--seed", "42"])):
                with self.subTest(entrypoint=entrypoint.__module__), self.assertRaises(PublicError) as caught:
                    entrypoint(args + suffix)
                self.assertEqual(caught.exception.reason_code, "BASELINE_PROFILE_INVALID")
        for parser, suffix in ((parse_run, []), (parse_train, ["--seed", "42"])):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser(self.cli() + suffix + ["--baseline-profile", "B"])

    def test_dry_run_does_not_start_child_and_failure_stops_sweep(self):
        with contextlib.redirect_stdout(io.StringIO()), patch("ablations.run.subprocess.run") as dispatch:
            run_main(self.cli() + ["--dry-run"])
            dispatch.assert_not_called()
        with contextlib.redirect_stdout(io.StringIO()), patch("ablations.run.subprocess.run") as dispatch:
            dispatch.return_value = subprocess.CompletedProcess([], 7)
            with self.assertRaises(PublicError) as caught:
                run_main(self.cli())
            self.assertEqual(caught.exception.reason_code, "SEED_PROCESS_FAILED")
            self.assertEqual(dispatch.call_count, 1)

    def test_seed_validation_matrix_and_single_run_plan(self):
        for seeds in ([], [0, 0], [True], [1.5], ["1"], [-1], [2**32], None, "0 42"):
            with self.subTest(invalid=repr(seeds)), self.assertRaises(ValueError):
                validate_seeds(seeds)
        for seed in (True, False, 1.1, "NaN", "1.0", "abc", "", " 42 ", -1, 2**32):
            with self.subTest(seed=repr(seed)), self.assertRaises(ValueError):
                seed_value(seed)
        for tokens in (["--seeds"], ["--seeds", "1", "1"], ["--seeds", "true"],
                       ["--seeds", "2.5"], ["--seeds", "-1"], ["--seeds", str(2**32)]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_run(self.cli() + tokens)
        self.assertEqual(parse_train(self.cli() + ["--seed", "19"]).requested_seeds, [19])
        self.assertEqual(parse_train(self.cli() + ["--seed", "19", "--requested-seeds", "7", "19"]).seed, 19)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_train(self.cli() + ["--seed", "19", "--requested-seeds", "7"])

    def test_real_completed_fixture_reuse_and_incomplete_refusal(self):
        from ablations.tests.artifact_fixtures import make_artifact
        from ablations.repository import sha256_file
        with tempfile.TemporaryDirectory() as directory:
            path, contract = make_artifact(Path(directory), seed=42)
            summary = json.loads(path.read_text(encoding="utf-8"))
            hashes = {p.name: sha256_file(p) for p in path.parent.iterdir() if p.is_file()}
            arguments = dict(dataset="burgers1d", model_variant="dense", condition="full", seed=42,
                             diagnostic=True, test_contract=contract,
                             expected_context={"source": summary["source"], "runtime": summary["runtime"]})
            # Sweep plan is not a skip/compatibility argument: changing how many
            # sibling seeds are requested must not change this single run.
            for plan in ([0, 42, 73], [0, 42, 73, 108, 202], [42]):
                self.assertIn(42, plan)
                self.assertTrue(skip_completed_run(path.parent, **arguments))
            self.assertEqual(hashes, {p.name: sha256_file(p) for p in path.parent.iterdir() if p.is_file()})
            (path.parent / "training_log.csv").unlink()
            with self.assertRaises(PublicError) as caught:
                skip_completed_run(path.parent, **arguments)
            self.assertEqual(caught.exception.reason_code, "INCOMPLETE_RUN_REFUSES_OVERWRITE")

    def test_sigma_optimizer_record_is_actual_and_does_not_consume_rng(self):
        import torch
        from ablations.models import build_model_from_kwargs, preserved_rng_state
        from ablations.tests.test_models import tiny_config
        with preserved_rng_state():
            built = build_model_from_kwargs(spatial_dim=1, configuration=tiny_config(1), condition="learnable_sigma")
            optimizer = torch.optim.AdamW(built.model.parameters(), lr=0.003, weight_decay=0.007)
            before = torch.get_rng_state().clone()
            record = sigma_optimizer_record(built.model, optimizer)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            expected = sum(p.numel() for name, p in built.model.named_parameters() if name.endswith("log_sigma"))
            self.assertGreater(expected, 0)
            self.assertEqual(record["trainable_scalar_count"], expected)
            self.assertFalse(record["gaussian_G_trainable"])
            for encoder in record["encoders"]:
                self.assertTrue(encoder["optimizer_included"])
                self.assertEqual(encoder["initial_learning_rate"], 0.003)
                self.assertEqual(encoder["weight_decay"], 0.007)
            without_sigma = torch.optim.AdamW([p for n, p in built.model.named_parameters() if not n.endswith("log_sigma")])
            with self.assertRaises(PublicError):
                sigma_optimizer_record(built.model, without_sigma)


if __name__ == "__main__":
    unittest.main()
