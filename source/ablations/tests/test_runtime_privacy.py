from __future__ import annotations

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from ablations import runtime
from ablations.public import (PublicError, PublicStream, SafeArgumentParser,
                              error_record, public_main, public_path, public_payload)
from ablations.runtime_metadata import capture_runtime_metadata


SYNTHETIC_PATHS = (
    r"C:\Users\identity_alpha\private_bundle.pt",
    "/home/identity_alpha/private_bundle.pt",
    "/mnt/c/Users/identity_alpha/private_bundle.pt",
    r"\\identity_host\private_share\identity_alpha\private_bundle.pt",
)


class RuntimePrivacyTests(unittest.TestCase):
    def setUp(self):
        self.policy = runtime.get_runtime_policy()
        self.tempdir = tempfile.tempdir
        self.environment = patch.dict(os.environ)
        self.environment.start()

    def tearDown(self):
        runtime._RUNTIME_POLICY = self.policy
        tempfile.tempdir = self.tempdir
        self.environment.stop()

    def test_formal_preserves_inherited_threads_and_numerical_policy(self):
        os.environ.update({name: "7" for name in runtime.THREAD_ENVIRONMENT})
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        fake = types.SimpleNamespace(set_num_threads=lambda value: self.fail("implicit torch override"),
                                     set_num_interop_threads=lambda value: self.fail("implicit torch override"))
        runtime.configure_runtime(cpu=False, profile="formal")
        with patch.dict(sys.modules, {"torch": fake}):
            runtime.apply_torch_runtime()
        self.assertTrue(all(os.environ[name] == "7" for name in runtime.THREAD_ENVIRONMENT))
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(runtime.get_runtime_policy()["inherited_thread_environment"],
                         runtime.get_runtime_policy()["applied_thread_environment"])
        self.assertEqual(runtime.get_runtime_policy()["numerical_policy"], "inherited_without_override")
        self.assertIn("results", Path(os.environ["TMPDIR"]).parts)

    def test_diagnostic_cpu_is_one_thread_and_idempotent(self):
        os.environ.update({name: "9" for name in runtime.THREAD_ENVIRONMENT})
        runtime.configure_runtime(cpu=True, profile="diagnostic")
        first = runtime.get_runtime_policy()
        runtime.configure_runtime(cpu=True, profile="diagnostic")
        self.assertEqual(first, runtime.get_runtime_policy())
        self.assertEqual(first["inherited_thread_environment"]["OMP_NUM_THREADS"], "9")
        self.assertTrue(all(os.environ[name] == "1" for name in runtime.THREAD_ENVIRONMENT))
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "")
        with self.assertRaises(ValueError):
            runtime.configure_runtime(cpu=True, profile="diagnostic", threads=2)

    def test_explicit_formal_override_is_recorded_and_applied(self):
        calls = []
        fake = types.SimpleNamespace(set_num_threads=lambda value: calls.append(("intra", value)),
                                     get_num_interop_threads=lambda: 8,
                                     set_num_interop_threads=lambda value: calls.append(("inter", value)))
        runtime.configure_runtime(cpu=True, profile="formal", threads=2)
        with patch.dict(sys.modules, {"torch": fake}):
            runtime.apply_torch_runtime()
        self.assertEqual(calls, [("intra", 2), ("inter", 2)])
        self.assertEqual(runtime.get_runtime_policy()["thread_override"], 2)
        for value in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                runtime.configure_runtime(cpu=True, threads=value)

    def test_public_ids_drop_external_basename_for_all_path_styles(self):
        for value in SYNTHETIC_PATHS:
            with self.subTest(value=value):
                self.assertEqual(public_path(value, root="/allowed"), "external-artifact")
                serialized = json.dumps(public_payload({"artifact": value, "error": "Failed to read " + value}))
                for marker in ("identity_alpha", "private_bundle", "identity_host", "private_share"):
                    self.assertNotIn(marker, serialized)
        self.assertEqual(public_path("/allowed/results/summary.json", root="/allowed"), "results/summary.json")
        self.assertEqual(public_path(r"C:\allowed\results\summary.json", root=r"C:\allowed"), "results/summary.json")
        self.assertEqual(public_path("/allowed/../outside/secret", root="/allowed"), "external-artifact")
        source_url = "https://example.org/upstream/LICENSE"
        self.assertEqual(public_payload(source_url), source_url)

    def test_editable_assign_and_annotated_mapping_cover_all_project_prefixes(self):
        from ablations.isolated import audit_site
        from ablations.repository import AUDITED_PREFIXES
        with tempfile.TemporaryDirectory(dir=runtime.workspace_root() / "tmp") as directory:
            site = Path(directory)
            finder = site / "__editable__synthetic_finder.py"
            for prefix in AUDITED_PREFIXES:
                for annotation in ("", ": dict"):
                    external = str(runtime.workspace_root().parent / "unopened_external_project" / prefix)
                    finder.write_text(f"MAPPING{annotation} = {{{prefix!r}: {external!r}}}\n", encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, "editable mapping"):
                        audit_site(site, runtime.workspace_root())

    def test_failures_are_safe_and_keep_nonzero_exit(self):
        for value in SYNTHETIC_PATHS:
            with self.subTest(value=value):
                stream = io.StringIO()
                def fail():
                    raise FileNotFoundError(value)
                with contextlib.redirect_stderr(stream), self.assertRaises(SystemExit) as caught:
                    public_main(fail)
                self.assertEqual(caught.exception.code, 1)
                self.assertEqual(json.loads(stream.getvalue())["reason_code"], "ARTIFACT_NOT_FOUND")
                self.assertNotIn("identity_alpha", stream.getvalue())
        self.assertEqual(error_record(PublicError("CHECKPOINT_INVALID", seed=42))["seed"], 42)

    def test_argument_errors_and_progress_are_public_safe(self):
        parser = SafeArgumentParser(prog="ablations.test")
        parser.add_argument("--seed", type=int)
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), self.assertRaises(SystemExit) as caught:
            parser.parse_args(["--seed", SYNTHETIC_PATHS[0]])
        self.assertEqual(caught.exception.code, 2)
        self.assertNotIn("identity_alpha", errors.getvalue())
        output = io.StringIO()
        stream = PublicStream(output)
        stream.write("Epoch 17/500 loss=0.25 ")
        stream.write("artifact=" + SYNTHETIC_PATHS[0] + "\n")
        stream.flush()
        self.assertIn("Epoch 17/500 loss=0.25", output.getvalue())
        self.assertNotIn("identity_alpha", output.getvalue())

    def test_stale_or_invalid_baseline_environment_has_safe_error(self):
        from ablations.repository import baseline_profile
        for value in ("A", "B", *SYNTHETIC_PATHS):
            with self.subTest(value=value), patch.dict(os.environ, {"CAFE_ABLATION_BASELINE_PROFILE": value}):
                output = io.StringIO()
                with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as caught:
                    public_main(baseline_profile)
                self.assertEqual(caught.exception.code, 1)
                self.assertEqual(json.loads(output.getvalue())["reason_code"], "BASELINE_PROFILE_INVALID")
                self.assertNotIn("identity_alpha", output.getvalue())
                self.assertNotIn("private_bundle", output.getvalue())

    def test_metadata_records_observed_versions_without_personal_paths(self):
        runtime.configure_runtime(cpu=True, profile="diagnostic")
        metadata = capture_runtime_metadata(profile="diagnostic", device="cpu", seed=42)
        self.assertEqual(metadata["schema_version"], 1)
        self.assertIn("numpy", metadata["packages"])
        self.assertIsNone(metadata["cuda"]["gpu_product_names"])
        self.assertIn("cuda.gpu_product_names", metadata["unavailable"])
        self.assertEqual(metadata["seed_policy"]["selection"], "explicit_run_seed")
        serialized = json.dumps(metadata, allow_nan=False)
        self.assertNotIn(str(runtime.workspace_root()), serialized)
        self.assertNotIn("hostname", serialized)
        self.assertNotIn("executable", serialized)
        runtime.validate_runtime_metadata(metadata, formal=False)
        with self.assertRaises(PublicError):
            runtime.validate_runtime_metadata(metadata, formal=True)
        damaged = copy.deepcopy(metadata)
        del damaged["packages"]["numpy"]
        with self.assertRaises(PublicError):
            runtime.validate_runtime_metadata(damaged, formal=False)
        formal = copy.deepcopy(metadata)
        formal["profile"] = "formal"
        formal["runtime_policy"]["profile"] = "formal"
        formal["packages"]["torch"] = None
        formal["unavailable"]["packages.torch"] = "not_reported_by_runtime"
        runtime.validate_runtime_metadata(formal, formal=True)
        with self.assertRaises(PublicError) as caught:
            runtime.require_runtime_comparable(formal)
        self.assertEqual(caught.exception.reason_code, "RUNTIME_ACCURACY_EVIDENCE_INCOMPLETE")
        damaged = copy.deepcopy(metadata)
        damaged["cuda"]["cudnn"] = float("nan")
        with self.assertRaises(PublicError):
            runtime.validate_runtime_metadata(damaged, formal=False)


if __name__ == "__main__":
    unittest.main()
