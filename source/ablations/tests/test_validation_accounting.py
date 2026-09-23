"""Observed unittest accounting and the seven-native-adapter execution gate."""
from __future__ import annotations

from copy import deepcopy
import io
import json
import unittest

from ablations.contracts import DATASETS
from ablations.validate import (
    PublicTestResult, REQUIRED_NATIVE_STAGES, native_adapter_requirements,
    suite_inventory, validation_status,
)


def run_cases(*classes):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes)
    inventory = suite_inventory(suite)
    return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0,
        resultclass=lambda *args, **kwargs: PublicTestResult(*args, inventory=inventory, **kwargs)).run(suite)


def completed_native_evidence():
    # Deliberately local input for testing the gate, never the execution ledger.
    return {dataset: {"dataset": dataset, "condition": "full", "status": "PASS", "executed": True,
                      "stages": {stage: True for stage in REQUIRED_NATIVE_STAGES}}
            for dataset in DATASETS}


class ValidationAccountingTests(unittest.TestCase):
    def test_method_subtest_and_skip_counts_are_separate(self):
        class Mixed(unittest.TestCase):
            def test_pass(self):
                self.assertTrue(True)

            def test_skip_method(self):
                self.skipTest("C:/Users/SyntheticIdentity/private-input.bin")

            def test_partial_subtests(self):
                for index in range(3):
                    with self.subTest(index=index):
                        if index == 1:
                            self.skipTest("NOT_RUN_EXPECTED_OPTIONAL_CASE")

            def test_failed_subtests(self):
                for index in range(3):
                    with self.subTest(index=index):
                        self.assertEqual(index, 0)

            def test_import_error(self):
                raise ImportError("broken implementation C:/Users/SyntheticIdentity/private-module.py")

        result = run_cases(Mixed)
        counts = result.accounting()
        methods = counts["method_counts"]
        self.assertEqual(result.testsRun, 5)
        self.assertEqual((methods["discovered"], methods["started"], methods["passed"],
                          methods["failed"], methods["errors"], methods["not_run"], methods["partial"]),
                         (5, 5, 1, 1, 1, 1, 1))
        self.assertEqual(counts["subtest_counts"]["observed"], 6)
        self.assertEqual(counts["subtest_counts"]["passed"], 3)
        self.assertEqual(counts["subtest_counts"]["failed"], 2)
        self.assertEqual(counts["subtest_counts"]["not_run"], 1)
        self.assertEqual(len(result.skipped), 2)
        self.assertNotEqual(result.testsRun - len(result.skipped), methods["passed"])
        serialized = json.dumps(counts) + json.dumps(result.errors[0][1])
        self.assertNotIn("SyntheticIdentity", serialized)
        self.assertNotIn("private-module", serialized)
        self.assertEqual(counts["fixture_counts"]["events"], 0)

    def test_class_setup_skip_accounts_for_unstarted_methods(self):
        class MissingFixture(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise unittest.SkipTest("NOT_RUN_EXPECTED_CLASS_DEPENDENCY")

            def test_one(self):
                self.fail("must not execute")

            def test_two(self):
                self.fail("must not execute")

        result = run_cases(MissingFixture)
        accounting = result.accounting()
        methods = accounting["method_counts"]
        self.assertEqual(result.testsRun, 0)
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(methods["discovered"], 2)
        self.assertEqual(methods["started"], 0)
        self.assertEqual(methods["not_run"], 0)
        self.assertEqual(methods["not_started"], 2)
        self.assertEqual(methods["not_started_due_to_fixture_skip"], 2)
        self.assertEqual(accounting["fixture_counts"]["class_skips"], 1)

    def test_class_decorator_skips_method_slots_without_fixture_event(self):
        @unittest.skip("NOT_RUN_EXPECTED_DECORATED_CLASS")
        class Disabled(unittest.TestCase):
            def test_one(self):
                self.fail("must not execute")

            def test_two(self):
                self.fail("must not execute")

        accounting = run_cases(Disabled).accounting()
        self.assertEqual(accounting["method_counts"]["started"], 2)
        self.assertEqual(accounting["method_counts"]["not_run"], 2)
        self.assertEqual(accounting["method_counts"]["not_started"], 0)
        self.assertEqual(accounting["fixture_counts"]["class_skips"], 0)

    def test_fixture_import_error_is_error_not_missing_dependency_skip(self):
        class BrokenFixture(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise ImportError("actual implementation bug")

            def test_one(self):
                self.fail("must not execute")

        result = run_cases(BrokenFixture)
        accounting = result.accounting()
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(result.skipped, [])
        self.assertEqual(accounting["fixture_counts"]["failures_or_errors"], 1)
        self.assertEqual(accounting["method_counts"]["not_started_due_to_fixture_error"], 1)
        self.assertEqual(validation_status(result, native_adapter_requirements(completed_native_evidence())), "FAIL")

    def test_required_native_evidence_demands_all_seven_full_paths(self):
        evidence = completed_native_evidence()
        complete = native_adapter_requirements(evidence)
        self.assertEqual(complete["status"], "PASS")
        self.assertEqual(complete["passed"], len(DATASETS))
        self.assertEqual(complete["required_count"], 7)
        mutations = (
            lambda records: records["darcy"].update(executed=False),
            lambda records: records["darcy"].update(condition="fourier_only"),
            lambda records: records["darcy"].update(dataset="ns2d"),
            lambda records: records["darcy"]["stages"].pop("strict_reload"),
            lambda records: records["darcy"]["stages"].update(native_training="true"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = deepcopy(evidence)
                mutate(changed)
                self.assertEqual(native_adapter_requirements(changed)["status"], "FAIL")
        absent = deepcopy(evidence)
        absent.pop("darcy")
        requirements = native_adapter_requirements(absent)
        self.assertEqual((requirements["status"], requirements["passed"], requirements["not_run"]),
                         ("NOT_RUN", 6, 1))

    def test_all_run_can_pass_and_missing_native_evidence_is_never_inferred(self):
        class SuccessfulImportSignatureOnly(unittest.TestCase):
            def test_pass(self):
                self.assertTrue(True)

        result = run_cases(SuccessfulImportSignatureOnly)
        missing = native_adapter_requirements({})
        self.assertEqual(validation_status(result, missing), "PASS_WITH_NOT_RUN")
        self.assertEqual(validation_status(result, missing, require_adapters=True), "FAIL")
        complete = native_adapter_requirements(completed_native_evidence())
        self.assertEqual(validation_status(result, complete, require_adapters=True), "PASS")


if __name__ == "__main__":
    unittest.main()
