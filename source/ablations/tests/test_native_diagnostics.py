"""Executed native adapter evidence, separate from import/signature checks."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ablations.contracts import DATASETS
from ablations.native_diagnostics import ADDITIONAL_CASES, REQUIRED_STAGES, run_native_diagnostic


class NativeDiagnosticExecutionTests(unittest.TestCase):
    def execute_case(self, dataset, condition):
        with tempfile.TemporaryDirectory(prefix="native-adapter-") as directory:
            record = run_native_diagnostic(dataset, condition=condition, output_root=Path(directory))
        if record["status"] == "NOT_RUN":
            self.skipTest(record["reason_code"])
        self.assertEqual(record["status"], "PASS")
        self.assertTrue(record["executed"])
        self.assertTrue(all(record["stages"][stage] for stage in REQUIRED_STAGES))
        self.assertEqual(record["loader_execution"]["train"]["samples"], 2)
        self.assertEqual(record["completed_epochs"], 1)
        self.assertEqual(record["parent_epochs"], 500)
        self.assertGreater(record["gradients"]["finite_tensors"], 0)
        self.assertGreater(record["gradients"]["nonzero_tensors"], 0)
        self.assertGreater(record["changed_parameter_tensors"], 0)
        self.assertTrue(record["finite_metrics"])
        self.assertEqual(record["runtime"]["device"], "cpu")


def _case(dataset, condition):
    def test(self):
        self.execute_case(dataset, condition)
    test.__doc__ = f"TEST/DIAGNOSTIC: {dataset} Dense {condition}, one actual native CPU epoch."
    return test


for _dataset in DATASETS:
    setattr(NativeDiagnosticExecutionTests, "test_native_full_" + _dataset, _case(_dataset, "full"))
for _dataset, _condition in ADDITIONAL_CASES:
    setattr(NativeDiagnosticExecutionTests, "test_native_variant_" + _dataset + "_" + _condition,
            _case(_dataset, _condition))


if __name__ == "__main__":
    unittest.main()
