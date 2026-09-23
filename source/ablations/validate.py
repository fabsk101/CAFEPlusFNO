"""Small CPU-only validation with explicit PASS/FAIL/NOT RUN accounting."""
from __future__ import annotations

from collections import Counter
import json
import unittest
import re
from pathlib import Path

from .repository import PACKAGE_ROOT, resolve_repository_root, activate_repository, source_state, import_audit
from .runtime import configure_runtime, output_path
from .public import SafeArgumentParser, emit_json, public_main, public_payload, error_record, safe_text
from .runtime_metadata import capture_runtime_metadata
from .contracts import DATASETS


REQUIRED_NATIVE_STAGES = (
    "native_training", "native_loss", "native_evaluation",
    "checkpoint_write", "strict_reload", "aggregate",
)


def suite_inventory(suite):
    """Capture discoverable methods before unittest consumes its suite.

    Subtests are dynamic and are intentionally not included in this inventory.
    A setUpClass/setUpModule skip can prevent inventoried methods from starting.
    """
    records = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            records.extend(suite_inventory(item))
        else:
            records.append({"test": safe_text(item.id()),
                            "class": type(item).__module__ + "." + type(item).__qualname__,
                            "module": type(item).__module__})
    return records


class PublicTestResult(unittest.TextTestResult):
    """Count methods, observed subtests and fixture events independently."""

    def __init__(self, *args, inventory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inventory = inventory
        self.method_records = []
        self.subtest_records = []
        self.fixture_events = []
        self._active_methods = {}

    def _exc_info_to_string(self, err, test):
        return json.dumps(error_record(err[1]), sort_keys=True)

    @staticmethod
    def _safe_skip_reason(reason):
        # Do not publish third-party exceptions or arbitrary SkipTest messages.
        return reason if isinstance(reason, str) and re.fullmatch(r"NOT_RUN_[A-Z0-9_]+", reason) else "NOT_RUN_DEPENDENCY_OR_RUNTIME_UNAVAILABLE"

    def startTest(self, test):
        record = {"test": safe_text(test.id()), "status": "RUNNING"}
        self.method_records.append(record)
        self._active_methods[id(test)] = record
        super().startTest(test)

    def stopTest(self, test):
        record = self._active_methods.pop(id(test), None)
        if record is not None and record["status"] == "RUNNING":
            record["status"] = "INCOMPLETE"
        super().stopTest(test)

    def _set_method_outcome(self, test, status):
        record = self._active_methods.get(id(test))
        if record is None:
            return False
        # A successful later subtest must not erase an earlier failed one.
        priority = {"RUNNING": 0, "PASS": 1, "EXPECTED_FAILURE": 1,
                    "NOT_RUN": 2, "PARTIAL": 2, "FAIL": 3,
                    "UNEXPECTED_SUCCESS": 3, "ERROR": 4, "INCOMPLETE": 4}
        if priority[status] >= priority[record["status"]]:
            record["status"] = status
        return True

    def _fixture_event(self, test, status, reason):
        # unittest reports fixture failures/skips as _ErrorHolder objects.
        identity = safe_text(test.id())
        match = re.fullmatch(r"(setUpClass|tearDownClass|setUpModule|tearDownModule) \((.+)\)", identity)
        phase, owner = match.groups() if match else ("unknown", identity)
        scope = "class" if phase.endswith("Class") else "module" if phase.endswith("Module") else "unknown"
        self.fixture_events.append({"phase": phase, "scope": scope, "owner": owner,
                                    "status": status, "reason_code": reason})

    def addSuccess(self, test):
        self._set_method_outcome(test, "PASS")
        super().addSuccess(test)

    def addFailure(self, test, err):
        if not self._set_method_outcome(test, "FAIL"):
            self._fixture_event(test, "FAIL", error_record(err[1])["reason_code"])
        super().addFailure(test, err)

    def addError(self, test, err):
        if not self._set_method_outcome(test, "ERROR"):
            self._fixture_event(test, "ERROR", error_record(err[1])["reason_code"])
        super().addError(test, err)

    def addSkip(self, test, reason):
        safe_reason = self._safe_skip_reason(reason)
        parent = getattr(test, "test_case", None)
        if parent is not None and id(parent) in self._active_methods:
            self.subtest_records.append({"test": safe_text(parent.id()), "status": "NOT_RUN",
                                         "reason_code": safe_reason})
            self._set_method_outcome(parent, "PARTIAL")
        elif self._set_method_outcome(test, "NOT_RUN"):
            self._active_methods[id(test)]["reason_code"] = safe_reason
        else:
            self._fixture_event(test, "NOT_RUN", safe_reason)
        super().addSkip(test, safe_reason)

    def addSubTest(self, test, subtest, err):
        if err is None:
            status = "PASS"
        else:
            status = "FAIL" if issubclass(err[0], test.failureException) else "ERROR"
            self._set_method_outcome(test, status)
        record = {"test": safe_text(test.id()), "status": status}
        if err is not None:
            record["reason_code"] = error_record(err[1])["reason_code"]
        self.subtest_records.append(record)
        super().addSubTest(test, subtest, err)

    def addExpectedFailure(self, test, err):
        self._set_method_outcome(test, "EXPECTED_FAILURE")
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):
        self._set_method_outcome(test, "UNEXPECTED_SUCCESS")
        super().addUnexpectedSuccess(test)

    def accounting(self):
        """Return observed counts; never infer method passes from skip events."""
        outcomes = Counter(record["status"] for record in self.method_records)
        subtests = Counter(record["status"] for record in self.subtest_records)
        unstarted = []
        consumed = Counter(record["test"] for record in self.method_records)
        for entry in self.inventory or ():
            if consumed[entry["test"]]:
                consumed[entry["test"]] -= 1
                continue
            reason = "NOT_STARTED"
            for fixture in self.fixture_events:
                if fixture["phase"] in {"setUpClass", "setUpModule"} and entry.get(fixture["scope"]) == fixture["owner"]:
                    reason = "FIXTURE_SKIP" if fixture["status"] == "NOT_RUN" else "FIXTURE_ERROR"
                    break
            unstarted.append({"test": entry["test"], "reason": reason})
        return {
            "schema_version": 1,
            "legacy_tests_run_semantics": "unittest_testsRun_started_method_slots_including_method_skips",
            "method_counts": {
                "discovered": None if self.inventory is None else len(self.inventory),
                "started": len(self.method_records), "passed": outcomes["PASS"],
                "failed": outcomes["FAIL"], "errors": outcomes["ERROR"],
                "not_run": outcomes["NOT_RUN"], "partial": outcomes["PARTIAL"],
                "expected_failures": outcomes["EXPECTED_FAILURE"],
                "unexpected_successes": outcomes["UNEXPECTED_SUCCESS"],
                "incomplete": outcomes["INCOMPLETE"],
                "not_started": None if self.inventory is None else len(unstarted),
                "not_started_due_to_fixture_skip": sum(row["reason"] == "FIXTURE_SKIP" for row in unstarted),
                "not_started_due_to_fixture_error": sum(row["reason"] == "FIXTURE_ERROR" for row in unstarted),
            },
            "subtest_counts": {"observed": len(self.subtest_records), "passed": subtests["PASS"],
                               "failed": subtests["FAIL"], "errors": subtests["ERROR"],
                               "not_run": subtests["NOT_RUN"],
                               "planned": None, "planned_unavailable_reason": "subtests_are_created_during_execution"},
            "fixture_counts": {"events": len(self.fixture_events),
                               "class_skips": sum(row["scope"] == "class" and row["status"] == "NOT_RUN" for row in self.fixture_events),
                               "module_skips": sum(row["scope"] == "module" and row["status"] == "NOT_RUN" for row in self.fixture_events),
                               "failures_or_errors": sum(row["status"] in {"FAIL", "ERROR"} for row in self.fixture_events)},
            "methods": list(self.method_records), "subtests": list(self.subtest_records),
            "fixture_events": list(self.fixture_events), "not_started_methods": unstarted,
        }


def native_adapter_requirements(evidence):
    """Require explicit completed Full diagnostics, never test-name inference."""
    datasets = []
    for dataset in DATASETS:
        record = evidence.get(dataset) if isinstance(evidence, dict) else None
        if not isinstance(record, dict):
            datasets.append({"dataset": dataset, "status": "NOT_RUN", "executed": False,
                             "reason_code": "NATIVE_ADAPTER_DIAGNOSTIC_NOT_EXECUTED",
                             "missing_stages": list(REQUIRED_NATIVE_STAGES)})
            continue
        stages = record.get("stages")
        missing = [name for name in REQUIRED_NATIVE_STAGES
                   if not isinstance(stages, dict) or stages.get(name) is not True]
        identity_ok = record.get("dataset") == dataset and record.get("condition") == "full"
        complete = identity_ok and record.get("status") == "PASS" and record.get("executed") is True and not missing
        status = "PASS" if complete else "NOT_RUN" if record.get("status") == "NOT_RUN" and record.get("executed") is False else "FAIL"
        reason = record.get("reason_code")
        if not isinstance(reason, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", reason):
            reason = "NATIVE_ADAPTER_EVIDENCE_INCOMPLETE"
        row = {"dataset": dataset, "status": status, "executed": record.get("executed") is True,
               "missing_stages": missing}
        if not complete:
            row["reason_code"] = reason
        datasets.append(row)
    counts = Counter(row["status"] for row in datasets)
    return {"status": "FAIL" if counts["FAIL"] else "NOT_RUN" if counts["NOT_RUN"] else "PASS",
            "required_datasets": list(DATASETS), "required_condition": "full",
            "required_stages": list(REQUIRED_NATIVE_STAGES), "required_count": len(DATASETS),
            "passed": counts["PASS"], "failed": counts["FAIL"], "not_run": counts["NOT_RUN"],
            "datasets": datasets}


def validation_status(result, requirements, *, require_adapters=False):
    accounting = result.accounting()
    methods = accounting["method_counts"]
    if (not result.wasSuccessful() or methods["incomplete"]
            or requirements["status"] == "FAIL"
            or require_adapters and requirements["status"] != "PASS"):
        return "FAIL"
    if result.skipped or methods["not_started"] or requirements["status"] != "PASS":
        return "PASS_WITH_NOT_RUN"
    return "PASS"


def main(argv=None):
    parser = SafeArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-adapters", action="store_true",
                        help="Require successful executed Full native diagnostics for all seven datasets")
    args = parser.parse_args(argv)
    root = resolve_repository_root(args.repository_root)
    activate_repository(root)
    provenance = source_state(root, formal=False)
    configure_runtime(cpu=True, profile="diagnostic")
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    from .native_diagnostics import reset_execution_evidence, execution_evidence
    reset_execution_evidence()
    suite = unittest.defaultTestLoader.discover(str(PACKAGE_ROOT / "tests"), top_level_dir=str(PACKAGE_ROOT.parent))
    inventory = suite_inventory(suite)

    def result_factory(*result_args, **result_kwargs):
        return PublicTestResult(*result_args, inventory=inventory, **result_kwargs)

    result = unittest.TextTestRunner(verbosity=2, resultclass=result_factory).run(suite)
    evidence = execution_evidence()
    requirements = native_adapter_requirements(evidence)
    report = {
        "status": validation_status(result, requirements, require_adapters=args.require_adapters),
        "tests_run": result.testsRun, "failures": [(str(test), trace) for test, trace in result.failures],
        "errors": [(str(test), trace) for test, trace in result.errors],
        "not_run": [(str(test), why) for test, why in result.skipped],
        "accounting": result.accounting(),
        "require_adapters": args.require_adapters,
        "required_native_adapters": requirements,
        "native_adapter_evidence": evidence,
        "torch_version": str(torch.__version__),
        "runtime": capture_runtime_metadata(profile="diagnostic", device="cpu"),
        "device": "cpu", "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "source": provenance, "imports": import_audit(root),
        "gpu_training": "NOT RUN", "500_epoch_training": "NOT RUN",
    }
    report = public_payload(report)
    if args.report:
        path = output_path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    compact = {key: report[key] for key in ("status", "tests_run", "torch_version", "device", "not_run",
                                           "required_native_adapters")}
    compact["accounting"] = {key: report["accounting"][key]
                             for key in ("method_counts", "subtest_counts", "fixture_counts")}
    emit_json(compact)
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    public_main(main)
