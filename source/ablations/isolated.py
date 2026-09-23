"""Start with: python -I -S ablations/isolated.py --module validate ...

No .pth, sitecustomize, editable finder, or user-site code is executed. An
external project editable mapping is rejected even though site is disabled.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import runpy
import sys
import sysconfig


def audit_site(site: Path, workspace: Path) -> None:
    from ablations.repository import AUDITED_PREFIXES
    for path in sorted(site.glob("*.egg-link")):
        raise RuntimeError("Unverified editable install; choose an independent environment")
    for path in sorted(site.glob("__editable__*finder.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            target_names = ([node.target.id] if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                            else [target.id for target in node.targets if isinstance(target, ast.Name)]
                            if isinstance(node, ast.Assign) else [])
            if "MAPPING" in target_names:
                mapping = ast.literal_eval(node.value)
                if not isinstance(mapping, dict):
                    raise RuntimeError("Invalid editable project mapping")
                for name, location in mapping.items():
                    if name.split(".")[0] in AUDITED_PREFIXES:
                        # Do not stat or resolve the mapped path.
                        if not Path(os.path.abspath(location)).is_relative_to(workspace):
                            raise RuntimeError("External project editable mapping detected; stop this environment's test")
    for path in sorted(site.glob("*.pth")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "import ", "import\t")):
                target = Path(os.path.abspath(site / line))
                if not target.is_relative_to(site) and not target.is_relative_to(workspace):
                    raise RuntimeError("External .pth path detected; use an isolated environment")


def main() -> None:
    if not sys.flags.isolated or not sys.flags.no_site:
        raise SystemExit("Required invocation: python -I -S ablations/isolated.py ...")
    # Load the sibling pure-stdlib output helper before importing project code.
    from ablations.public import SafeArgumentParser
    parser = SafeArgumentParser(description=__doc__)
    parser.add_argument("--module", required=True, choices=("check", "parameters", "validate", "run", "train", "aggregate", "artifacts", "smoke", "native_diagnostics"))
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--dataset")
    options, remaining = parser.parse_known_args()
    # The package has one reviewed baseline; stale A settings must not select it.
    from ablations.repository import baseline_profile, require_dataset_baseline
    baseline_profile()
    if options.dataset is not None:
        require_dataset_baseline(options.dataset)
    package_parent = Path(__file__).absolute().parent.parent
    workspace = Path(os.path.abspath(options.workspace_root or package_parent))
    for item in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if item and not Path(os.path.abspath(item)).is_relative_to(workspace):
            raise RuntimeError("External PYTHONPATH detected; stop and use an independent shell")
    sites = {Path(sysconfig.get_path(name)) for name in ("purelib", "platlib")}
    for site in sites:
        audit_site(site, workspace)
    # Add dependencies without executing their startup files.
    sys.path.extend(str(site) for site in sorted(sites))
    sys.path.insert(0, str(package_parent))
    os.environ["CAFE_ABLATION_WORKSPACE_ROOT"] = str(workspace)
    os.environ["CAFE_ABLATION_REPOSITORY_ROOT"] = str(Path(os.path.abspath(options.repository_root or package_parent)))
    os.environ["CAFE_ABLATION_ISOLATED"] = "1"
    from ablations.runtime import configure_runtime
    from ablations.repository import activate_repository, resolve_repository_root, verify_base_sources

    root = resolve_repository_root()
    verify_base_sources(root)
    # Local process only; inherited Git settings cannot redirect source checks.
    for name in tuple(os.environ):
        if name.startswith("GIT_"):
            del os.environ[name]
    os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
    os.environ["GIT_CONFIG_GLOBAL"] = os.devnull
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    os.environ["GIT_CONFIG_COUNT"] = "2"
    for index, path in enumerate((root, root / "third_party" / "SirenFNO")):
        os.environ[f"GIT_CONFIG_KEY_{index}"] = "safe.directory"
        os.environ[f"GIT_CONFIG_VALUE_{index}"] = str(path)
    cpu = options.module not in {"train", "run", "smoke"} or not any(
        item == "cuda" or item == "--device=cuda" for item in remaining
    )
    profile = "formal" if (options.module in {"train", "run", "artifacts", "aggregate"}
                            or options.module == "check" and "--formal" in remaining) else "diagnostic"
    configure_runtime(cpu=cpu, profile=profile,
                      cache_area="results" if options.module == "smoke" else None)
    activate_repository(root)
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    if options.dataset is not None:
        remaining.extend(("--dataset", options.dataset))
    sys.argv = [f"ablations.{options.module}", *remaining]
    runpy.run_module(f"ablations.{options.module}", run_name="__main__")


if __name__ == "__main__":
    # -I excludes the script's parent from sys.path; only this packaged sibling
    # is needed to protect startup failures, including dependency/site audits.
    sys.path.insert(0, str(Path(__file__).absolute().parent.parent))
    from ablations.public import public_main
    public_main(main)
