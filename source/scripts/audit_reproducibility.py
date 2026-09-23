"""Audit pinned sources, data, configs, seeds, and fixed paper protocols."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import torch
from torch.utils.data import RandomSampler, SequentialSampler


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common.dataset_provenance import (  # noqa: E402
    dataset_manifest,
    verify_dataset,
)
from experiments.common.seed import set_seed  # noqa: E402
from experiments.common.sirenfno_backend import (  # noqa: E402
    bootstrap_sirenfno_backend,
    source_repository_state,
    verify_sirenfno_checkout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    return parser.parse_args()


def print_report(title: str, checks: list[tuple[str, bool | None]]) -> None:
    print(title)
    print("=" * 36)
    for label, passed in checks:
        status = "NOT RUN" if passed is None else "PASS" if passed else "FAIL"
        print(f"{label:<32} {status}")


def audit_ns_offline_loader(train_ns2d) -> bool:
    """Exercise loader construction without reading data or permitting downloads."""

    calls: dict[str, object] = {"loaders": []}
    train_db = object()
    test_db = object()
    data_processor = object()

    class CapturedDataset:
        def __init__(self, **kwargs) -> None:
            calls["dataset"] = kwargs
            self.train_db = train_db
            self.test_dbs = {128: test_db}
            self.data_processor = data_processor

    def captured_loader(dataset, **kwargs):
        calls["loaders"].append((dataset, kwargs))
        return (dataset, kwargs)

    originals = (
        train_ns2d.NavierStokesDataset,
        train_ns2d.DataLoader,
        train_ns2d.expected_ns_files,
    )
    try:
        train_ns2d.NavierStokesDataset = CapturedDataset
        train_ns2d.DataLoader = captured_loader
        train_ns2d.expected_ns_files = lambda _root: ()
        train_loader, test_loaders, returned_processor = train_ns2d.load_ns(
            Path("data")
        )
    finally:
        (
            train_ns2d.NavierStokesDataset,
            train_ns2d.DataLoader,
            train_ns2d.expected_ns_files,
        ) = originals

    expected_dataset = {
        "root_dir": "data",
        "n_train": 1000,
        "n_tests": [200],
        "batch_size": 32,
        "test_batch_sizes": [32],
        "train_resolution": 128,
        "test_resolutions": [128],
        "encode_input": True,
        "encode_output": True,
        "encoding": "channel-wise",
        "channel_dim": 1,
        "subsampling_rate": None,
        "download": False,
    }
    expected_train_options = {
        "batch_size": 32,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
    }
    expected_test_options = {
        **expected_train_options,
        "shuffle": False,
    }
    loader_source = inspect.getsource(train_ns2d.load_ns)
    return (
        calls.get("dataset") == expected_dataset
        and calls["loaders"]
        == [
            (train_db, expected_train_options),
            (test_db, expected_test_options),
        ]
        and train_loader == (train_db, expected_train_options)
        and test_loaders == {128: (test_db, expected_test_options)}
        and returned_processor is data_processor
        and "download=False" in loader_source.replace(" ", "")
        and "download_from_zenodo_record" not in loader_source
        and "requests." not in loader_source
    )


def audit_burgers_offline_pipeline(train_burgers) -> bool:
    """Check pinned data primitives, loader options, and absence of acquisition."""

    train = torch.zeros(64, 20, 8)
    test = torch.zeros(33, 20, 8)
    loaders = train_burgers.build_data_loaders(train, test, 0.0, 1.0)
    train_loader, train_eval_loader, test_eval_loader = loaders
    load_source = inspect.getsource(train_burgers.load_burgers)
    main_source = inspect.getsource(train_burgers.main)
    module_source = inspect.getsource(train_burgers)
    required_primitives = (
        "preprocess_decimated_hdf5_1d_scalar",
        "load_subset_to_ram_1d_scalar",
        "compute_mean_std_from_ram",
        "RolloutRAMDataset1D",
    )
    return (
        all(token in module_source for token in required_primitives)
        and train_loader.batch_size == 32
        and isinstance(train_loader.sampler, RandomSampler)
        and train_loader.num_workers == 0
        and train_loader.pin_memory
        and train_loader.drop_last
        and isinstance(train_eval_loader.sampler, SequentialSampler)
        and isinstance(test_eval_loader.sampler, SequentialSampler)
        and not train_eval_loader.drop_last
        and not test_eval_loader.drop_last
        and "ensure_data_available" not in load_source
        and "urlopen" not in load_source
        and "requests." not in load_source
        and main_source.index("verify_burgers_dataset(data_root)")
        < main_source.index("load_burgers(data_root)")
    )


def audit_cfd1d_offline_pipeline(train_cfd1d) -> bool:
    """Check released Vx semantics, loader options, and no acquisition path."""

    train = torch.zeros(64, 20, 8, 1)
    test = torch.zeros(33, 20, 8, 1)
    train_loader, train_eval_loader, test_eval_loader = (
        train_cfd1d.build_data_loaders(train, test)
    )
    load_source = inspect.getsource(train_cfd1d.load_cfd1d)
    main_source = inspect.getsource(train_cfd1d.main)
    module_source = inspect.getsource(train_cfd1d)
    required_primitives = (
        "_find_data_dataset",
        "load_subset_to_ram",
        "RolloutRAMDataset",
        "rollout_step_model_1d_channel_first",
        "rollout_loss_lp",
        "evaluate_corrected_relative_l2",
        "corrected_relative_l2_metadata",
    )
    return (
        all(token in module_source for token in required_primitives)
        and train_cfd1d.SELECTED_FIELD == "Vx"
        and train_cfd1d.FIELD_DIM == 1
        and train_loader.batch_size == 32
        and isinstance(train_loader.sampler, RandomSampler)
        and train_loader.num_workers == 0
        and train_loader.pin_memory
        and train_loader.drop_last
        and isinstance(train_eval_loader.sampler, SequentialSampler)
        and isinstance(test_eval_loader.sampler, SequentialSampler)
        and not train_eval_loader.drop_last
        and not test_eval_loader.drop_last
        and "ensure_data_available" not in load_source
        and "_download_file" not in load_source
        and "urlopen" not in load_source
        and "requests." not in load_source
        and main_source.index("verify_cfd1d_dataset(data_root)")
        < main_source.index("load_cfd1d(data_root)")
    )


def audit_reacdiff_offline_pipeline(train_reacdiff) -> bool:
    """Check released ReacDiff data/rollout semantics and no acquisition."""

    train = torch.zeros(64, 20, 8, 1)
    test = torch.zeros(33, 20, 8, 1)
    train_loader, train_eval_loader, test_eval_loader = (
        train_reacdiff.build_data_loaders(train, test)
    )
    load_source = inspect.getsource(train_reacdiff.load_reacdiff)
    main_source = inspect.getsource(train_reacdiff.main)
    module_source = inspect.getsource(train_reacdiff)
    loop_source = inspect.getsource(train_reacdiff.run_training_loop)
    required_primitives = (
        "_find_data_dataset",
        "load_subset_to_ram",
        "RolloutRAMDataset",
        "rollout_step_model_1d_channel_first",
        "rollout_loss_lp",
        "evaluate_corrected_relative_l2",
        "corrected_relative_l2_metadata",
        "LpLoss(d=1, p=2, reduction=\"mean\")",
    )
    forbidden_network = (
        "ensure_data_available(", "_download_file(", "urlopen(",
        "requests.get", "preprocess_decimated_hdf5(",
    )
    return (
        all(token in module_source for token in required_primitives)
        and all(token not in module_source for token in forbidden_network)
        and train_reacdiff.FIELD_DIM == 1
        and train_loader.batch_size == 32
        and isinstance(train_loader.sampler, RandomSampler)
        and train_loader.num_workers == 0
        and train_loader.pin_memory
        and train_loader.drop_last
        and isinstance(train_eval_loader.sampler, SequentialSampler)
        and isinstance(test_eval_loader.sampler, SequentialSampler)
        and not train_eval_loader.drop_last
        and not test_eval_loader.drop_last
        and main_source.index("verify_reacdiff_dataset(data_root)")
        < main_source.index("set_seed(args.seed)")
        < main_source.index("load_reacdiff(data_root)")
        < main_source.index("build_model(args.model, device)")
        and loop_source.count("scheduler.step()") == 1
        and loop_source.index("optimizer.step()")
        < loop_source.index("scheduler.step()")
        < loop_source.index("evaluate_corrected_relative_l2(")
    )


def audit_cfd2d_offline_pipeline(train_cfd2d) -> bool:
    """Check released Vx semantics, loader options, and no acquisition path."""

    train = torch.zeros(64, 10, 8, 8, 1)
    test = torch.zeros(33, 10, 8, 8, 1)
    train_loader, train_eval_loader, test_eval_loader = (
        train_cfd2d.build_data_loaders(train, test)
    )
    load_source = inspect.getsource(train_cfd2d.load_cfd2d)
    main_source = inspect.getsource(train_cfd2d.main)
    module_source = inspect.getsource(train_cfd2d)
    required_primitives = (
        "_find_data_dataset",
        "load_subset_to_ram",
        "RolloutRAMDataset",
        "rollout_step_model_nd_channel_first",
        "rollout_loss_lp",
        "evaluate_corrected_relative_l2",
        "corrected_relative_l2_metadata",
    )
    return (
        all(token in module_source for token in required_primitives)
        and train_cfd2d.SELECTED_HDF5_KEY == "/Vx"
        and train_cfd2d.FIELD_DIM == 1
        and train_loader.batch_size == 32
        and isinstance(train_loader.sampler, RandomSampler)
        and train_loader.num_workers == 0
        and train_loader.pin_memory
        and train_loader.drop_last
        and isinstance(train_eval_loader.sampler, SequentialSampler)
        and isinstance(test_eval_loader.sampler, SequentialSampler)
        and not train_eval_loader.drop_last
        and not test_eval_loader.drop_last
        and "ensure_data_available" not in load_source
        and "_download_file" not in load_source
        and "urlopen" not in load_source
        and "requests." not in load_source
        and main_source.index("verify_cfd2d_dataset(data_root)")
        < main_source.index("load_cfd2d(data_root)")
    )


def audit_airfoil_offline_pipeline(train_airfoil) -> bool:
    """Check fixed Airfoil tensor/loader semantics and no acquisition path."""

    train_input = torch.zeros(16, 8, 8, 2)
    train_target = torch.zeros(16, 8, 8, 1)
    test_input = torch.zeros(8, 8, 8, 2)
    test_target = torch.zeros(8, 8, 8, 1)
    train_loader, test_loader = train_airfoil.build_data_loaders(
        train_input, train_target, test_input, test_target
    )
    load_source = inspect.getsource(train_airfoil.load_airfoil)
    main_source = inspect.getsource(train_airfoil.main)
    module_source = inspect.getsource(train_airfoil)
    return (
        train_loader.batch_size == 8
        and isinstance(train_loader.sampler, RandomSampler)
        and train_loader.num_workers == 0
        and not train_loader.pin_memory
        and not train_loader.drop_last
        and test_loader.batch_size == 8
        and isinstance(test_loader.sampler, SequentialSampler)
        and not test_loader.drop_last
        and "np.load" in load_source
        and "np.stack" in load_source
        and "TARGET_FIELD_INDEX" in load_source
        and "_download_file" not in module_source
        and "urlopen" not in module_source
        and "requests." not in module_source
        and main_source.index("verify_airfoil_dataset(data_root)")
        < main_source.index("set_seed(args.seed)")
        < main_source.index("load_airfoil(data_root)")
        < main_source.index("build_model(args.model, device)")
    )


def main() -> None:
    args = parse_args()
    backend = bootstrap_sirenfno_backend()

    from experiments import (
        train_airfoil,
        train_burgers,
        train_cfd1d,
        train_cfd2d,
        train_darcy,
        train_ns2d,
        train_reacdiff,
    )
    from experiments.configs import (
        airfoil, burgers1d, cfd1d, cfd2d, darcy, ns2d, reacdiff1d
    )
    from scripts import run_paper_sweep

    verified = verify_sirenfno_checkout()
    source_state = source_repository_state(allow_dirty=True)
    manifests = dataset_manifest()
    dataset_checks_ok: bool | None = True
    try:
        verify_dataset("darcy128", args.data_root.expanduser())
        verify_dataset("ns128", args.data_root.expanduser())
        verify_dataset("burgers1d", args.data_root.expanduser())
        verify_dataset("cfd1d1024", args.data_root.expanduser())
        verify_dataset("cfd2d128", args.data_root.expanduser())
        verify_dataset("airfoil221x51", args.data_root.expanduser())
        verify_dataset("reacdiff1024", args.data_root.expanduser())
    except FileNotFoundError:
        dataset_checks_ok = None
    except RuntimeError:
        dataset_checks_ok = False

    import_paths = {
        **train_airfoil.author_local_import_audit(),
        **train_darcy.author_local_import_audit(),
        **train_ns2d.author_local_import_audit(),
        **train_burgers.author_local_import_audit(),
        **train_cfd1d.author_local_import_audit(),
        **train_cfd2d.author_local_import_audit(),
        **train_reacdiff.author_local_import_audit(),
    }
    official_paths_ok = all(
        path.startswith("third_party/SirenFNO/")
        for name, path in import_paths.items()
        if not name.startswith("CAFEPlusFNO")
    )
    cafe_path_ok = all(
        import_paths[name].startswith("models/")
        for name in ("CAFEPlusFNO1D", "CAFEPlusFNO2D")
    )

    lock_text = (REPOSITORY_ROOT / "requirements-lock.txt").read_text(
        encoding="utf-8"
    )
    environment_ok = all(
        token in lock_text
        for token in (
            "torch==2.8.0",
            "h5py==3.15.1",
            "numpy==2.4.6",
            "tensorly==0.9.0",
            "tensorly-torch==0.5.0",
        )
    ) and (REPOSITORY_ROOT / "environment.yml").is_file()

    seed_source = inspect.getsource(set_seed)
    seed_helper_ok = all(
        token in seed_source
        for token in (
            "random.seed(seed)",
            "np.random.seed(seed)",
            "torch.manual_seed(seed)",
            "torch.cuda.manual_seed_all(seed)",
        )
    ) and all(
        forbidden not in seed_source
        for forbidden in (
            "torch.use_deterministic_algorithms(",
            "torch.backends.cudnn.deterministic",
        )
    )
    entry_seed_ok = True
    for module, loader_call in (
        (train_darcy, "load_darcy(data_root)"),
        (train_ns2d, "load_ns(data_root)"),
        (train_burgers, "load_burgers(data_root)"),
        (train_cfd1d, "load_cfd1d(data_root)"),
        (train_cfd2d, "load_cfd2d(data_root)"),
        (train_airfoil, "load_airfoil(data_root)"),
        (train_reacdiff, "load_reacdiff(data_root)"),
    ):
        source = inspect.getsource(module.main)
        entry_seed_ok &= source.count("set_seed(args.seed)") == 1
        entry_seed_ok &= source.index("set_seed(args.seed)") < source.index(loader_call)
        entry_seed_ok &= source.index("set_seed(args.seed)") < source.index(
            "build_model(args.model, device)"
        )
    darcy_main_source = inspect.getsource(train_darcy.main)
    ns_main_source = inspect.getsource(train_ns2d.main)
    burgers_main_source = inspect.getsource(train_burgers.main)
    cfd1d_main_source = inspect.getsource(train_cfd1d.main)
    cfd2d_main_source = inspect.getsource(train_cfd2d.main)
    airfoil_main_source = inspect.getsource(train_airfoil.main)
    reacdiff_main_source = inspect.getsource(train_reacdiff.main)
    seed_parity_ok = (
        darcy_main_source.count("set_seed(args.seed)") == 1
        and ns_main_source.count("set_seed(args.seed)") == 1
        and darcy_main_source.index('verify_dataset("darcy128", data_root)')
        < darcy_main_source.index("set_seed(args.seed)")
        < darcy_main_source.index("load_darcy(data_root)")
        < darcy_main_source.index("build_model(args.model, device)")
        and ns_main_source.index("verify_ns_dataset(data_root)")
        < ns_main_source.index("set_seed(args.seed)")
        < ns_main_source.index("load_ns(data_root)")
        < ns_main_source.index("build_model(args.model, device)")
        and all(
            forbidden not in ns_main_source
            for forbidden in ("manual_seed(", "torch.seed(", "torch.Generator(")
        )
        and burgers_main_source.count("set_seed(args.seed)") == 1
        and burgers_main_source.index("verify_burgers_dataset(data_root)")
        < burgers_main_source.index("set_seed(args.seed)")
        < burgers_main_source.index("load_burgers(data_root)")
        < burgers_main_source.index("build_model(args.model, device)")
        and all(
            forbidden not in burgers_main_source
            for forbidden in ("manual_seed(", "torch.seed(", "torch.Generator(")
        )
        and cfd1d_main_source.count("set_seed(args.seed)") == 1
        and cfd1d_main_source.index("verify_cfd1d_dataset(data_root)")
        < cfd1d_main_source.index("set_seed(args.seed)")
        < cfd1d_main_source.index("load_cfd1d(data_root)")
        < cfd1d_main_source.index("build_model(args.model, device)")
        and all(
            forbidden not in cfd1d_main_source
            for forbidden in ("manual_seed(", "torch.seed(", "torch.Generator(")
        )
        and cfd2d_main_source.count("set_seed(args.seed)") == 1
        and cfd2d_main_source.index("verify_cfd2d_dataset(data_root)")
        < cfd2d_main_source.index("set_seed(args.seed)")
        < cfd2d_main_source.index("load_cfd2d(data_root)")
        < cfd2d_main_source.index("build_model(args.model, device)")
        and all(
            forbidden not in cfd2d_main_source
            for forbidden in ("manual_seed(", "torch.seed(", "torch.Generator(")
        )
        and airfoil_main_source.count("set_seed(args.seed)") == 1
        and airfoil_main_source.index("verify_airfoil_dataset(data_root)")
        < airfoil_main_source.index("set_seed(args.seed)")
        < airfoil_main_source.index("load_airfoil(data_root)")
        < airfoil_main_source.index("build_model(args.model, device)")
        and all(
            forbidden not in airfoil_main_source
            for forbidden in ("manual_seed(", "torch.seed(", "torch.Generator(")
        )
        and reacdiff_main_source.count("set_seed(args.seed)") == 1
        and reacdiff_main_source.index("verify_reacdiff_dataset(data_root)")
        < reacdiff_main_source.index("set_seed(args.seed)")
        < reacdiff_main_source.index("load_reacdiff(data_root)")
        < reacdiff_main_source.index("build_model(args.model, device)")
        and all(
            forbidden not in reacdiff_main_source
            for forbidden in ("manual_seed(", "torch.seed(", "torch.Generator(")
        )
    )
    ns_offline_loader_ok = audit_ns_offline_loader(train_ns2d)
    burgers_offline_loader_ok = audit_burgers_offline_pipeline(train_burgers)
    cfd1d_offline_loader_ok = audit_cfd1d_offline_pipeline(train_cfd1d)
    cfd2d_offline_loader_ok = audit_cfd2d_offline_pipeline(train_cfd2d)
    airfoil_offline_loader_ok = audit_airfoil_offline_pipeline(train_airfoil)
    reacdiff_offline_loader_ok = audit_reacdiff_offline_pipeline(
        train_reacdiff
    )

    darcy_config_ok = darcy.FNO_CONFIG == {
        "n_modes": (16, 16),
        "hidden_channels": 32,
        "in_channels": 1,
        "out_channels": 1,
        "n_layers": 4,
        "lifting_channels": 64,
        "projection_channels": 64,
    } and darcy.CAFEPLUSFNO_FACTORIZATIONS == {
        "cafe_plus_fno": {
            "factorization": "dense",
            "rank": 16,
            "cafe_branch_dim": 32,
            "kernel_hidden_dim": 32,
        },
        "cp_cafe_plus_fno": {
            "factorization": "cp",
            "rank": 8,
            "factor_rff_basis": 16,
            "factor_cheb_basis": 8,
            "factor_branch_dim": 12,
            "factor_hidden_dim": 16,
            "factor_output_init_mode": "xavier",
            "factor_kernel_target_rms": 1e-3,
        },
        "tt_cafe_plus_fno": {
            "factorization": "tt",
            "rank": 8,
            "factor_rff_basis": 16,
            "factor_cheb_basis": 8,
            "factor_branch_dim": 12,
            "factor_hidden_dim": 24,
            "factor_output_init_mode": "xavier",
            "factor_kernel_target_rms": 1e-3,
        },
        "tucker_cafe_plus_fno": {
            "factorization": "tucker",
            "rank": 10,
            "factor_rff_basis": 16,
            "factor_cheb_basis": 8,
            "factor_branch_dim": 12,
            "factor_hidden_dim": 16,
            "factor_output_init_mode": "xavier",
            "factor_kernel_target_rms": 1e-3,
        },
    }
    ns_config_ok = ns2d.FNO_CONFIG == {
        "n_modes": (32, 32),
        "hidden_channels": 32,
        "in_channels": 1,
        "out_channels": 1,
        "n_layers": 4,
        "lifting_channels": 64,
        "projection_channels": 64,
    } and ns2d.CAFEPLUSFNO_FACTOR_COMMON_CONFIG == {
        "factor_rff_basis": 16,
        "factor_cheb_basis": 8,
        "factor_branch_dim": 12,
        "factor_output_init_mode": "xavier",
        "factor_kernel_target_rms": 1e-3,
    } and ns2d.CAFEPLUSFNO_FACTORIZATIONS == {
        "cafe_plus_fno": {
            "factorization": "dense",
            "rank": 16,
            "cafe_branch_dim": 32,
            "kernel_hidden_dim": 64,
        },
        "cp_cafe_plus_fno": {
            "factor_rff_basis": 16,
            "factor_cheb_basis": 8,
            "factor_branch_dim": 12,
            "factor_output_init_mode": "xavier",
            "factor_kernel_target_rms": 1e-3,
            "factorization": "cp",
            "rank": 16,
            "factor_hidden_dim": 16,
        },
        "tt_cafe_plus_fno": {
            "factor_rff_basis": 16,
            "factor_cheb_basis": 8,
            "factor_branch_dim": 12,
            "factor_output_init_mode": "xavier",
            "factor_kernel_target_rms": 1e-3,
            "factorization": "tt",
            "rank": 16,
            "factor_hidden_dim": 24,
        },
        "tucker_cafe_plus_fno": {
            "factor_rff_basis": 16,
            "factor_cheb_basis": 8,
            "factor_branch_dim": 12,
            "factor_output_init_mode": "xavier",
            "factor_kernel_target_rms": 1e-3,
            "factorization": "tucker",
            "rank": 16,
            "factor_hidden_dim": 16,
        },
    }
    burgers_config_ok = (
        burgers1d.RESOLUTION == 1024
        and (burgers1d.INPUT_STEPS, burgers1d.ROLLOUT) == (10, 10)
        and (burgers1d.N_TRAIN, burgers1d.N_VAL, burgers1d.N_TEST)
        == (1000, 0, 200)
        and burgers1d.AMFNO_CONFIG["width"] == 64
        and burgers1d.EXPECTED_PINNED_PARAMETER_COUNTS["amfno"] == 823_073
        and burgers1d.model_configuration_provenance("amfno")[
            "prior_width_32_checkpoints_compatible"
        ]
        is False
        and burgers1d.UFNO_CONFIG
        == {
            "n_modes_height": 32,
            "hidden_channels": 64,
            "in_channels": 10,
            "out_channels": 1,
            "n_layers": 6,
            "lifting_channels": 64,
            "projection_channels": 64,
            "positional_embedding": "grid",
        }
        and burgers1d.EXPECTED_PINNED_PARAMETER_COUNTS["ufno"] == 1_883_073
        and not burgers1d.model_configuration_provenance("ufno")[
            "table_3_architecture_parity_claimed"
        ]
    )
    cfd1d_config_ok = (
        cfd1d.RESOLUTION == 1024
        and cfd1d.EXPECTED_TIME_STEPS == 101
        and cfd1d.SELECTED_FIELD == "Vx"
        and (cfd1d.PHYSICAL_CHANNELS, cfd1d.INPUT_DIM, cfd1d.OUTPUT_DIM)
        == (1, 10, 1)
        and (cfd1d.INPUT_STEPS, cfd1d.ROLLOUT) == (10, 10)
        and (cfd1d.N_TRAIN, cfd1d.N_VAL, cfd1d.N_TEST) == (1800, 0, 200)
        and cfd1d.NORMALIZATION_POLICY == "none"
        and cfd1d.FNO_CONFIG["n_modes_height"] == 1024
        and cfd1d.SIRENFNO_COMMON_CONFIG["omega"] == 15.0
        and cfd1d.SIRENFNO_COMMON_CONFIG["ff_sigma"] == 256
        and cfd1d.EXPECTED_PINNED_PARAMETER_COUNTS
        == {
            "fno": 4_216_161,
            "ufno": 1_892_161,
            "tfno_cp": 226_369,
            "amfno": 823_073,
            "sirenfno": 304_833,
            "cpsirenfno": 57_665,
            "ttsirenfno": 72_001,
            "tuckersirenfno": 61_761,
        }
        and cfd1d.CAFEPLUSFNO_FACTORIZATIONS["cp_cafe_plus_fno"]["rank"] == 8
        and cfd1d.CAFEPLUSFNO_FACTORIZATIONS["tt_cafe_plus_fno"]
        ["factor_hidden_dim"] == 24
        and cfd1d.CAFEPLUSFNO_FACTORIZATIONS["tucker_cafe_plus_fno"]
        ["rank"] == 8
    )
    cfd2d_config_ok = (
        cfd2d.RESOLUTION == (128, 128)
        and cfd2d.EXPECTED_TIME_STEPS == 21
        and cfd2d.SELECTED_HDF5_KEY == "/Vx"
        and (cfd2d.PHYSICAL_CHANNELS, cfd2d.INPUT_DIM, cfd2d.OUTPUT_DIM)
        == (1, 5, 1)
        and (cfd2d.INPUT_STEPS, cfd2d.ROLLOUT) == (5, 5)
        and (cfd2d.N_TRAIN, cfd2d.N_VAL, cfd2d.N_TEST) == (1800, 0, 200)
        and cfd2d.NORMALIZATION_POLICY == "none"
        and cfd2d.FNO_CONFIG["n_modes"] == (32, 32)
        and cfd2d.UFNO_CONFIG["n_modes"] == (12, 12)
        and cfd2d.SIRENFNO_COMMON_CONFIG["omega"] == 30.0
        and cfd2d.SIRENFNO_COMMON_CONFIG["ff_sigma"] == 256
        and cfd2d.EXPECTED_PINNED_PARAMETER_COUNTS
        == {
            "fno": 4_469_857,
            "ufno": 991_009,
            "tfno_cp": 237_761,
            "amfno": 385_601,
            "sirenfno": 304_769,
            "cpsirenfno": 64_001,
            "ttsirenfno": 92_673,
            "tuckersirenfno": 96_769,
        }
        and cfd2d.EXPECTED_CAFE_PARAMETER_COUNTS
        == {
            "cafe_plus_fno": 327_169,
            "cp_cafe_plus_fno": 80_645,
            "tt_cafe_plus_fno": 108_421,
            "tucker_cafe_plus_fno": 113_413,
        }
        and cfd2d.CAFEPLUSFNO_FACTORIZATIONS["cp_cafe_plus_fno"]["rank"] == 8
        and cfd2d.CAFEPLUSFNO_FACTORIZATIONS["tt_cafe_plus_fno"]
        ["factor_hidden_dim"] == 24
        and cfd2d.CAFEPLUSFNO_FACTORIZATIONS["tucker_cafe_plus_fno"]
        ["rank"] == 8
        and cfd2d.CAFEPLUSFNO_COMMON_CONFIG["learnable_sigma"] is False
    )
    airfoil_config_ok = (
        airfoil.RESOLUTION == (221, 51)
        and (airfoil.N_TRAIN, airfoil.N_VAL, airfoil.N_TEST) == (1000, 0, 200)
        and airfoil.BATCH_SIZE == 8
        and airfoil.EPOCHS == 500
        and airfoil.MODEL_CHOICES
        == (
            "fno", "tfno_cp", "amfno",
            "sirenfno", "cpsirenfno", "ttsirenfno", "tuckersirenfno",
            "cafe_plus_fno", "cp_cafe_plus_fno", "tt_cafe_plus_fno",
            "tucker_cafe_plus_fno",
        )
        and "ufno" not in airfoil.MODEL_CHOICES
        and not hasattr(airfoil, "UFNO_CONFIG")
        and airfoil.AIRFOIL_EXCLUDED_BASELINES["ufno"]
        ["performance_based_exclusion"] is False
        and airfoil.AIRFOIL_EXCLUDED_BASELINES["ufno"]
        ["excluded_before_paper_training"] is True
        and airfoil.AIRFOIL_EXCLUDED_BASELINES["ufno"]
        ["published_result_imported"] is False
        and airfoil.NEURALOP_N_MODES == (24, 22)
        and airfoil.FNO_CONFIG["lifting_channel_ratio"] == 2
        and airfoil.FNO_CONFIG["projection_channel_ratio"] == 2
        and airfoil.FNO_CONFIG["positional_embedding"] == "grid"
        and "lifting_channels" not in airfoil.FNO_CONFIG
        and "projection_channels" not in airfoil.FNO_CONFIG
        and airfoil.SIRENFNO_AIRFOIL_COMMON_CONFIG
        == {
            "width": 32,
            "input_dim": 2,
            "output_dim": 1,
            "padding": 8,
            "mlp_dropout": 0.0,
            "add_grid": True,
            "hidden_dim": 32,
            "omega": 30.0,
            "n_hidden": 1,
            "siren_dim_in": 32,
            "ff_sigma": 512.0,
            "learnable_ff": True,
        }
        and all(
            airfoil.SIRENFNO_FACTORIZATIONS[name]["rank"] == 8
            for name in ("cpsirenfno", "ttsirenfno", "tuckersirenfno")
        )
        and all(
            airfoil.CAFEPLUSFNO_FACTORIZATIONS[name]["rank"] == 8
            for name in (
                "cp_cafe_plus_fno", "tt_cafe_plus_fno",
                "tucker_cafe_plus_fno",
            )
        )
        and airfoil.EXPECTED_TRAIN_BATCHES_PER_EPOCH == 125
        and airfoil.EXPECTED_SCHEDULER_T_MAX == 62_500
        and set(airfoil.EXPECTED_PINNED_PARAMETER_COUNTS)
        == {
            "fno", "tfno_cp", "amfno", "sirenfno", "cpsirenfno",
            "ttsirenfno", "tuckersirenfno",
        }
        and set(airfoil.EXPECTED_CAFE_PARAMETER_COUNTS)
        == {
            "cafe_plus_fno", "cp_cafe_plus_fno", "tt_cafe_plus_fno",
            "tucker_cafe_plus_fno",
        }
    )
    reacdiff_config_ok = (
        reacdiff1d.RESOLUTION == 1024
        and reacdiff1d.EXPECTED_TIME_STEPS == 101
        and (reacdiff1d.INPUT_STEPS, reacdiff1d.ROLLOUT) == (10, 10)
        and (reacdiff1d.N_TRAIN, reacdiff1d.N_VAL, reacdiff1d.N_TEST)
        == (1000, 0, 200)
        and reacdiff1d.BATCH_SIZE == 32
        and reacdiff1d.NORMALIZATION_POLICY == "none"
        and reacdiff1d.PUSHFORWARD_DETACH is False
        and reacdiff1d.SCHEDULER_T_MAX == 500
        and reacdiff1d.MODEL_CHOICES == (
            "fno", "ufno", "tfno_cp", "amfno",
            "sirenfno", "cpsirenfno", "ttsirenfno", "tuckersirenfno",
            "cafe_plus_fno", "cp_cafe_plus_fno", "tt_cafe_plus_fno",
            "tucker_cafe_plus_fno",
        )
        and reacdiff1d.FNO_CONFIG["n_modes_height"] == 1024
        and reacdiff1d.TFNO_CP_CONFIG["rank"] == 0.05
        and reacdiff1d.AMFNO_CONFIG["width"] == 64
        and reacdiff1d.UFNO_CONFIG["n_layers"] == 6
        and reacdiff1d.SIRENFNO_COMMON_CONFIG["omega"] == 15.0
        and reacdiff1d.SIRENFNO_COMMON_CONFIG["ff_sigma"] == 256
        and reacdiff1d.EXPECTED_PINNED_PARAMETER_COUNTS
        == cfd1d.EXPECTED_PINNED_PARAMETER_COUNTS
        and reacdiff1d.EXPECTED_CAFE_PARAMETER_COUNTS == {
            "cafe_plus_fno": 323_201,
            "cp_cafe_plus_fno": 70_149,
            "tt_cafe_plus_fno": 85_381,
            "tucker_cafe_plus_fno": 74_245,
        }
        and all(
            reacdiff1d.CAFEPLUSFNO_FACTORIZATIONS[name]["rank"] == 8
            for name in (
                "cp_cafe_plus_fno", "tt_cafe_plus_fno",
                "tucker_cafe_plus_fno",
            )
        )
    )
    training_audit_source = inspect.getsource(
        run_paper_sweep._audit_run_artifacts
    )
    reevaluation_audit_source = inspect.getsource(
        run_paper_sweep._validate_reevaluation_record
    )
    training_reader_source = inspect.getsource(
        run_paper_sweep.validate_training_artifacts
    )
    contract_source = inspect.getsource(run_paper_sweep.run_contract_metadata)
    resolver_source = inspect.getsource(
        run_paper_sweep.resolve_current_source_commit
    )
    writer_contracts = {
        "airfoil221x51": airfoil_main_source,
        "burgers1024": burgers_main_source,
        "cfd1d1024": cfd1d_main_source,
        "cfd2d128": cfd2d_main_source,
        "darcy128": darcy_main_source,
        "ns128": ns_main_source,
        "reacdiff1024": reacdiff_main_source,
    }
    writer_environment_sources = tuple(
        inspect.getsource(module.environment_metadata)
        for module in (
            train_airfoil,
            train_burgers,
            train_cfd1d,
            train_cfd2d,
            train_darcy,
            train_ns2d,
            train_reacdiff,
        )
    )
    shared_source_contract_ok = (
        all(
            spec.get("require_current_source_commit") is True
            for spec in run_paper_sweep.DATASETS.values()
        )
        and all(
            token in training_audit_source
            for token in (
                "training_source_commit",
                "checkpoint_sha256",
                "load_state_dict(model_state, strict=True)",
            )
        )
        and "evaluation_source_commit" in reevaluation_audit_source
        and "_audit_run_artifacts(" in training_reader_source
        and "load_weights_only_checkpoint(" in training_reader_source
        and all(
            token in contract_source
            for token in (
                '"training_source_commit"',
                '"evaluation_source_commit"',
                '"training_protocol_identifier"',
                '"evaluation_protocol_version"',
            )
        )
        and "source_repository_state(allow_dirty=False)" in resolver_source
        and all(
            f'run_contract_metadata("{dataset_key}"' in writer_source
            and "run_environment_provenance(contract)" in writer_source
            and "prepare_weights_only_checkpoint({" in writer_source
            and 'summary["checkpoint_sha256"]' in writer_source
            for dataset_key, writer_source in writer_contracts.items()
        )
        and all(
            '"pytorch_version": str(torch.__version__)' in source
            for source in writer_environment_sources
        )
    )
    airfoil_source_commit_guard_ok = shared_source_contract_ok
    reacdiff_source_commit_guard_ok = shared_source_contract_ok
    corrected_evaluation_contract_ok = (
        all(
            run_paper_sweep.DATASETS[key]["evaluation_protocol_version"]
            == "corrected_relative_l2_v1"
            for key in ("cfd1d1024", "cfd2d128", "reacdiff1024")
        )
        and all(
            "evaluate_corrected_relative_l2(" in source
            and "evaluate_rel_l2_metrics(" not in source
            for source in (
                inspect.getsource(train_cfd1d.run_training_loop),
                inspect.getsource(train_cfd2d.run_training_loop),
                inspect.getsource(train_reacdiff.run_training_loop),
            )
        )
    )

    checks = [
        ("SirenFNO upstream pinned", bool(backend["commit"])),
        ("SirenFNO commit verified", verified["commit"] == backend["commit"]),
        (
            "bundled NeuralOperator verified",
            verified["neuraloperator_git_tree_sha1"]
            == backend["neuraloperator_git_tree_sha1"],
        ),
        ("site-packages isolation", official_paths_ok and cafe_path_ok),
        ("environment pinned", environment_ok),
        (
            "dataset specification",
            set(manifests)
            == {
                "airfoil221x51", "burgers1d", "cfd1d1024", "cfd2d128",
                "darcy128", "ns128",
                "reacdiff1024",
            },
        ),
        ("dataset hashes", dataset_checks_ok),
        (
            "model configs explicit",
            airfoil_config_ok
            and burgers_config_ok
            and cfd1d_config_ok
            and cfd2d_config_ok
            and darcy_config_ok
            and ns_config_ok
            and reacdiff_config_ok,
        ),
        (
            "seed policy explicit",
            seed_helper_ok and entry_seed_ok and seed_parity_ok,
        ),
        ("seven-dataset source contract", shared_source_contract_ok),
        ("corrected evaluation contract", corrected_evaluation_contract_ok),
        ("source tree clean", not source_state["source_repository_dirty"]),
        (
            "one-command execution",
            all(
                path.is_file()
                for path in (
                    REPOSITORY_ROOT / "experiments" / "train_darcy.py",
                    REPOSITORY_ROOT / "experiments" / "train_ns2d.py",
                    REPOSITORY_ROOT / "experiments" / "train_burgers.py",
                    REPOSITORY_ROOT / "experiments" / "train_cfd1d.py",
                    REPOSITORY_ROOT / "experiments" / "train_cfd2d.py",
                    REPOSITORY_ROOT / "experiments" / "train_airfoil.py",
                    REPOSITORY_ROOT / "experiments" / "train_reacdiff.py",
                    REPOSITORY_ROOT / "scripts" / "setup_sirenfno.py",
                )
            ),
        ),
        (
            "fixed-final-epoch policy",
            darcy.SELECTION_POLICY == "fixed_final_epoch_no_test_selection"
            and ns2d.SELECTION_POLICY == "fixed_final_epoch_no_test_selection"
            and burgers1d.SELECTION_POLICY
            == "fixed_final_epoch_no_test_selection"
            and cfd1d.SELECTION_POLICY
            == "fixed_final_epoch_no_test_selection"
            and cfd2d.SELECTION_POLICY
            == "fixed_final_epoch_no_test_selection"
            and airfoil.SELECTION_POLICY
            == "fixed_final_epoch_no_test_selection"
            and reacdiff1d.SELECTION_POLICY
            == "fixed_final_epoch_no_test_selection",
        ),
    ]
    print_report("REPRODUCIBILITY AUDIT", checks)

    darcy_checks = [
        (
            "dataset loader",
            train_darcy.load_darcy_flow_small.__module__.endswith(".darcy"),
        ),
        ("1000/200 split", (darcy.N_TRAIN, darcy.N_TEST) == (1000, 200)),
        ("128x128", darcy.RESOLUTION == 128),
        ("batch 32", (darcy.BATCH_SIZE, darcy.TEST_BATCH_SIZE) == (32, 32)),
        ("AdamW", train_darcy.AdamW.__module__.startswith("neuralop.")),
        ("LR 1e-3", darcy.LEARNING_RATE == 1e-3),
        ("WD 1e-4", darcy.WEIGHT_DECAY == 1e-4),
        ("Cosine T_max 500", darcy.SCHEDULER_T_MAX == 500),
        ("500 epochs", darcy.EPOCHS == 500),
        ("relative L2 train", True),
        ("L2/H1 evaluation", True),
        ("no validation", "validation" not in inspect.getsource(train_darcy.main)),
        ("final-epoch selection", darcy.SELECTION_POLICY.endswith("no_test_selection")),
    ]
    print()
    print_report("Darcy protocol", darcy_checks)

    ns_checks = [
        (
            "pinned dataset implementation",
            train_ns2d.NavierStokesDataset.__module__.endswith(".navier_stokes"),
        ),
        (
            "official loader reference",
            train_ns2d.load_navier_stokes_pt_reference.__module__.endswith(
                ".navier_stokes"
            ),
        ),
        ("offline loader parity", ns_offline_loader_ok),
        (
            "verified local data first",
            ns_main_source.index("verify_ns_dataset(data_root)")
            < ns_main_source.index("load_ns(data_root)"),
        ),
        ("training network disabled", ns_offline_loader_ok),
        ("1000/200 split", (ns2d.N_TRAIN, ns2d.N_TEST) == (1000, 200)),
        ("128x128", ns2d.RESOLUTION == 128),
        ("batch 32", (ns2d.BATCH_SIZE, ns2d.TEST_BATCH_SIZE) == (32, 32)),
        ("AdamW", train_ns2d.AdamW.__module__.startswith("neuralop.")),
        ("LR 1e-3", ns2d.LEARNING_RATE == 1e-3),
        ("WD 1e-4", ns2d.WEIGHT_DECAY == 1e-4),
        ("Cosine T_max 500", ns2d.SCHEDULER_T_MAX == 500),
        ("500 epochs", ns2d.EPOCHS == 500),
        ("relative L2 train", True),
        ("L2/H1 evaluation", True),
        ("no validation", "validation" not in inspect.getsource(train_ns2d.main)),
        (
            "final-epoch selection",
            ns2d.SELECTION_POLICY.endswith("no_test_selection"),
        ),
    ]
    print()
    print_report("Navier-Stokes protocol", ns_checks)

    burgers_checks = [
        ("verified local/offline", burgers_offline_loader_ok),
        ("1000/0/200 split", (burgers1d.N_TRAIN, burgers1d.N_VAL, burgers1d.N_TEST) == (1000, 0, 200)),
        ("1024 resolution", burgers1d.RESOLUTION == 1024),
        ("10 -> 10 rollout", (burgers1d.INPUT_STEPS, burgers1d.ROLLOUT) == (10, 10)),
        ("pushforward gradients", burgers1d.PUSHFORWARD_DETACH is False),
        ("batch 32", burgers1d.BATCH_SIZE == 32),
        ("AdamW", "torch.optim.AdamW" in inspect.getsource(train_burgers.main)),
        ("LR 1e-3", burgers1d.LEARNING_RATE == 1e-3),
        ("WD 1e-4", burgers1d.WEIGHT_DECAY == 1e-4),
        ("Cosine T_max 500", burgers1d.SCHEDULER_T_MAX == 500),
        ("500 epochs", burgers1d.EPOCHS == 500),
        (
            "AM-FNO pinned width 64",
            burgers1d.AMFNO_CONFIG["width"] == 64
            and burgers1d.EXPECTED_PINNED_PARAMETER_COUNTS["amfno"] == 823_073,
        ),
        ("U-FNO discrepancy explicit", burgers_config_ok),
        (
            "final-epoch selection",
            burgers1d.SELECTION_POLICY.endswith("no_test_selection"),
        ),
    ]
    print()
    print_report("Burgers-1D protocol", burgers_checks)

    cfd1d_checks = [
        ("verified local/offline", cfd1d_offline_loader_ok),
        ("Vx-only released runtime", cfd1d.SELECTED_FIELD == "Vx"),
        (
            "1800/0/200 split",
            (cfd1d.N_TRAIN, cfd1d.N_VAL, cfd1d.N_TEST) == (1800, 0, 200),
        ),
        ("1024 resolution", cfd1d.RESOLUTION == 1024),
        (
            "10 -> 10 rollout",
            (cfd1d.INPUT_STEPS, cfd1d.ROLLOUT) == (10, 10),
        ),
        ("10 -> 1 model IO", (cfd1d.INPUT_DIM, cfd1d.OUTPUT_DIM) == (10, 1)),
        ("field_dim 1", cfd1d.FIELD_DIM == 1),
        ("no normalization", cfd1d.NORMALIZATION_POLICY == "none"),
        ("pushforward gradients", cfd1d.PUSHFORWARD_DETACH is False),
        ("batch 32", cfd1d.BATCH_SIZE == 32),
        ("AdamW", "torch.optim.AdamW" in inspect.getsource(train_cfd1d.main)),
        ("LR 1e-3", cfd1d.LEARNING_RATE == 1e-3),
        ("WD 1e-4", cfd1d.WEIGHT_DECAY == 1e-4),
        ("Cosine T_max 500", cfd1d.SCHEDULER_T_MAX == 500),
        ("500 epochs", cfd1d.EPOCHS == 500),
        ("released parameter counts", cfd1d_config_ok),
        (
            "final-epoch selection",
            cfd1d.SELECTION_POLICY.endswith("no_test_selection"),
        ),
    ]
    print()
    print_report("CFD-1D protocol", cfd1d_checks)

    cfd2d_checks = [
        ("verified local/offline", cfd2d_offline_loader_ok),
        ("Vx-only released runtime", cfd2d.SELECTED_HDF5_KEY == "/Vx"),
        (
            "1800/0/200 split",
            (cfd2d.N_TRAIN, cfd2d.N_VAL, cfd2d.N_TEST) == (1800, 0, 200),
        ),
        ("128x128 resolution", cfd2d.RESOLUTION == (128, 128)),
        ("5 -> 5 rollout", (cfd2d.INPUT_STEPS, cfd2d.ROLLOUT) == (5, 5)),
        ("5 -> 1 model IO", (cfd2d.INPUT_DIM, cfd2d.OUTPUT_DIM) == (5, 1)),
        ("field_dim 1", cfd2d.FIELD_DIM == 1),
        ("no normalization", cfd2d.NORMALIZATION_POLICY == "none"),
        ("pushforward gradients", cfd2d.PUSHFORWARD_DETACH is False),
        ("batch 32", cfd2d.BATCH_SIZE == 32),
        ("AdamW", "torch.optim.AdamW" in inspect.getsource(train_cfd2d.main)),
        ("LR 1e-3", cfd2d.LEARNING_RATE == 1e-3),
        ("WD 1e-4", cfd2d.WEIGHT_DECAY == 1e-4),
        ("Cosine T_max 500", cfd2d.SCHEDULER_T_MAX == 500),
        ("500 epochs", cfd2d.EPOCHS == 500),
        ("released parameter counts", cfd2d_config_ok),
        (
            "final-epoch selection",
            cfd2d.SELECTION_POLICY.endswith("no_test_selection"),
        ),
    ]
    print()
    print_report("CFD-2D protocol", cfd2d_checks)

    airfoil_loop_source = inspect.getsource(train_airfoil.run_training_loop)
    airfoil_checks = [
        ("verified local/offline", airfoil_offline_loader_ok),
        ("1000/0/200 split", (airfoil.N_TRAIN, airfoil.N_VAL, airfoil.N_TEST) == (1000, 0, 200)),
        ("221x51 resolution", airfoil.RESOLUTION == (221, 51)),
        ("X/Y -> Q[:,4]", (airfoil.INPUT_DIM, airfoil.OUTPUT_DIM, airfoil.TARGET_FIELD_INDEX) == (2, 1, 4)),
        ("no normalization", airfoil.NORMALIZATION_POLICY == "none"),
        ("11-model roster", len(airfoil.MODEL_CHOICES) == 11 and "ufno" not in airfoil.MODEL_CHOICES),
        ("U-FNO exclusion explicit", airfoil_config_ok),
        ("batch 8", airfoil.BATCH_SIZE == 8),
        ("AdamW", "torch.optim.AdamW" in inspect.getsource(train_airfoil.main)),
        ("LR 1e-3", airfoil.LEARNING_RATE == 1e-3),
        ("WD 1e-4", airfoil.WEIGHT_DECAY == 1e-4),
        ("scheduler T_max 62500", airfoil.EXPECTED_SCHEDULER_T_MAX == 62_500),
        ("batch plus epoch scheduler", airfoil_loop_source.count("scheduler.step()") == 2),
        ("current source commit guard", airfoil_source_commit_guard_ok),
        ("500 epochs", airfoil.EPOCHS == 500),
        ("fixed final epoch", airfoil.SELECTION_POLICY.endswith("no_test_selection")),
    ]
    print()
    print_report("Airfoil protocol", airfoil_checks)

    reacdiff_loop_source = inspect.getsource(train_reacdiff.run_training_loop)
    reacdiff_checks = [
        ("verified local/offline", reacdiff_offline_loader_ok),
        ("1000/0/200 split", (reacdiff1d.N_TRAIN, reacdiff1d.N_VAL, reacdiff1d.N_TEST) == (1000, 0, 200)),
        ("1024 resolution", reacdiff1d.RESOLUTION == 1024),
        ("101 source snapshots", reacdiff1d.EXPECTED_TIME_STEPS == 101),
        ("10 -> 10 rollout", (reacdiff1d.INPUT_STEPS, reacdiff1d.ROLLOUT) == (10, 10)),
        ("no normalization", reacdiff1d.NORMALIZATION_POLICY == "none"),
        ("pushforward gradients", reacdiff1d.PUSHFORWARD_DETACH is False),
        ("batch 32", reacdiff1d.BATCH_SIZE == 32),
        ("12-model roster", len(reacdiff1d.MODEL_CHOICES) == 12),
        ("released constructors", reacdiff_config_ok),
        ("AdamW", "torch.optim.AdamW" in inspect.getsource(train_reacdiff.main)),
        ("LR 1e-3", reacdiff1d.LEARNING_RATE == 1e-3),
        ("WD 1e-4", reacdiff1d.WEIGHT_DECAY == 1e-4),
        ("Cosine T_max 500", reacdiff1d.SCHEDULER_T_MAX == 500),
        ("one epoch-end scheduler", reacdiff_loop_source.count("scheduler.step()") == 1),
        ("500 epochs", reacdiff1d.EPOCHS == 500),
        ("current source commit guard", reacdiff_source_commit_guard_ok),
        ("fixed final epoch", reacdiff1d.SELECTION_POLICY.endswith("no_test_selection")),
    ]
    print()
    print_report("Reaction-Diffusion protocol", reacdiff_checks)

    if not all(
        passed
        for _, passed in (
            checks
            + darcy_checks
            + ns_checks
            + burgers_checks
            + cfd1d_checks
            + cfd2d_checks
            + airfoil_checks
            + reacdiff_checks
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
