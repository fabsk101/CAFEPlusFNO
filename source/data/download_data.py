"""Acquire the pinned public datasets in a separate, explicit stage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.common.dataset_provenance import (
    dataset_manifest,
    sha256_file,
    verify_dataset,
)
from experiments.common.sirenfno_backend import bootstrap_sirenfno_backend

bootstrap_sirenfno_backend()

from neuralop.data.datasets.darcy import DarcyDataset
from neuralop.data.datasets.navier_stokes import NavierStokesDataset
from utils import _download_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned public paper-experiment datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=(
            "darcy128",
            "ns128",
            "burgers1d",
            "cfd1d1024",
            "cfd2d128",
            "airfoil221x51",
            "reacdiff1024",
            "both",
            "all",
        ),
        default="both",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Destination directory for the public dataset files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser()
    data_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "both":
        selected = ("darcy128", "ns128")
    elif args.dataset == "all":
        selected = (
            "darcy128",
            "ns128",
            "burgers1d",
            "cfd1d1024",
            "cfd2d128",
            "airfoil221x51",
            "reacdiff1024",
        )
    else:
        selected = (args.dataset,)
    if "darcy128" in selected:
        DarcyDataset(
            root_dir=data_root,
            n_train=1000,
            n_tests=[200],
            batch_size=32,
            test_batch_sizes=[32],
            train_resolution=128,
            test_resolutions=[128],
            encode_input=False,
            encode_output=True,
            encoding="channel-wise",
            channel_dim=1,
            download=True,
        )
        verify_dataset("darcy128", data_root)
        print("Darcy128 checksum verification: PASS")

    if "ns128" in selected:
        NavierStokesDataset(
            root_dir=data_root,
            n_train=1000,
            n_tests=[200],
            batch_size=32,
            test_batch_sizes=[32],
            train_resolution=128,
            test_resolutions=[128],
            encode_input=True,
            encode_output=True,
            encoding="channel-wise",
            channel_dim=1,
            download=True,
        )
        verify_dataset("ns128", data_root)
        print("NS128 checksum verification: PASS")

    if "burgers1d" in selected:
        specification = dataset_manifest()["burgers1d"]
        filename, identity = next(iter(specification["files"].items()))
        destination = data_root / filename
        try:
            verify_dataset("burgers1d", data_root)
        except (FileNotFoundError, RuntimeError):
            temporary = destination.with_name(destination.name + ".part")
            if temporary.exists():
                temporary.unlink()
            try:
                _download_file(specification["source"], str(temporary))
                if temporary.stat().st_size != identity["size_bytes"]:
                    raise RuntimeError("Downloaded Burgers dataset size mismatch.")
                if sha256_file(temporary) != identity["sha256"]:
                    raise RuntimeError("Downloaded Burgers dataset SHA256 mismatch.")
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        verify_dataset("burgers1d", data_root)
        print("Burgers-1D checksum verification: PASS")

    if "cfd1d1024" in selected:
        specification = dataset_manifest()["cfd1d1024"]
        filename, identity = next(iter(specification["files"].items()))
        destination = data_root / filename
        try:
            verify_dataset("cfd1d1024", data_root)
        except (FileNotFoundError, RuntimeError):
            temporary = destination.with_name(destination.name + ".part")
            if temporary.exists():
                temporary.unlink()
            try:
                _download_file(specification["source"], str(temporary))
                if temporary.stat().st_size != identity["size_bytes"]:
                    raise RuntimeError("Downloaded CFD-1D dataset size mismatch.")
                if sha256_file(temporary) != identity["sha256"]:
                    raise RuntimeError("Downloaded CFD-1D dataset SHA256 mismatch.")
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        verify_dataset("cfd1d1024", data_root)
        print("CFD-1D checksum verification: PASS")

    if "cfd2d128" in selected:
        specification = dataset_manifest()["cfd2d128"]
        filename, identity = next(iter(specification["files"].items()))
        destination = data_root / filename
        try:
            verify_dataset("cfd2d128", data_root)
        except (FileNotFoundError, RuntimeError):
            temporary = destination.with_name(destination.name + ".part")
            if temporary.exists():
                temporary.unlink()
            try:
                _download_file(specification["source"], str(temporary))
                if temporary.stat().st_size != identity["size_bytes"]:
                    raise RuntimeError("Downloaded CFD-2D dataset size mismatch.")
                if sha256_file(temporary) != identity["sha256"]:
                    raise RuntimeError("Downloaded CFD-2D dataset SHA256 mismatch.")
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        verify_dataset("cfd2d128", data_root)
        print("CFD-2D checksum verification: PASS")

    if "airfoil221x51" in selected:
        specification = dataset_manifest()["airfoil221x51"]
        try:
            verify_dataset("airfoil221x51", data_root)
        except (FileNotFoundError, RuntimeError):
            for filename, identity in specification["files"].items():
                destination = data_root / filename
                if (
                    destination.is_file()
                    and destination.stat().st_size == identity["size_bytes"]
                    and sha256_file(destination) == identity["sha256"]
                ):
                    continue
                temporary = destination.with_name(destination.name + ".part")
                if temporary.exists():
                    temporary.unlink()
                try:
                    _download_file(identity["source_url"], str(temporary))
                    if temporary.stat().st_size != identity["size_bytes"]:
                        raise RuntimeError(
                            f"Downloaded Airfoil file size mismatch: {filename}"
                        )
                    if sha256_file(temporary) != identity["sha256"]:
                        raise RuntimeError(
                            f"Downloaded Airfoil file SHA256 mismatch: {filename}"
                        )
                    temporary.replace(destination)
                finally:
                    if temporary.exists():
                        temporary.unlink()
        verify_dataset("airfoil221x51", data_root)
        print("Airfoil 221x51 checksum verification: PASS")

    if "reacdiff1024" in selected:
        specification = dataset_manifest()["reacdiff1024"]
        filename, identity = next(iter(specification["files"].items()))
        destination = data_root / filename
        try:
            verify_dataset("reacdiff1024", data_root)
        except (FileNotFoundError, RuntimeError):
            temporary = destination.with_name(destination.name + ".part")
            if temporary.exists():
                temporary.unlink()
            try:
                _download_file(specification["source"], str(temporary))
                if temporary.stat().st_size != identity["size_bytes"]:
                    raise RuntimeError(
                        "Downloaded Reaction-Diffusion dataset size mismatch."
                    )
                if sha256_file(temporary) != identity["sha256"]:
                    raise RuntimeError(
                        "Downloaded Reaction-Diffusion dataset SHA256 mismatch."
                    )
                temporary.replace(destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
        verify_dataset("reacdiff1024", data_root)
        print("Reaction-Diffusion-1024 checksum verification: PASS")


if __name__ == "__main__":
    main()
