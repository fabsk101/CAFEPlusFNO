import contextlib
import os
import random
import time
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _find_data_dataset(f: h5py.File):
    """Locate the primary PDE dataset stored in an HDF5 file.

    Heuristic: scan all datasets and choose the one with rank >= 3 that has the
    largest time dimension (axis=1). This avoids accidentally picking coordinate
    vectors or metadata arrays.
    """

    candidates: List[Tuple[Tuple[int, ...], h5py.Dataset]] = []

    def dfs(g):
        for _, v in g.items():
            if isinstance(v, h5py.Dataset):
                shp = v.shape
                if shp is not None and len(shp) >= 3:
                    candidates.append((tuple(int(s) for s in shp), v))
            elif isinstance(v, h5py.Group):
                dfs(v)

    dfs(f)
    if not candidates:
        raise RuntimeError("No [N, T, ...] dataset found in HDF5.")

    def score(item):
        shp, _ = item
        t = shp[1] if len(shp) > 1 else 0
        numel = 1
        for s in shp:
            numel *= int(s)
        return (t, len(shp), numel)

    candidates.sort(key=score, reverse=True)
    return candidates[0][1]


def _is_valid_hdf5(path: str) -> bool:
    try:
        with h5py.File(path, "r") as f:
            _ = _find_data_dataset(f)
        return True
    except Exception:
        return False


def _download_file(url: str, dest_path: str, chunk_size: int = 1 << 20):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with (
        contextlib.closing(urllib.request.urlopen(req)) as r,
        open(dest_path, "wb") as f,
    ):
        total_str = r.headers.get("Content-Length")
        total = int(total_str) if total_str and total_str.isdigit() else None
        downloaded = 0
        t0 = time.perf_counter()
        while True:
            chunk = r.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = 100.0 * downloaded / total
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                dt = max(1e-6, time.perf_counter() - t0)
                speed = (downloaded / (1024 * 1024)) / dt
                print(
                    f"\rDownloading: {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%) at {speed:.1f} MB/s",
                    end="",
                )
        print()


def ensure_data_available(data_file: str, url: str):
    data_dir = os.path.dirname(data_file) or "."
    os.makedirs(data_dir, exist_ok=True)

    if os.path.isfile(data_file) and _is_valid_hdf5(data_file):
        return

    print(
        f"Data file not found or invalid at '{data_file}'. Attempting download from:\n  {url}"
    )
    tmp_path = data_file + ".part"
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    try:
        _download_file(url, tmp_path)
        os.replace(tmp_path, data_file)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise RuntimeError(
            "Failed to download dataset. Please download manually and place it at "
            f"'{data_file}'. Original error: {e}"
        )

    if not _is_valid_hdf5(data_file):
        try:
            os.remove(data_file)
        except OSError:
            pass
        raise RuntimeError(
            "Downloaded file is not a valid HDF5 dataset or has no dataset inside. "
            f"Please place the correct file manually at '{data_file}'."
        )


def reduced_file_path(in_path: str, rx: int, rt: int) -> str:
    root, ext = os.path.splitext(in_path)
    return f"{root}_rx{rx}_rt{rt}{ext}"


def _infer_spatial_structure(sample_shape: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int, bool]:
    """Infer spatial dimensions and channel count from a single sample shape.

    Parameters
    ----------
    sample_shape : Tuple[int, ...]
        Shape of a single trajectory as stored in the HDF5 file (e.g. ``(T, X, Y, C)``).

    Returns
    -------
    spatial_shape : tuple of int
        Spatial resolution per axis (without time/channel dimensions).
    channels : int
        Number of physical channels; returns 1 when the dataset omits an explicit channel axis.
    has_channel_axis : bool
        Indicates whether an explicit channel axis was present in the sample.
    """

    if len(sample_shape) < 2:
        raise ValueError(f"Expected at least time + one spatial dim, got shape={sample_shape}")

    has_channel_axis = len(sample_shape) >= 3 and sample_shape[-1] <= 8
    if has_channel_axis:
        spatial_shape = sample_shape[1:-1]
        channels = sample_shape[-1]
    else:
        spatial_shape = sample_shape[1:]
        channels = 1

    if len(spatial_shape) == 0:
        raise ValueError(f"Unable to infer spatial dimensions from shape={sample_shape}")

    return spatial_shape, channels, has_channel_axis


def preprocess_decimated_hdf5(
    in_path: str,
    rx: int,
    rt: int,
    out_path: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Decimate time and space from an HDF5 trajectory dataset.

    - Time: `u[::rt]`
    - Space: every spatial axis `[..., ::rx]`

    Writes a new HDF5 file with dataset name `'u'` and shape `[N, T', *spatial', C]`,
    always keeping an explicit channel dimension (with `C=1` when absent in source).
    """

    assert rx >= 1 and rt >= 1
    if out_path is None:
        out_path = reduced_file_path(in_path, rx, rt)
    if (not overwrite) and os.path.isfile(out_path):
        print(f"[preprocess] Found existing reduced file: {out_path}")
        return out_path

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"[preprocess] Creating reduced file -> {out_path}")
    with h5py.File(in_path, "r") as fin:
        ds = _find_data_dataset(fin)
        sample = np.asarray(ds[0])
        spatial_shape, channels, has_channel_axis = _infer_spatial_structure(sample.shape)
        N = ds.shape[0]
        T = sample.shape[0]
        T_red = (T + rt - 1) // rt
        spatial_red = tuple((s + rx - 1) // rx for s in spatial_shape)

        target_shape = (N, T_red, *spatial_red, channels)
        chunk_shape = (1, T_red, *spatial_red, channels)

        with h5py.File(out_path, "w") as fout:
            dset = fout.create_dataset(
                "u",
                shape=target_shape,
                dtype="f4",
                chunks=chunk_shape,
                compression=None,
            )
            dset.attrs["reduce_x"] = rx
            dset.attrs["reduce_t"] = rt
            dset.attrs["source"] = os.path.abspath(in_path)

            for n in range(N):
                u = np.asarray(ds[n])
                if rt > 1:
                    u = u[::rt, ...]

                spatial_slices = tuple(slice(None, None, rx) for _ in spatial_shape)
                if has_channel_axis:
                    slicing = (slice(None),) + spatial_slices + (slice(None),)
                    u = u[slicing]
                else:
                    slicing = (slice(None),) + spatial_slices
                    u = u[slicing][..., None]

                expected_shape = (T_red, *spatial_red, channels)
                assert u.shape == expected_shape, (
                    f"Unexpected reduced shape {u.shape}, expected {expected_shape}"
                )
                dset[n] = u.astype("f4")
    return out_path


def preprocess_decimated_hdf5_1d_scalar(
    in_path: str,
    rx: int,
    rt: int,
    out_path: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """1D scalar decimation (Burgers-style): writes `[N, T', X']` without channel axis."""

    assert rx >= 1 and rt >= 1
    if out_path is None:
        out_path = reduced_file_path(in_path, rx, rt)
    if (not overwrite) and os.path.isfile(out_path):
        print(f"[preprocess] Found existing reduced file: {out_path}")
        return out_path

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"[preprocess] Creating reduced file -> {out_path}")
    with h5py.File(in_path, "r") as fin:
        ds = _find_data_dataset(fin)
        N, T, X = ds.shape[0], ds.shape[1], ds.shape[2]
        T_red = (T + rt - 1) // rt
        X_red = (X + rx - 1) // rx

        with h5py.File(out_path, "w") as fout:
            dset = fout.create_dataset(
                "u",
                shape=(N, T_red, X_red),
                dtype="f4",
                chunks=(1, T_red, X_red),
                compression=None,
            )
            dset.attrs["reduce_x"] = rx
            dset.attrs["reduce_t"] = rt
            dset.attrs["source"] = os.path.abspath(in_path)

            for n in range(N):
                u = np.asarray(ds[n])  # [T, X] or [T, X, 1]
                if u.ndim == 3:
                    u = u[..., 0]
                if rt > 1:
                    u = u[::rt, :]
                if rx > 1:
                    u = u[:, ::rx]
                assert u.ndim == 2 and u.shape == (T_red, X_red)
                dset[n] = u.astype("f4")
    return out_path


def load_subset_to_ram(path: str, indices: np.ndarray) -> torch.Tensor:
    """Load selected trajectories into RAM as a CPU float32 tensor `[N, T, *spatial, C]`."""

    indices = np.asarray(indices)
    with h5py.File(path, "r") as f:
        ds = _find_data_dataset(f)
        if len(indices) == 0:
            raise ValueError("Requested empty index set for RAM loading.")

        first = np.asarray(ds[int(indices[0])], dtype=np.float32)
        spatial_shape, _, has_channel_axis = _infer_spatial_structure(first.shape)
        if not has_channel_axis:
            first = first[..., None]

        sample_shape = first.shape
        out = np.empty((len(indices),) + sample_shape, dtype=np.float32)
        out[0] = first

        for i, tr in enumerate(indices[1:], start=1):
            arr = np.asarray(ds[int(tr)], dtype=np.float32)
            if not has_channel_axis:
                arr = arr[..., None]
            out[i] = arr
    return torch.from_numpy(out)


def load_subset_to_ram_1d_scalar(path: str, indices: np.ndarray) -> torch.Tensor:
    """Load selected 1D scalar trajectories into RAM as `[N, T, X]` (CPU float32)."""

    indices = np.asarray(indices)
    with h5py.File(path, "r") as f:
        ds = _find_data_dataset(f)
        T, X = ds.shape[1], ds.shape[2]
        out = np.empty((len(indices), T, X), dtype=np.float32)
        for i, tr in enumerate(indices):
            arr = np.asarray(ds[int(tr)])
            if arr.ndim == 3:
                arr = arr[..., 0]
            out[i] = arr.astype(np.float32, copy=False)
    return torch.from_numpy(out)


def compute_mean_std_from_ram(
    data: torch.Tensor, max_traj_for_stats: int = 200
) -> Tuple[float, float]:
    """Compute global mean/std from a random subset of trajectories already in RAM."""

    N = data.shape[0]
    k = min(max_traj_for_stats, N)
    idx = torch.randperm(N)[:k]
    subset = data[idx]
    mean = subset.mean().item()
    std = subset.std(unbiased=False).item()
    if std < 1e-12:
        std = 1.0
    return mean, std


class RolloutRAMDataset(Dataset):
    """RAM-backed dataset: `[N, T, *spatial, C]` -> `(x_hist, y_seq)` windows."""

    def __init__(
        self,
        data_cpu: torch.Tensor,
        input_steps: int,
        rollout: int,
        mean: Optional[float],
        std: Optional[float],
    ):
        assert data_cpu.device.type == "cpu"
        if data_cpu.ndim == 3:
            data_cpu = data_cpu.unsqueeze(-1)
        if data_cpu.ndim < 4:
            raise ValueError(
                f"Expected data with explicit channel dimension, got shape={tuple(data_cpu.shape)}"
            )

        self.data = data_cpu
        self.N = data_cpu.shape[0]
        self.T = data_cpu.shape[1]
        self.spatial_shape = tuple(data_cpu.shape[2:-1])
        self.C = data_cpu.shape[-1]
        self.input_steps = input_steps
        self.rollout = rollout
        self.mean = mean
        self.std = std
        self.channels = self.C

        self.max_start = self.T - (self.input_steps + self.rollout)
        if self.max_start < 0:
            raise ValueError(
                f"T'={self.T} < input_steps+rollout={self.input_steps + self.rollout}"
            )

        self._build_pairs()

    def _build_pairs(self):
        self.pairs: List[Tuple[int, int]] = [(int(tr), 0) for tr in range(self.N)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        tr, start = self.pairs[idx]
        u = self.data[tr, start : start + self.input_steps + self.rollout]
        if (self.mean is not None) and (self.std is not None):
            u = (u - self.mean) / (self.std + 1e-8)
        x_hist = u[: self.input_steps]
        y_seq = u[self.input_steps :]

        x_hist = x_hist.movedim(-1, 1).reshape(
            self.input_steps * self.C, *self.spatial_shape
        )
        y_seq = y_seq.movedim(-1, 1)

        return x_hist.contiguous(), y_seq.contiguous()


class RolloutRAMDataset1D(Dataset):
    """Burgers-style RAM-backed dataset: `[N, T, X]` -> `(x_hist, y_seq)` windows."""

    def __init__(
        self,
        data_cpu: torch.Tensor,
        input_steps: int,
        rollout: int,
        mean: Optional[float],
        std: Optional[float],
    ):
        assert data_cpu.device.type == "cpu"
        self.data = data_cpu
        self.N, self.T, self.X = data_cpu.shape
        self.input_steps = input_steps
        self.rollout = rollout
        self.mean = mean
        self.std = std

        self.max_start = self.T - (self.input_steps + self.rollout)
        if self.max_start < 0:
            raise ValueError(
                f"T'={self.T} < input_steps+rollout={self.input_steps + self.rollout}"
            )

        self._build_pairs()

    def _build_pairs(self):
        self.pairs: List[Tuple[int, int]] = [(int(tr), 0) for tr in range(self.N)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        tr, start = self.pairs[idx]
        u = self.data[
            tr, start : start + self.input_steps + self.rollout, :
        ]  # [S, X], CPU
        if (self.mean is not None) and (self.std is not None):
            u = (u - self.mean) / (self.std + 1e-8)
        x = u[: self.input_steps]
        y = u[self.input_steps :]
        return x, y

