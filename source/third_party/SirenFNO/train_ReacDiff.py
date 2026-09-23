import argparse
import time
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline.AMFNO import FNO1dMLP
from neuralop.losses import LpLoss
from neuralop.models import FNO, UNO, FNO1d
from neuralop.utils import (
    count_model_params,
    evaluate_rel_l2_metrics,
    full_rel_l2_lp,
    rollout_loss_lp,
    rollout_step_model_1d_channel_first,
)
from SirenFNO1D import SirenFNO1d
from SirenSpectralConv_ablation import SirenHyperConv
from baseline.UFNO import UFNO1d
from utils import (
    RolloutRAMDataset,
    _find_data_dataset,
    ensure_data_available,
    load_subset_to_ram,
    preprocess_decimated_hdf5,
    set_seed,
)

MODEL_CHOICES = (
    "sirenfno",
    "cpsirenfno",
    "ttsirenfno",
    "tuckersirenfno",
    "uno",
    "fno",
    "amfno",
    "tfno",
    "ufno",
    "sirenfno_ablation",
    "cpsirenfno_ablation",
    "ttsirenfno_ablation",
    "tuckersirenfno_ablation",
)


@dataclass
class CFG:
    data_file: str = "Data/ReacDiff_Nu0.5_Rho1.0.hdf5"
    dataset_url: str = "https://darus.uni-stuttgart.de/api/access/datafile/133177"

    reduce_x: int = 1
    reduce_t: int = 1  # 201 -> 201 (no temporal decimation by default)

    input_steps: int = 10
    rollout: int = 10
    batch_size: int = 32
    epochs: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-4
    modes: int = 16
    hidden: int = 64
    n_layers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    physical_channels: int = 1

    n_train: int = 1000
    n_val: int = 0
    n_test: int = 200

    num_workers: int = 0
    pin_memory: bool = True
    use_amp: bool = False

    model: str = "sirenfno"

cfg = CFG()


def make_model(
    in_channels: int,
    out_channels: int,
    modes: int,
    hidden: int,
    n_layers: int,
    model_type: str = "sirenfno",
) -> nn.Module:
    model_key = model_type.lower()

    if model_key == "sirenfno":
        model = SirenFNO1d(
            width=32,
            padding=0,
            input_dim=in_channels,
            output_dim=out_channels,
            mlp_dropout=0.0,
            add_grid=True,
            siren_dim_in=16,
            hidden_dim=32,
            omega=15.0,
            n_hidden=1,
            ff_sigma=256,
            learnable_ff=True,
            factorization="dense",
        )
    elif model_key == "cpsirenfno":
        model = SirenFNO1d(
            width=32,
            padding=0,
            input_dim=in_channels,
            output_dim=out_channels,
            mlp_dropout=0.0,
            add_grid=True,
            siren_dim_in=16,
            hidden_dim=32,
            omega=15.0,
            n_hidden=1,
            ff_sigma=256,
            learnable_ff=True,
            factorization="cp",
            rank=8,
        )
    elif model_key == "ttsirenfno":
        model = SirenFNO1d(
            width=32,
            padding=0,
            input_dim=in_channels,
            output_dim=out_channels,
            mlp_dropout=0.0,
            add_grid=True,
            siren_dim_in=16,
            hidden_dim=32,
            omega=15.0,
            n_hidden=1,
            ff_sigma=256,
            learnable_ff=True,
            factorization="tt",
            rank=8,
        )
    elif model_key == "tuckersirenfno":
        model = SirenFNO1d(
            width=32,
            padding=0,
            input_dim=in_channels,
            output_dim=out_channels,
            mlp_dropout=0.0,
            add_grid=True,
            siren_dim_in=16,
            hidden_dim=32,
            omega=15.0,
            n_hidden=1,
            ff_sigma=256,
            learnable_ff=True,
            factorization="tucker",
            rank=8,
        )

    elif model_key == "uno":
        model = UNO(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=64,
            lifting_channels=128,
            projection_channels=128,
            uno_out_channels=[64, 64, 32, 32, 64, 64],
            uno_n_modes=[[1024,],[64,],[32,],[32,],[64,],[64,],],
            uno_scalings=[[1,],[1,],[1,],[1,],[1,],[1,],],
            horizontal_skips_map=None,
            channel_mlp_skip="linear",
            n_layers=6,
            #  channel_mlp_dropout=0.0,
            # domain_padding=0.2
        )

    elif model_key == "fno":
        model = FNO1d(
            n_modes_height=1024,
            hidden_channels=32,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=4,
            lifting_channels=64,
            projection_channels=64,
        )

    elif model_key == "amfno":
        model = FNO1dMLP(
            width=64,
            n1=10,
            padding=0,
            input_dim=in_channels,
            output_dim=out_channels,
            mlp_dropout=0,
        )

    elif model_key == "tfno":
        model = FNO1d(
            n_modes_height=1024,
            hidden_channels=32,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=4,
            lifting_channels=64,
            projection_channels=64,
            implementation="factorized",
            factorization="cp",
            rank=0.05,
        )

    elif model_key == "ufno":
        model = UFNO1d(
            n_modes_height=32,
            hidden_channels=64,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=6,
            lifting_channels=128,
            projection_channels=128,
            positional_embedding="grid",
        )

    elif model_key == "sirenfno_ablation":
        model = FNO(
            n_modes=(1024,),
            # n_modes_height=1024,
            hidden_channels=32,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=4,
            implementation="factorized",
            factorization=None,
            lifting_channels=64,
            projection_channels=64,
            conv_module=SirenHyperConv,
        )

    elif model_key == "cpsirenfno_ablation":
        model = FNO(
            n_modes=(1024,),
            # n_modes_height=1024,
            hidden_channels=32,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=4,
            implementation="factorized",
            factorization="cp",
            rank=8,
            lifting_channels=64,
            projection_channels=64,
            conv_module=SirenHyperConv,
        )

    elif model_key == "ttsirenfno_ablation":
        model = FNO(
            n_modes=(1024,),
            # n_modes_height=1024,
            hidden_channels=32,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=4,
            implementation="factorized",
            factorization="tt",
            rank=8,
            lifting_channels=64,
            projection_channels=64,
            conv_module=SirenHyperConv,
        )

    elif model_key == "tuckersirenfno_ablation":
        model = FNO(
            n_modes=(1024,),
            # n_modes_height=1024,
            hidden_channels=32,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=4,
            implementation="factorized",
            factorization="tucker",
            rank=8,
            lifting_channels=64,
            projection_channels=64,
            conv_module=SirenHyperConv,
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from: {', '.join(MODEL_CHOICES)}"
        )

    return model.to(cfg.device)


lp_rel_train = LpLoss(d=1, p=2, reduction="mean")
lp_rel_eval_sum = LpLoss(d=1, p=2, reduction="mean")


def train():
    ensure_data_available(cfg.data_file, cfg.dataset_url)

    if cfg.reduce_x == 1 and cfg.reduce_t == 1:
        reduced_path = cfg.data_file
        print("[preprocess] reduce_x=reduce_t=1 -> using original file without decimation.")
    else:
        reduced_path = preprocess_decimated_hdf5(cfg.data_file, cfg.reduce_x, cfg.reduce_t)

    with h5py.File(reduced_path, "r") as f:
        ds = _find_data_dataset(f)
        N_total, T_red, X_red = ds.shape[0], ds.shape[1], ds.shape[2]

    need_T = cfg.input_steps + cfg.rollout
    if T_red < need_T:
        raise ValueError(
            f"After temporal decimation (rt={cfg.reduce_t}), T'={T_red} < input_steps+rollout={need_T}. "
            f"Decrease 'rollout' or use smaller 'reduce_t'."
        )

    assert cfg.n_train + cfg.n_val + cfg.n_test <= N_total, (
        "Split exceeds dataset size."
    )
    all_idx = np.arange(N_total)
    train_idx = all_idx[: cfg.n_train]
    test_idx = all_idx[cfg.n_train + cfg.n_val : cfg.n_train + cfg.n_val + cfg.n_test]
    train_cpu = load_subset_to_ram(reduced_path, train_idx)  # [Ntr, T', X', C]
    test_cpu = load_subset_to_ram(reduced_path, test_idx)  # [Nte, T', X', C]

    cfg.physical_channels = int(train_cpu.shape[-1]) if train_cpu.ndim == 4 else 1

    mean, std = None, None  # disable standardization

    ds_train = RolloutRAMDataset(
        train_cpu, cfg.input_steps, cfg.rollout, mean, std
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )

    ds_train_eval = RolloutRAMDataset(
        train_cpu, cfg.input_steps, cfg.rollout, mean, std
    )
    dl_train_eval = DataLoader(
        ds_train_eval,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    ds_test_eval = RolloutRAMDataset(
        test_cpu, cfg.input_steps, cfg.rollout, mean, std
    )
    dl_test_eval = DataLoader(
        ds_test_eval,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    in_channels = cfg.input_steps * cfg.physical_channels
    out_channels = cfg.physical_channels
    model = make_model(in_channels, out_channels, cfg.modes, cfg.hidden, cfg.n_layers, cfg.model)
    n_params = count_model_params(model)
    print(f"\nOur model has {n_params} parameters.")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp) if cfg.use_amp else None

    field_dim = max(1, getattr(cfg, "physical_channels", 1))
    rollout_fn = lambda m, x, steps, pushforward_detach=True: rollout_step_model_1d_channel_first(
        m, x, steps, pushforward_detach=pushforward_detach, field_dim=field_dim
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        for xb, yb in dl_train:
            xb = xb.to(cfg.device, non_blocking=True).float()  # [B, C_in, X']
            yb = yb.to(cfg.device, non_blocking=True).float()  # [B, T, X']
            opt.zero_grad(set_to_none=True)

            if cfg.use_amp:
                with torch.amp.autocast("cuda"):
                    pred_seq = rollout_fn(
                        model, xb, cfg.rollout, pushforward_detach=False
                    )

                    loss = rollout_loss_lp(pred_seq, yb, lp_rel_train)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred_seq = rollout_fn(
                    model, xb, cfg.rollout, pushforward_detach=False
                )

                loss = rollout_loss_lp(pred_seq, yb, lp_rel_train)
                loss.backward()
                opt.step()

        if cfg.device == "cuda":
            torch.cuda.synchronize()
        train_time = time.perf_counter() - t0
        sched.step()

        train_step_rel, train_full_rel = evaluate_rel_l2_metrics(
            model,
            dl_train_eval,
            cfg.rollout,
            rollout_fn,
            lp_rel_train,
            lambda pred, y: full_rel_l2_lp(pred, y, lp_rel_train),
            cfg.device,
        )
        test_step_rel, test_full_rel = evaluate_rel_l2_metrics(
            model,
            dl_test_eval,
            cfg.rollout,
            rollout_fn,
            lp_rel_train,
            lambda pred, y: full_rel_l2_lp(pred, y, lp_rel_train),
            cfg.device,
        )
        print(
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | T'={T_red} X'={X_red} | train_time {train_time:.2f}s | step_rel(train/test) {train_step_rel:.10f}/{test_step_rel:.10f} | full_rel(train/test) {train_full_rel:.10f}/{test_full_rel:.10f}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Reaction Diffusion Training")
    p.add_argument("--data_file", type=str, default=cfg.data_file)
    p.add_argument("--reduce_x", type=int, default=cfg.reduce_x)
    p.add_argument("--reduce_t", type=int, default=cfg.reduce_t)
    p.add_argument("--dataset_url", type=str, default=cfg.dataset_url)
    p.add_argument("--input_steps", type=int, default=cfg.input_steps)
    p.add_argument("--rollout", type=int, default=cfg.rollout)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--modes", type=int, default=cfg.modes)
    p.add_argument("--hidden", type=int, default=cfg.hidden)
    p.add_argument("--n_layers", type=int, default=cfg.n_layers)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--weight_decay", type=float, default=cfg.weight_decay)
    p.add_argument("--n_train", type=int, default=cfg.n_train)
    p.add_argument("--n_val", type=int, default=cfg.n_val)
    p.add_argument("--n_test", type=int, default=cfg.n_test)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument(
        "--use_amp", action="store_true", help="Enable AMP (off by default)."
    )
    p.add_argument(
        "--model",
        type=str,
        default=cfg.model,
        choices=MODEL_CHOICES,
        help=f"Model type: {', '.join(MODEL_CHOICES)}",
    )
    args = p.parse_args()

    for k, v in vars(args).items():
        setattr(cfg, k, v)

    set_seed(cfg.seed)

    train()
