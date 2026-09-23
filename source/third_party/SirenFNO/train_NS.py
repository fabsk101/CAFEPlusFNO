import argparse
import sys

import torch

from baseline.AMFNO import FNO2dMLP
from baseline.UFNO import UFNO
from neuralop import H1Loss, LpLoss, Trainer
from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from neuralop.models import FNO, UNO
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from SirenFNO2D import SirenFNO2d
from SirenSpectralConv_ablation import SirenHyperConv

device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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

def parse_args():
    parser = argparse.ArgumentParser(description="Train Navier-Stokes models.")
    parser.add_argument(
        "--model",
        default="tuckersirenfno",
        choices=MODEL_CHOICES,
        help="Model identifier to train.",
    )
    return parser.parse_args()


def build_model(model_name: str) -> torch.nn.Module:
    name = model_name.lower()

    if name == "sirenfno":
        
        model = SirenFNO2d(
            width=32,
            input_dim=1,
            output_dim=1,
            add_grid=True,
            padding=0,
            mlp_dropout=0.0,
            hidden_dim=64,
            omega=30.0,
            n_hidden=1,
            siren_dim_in=32,
            ff_sigma=512.0,
            learnable_ff=True,
            factorization="dense",
        )
    elif name == "cpsirenfno":
        model = SirenFNO2d(
            width=32,
            input_dim=1,
            output_dim=1,
            add_grid=True,
            padding=0,
            mlp_dropout=0.0,
            hidden_dim=64,
            omega=30.0,
            n_hidden=1,
            siren_dim_in=32,
            ff_sigma=512.0,
            learnable_ff=True,
            factorization="cp",
            rank=16,
        )
    elif name == "ttsirenfno":
        model = SirenFNO2d(
            width=32,
            input_dim=1,
            output_dim=1,
            add_grid=True,
            padding=0,
            mlp_dropout=0.0,
            hidden_dim=32,
            omega=30.0,
            n_hidden=1,
            siren_dim_in=32,
            ff_sigma=512.0,
            learnable_ff=True,
            factorization="tt",
            rank=16,
        )
    elif name == "tuckersirenfno":
        model = SirenFNO2d(
            width=32,
            input_dim=1,
            output_dim=1,
            add_grid=True,
            padding=0,
            mlp_dropout=0.0,
            hidden_dim=32,
            omega=30.0,
            n_hidden=1,
            siren_dim_in=16,
            ff_sigma=512.0,
            learnable_ff=True,
            factorization="tucker",
            rank=16,
        )
    elif name == "fno":
        model = FNO(
            n_modes=(32,32),
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
            n_layers=4,
            lifting_channels=64,
            projection_channels=64,
        )
    
    elif name == "tfno":
        model = FNO(
            n_modes=(32,32),
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
            n_layers=4,
            lifting_channels=64,
            projection_channels=64,
            implementation="factorized",
            factorization="cp",
            rank=0.05,
        )
        
    elif name == "sirenfno_ablation":
        model = FNO(
            n_modes=(32,32),
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
            n_layers=4,
            lifting_channels=64,
            projection_channels=64,
           # implementation="factorized",
            factorization=None,
            conv_module=SirenHyperConv,
        )
    elif name == "cpsirenfno_ablation":
        model = FNO(
            n_modes=(32,32),
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
            n_layers=4,
            lifting_channels=64,
            projection_channels=64,
            implementation="factorized",
            factorization="cp",
            rank=16,
            conv_module=SirenHyperConv,
        )
    elif name == "ttsirenfno_ablation":
        model = FNO(
            n_modes=(32,32),
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
            n_layers=4,
            lifting_channels=64,
            projection_channels=64,
            implementation="factorized",
            factorization="tt",
            rank=16,
            conv_module=SirenHyperConv,
        )
    elif name == "tuckersirenfno_ablation":
        model = FNO(
            n_modes=(32,32),
            hidden_channels=32,
            in_channels=1,
            out_channels=1,
            n_layers=4,
            lifting_channels=64,
            projection_channels=64,
            implementation="factorized",
            factorization="tucker",
            rank=14,
            conv_module=SirenHyperConv,
        )
    
    elif name == "amfno":
        model = FNO2dMLP(
            width=64,
            n1=32,
            n2=32,
            padding=0,
            input_dim=1,
            output_dim=1,
            mlp_dropout=0,
        )
        
    elif name == "ufno":
        model = UFNO(n_modes=(12,12),
                    in_channels=1,
                    out_channels=1,
                    hidden_channels=32,
                    n_layers=4,
                    positional_embedding="grid",
                    use_channel_mlp=True,
                    #channel_mlp_dropout=0.0,
                    #channel_mlp_expansion=0.5,
                    use_unet_from=2,
                    unet_dropout=0,
                    domain_padding=None,
                    fno_block_precision="full")
    
    elif name == "uno":
        model = UNO(in_channels=1,
            out_channels=1,
            hidden_channels=64,
            lifting_channels=256,
            projection_channels=256,
            uno_out_channels=[32,32,32,32,32,32],
            uno_n_modes=[[32,32],[32,32],[16,16],[16,16],[32,32],[32,32]],
            uno_scalings=[[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]],
            horizontal_skips_map=None,
            channel_mlp_skip="linear",
            n_layers = 6,
            channel_mlp_dropout=0.0,
            domain_padding=0.0
        )
        
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    return model.to(device)


def main():
    args = parse_args()
    # set_seed(42)

    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        n_train=1000, batch_size=32, train_resolution=128,
        test_resolutions=[128], n_tests=[200],
        test_batch_sizes=[32], data_root='./data/'
    )
    data_processor = data_processor.to(device)

    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = l2loss
    eval_losses = {'h1': h1loss, 'l2': l2loss}

    model_name = args.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f'\n========== Training model: {model_name} ==========\n')
    model = build_model(model_name)

    n_params = count_model_params(model)
    print(f'\nOur {model_name} model has {n_params} parameters.')
    sys.stdout.flush()

    optimizer = AdamW(model.parameters(),
                      lr=1e-3,
                      weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)

    print('\n### MODEL ###\n', model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')
    sys.stdout.flush()

    trainer = Trainer(model=model, n_epochs=500,
                      device=device,
                      data_processor=data_processor,
                      wandb_log=False,
                      eval_interval=1,
                      use_distributed=False,
                      verbose=True)

    trainer.train(train_loader=train_loader,
                  test_loaders=test_loaders,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=False,
                  training_loss=train_loss,
                  eval_losses=eval_losses)


if __name__ == "__main__":
    main()
