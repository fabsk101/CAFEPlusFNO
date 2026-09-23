# SirenFNO: Efficient and Full Frequency Learning of Fourier Neural Operators

Official code for *SirenFNO: Efficient and Full Frequency Learning of Fourier Neural Operators* (IJCAI 2026).

## Installation

```bash
pip install -r requirements.txt
```

## Running experiments

```bash
python train_Burgers.py        # 1D Burgers
python train_CFD.py            # 1D CFD
python train_CFD2D.py          # 2D CFD
python train_Darcy32.py        # Darcy (zero-shot super-resolution)
python train_Darcy128.py       # Darcy
python train_NS.py             # Navier–Stokes
python train_ReacDiff.py       # Reaction–Diffusion
```

Datasets download automatically on first run. Switch models with `--model`, e.g.:

```bash
python train_Burgers.py --model cpsirenfno
python train_NS.py --model fno
```

Available models:

- **SirenFNO:** `sirenfno`, `cpsirenfno`, `ttsirenfno`, `tuckersirenfno`
- **Baselines:** `fno`, `tfno`, `uno`, `amfno`, `ufno`
- **Ablations:** `sirenfno_ablation`, `cpsirenfno_ablation`, `ttsirenfno_ablation`, `tuckersirenfno_ablation`
