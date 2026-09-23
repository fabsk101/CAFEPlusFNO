# Dataset acquisition and identity

This directory tracks acquisition metadata, not dataset binaries. Dataset
acquisition is explicitly separated from every training entry point.

Download and verify all datasets:

```bash
python data/download_data.py --dataset all --data-root data
```

Individual choices are `--dataset airfoil221x51`, `--dataset burgers1d`, `--dataset cfd1d1024`,
`--dataset cfd2d128`, `--dataset darcy128`, `--dataset ns128`, and
`--dataset reacdiff1024`; `both`
retains its original Darcy+NS meaning. Every download and training entry point computes SHA256 and
rejects a mismatch.

## Geo-FNO Airfoil 221x51

- Source: <https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8>
- Fixed split: first 1,000 training / next 200 test samples
- Input: `stack([X, Y], dim=-1)`; target: `Q[:, 4]`
- Resolution: 221x51; no normalization, transpose, crop, or downsampling
- `NACA_Cylinder_X.npy`: 224,518,448 bytes, SHA256
  `23902420694ba8552e427480aa6b26c08ea4e7e0b2454ee83cd9e1d29060f35f`
- `NACA_Cylinder_Y.npy`: 224,518,448 bytes, SHA256
  `710c77e42c3fe49edcfc951b59693511e12f5491f9dcb77a506f395178481155`
- `NACA_Cylinder_Q.npy`: 1,122,591,728 bytes, SHA256
  `f79dfdaa3be95a5183981531e59b5a372cf3dd69986109f45426b3eb5410c58c`
- Training reads only verified local files and has no download or network
  fallback.

## PDEBench Burgers-1D

- Source: <https://darus.uni-stuttgart.de/api/access/datafile/268190>
- Fixed split: first 1,000 training / next 200 test trajectories
- Resolution: 1,024; 201 source time steps
- `1D_Burgers_Sols_Nu0.001.hdf5`: 8,232,968,312 bytes,
  SHA256 `afeeed1c40ce01d2ba5e1702f4f66b6bf95c65943eb0e4847691e9a10d1cb50d`
- Training is offline after verification. The pinned SirenFNO preprocessor may
  create the ignored derived cache `1D_Burgers_Sols_Nu0.001_rx1_rt1.hdf5`.

## PDEBench CFD-1D released-runtime task

- Source: <https://darus.uni-stuttgart.de/api/access/datafile/164672>
- Fixed split: first 1,800 training / next 200 test trajectories
- Resolution: 1,024; 101 field snapshots in the verified artifact
- Selected field: `Vx` only, matching the released SirenFNO runtime; density
  and pressure are not concatenated
- `1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5`:
  12,410,888,600 bytes, SHA256
  `86a2b8cf81f40191dbc40a7c2a9b268784979f2c1c269b59daa23c83885ebe8f`
- Training reads the verified raw file offline. Since `reduce_x=reduce_t=1`,
  the no-op preprocessing cache is elided and no absolute source path is
  written into generated HDF5 metadata.

## PDEBench CFD-2D released-runtime task

- Source: <https://darus.uni-stuttgart.de/api/access/datafile/164687>
- Fixed split: first 1,800 training / next 200 test trajectories
- Resolution: 128x128; 21 field snapshots in the verified artifact
- Selected HDF5 key: `/Vx` only, a single Vx velocity-component channel,
  matching the released SirenFNO runtime
- `2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`:
  55,050,245,208 bytes, SHA256
  `8f21323cb7b61e80dd6d5ed93190bbba4b1a3e462b4be927d75a8fb0e286f3e4`
- The official DaRUS file metadata reports MD5
  `5b21dcccaef4d2145ca579a71153c580`; the audited local file matches it.
- Training reads the verified raw file offline. Since `reduce_x=reduce_t=1`,
  the no-op preprocessing cache is elided and no absolute source path is
  written into generated HDF5 metadata.

## Darcy Flow 128x128

- Source: <https://zenodo.org/records/12784353>
- Fixed split: 1,000 training / 200 test examples
- Resolution: 128x128
- `darcy_train_128.pt`: 655,360,935 bytes,
  SHA256 `b0b98f29679459f9eeb03657835b683140abf186f62bcc92efb886f022d1447c`
- `darcy_test_128.pt`: 131,072,935 bytes,
  SHA256 `549ac096e1a772ac355297deed98955f7ca6d03c97295705f66e3c194cf20382`

## Navier--Stokes forcing 128x128

- Source: <https://zenodo.org/records/12825163>
- Fixed split: 1,000 training / 200 test examples
- Resolution: 128x128
- `nsforcing_train_128.pt`: 1,310,720,935 bytes,
  SHA256 `6c145749f40aa1f00c2088e662f6668e1ca98018353a0f27a814732278c1a964`
- `nsforcing_test_128.pt`: 262,144,935 bytes,
  SHA256 `a7e1ed8ebc050317b8ccb7ef35dfc488e835337fcdefc62f56e01596accccbe5`

## PDEBench Reaction-Diffusion 1D

- Source: <https://darus.uni-stuttgart.de/api/access/datafile/133177>
- Fixed split: first 1,000 training / next 200 test trajectories
- Resolution: 1,024; 101 source snapshots; 10 input states to 10 rollout states
- No normalization or spatial/temporal decimation
- `ReacDiff_Nu0.5_Rho1.0.hdf5`: 4,136,968,376 bytes, SHA256
  `0ccd649b1d5ecca8a5ae417a506dec5ef57ea365f7bce46afa190d34fe02dcc7`
- The official DaRUS metadata MD5 is
  `69a429239778d529cd419ed5888ea835`; the audited local file matches it.
- Training reads only the verified raw file and has no acquisition fallback.

Exact machine-readable identities and public sources are in `DATASETS.json`.
Use repository-relative `--data-root` examples for publication; local dataset
and output roots are never embedded in result metadata.
