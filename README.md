# CAFE+FNO review artifact

This repository provides the source artifact accompanying the submitted paper.

## Contents

- `source/`: a browsable copy of the source archive.
- `artifacts/anonymous-cafe-plus-fno.zip`: the unchanged source archive.
- `SHA256SUMS.txt`: the SHA-256 checksum of that archive.

The manuscript reports experiments on Darcy, Navier-Stokes, Burgers,
Reaction-Diffusion, and Airfoil. CFD-1D and CFD-2D code is retained,
but their results are not included in the submitted comparison.

## Reproduction

Use `artifacts/anonymous-cafe-plus-fno.zip` for reproduction.
Extract it into a standalone directory outside any Git repository.
Follow the documentation inside the extracted archive.

The publication repository is a distribution wrapper. Its commit is not
the training-source commit recorded in the experimental artifacts.

Original training-source identifiers and the deployment compatibility
record are preserved. The source archive does not include dataset
binaries or trained checkpoints.