# CAFE+FNO: anonymous ICLR 2027 experiment repository

This repository provides anonymous, reproducible Airfoil, Burgers-1D, CFD-1D,
CFD-2D, Darcy Flow, Navier--Stokes, and Reaction-Diffusion experiments for
CAFE+FNO. Exact source revisions,
configurations, data splits, seeds, and a reference environment are pinned for
reproducible reruns; bitwise equality across GPU architectures is not claimed.

## Unified experimental stack

- **Experimental backend:** official SirenFNO repository at pinned commit
  `81918ecce323a2fd5c5a54db917598bda088574b`.
- **NeuralOperator:** bundled implementation distributed with that SirenFNO
  revision (`neuralop` version `1.0.2`, Git tree
  `fbc6aa738d6ffc2e2ca6808e724b0083c3a688eb`).
- **Baseline models:** official AM-FNO, U-FNO, and dense/CP/TT/Tucker SirenFNO
  implementations from that SirenFNO revision.
- **Proposed model:** the CAFE+FNO implementation in this anonymous repository.

The pinned upstream is the public repository
<https://github.com/pengqingshi/SirenFNO.git>, included as the Git submodule
`third_party/SirenFNO`. `third_party/UPSTREAM_VERSIONS.json` records its commit,
bundled NeuralOperator tree, and canonical Git blob IDs for every upstream
source file used by these experiments. Git-object verification is independent
of CRLF/LF checkout style.

Darcy and Navier--Stokes use the same bundled Trainer, AdamW, LpLoss/H1Loss,
dataset processing semantics, and scheduler path. Airfoil, Burgers, CFD-1D,
CFD-2D, and Reaction-Diffusion retain the pinned author training and rollout
paths and `torch.optim.AdamW`. CFD-1D, CFD-2D, and Reaction-Diffusion use the
repository-owned corrected paper evaluator described below; their training
loss remains the pinned mean-reduced `LpLoss`. A
preloaded or installed `site-packages/neuralop` is rejected; it is never
silently reloaded or mixed with the pinned backend.
All reported comparison results must be produced by rerunning the models under
this unified stack. Published baseline numbers must not be mixed with these
reproduced results.

## Clone and environment

For the anonymous Git repository:

```bash
git clone --recurse-submodules <ANONYMOUS_REPOSITORY_URL>
cd <repository>
conda env create -f environment.yml
conda activate anonymous-cafe-plus-fno
python -m pip install -e . --no-deps
python scripts/setup_sirenfno.py
python scripts/verify_environment.py
```

The reference environment is Python 3.13.11, PyTorch 2.8.0 with CUDA 12.8,
NumPy 2.4.6, h5py 3.15.1, tensorly 0.9.0, tensorly-torch 0.5.0, opt-einsum
3.4.0, and the versions in `requirements-lock.txt`. The CUDA build is selected
from PyTorch's public cu128 wheel index. `scripts/verify_environment.py` fails
if the active runtime differs from the reference or CUDA is unavailable.

For an ICLR supplementary ZIP, the release builder bundles the exact
SirenFNO snapshot at commit
`81918ecce323a2fd5c5a54db917598bda088574b` without any Git metadata. After
extraction, verify the bundled tree and environment with:

```bash
python scripts/setup_sirenfno.py
python scripts/verify_environment.py
```

The setup script refuses a wrong or modified upstream source tree. In an
anonymous GitHub clone, the superproject commit and dirty state are verified
from Git metadata. In the Git-less supplementary ZIP,
`RELEASE_PROVENANCE.json` format v2 records the committed superproject and
upstream identities and hashes every regular archive member except the
manifest itself (avoiding a circular self-hash).

Do not submit a raw working-directory archive containing `.git/`. Use only the
archive produced by `scripts/build_anonymous_release.py`. It is built from a
clean committed HEAD, rejects dirty or untracked source, excludes `.git`
directories and pointer files at every depth, merges separately verified
superproject and pinned-upstream archives, scans the actual ZIP contents, and
performs a fresh Git-less extraction smoke test before atomic publication:

```bash
python scripts/build_anonymous_release.py --data-root data
```

Use `--overwrite` only to replace an older generated `dist/` artifact after the
new temporary archive passes every verification. An existing archive can be
checked again with
`python scripts/build_anonymous_release.py --verify-archive <archive.zip>` and
`python scripts/audit_anonymity.py --ref HEAD --archive <archive.zip>`.

## Data

Dataset binaries are not redistributed. Acquire all public files through the
explicit preparation entry point and verify their SHA256 values:

```bash
python data/download_data.py --dataset all --data-root data
```

| Dataset | Public source | Files | Fixed split | Resolution |
| --- | --- | --- | --- | --- |
| Geo-FNO Airfoil | <https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8> | `NACA_Cylinder_X.npy`, `NACA_Cylinder_Y.npy`, `NACA_Cylinder_Q.npy` | first 1,000 train / next 200 test | 221x51 |
| PDEBench Burgers | <https://darus.uni-stuttgart.de/api/access/datafile/268190> | `1D_Burgers_Sols_Nu0.001.hdf5` | first 1,000 train / next 200 test | 1,024 |
| PDEBench CFD-1D (`Vx`) | <https://darus.uni-stuttgart.de/api/access/datafile/164672> | `1D_CFD_Rand_Eta0.01_Zeta0.01_periodic_Train.hdf5` | first 1,800 train / next 200 test | 1,024 |
| PDEBench CFD-2D (`Vx`) | <https://darus.uni-stuttgart.de/api/access/datafile/164687> | `2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5` | first 1,800 train / next 200 test | 128x128 |
| Darcy Flow | <https://zenodo.org/records/12784353> | `darcy_train_128.pt`, `darcy_test_128.pt` | 1,000 train / 200 test | 128x128 |
| Navier--Stokes forcing | <https://zenodo.org/records/12825163> | `nsforcing_train_128.pt`, `nsforcing_test_128.pt` | 1,000 train / 200 test | 128x128 |
| PDEBench Reaction-Diffusion | <https://darus.uni-stuttgart.de/api/access/datafile/133177> | `ReacDiff_Nu0.5_Rho1.0.hdf5` | first 1,000 train / next 200 test | 1,024 |

Exact sizes and hashes are recorded in `data/DATASETS.json` and checked before
each experiment. See `data/README.md` for the individual values.

NS, Airfoil, Burgers, CFD-1D, CFD-2D, and Reaction-Diffusion training are
strictly offline after that verification step. The
NS loader reproduces the pinned `load_navier_stokes_pt` pipeline with
`download=False`. The Burgers loader reuses the pinned preprocessing, RAM
loading, train-only normalization, rollout dataset, and DataLoader utilities.
The CFD-1D loader reproduces the released SirenFNO runtime's `Vx`-only field
selection and elides only its no-op `reduce_x=reduce_t=1` cache materialization.
The CFD-2D loader applies the same audited policy to the released 2-D runtime:
the pinned selector chooses `/Vx`, yielding one physical channel, and the no-op
cache materialization is elided.
The Reaction-Diffusion loader reuses the released SirenFNO dataset discovery,
RAM loading, rollout dataset, and DataLoader behavior directly on the verified
raw file (`reduce_x=reduce_t=1`). The artifact contains 101 snapshots although
the source comment says 201; the released 10-to-10 window requires no
interpolation or decimation.
Missing or mismatched files stop the run; only `data/download_data.py` performs
dataset acquisition.

Airfoil stacks the structured X/Y coordinates as two input channels and uses
`Q[:, 4]` as the scalar target. It applies no normalization, transpose, crop,
or downsampling. The fixed split is `[0:1000]` for training and `[1000:1200]`
for testing; there is no validation split.

## Paper protocol

All seven entry points run one model and one explicit seed per process. Darcy
and Navier--Stokes use batch size 32, 500 epochs, bundled AdamW (`lr=1e-3`,
`weight_decay=1e-4`), cosine scheduling (`T_max=500`), relative L2 training,
relative L2 plus H1 evaluation every epoch, no validation set, no regularizer,
and fixed-final-epoch selection. Darcy uses modes 16 and the official
SirenFNO/CAFE ranks 8 (CP), 8 (TT), and 10 (Tucker). Navier--Stokes uses modes
32 and ranks 16 for all three factorized SirenFNO and CAFE+FNO variants. Main
CAFE comparisons keep `learnable_sigma=False`.

Burgers uses the pinned 10-input-step to 10-step autoregressive rollout with
`pushforward_detach=False`, train-only global mean/std normalization, batch 32,
500 epochs, `torch.optim.AdamW` (`lr=1e-3`, `weight_decay=1e-4`), cosine
scheduling (`T_max=500`), step/full relative-L2 evaluation, no validation, and
fixed-final-epoch selection. Its AM-FNO baseline uses the pinned public
constructor setting `width=64`.

CFD-1D follows the released SirenFNO runtime: `Vx` only, 10 input states to 10
autoregressive predictions, `field_dim=1`, no normalization, batch 32, 500
epochs, `torch.optim.AdamW` (`lr=1e-3`, `weight_decay=1e-4`), cosine scheduling
(`T_max=500`), corrected step-average/trajectory relative-L2 evaluation, no
validation, and fixed-final-epoch selection. The public artifact contains 101 field snapshots although the
pinned source comment refers to 201; no interpolation is performed.

CFD-2D follows the released SirenFNO runtime: `/Vx` only, 5 input states to 5
autoregressive predictions, `field_dim=1`, no normalization, batch 32, 500
epochs, `torch.optim.AdamW` (`lr=1e-3`, `weight_decay=1e-4`), cosine scheduling
(`T_max=500`), corrected step-average/trajectory relative-L2 evaluation, no
validation, and fixed-final-epoch selection. Its main reporting metric is
`final_test_corrected_trajectory_relative_l2`;
`final_test_corrected_step_relative_l2` is secondary.

Airfoil uses batch size 8, 500 epochs, `torch.optim.AdamW` (`lr=1e-3`,
`weight_decay=1e-4`), and relative-L2 training/evaluation. Its cosine scheduler
uses `T_max = epochs * len(train_loader)`, steps after every optimizer step,
and preserves the additional epoch-end step in the audited AM-FNO Airfoil
source. This author-source-exact scheduler policy is applied to all compared
models. The experiment has no validation set and uses fixed-final-epoch
selection.

Reaction-Diffusion follows the released SirenFNO runtime: one scalar field at
resolution 1,024, 10 input states to 10 autoregressive predictions,
`pushforward_detach=False`, no normalization, batch 32, and 500 epochs. It uses
`torch.optim.AdamW` (`lr=1e-3`, `weight_decay=1e-4`),
`CosineAnnealingLR(T_max=500)` stepped exactly once after each epoch's training
loop, and the pinned time-summed
`LpLoss(d=1, p=2, reduction="mean")` rollout training loss. Paper evaluation
uses the corrected step-average and trajectory
metrics. There is no validation or best-test selection; the precommitted main
metric is `final_test_corrected_trajectory_relative_l2` at the fixed final
epoch. The released
pipeline is reused for all twelve comparison models, but all reported values
must come from our unified reruns; published SirenFNO table values are not
mixed with them.

### Training loss versus paper evaluation

The evaluation correction does not alter any training loss, optimizer,
scheduler, rollout, normalization, or model parameter update. For CFD-1D,
CFD-2D, and Reaction-Diffusion, predictions and targets are explicitly arranged
as `[B,T,C,*spatial]` in physical/source-value space (these three pipelines do
not normalize their fields). With `epsilon=1e-12`, the reported definitions are:

```text
step[i,t] = ||prediction[i,t] - target[i,t]||_2 over (C,*spatial)
            / max(||target[i,t]||_2 over (C,*spatial), epsilon)
corrected_step_relative_l2 = arithmetic mean of step[i,t] over samples and time

trajectory[i] = ||prediction[i] - target[i]||_2 over (T,C,*spatial)
                / max(||target[i]||_2 over (T,C,*spatial), epsilon)
corrected_trajectory_relative_l2 = arithmetic mean of trajectory[i] over samples
```

Every sample has equal weight, including an incomplete final batch. Channel
norms are joint rather than channel-wise, the actual processed sample count and
horizon are stored, and non-finite inputs or outputs are rejected. Evaluation
runs under `no_grad`, then restores all module train/eval flags and CPU/CUDA RNG
states. Darcy and Navier--Stokes retain their decoded physical-space relative
L2/H1 protocol, Burgers retains its train-normalized rollout protocol, and
Airfoil retains its unnormalized sample-relative-L2 protocol.

The unchanged protocols are versioned separately. Burgers computes joint
space (step) and joint time/space (full trajectory) ratios in normalized space
with denominator `target_norm + 1e-12`, sums per-sample ratios, then divides by
the actual sample/time counts. Airfoil uses the pinned
`LpLoss(d=2,p=2,reduction="sum",eps=1e-8)` on its single physical output
channel and divides by the sample count. Darcy and Navier--Stokes use the same
pinned relative-L2 epsilon/reduction plus pinned H1Loss after the data
processor decodes outputs and targets to physical space. Their versioned
re-evaluation metadata records these definitions instead of presenting them as
the corrected three-dataset protocol.

At the pinned SirenFNO commit, `evaluate_rel_l2_metrics()` combines a
mean-reduced batch loss with a second division by sample count, and
`full_rel_l2_lp()` does not compute the joint time trajectory norm used above.
Those upstream files remain byte-for-byte unchanged. Records made with that
path are classified as
`legacy_sirenfno_batch_reduced_relative_l2_v0` and are never mixed with
`corrected_relative_l2_v1`. This repository observation does not establish that
the numerical values reported in the SirenFNO paper used the affected path or
are incorrect.

Airfoil's SirenFNO dense/CP/TT/Tucker variants use the explicit,
researcher-precommitted common capacity `hidden_dim=32`, `siren_dim_in=32`,
`ff_sigma=512`, and `rank=8` for each factorized variant. Airfoil was not
evaluated by SirenFNO, so this is not an author-provided Airfoil configuration;
it was fixed without validation/test results to keep generator capacity equal
across factorization comparisons. FNO and TFNO-CP are unified reruns
using the pinned project implementations with width 32, four layers, and the
legacy effective retained support of 12 modes. TFNO-CP is not identified with
AM-FNO's F-FNO, and AM-FNO Table 2 reproduction is not claimed.

U-FNO is excluded from the unified Airfoil rerun because the exact
implementation behind the published result cannot be reconstructed from the
released artifacts without an Airfoil-specific modification. This exclusion
was fixed before full paper training, is not performance-based, and the
published U-FNO number is not mixed into the unified rerun table.

For CFD-1D, `final_test_corrected_trajectory_relative_l2` is the precommitted
main-table reporting metric. `final_test_corrected_step_relative_l2` is
secondary. This choice fixes the repository's reporting convention; it does
not claim to reconstruct an unpublished aggregation convention behind a
SirenFNO table.

The pinned AM-FNO constructor contains an unused tensor expression with an
unconditional `.cuda()` call. Paper AM-FNO runs therefore target the pinned
CUDA reference environment. CPU-only synthetic constructor/forward tests use
a narrowly scoped compatibility shim for that unused call; this is test
infrastructure and is not a claim that the released AM-FNO implementation has
full CPU support. The pinned upstream source is not modified.

The pinned public Burgers AM-FNO branch uses width 64. This repository therefore
restores `width=64` and records 823,073 parameters for that baseline. Earlier
local width-32 runs and their 207,617-parameter checkpoints remain distinct;
they must be retrained and must never be relabeled as width-64 results. The
released Burgers U-FNO constructor at the pinned SirenFNO
commit does not reproduce the Burgers U-FNO parameter count reported in Table
3. Because the published configuration cannot be uniquely reconstructed from
public artifacts, this experiment uses the released constructor without
reverse-engineering an undocumented alternative. Its recorded count is
1,883,073; Table 3 architecture parity is not claimed.

Most experiments use the following twelve-model roster:

```text
fno, ufno, tfno_cp, amfno,
sirenfno, cpsirenfno, ttsirenfno, tuckersirenfno,
cafe_plus_fno, cp_cafe_plus_fno, tt_cafe_plus_fno,
tucker_cafe_plus_fno
```

Airfoil uses the following eleven reproducibly rerunnable models:

```text
fno, tfno_cp, amfno,
sirenfno, cpsirenfno, ttsirenfno, tuckersirenfno,
cafe_plus_fno, cp_cafe_plus_fno, tt_cafe_plus_fno,
tucker_cafe_plus_fno
```

Canonical single runs:

```bash
python -m experiments.train_darcy --model fno --seed 42 --data-root data
python -m experiments.train_ns2d --model fno --seed 42 --data-root data
python -m experiments.train_burgers --model fno --seed 42 --data-root data
python -m experiments.train_cfd1d --model fno --seed 42 --data-root data
python -m experiments.train_cfd2d --model fno --seed 42 --data-root data
python -m experiments.train_airfoil --model fno --seed 42 --data-root data
python -m experiments.train_reacdiff --model fno --seed 42 --data-root data
```

Default outputs use revision-specific roots:

```text
results/darcy128_sirenfno_81918ec/<model>/seed_<seed>/
results/ns128_sirenfno_81918ec/<model>/seed_<seed>/
results/burgers1024_sirenfno_81918ec_protocol_v2/<model>/seed_<seed>/
results/cfd1d1024_sirenfno_81918ec/<model>/seed_<seed>/
results/cfd2d128_sirenfno_81918ec/<model>/seed_<seed>/
results/airfoil221x51_sirenfno_81918ec/<model>/seed_<seed>/
results/reacdiff1024_sirenfno_81918ec/<model>/seed_<seed>/
```

Each completed run contains `training_log.csv`, `summary.json`, and
`final_checkpoint.pt`. Existing artifacts are protected unless `--overwrite`
is supplied. Full runs require a clean source tree by default; the explicit
`--allow-dirty-source` option is for diagnostics only and records only a dirty
boolean, never raw status or local paths.

Checkpoint environment metadata is normalized to exact Python built-in values
before saving. For pinned neuraloperator models, the upstream mapping entry
`model_state_dict["_metadata"]` is separated as
`model_initialization_metadata`: ordinary initialization values stay built-in,
and the known GELU function and `SpectralConv` class become fixed identifiers.
This mapping entry is distinct from the state dict's `._metadata` attribute;
the latter's module-version data, all parameter/buffer tensors, ordering,
dtype, and shape are preserved. Optimizer and scheduler state structures are
unchanged. New checkpoints therefore load with
`torch.load(..., map_location="cpu", weights_only=True)` without an allowlist.
The shared reader never falls back to unrestricted pickle. For an older local
checkpoint whose only restricted-load incompatibility is PyTorch's historical
`TorchVersion` metadata, re-evaluation requires the explicit
`--allow-legacy-torch-version` option; the resulting record says that this
narrow compatibility path was used. A trusted legacy checkpoint containing the
pinned upstream neuraloperator initialization metadata instead requires
`--allow-legacy-neuraloperator-metadata`. That option first verifies the pinned
SirenFNO checkout, then scopes exactly `SpectralConv` and PyTorch GELU to one
`weights_only=True` read; it neither registers process-wide safe globals nor
changes the legacy file.

The independent-process five-seed sweep uses seeds `0 42 73 108 202`:

```bash
python scripts/run_paper_sweep.py --dataset darcy128 --data-root data
python scripts/run_paper_sweep.py --dataset ns128 --data-root data
python scripts/run_paper_sweep.py --dataset burgers1024 --data-root data
python scripts/run_paper_sweep.py --dataset cfd1d1024 --data-root data
python scripts/run_paper_sweep.py --dataset cfd2d128 --data-root data
python scripts/run_paper_sweep.py --dataset airfoil221x51 --data-root data
python scripts/run_paper_sweep.py --dataset reacdiff1024 --data-root data --device cuda
```

For the compact CAFE-only Darcy rerun, use the dedicated wrapper; it dispatches
only the four CAFE models and writes to a separate result root:

```bash
python scripts/run_cafe_darcy_sweep.py --data-root data
```

Dense CAFE+FNO retains its canonical joint CAFE+ mode generator unchanged.
CP, TT, and Tucker retain the SirenFNO functional tensor contractions but use
compact, independent CAFE+ factor generators (RFF 16, Chebyshev 8, branch width
12, and factorization-specific hidden width). The full dense Fourier kernel is
still reconstructed before spectral multiplication, so this change reduces
learnable generator parameters and improves initialization conditioning; it
does not claim reduced spectral-contraction FLOPs or peak kernel memory.

A run is complete only when its result path, summary, checkpoint, complete
epochs 1--500 CSV, and current manifests agree. The shared audit checks all
seven datasets for dataset/experiment/model/variant/seed identity; exact model,
training, data, and evaluation configurations; fixed-final-epoch selection;
clean training and evaluation provenance; the pinned SirenFNO revision; finite
metrics and tensors; and the externally stored checkpoint SHA256. It
reconstructs the expected model and strictly loads the nonempty state. Missing
or duplicate epochs, partial artifacts, metadata-only or corrupt checkpoints,
key/shape mismatch, source/config/data/protocol mismatch, and NaN/Infinity all
fail with explicit reason codes. Incomplete runs are never silently skipped,
deleted, or overwritten.

Training and evaluation provenance are independent:
`training_source_commit` remains the source that produced the weights, while
`evaluation_source_commit` records the evaluator. A structurally valid
historical checkpoint may therefore have a current corrected evaluation record
without pretending it was trained by current source. The audit distinguishes a
valid training run with legacy evaluation from an incomplete training run.
Unverifiable legacy provenance is not promoted to the current schema.
After a source commit containing only checkpoint storage/loading/validation
changes, reuse is never inferred from all repository history. It must name both
the exact recorded training commit and exact checkpoint digest with
`--reuse-training-source-commit <40-hex>` and
`--reuse-checkpoint-sha256 <64-hex>`. The reader also verifies the real Git diff
between that commit and the current evaluation commit and rejects every changed
path outside the checkpoint, sweep-validation, summarization, tests, and this
documentation. The changed paths and diff SHA256 remain in each evaluation
record and the aggregate JSON/CSV. Records using this exact bridge may share an
aggregation source commit with newly trained records, while their original
`training_source_commit` values and per-seed compatibility paths remain visible.

The same training-artifact audit is used by sweep completion and
re-evaluation: it reads the real CSV, requires every epoch through the fixed
final epoch, verifies required timing/learning-rate and version-appropriate
metric columns, and cross-checks source commit/dirty aliases in the summary,
checkpoint, and checkpoint environment. `--allow-dirty-source` applies only to
the evaluator checkout. A provenance-complete dirty training bundle can be
opened only with `--diagnostic-allow-dirty-training-artifact`, and its output is
marked diagnostic-only and cannot enter paper aggregation; missing dirty
provenance is never inferred as clean.

Run metadata includes the complete model configuration, factorization/rank,
parameter count, seed, dataset hashes and split, optimizer/scheduler/losses,
runtime versions and GPU model, anonymous repository commit, pinned SirenFNO
commit, sanitized repository-relative import sources, and source-tree dirty
state. It excludes usernames, home paths, institutions, hostnames, personal
emails/remotes, and raw failed-run tracebacks.

### Checkpoint re-evaluation and paper tables

`scripts/summarize_results.py` reads training artifacts without modifying them,
reconstructs the stored model configuration, performs strict state loading,
restores the verified test split/preprocessing, and writes versioned results
under a separate evaluation directory. Select one or more checkpoints
explicitly:

```bash
python scripts/summarize_results.py reevaluate \
  --dataset cfd1d1024 \
  --checkpoints results/cfd1d1024_sirenfno_81918ec/fno/seed_0/final_checkpoint.pt \
                results/cfd1d1024_sirenfno_81918ec/fno/seed_42/final_checkpoint.pt \
  --data-root data --batch-size 32 --device cuda
```

The result records include checkpoint SHA256, original training source,
evaluation source, exact model/training/data configurations, metric definition
and version, evaluation space, horizon, and actual test sample count. The CLI
does not rescale old scores, select a best-test checkpoint, recalibrate CAFE
bases, or replace the source checkpoint. A changed evaluation batch size keeps
the same metric definition.

Aggregate corrected records into seed-level CSV, aggregate CSV, JSON, Markdown,
and LaTeX tables with:

```bash
python scripts/summarize_results.py aggregate \
  --inputs results/cfd1d1024_sirenfno_81918ec \
  --output-dir paper_tables/cfd1d
```

The default required seeds are `0 42 73 108 202`. Results are grouped only when
model configuration, data identity, training protocol/source, and evaluation
definition/source all match. Duplicate or missing seeds and non-finite values
are rejected. Means use all selected seeds and standard deviations are sample
standard deviations (`ddof=1`); for one seed the standard deviation is
`undefined`, never zero. Partial aggregation requires `--allow-partial` and is
marked as partial. Missing timing remains missing and re-evaluation time is not
reported as training time.

Paper aggregation re-derives its contract from the repository-owned sweep
configuration, dataset manifest, and resolved model constructor. It requires
the approved 500-epoch experiment, exact split/preprocessing/normalization and
horizon/sample counts, matching factorization/rank/parameter count, clean
verified provenance, and finite nonnegative error metrics. Payload and grouping
hashes are recomputed rather than trusted. `--allow-partial` relaxes only seed
shortage; it does not admit diagnostic, dirty, wrong-epoch, wrong-split, or
otherwise incompatible records. Evaluation batch size alone does not split an
otherwise identical metric contract. CLI success and error envelopes show
repository-relative paths or fixed `<external-results>` identities and never
echo arbitrary external basenames or exception text.

## Verification and diagnostics

Run the release gates from the repository root:

```bash
python scripts/setup_sirenfno.py
python scripts/verify_environment.py
python scripts/audit_anonymity.py
python scripts/audit_reproducibility.py
python -B -m unittest discover -s tests -v
```

The dataset-dependent checks below verify official-data loading and
dataset-specific validation/smoke behavior. Model construction, output shape
and finiteness, backward, and parameterization/configuration checks are covered
by the corresponding synthetic/model tests.

```bash
DARCY_DATA_ROOT=data python -B -m unittest tests.test_official_data.DarcyModelSmokeTests -v
NS_DATA_ROOT=data python -B -m unittest tests.test_official_data.NSOfficialDataSmokeTests -v
BURGERS_DATA_ROOT=data python -B -m unittest tests.test_official_data.BurgersOfficialDataSmokeTests -v
CFD1D_DATA_ROOT=data python -B -m unittest tests.test_official_data.CFD1DOfficialDataSmokeTests -v
CFD2D_DATA_ROOT=data python -B -m unittest tests.test_official_data.CFD2DOfficialDataSmokeTests -v
AIRFOIL_DATA_ROOT=data python -B -m unittest tests.test_official_data.AirfoilOfficialDataSmokeTests -v
REACDIFF_DATA_ROOT=data python -B -m unittest tests.test_official_data.ReacDiffOfficialDataTests -v
```

PowerShell uses `$env:DARCY_DATA_ROOT = "data"` and
`$env:NS_DATA_ROOT = "data"`, `$env:BURGERS_DATA_ROOT = "data"`, or
`$env:CFD1D_DATA_ROOT = "data"`, `$env:CFD2D_DATA_ROOT = "data"`, or
`$env:AIRFOIL_DATA_ROOT = "data"` before the
corresponding command. Reaction-Diffusion uses
`$env:REACDIFF_DATA_ROOT = "data"`. The full official batch-32 CUDA smoke is
explicitly opt-in with `$env:RUN_REACDIFF_CUDA_SMOKE = "1"` before running
`tests.test_cuda_smoke.ReacDiffOfficialBatchCUDASmokeTests`.
For example:

```powershell
$env:CFD1D_DATA_ROOT = "data"
python -B -m unittest tests.test_official_data.CFD1DOfficialDataSmokeTests -v
$env:CFD2D_DATA_ROOT = "data"
python -B -m unittest tests.test_official_data.CFD2DOfficialDataSmokeTests -v
$env:AIRFOIL_DATA_ROOT = "data"
python -B -m unittest tests.test_official_data.AirfoilOfficialDataSmokeTests -v
$env:REACDIFF_DATA_ROOT = "data"
python -B -m unittest tests.test_official_data.ReacDiffOfficialDataTests -v
```

## Anonymous review preparation

Use a fresh/sanitized repository history and configure identity locally (never
with `--global`):

```bash
git config user.name "Anonymous Author"
git config user.email "anonymous@users.noreply.github.com"
```

Replace the example email with the anonymous account's noreply address if one
is available. `scripts/audit_anonymity.py` separately scans the selected public
ref/history and an explicitly supplied final archive. It checks commit authors
and committers, commit messages, reachable historical file contents (including
deleted content), annotated-tag taggers, secrets, unsafe artifacts, and every
archive member's name and bytes. `.git` directories and pointer files are
forbidden at every archive depth. Official SirenFNO/NeuralOperator author names,
public URLs, copyright, and license notices remain intact as required
third-party attribution.

OS account names are not treated as personal-name substrings; machine-local
identity is detected through absolute-path structure instead. For a private
pre-submission check, `ANONYMITY_PRIVATE_NAMES`, `ANONYMITY_PRIVATE_EMAILS`,
`ANONYMITY_PRIVATE_AFFILIATIONS`, and `ANONYMITY_PRIVATE_GITHUBS` may each be
set to a JSON array of private strings. These optional values are used only in
memory by both source and archive scans and are never printed or stored.
Machine-local Git config, reflogs, and worktree metadata are reported as
local-only review findings because ordinary Git push/archive does not export
them; never distribute a raw working-directory ZIP. The anonymity of the
eventual GitHub account, organization, and repository URL must be reviewed
separately and is not asserted by these local checks.

## Implementation notes

CAFE+FNO draws each fixed Gaussian RFF basis from the global PyTorch RNG. The
entry point calls the shared seed helper exactly once before loader and model
construction; no local generator, second model seed, or deterministic-algorithm
override is introduced. Dense and CP/TT/Tucker variants preserve the approved
kernel construction, factor order, Hermitian handling, and fixed-sigma policy.

All paths are derived from `Path(__file__).resolve()` and public metadata uses
only repository-relative notation. Dataset and result locations remain
explicit CLI arguments and do not depend on the caller's working directory.

## License

Original CAFE+FNO repository material is released under the MIT License in
`LICENSE`. The pinned SirenFNO submodule and its bundled NeuralOperator retain
their upstream MIT notices; see `THIRD_PARTY_NOTICES.md`.
