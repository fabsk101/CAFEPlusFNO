# CAFE+FNO ablation package

This directory contains the ablation runner distributed with the CAFE+FNO
review artifact. It reuses the repository's dataset loaders, training losses,
optimizers, schedulers, evaluation definitions, checkpoint helpers, and native
CAFE+FNO constructors. It does not replace or monkey-patch the parent training
modules.

The current and only active source profile is **baseline C**:

- active manifest: `source_manifest_baseline_c.json`
- base source commit: `5d1ec68dfe2774b74eec45f402b6f7f028a47b3e`
- experiment ID: `cafe_plus_fno_ablation_v1`
- default seeds: `0 42 73 108 202`
- formal protocol: the inherited fixed 500-epoch protocol for each dataset

The base source commit identifies the reviewed model and experiment source.
Each formal run also records its own `actual_ablation_source_commit`. Existing
formal paper-ablation artifacts record
`ba668d88c3cf207157abf1170bae1c9e65e9d9f0`; a later documentation or release
commit must not be presented as the source that trained those artifacts.

See [GUIDE.md](GUIDE.md) for the current reproduction sequence. The historical
B-to-C transition is recorded separately in
[BASELINE_SYNC_20260914.md](BASELINE_SYNC_20260914.md).

## Scope reported in the paper

The paper ablation matrix uses the Dense model on five datasets:

- `airfoil`
- `burgers1d`
- `darcy`
- `ns2d`
- `reacdiff1d`

The repository also retains working `cfd1d` and `cfd2d` adapters for code
completeness and additional experiments. Their ablation results are not part of
the submitted five-dataset comparison.

The paper matrix uses these five conditions:

- `full`
- `learnable_sigma`
- `fourier_only`
- `chebyshev_only`
- `without_linear_branches_input32`

All five paper conditions use `--model-variant dense`. The source archive does
not bundle the paper checkpoints, training CSV files, summaries, or aggregate
tables.

## Implemented conditions

The implementation exposes six condition identifiers. The first five are
available to Dense, CP, TT, and Tucker unless noted otherwise by the CLI. The
sixth condition is Dense-only.

| Condition | Implemented change |
| --- | --- |
| `full` | Fourier and Chebyshev features, fixed sigma, and the original Linear branches with Hadamard multiplication. |
| `learnable_sigma` | Only each encoder's `log_sigma` becomes a trainable scalar parameter. The sampled Gaussian basis `G` remains a fixed persistent buffer. |
| `fourier_only` | Removes Chebyshev coordinates while preserving the Fourier basis, fixed sigma, Linear branches, and Hadamard multiplication. |
| `chebyshev_only` | Removes random Fourier features while preserving Chebyshev normalization and order, Linear branches, and Hadamard multiplication. |
| `without_linear_branches` | Removes the Linear branches and their Hadamard product. The complete native embedding enters the kernel MLP directly. This historical condition remains supported but is not the branch-free condition used in the paper matrix. |
| `without_linear_branches_input32` | Dense-only fixed-input comparison used in the paper matrix. It sends 32 features directly to the kernel MLP: 16 Fourier features derived from the first eight columns of the native fixed `G`, plus 16 Chebyshev features. No Linear branch or Hadamard product remains. |

Neither branch-free condition zeroes a branch output. Kernel generation remains
active. The fixed-input condition is not an equal-total-parameter comparison;
see [INPUT32_CONDITION.md](INPUT32_CONDITION.md) for its exact basis selection
and checkpoint contract.

The code also supports model variants `dense`, `cp`, `tt`, and `tucker` for the
original five conditions. `without_linear_branches_input32` is rejected for
CP, TT, and Tucker before training. Use the `parameters` module to obtain the
current constructor dimensions and parameter counts instead of relying on an
old table:

```text
python -I -S ablations/isolated.py --workspace-root . --repository-root . --module parameters --dataset darcy --model-variant dense
```

## Quick reproduction outline

Run all commands from the root of an extracted official source archive or a
clean committed checkout. Use a dedicated environment created from the bundled
environment files, then keep `-I -S` on every ablation command.

1. Perform a formal source and import check for one explicit tuple.
2. Run the same tuple with `--dry-run`; this prints the five-seed plan and does
   not start a training subprocess.
3. Remove only `--dry-run` when the verified data and compute environment are
   ready for the fixed 500-epoch runs.
4. Validate and aggregate the completed artifacts without rewriting them.

Example source check:

```text
python -I -S ablations/isolated.py --workspace-root . --repository-root . --module check --dataset darcy --model-variant dense --condition full --formal
```

Example no-training plan for the paper's branch-free condition:

```text
python -I -S ablations/isolated.py --workspace-root . --repository-root . --module run --dataset darcy --model-variant dense --condition without_linear_branches_input32 --data-root data --results-root results/ablations_reproduction --device cuda --dry-run
```

The corresponding training command is identical except that `--dry-run` is
removed. A `run` command operates on one dataset/model/condition tuple only; it
never expands automatically across datasets, model variants, or conditions.

Formal output is stored under:

```text
<results-root>/cafe_plus_fno_ablation_v1/formal/
  <dataset>/<model-variant>/<condition>/seed_<seed>/
```

Each completed run contains `training_log.csv`, `summary.json`, and
`final_checkpoint.pt`. Existing complete runs are reused only after strict
validation; partial or incompatible artifacts are rejected rather than silently
overwritten.

## Source and release provenance

In a Git checkout, formal execution requires a clean committed source with the
packaged files tracked. In an official Git-less release, formal execution uses
the parent `RELEASE_PROVENANCE.json` member hashes and pinned upstream identity.
Creating a new Git repository inside an extracted release is neither required
nor a substitute for release verification.

`source_compatibility_anonymity_release_v1.json` records a narrowly reviewed,
deployment-only anonymity transition without changing the baseline manifest or
claiming that the deployment files trained historical results. Historical
`source_manifest*.json` files remain provenance records; only
`source_manifest_baseline_c.json` is active.

Dataset binaries are not included and training never downloads them implicitly.
Follow the repository [data instructions](../data/README.md). Third-party source,
copyright, and license attribution is preserved in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
