# CAFE+FNO ablation reproduction guide

This guide describes the current baseline-C workflow implemented by the files
in `ablations/`. Commands are intentionally explicit: one invocation selects
one dataset, one model variant, and one condition. No command in this guide
launches every dataset or every factorization automatically.

## 1. Current contract and provenance roles

The active contract is:

- baseline profile: `C`
- active manifest: `source_manifest_baseline_c.json`
- base source commit: `5d1ec68dfe2774b74eec45f402b6f7f028a47b3e`
- experiment ID: `cafe_plus_fno_ablation_v1`
- default seeds: `[0, 42, 73, 108, 202]`
- formal epochs: 500, inherited from each dataset configuration
- pinned SirenFNO commit: `81918ecce323a2fd5c5a54db917598bda088574b`

The base source commit identifies the reviewed parent model/configuration
contract. `actual_ablation_source_commit` identifies the clean integrated source
that executed a particular formal run. Existing formal paper-ablation artifacts
record `ba668d88c3cf207157abf1170bae1c9e65e9d9f0`. A later release or
documentation commit is not their training source and must not replace that
recorded value.

For a new reproduction, the current clean execution commit is recorded in the
new output. Reproduction must not rewrite the provenance of an older run.

## 2. Datasets, variants, and conditions

The CLI recognizes seven datasets:

```text
airfoil  burgers1d  cfd1d  cfd2d  darcy  ns2d  reacdiff1d
```

The paper reports the ablation comparison for five datasets only:

```text
airfoil  burgers1d  darcy  ns2d  reacdiff1d
```

The CFD-1D and CFD-2D adapters remain supported for additional runs and source
verification, but their results are not part of the submitted five-dataset
comparison.

The implemented model variants are:

```text
dense  cp  tt  tucker
```

The paper ablation matrix uses `dense` and these five conditions:

```text
full
learnable_sigma
fourier_only
chebyshev_only
without_linear_branches_input32
```

The implementation also retains `without_linear_branches`, the original
branch-free condition that passes the complete native embedding directly to the
kernel MLP. It remains available for Dense, CP, TT, and Tucker, but it is not the
branch-free condition reported in the paper matrix.

`without_linear_branches_input32` is Dense-only. It preserves the native
Gaussian draw, takes the first eight columns of fixed `G` to form eight cosine
and eight sine features, adds 16 Chebyshev features, and sends those 32 features
directly to the kernel MLP. In 1D the Chebyshev features are T0 through T15; in
2D they are T0 through T7 on each axis. No Linear branch or Hadamard product is
present. See [INPUT32_CONDITION.md](INPUT32_CONDITION.md) for the immutable
reconstruction contract.

The other conditions have the following narrow scopes:

- `full`: no architecture change from native CAFE+FNO.
- `learnable_sigma`: changes only `log_sigma` from a persistent buffer to a
  trainable scalar; `G` remains fixed.
- `fourier_only`: excludes Chebyshev features.
- `chebyshev_only`: excludes Fourier features.
- `without_linear_branches`: removes the Linear/Hadamard stage without reducing
  the native embedding to 32.

The code rejects an input-32 request for CP, TT, or Tucker before construction
or output creation. None of the conditions claims an equal total parameter
budget. Query the current code for dimensions and counts:

```text
python -I -S ablations/isolated.py --workspace-root . --repository-root . --module parameters --dataset darcy --model-variant dense
```

## 3. Prepare an isolated environment

Run from the root of an extracted official source archive or a clean committed
checkout. The release contains `environment.yml`, `requirements-lock.txt`, and
the complete pinned `third_party/SirenFNO` source.

Create a dedicated environment without an editable installation that points to
another checkout:

```text
conda env create --file environment.yml
```

The environment name declared by that file is `anonymous-cafe-plus-fno`. The
reference specification pins Python 3.13.11, PyTorch 2.8.0, and the remaining
public dependencies in `requirements-lock.txt`.

The examples below use `conda run` so the selected environment is explicit:

```text
conda run -n anonymous-cafe-plus-fno python -I -S ablations/isolated.py --help
```

`-I -S` is required. `isolated.py` audits the environment before adding its
site-packages and rejects external project editable mappings, unsafe `.pth`
paths, and external `PYTHONPATH` entries. Do not bypass that check by installing
the project from a different checkout.

## 4. Prepare data

Dataset binaries are not distributed. Follow [data/README.md](../data/README.md)
and use the repository downloader before training. For example, the documented
all-dataset preparation entry point is:

```text
conda run -n anonymous-cafe-plus-fno python data/download_data.py --dataset all --data-root data
```

Training loaders are offline. Missing or hash-mismatched inputs stop the run;
the ablation runner does not download data implicitly. A paper reproduction
needs only the five reported datasets, while CFD data is needed only if those
additional adapters are selected.

## 5. Perform the formal source check

Select one explicit tuple. The following command checks baseline C, the active
manifest, release or Git provenance, pinned imports, and the Dense Full model on
CPU. It does not train a model:

```text
conda run -n anonymous-cafe-plus-fno python -I -S ablations/isolated.py --workspace-root . --repository-root . --module check --dataset darcy --model-variant dense --condition full --formal
```

For a clean Git checkout, formal mode requires tracked ablation files and a
clean commit. For an official Git-less source archive, it verifies
`RELEASE_PROVENANCE.json` and every recorded member hash. Git ancestry is
reported as unavailable in a Git-less archive rather than being inferred.

The compatibility record permits only its reviewed deployment-only transition;
it does not replace the active baseline, weaken unrelated hash checks, or claim
that the deployment source trained older results.

## 6. Inspect the no-training plan

Use `run --dry-run` before removing the safety flag. This verifies the selected
tuple and prints the requested seed plan. It does not launch any `train` child
process and does not load the dataset:

```text
conda run -n anonymous-cafe-plus-fno python -I -S ablations/isolated.py --workspace-root . --repository-root . --module run --dataset darcy --model-variant dense --condition without_linear_branches_input32 --data-root data --results-root results/ablations_reproduction --device cuda --dry-run
```

With no `--seeds` argument, the requested seeds are exactly:

```text
0 42 73 108 202
```

An explicit `--seeds` list replaces the default list; it does not append to it.
Duplicate, empty, non-integer, or out-of-range seed selections are rejected.
The command still selects only one dataset/model/condition tuple.

## 7. Run formal training

Only after the source check, data verification, and dry-run succeed, remove
`--dry-run` from the same command:

```text
conda run -n anonymous-cafe-plus-fno python -I -S ablations/isolated.py --workspace-root . --repository-root . --module run --dataset darcy --model-variant dense --condition without_linear_branches_input32 --data-root data --results-root results/ablations_reproduction --device cuda
```

This launches one isolated 500-epoch process per requested seed. It reuses the
selected dataset's native loader, loss, optimizer, scheduler, and evaluation
definition. It does not shorten the formal epoch count or sweep other tuples.

To run one explicit seed instead of a seed list, use `train`:

```text
conda run -n anonymous-cafe-plus-fno python -I -S ablations/isolated.py --workspace-root . --repository-root . --module train --dataset darcy --model-variant dense --condition without_linear_branches_input32 --seed 0 --data-root data --results-root results/ablations_reproduction --device cuda
```

The default output layout is:

```text
<results-root>/cafe_plus_fno_ablation_v1/formal/
  <dataset>/<model-variant>/<condition>/seed_<seed>/
    training_log.csv
    final_checkpoint.pt
    summary.json
```

If `--results-root` is omitted, it defaults to `results/ablations`. Output and
data paths must remain inside the selected workspace. Existing complete runs
are reused only after strict validation. Incomplete or incompatible artifacts
cause an error; `--overwrite` is an explicit destructive choice for that one
run and is not required for normal reproduction.

For the submitted ablation matrix, repeat the command only for the chosen
five-dataset/five-condition Dense tuples. The runner intentionally provides no
automatic all-dataset or all-condition sweep.

## 8. Validate completed artifacts

Formal training performs the dataset-native final evaluation and writes its
metrics into the CSV, checkpoint, and summary contract. The artifact command
validates one saved bundle without training or recalculating test metrics:

```text
conda run -n anonymous-cafe-plus-fno python -I -S ablations/isolated.py --workspace-root . --repository-root . --module artifacts --summary results/ablations_reproduction/cafe_plus_fno_ablation_v1/formal/darcy/dense/without_linear_branches_input32/seed_0/summary.json --dataset darcy --model-variant dense --condition without_linear_branches_input32 --seed 0
```

The validator checks the run identity, source and runtime evidence, complete
1-500 epoch CSV, checkpoint hash, model configuration, tensor shapes and
finiteness, optimizer/scheduler state, and the stored final metrics. Optional
`--restore` additionally reconstructs the model on CPU; it still does not load
the dataset or recompute the metric.

## 9. Aggregate the selected seeds

Aggregate only after all selected runs validate:

```text
conda run -n anonymous-cafe-plus-fno python -I -S ablations/isolated.py --workspace-root . --repository-root . --module aggregate --results-root results/ablations_reproduction --dataset darcy --model-variant dense --condition without_linear_branches_input32 --output-dir results/ablation_tables/darcy_input32
```

Without `--seeds`, aggregation requires the default five seeds. It validates
each bundle and writes `aggregate.json`, `aggregate.csv`, `seed_values.csv`,
`table.md`, and `table.tex`. It reports means and sample standard deviations
with `ddof=1`; it does not retrain models or re-evaluate checkpoints.

Missing selected seeds fail by default. `--allow-partial` permits only missing
selected seeds, not corrupt or incompatible existing runs. Existing aggregate
outputs are protected unless `--overwrite-aggregate` is explicitly supplied;
that option is unrelated to training `--overwrite`.

The source archive does not include the paper's checkpoint, CSV, summary, or
aggregate files. A reviewer can reproduce them with the sequence above, but
their absence from the ZIP must not be described as a completed rerun.

## 10. Historical baseline B records

Baseline B is not selectable by the current runner. Its manifests and old
documentation references are retained only to preserve development provenance.
The current code accepts `CAFE_ABLATION_BASELINE_PROFILE` only when it is unset
or equal to `C`.

Do not use historical B parameter tables, temporary checkout paths, development
environment names, or old readiness reports as current reproduction evidence.
Do not rename B artifacts as C artifacts or replace their recorded source
commit. [BASELINE_SYNC_20260914.md](BASELINE_SYNC_20260914.md) summarizes the
historical transition without defining the current execution procedure.

## 11. Release boundaries

The official source ZIP contains code, configuration, manifests, environment
specifications, data metadata/download helpers, and pinned third-party source.
It does not contain datasets, trained checkpoints, training logs, summaries,
paper tables, internal development workspaces, or private diagnostic logs.

All public paths emitted by the ablation tools are workspace-relative or fixed
artifact identifiers. Third-party authorship, copyright, and license notices
must remain intact; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
