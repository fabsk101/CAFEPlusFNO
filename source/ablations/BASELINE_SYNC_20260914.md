# Historical record: baseline B to baseline C synchronization

This document records the source review performed on 2026-09-14 when the
ablation package moved from historical baseline B to baseline C. It is not the
current execution guide. Use [GUIDE.md](GUIDE.md) for current commands and
requirements.

## Recorded transition

- active profile after the transition: `C`
- active manifest: `source_manifest_baseline_c.json`
- baseline-C base source commit:
  `5d1ec68dfe2774b74eec45f402b6f7f028a47b3e`
- historical baseline-B commit:
  `647c70002e2cfda34878227fbde69b7cbdefbd09`
- pinned SirenFNO commit:
  `81918ecce323a2fd5c5a54db917598bda088574b`

The transition aligned the ablation parent configuration with the reviewed
final main-experiment source. It did not define a new model family, dataset
protocol, loss, optimizer, scheduler, evaluation metric, seed set, or epoch
count.

The historical review kept the 189-path source-verification scope. Eleven
reviewed source/configuration or regression-reference hashes changed and 178
remained unchanged. Earlier manifests were retained byte-for-byte as historical
records rather than rewritten to look like baseline C.

Dense constructor settings changed as part of adopting the reviewed baseline-C
configuration. Historical baseline-B parameter tables are therefore not valid
current instructions. Current dimensions and parameter counts must be obtained
from the active code with the `parameters` module documented in GUIDE.

CP, TT, and Tucker configurations and ranks were not redefined by this
transition. The source review also kept the current offline Darcy loader
contract instead of restoring obsolete loader names or download behavior.

## Scope of the transition review

The transition review used CPU constructor and synthetic-equivalence checks.
Those checks were not official-data validation, GPU validation, native
500-epoch training, or a release-builder result. Items recorded as NOT RUN at
that time remain historical NOT RUN entries; later experiments do not
retroactively change what was executed during the transition review.

The review did not authorize relabeling an older result with a newer source
commit. The base source commit and the integrated execution commit have
different roles:

- `base_source_commit` identifies the reviewed baseline-C parent source.
- `actual_ablation_source_commit` identifies the clean integrated source that
  performed a particular ablation run.
- a release or documentation commit identifies packaged source and must not be
  substituted for either value in an existing training artifact.

Existing formal paper-ablation artifacts record
`actual_ablation_source_commit = ba668d88c3cf207157abf1170bae1c9e65e9d9f0`.
This statement preserves their recorded provenance; it does not claim that the
current documentation commit trained them.

## Current status

Baseline C is now the only active profile, and
`source_manifest_baseline_c.json` is the only active baseline manifest. The
runner rejects stale profile selections. Historical baseline-B manifests remain
in the package solely for provenance and are not alternative runtime choices.

Current supported conditions, the Dense-only input-32 paper condition, the
five reported datasets, the two retained CFD adapters, and the complete
environment-to-aggregation workflow are documented in [GUIDE.md](GUIDE.md).
