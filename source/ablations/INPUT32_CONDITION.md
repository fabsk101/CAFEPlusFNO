# Direct input-32 branch-free Dense ablation

The new condition is `without_linear_branches_input32`. The original five
conditions, including `without_linear_branches`, retain their numerical
behavior and condition contracts.

- Only Dense is supported. CP/TT/Tucker requests fail before training.
- Start from the original native model, preserving the original Gaussian draw.
- Select columns 0 through 7 of each Dense encoder's fixed `G` (no redraw).
- Use 8 cosines and 8 sines, plus 16 Chebyshev features.
- In 1D, use T0 through T15. In 2D, use T0 through T7 on each axis.
- Feature order stays Chebyshev by axis, cosine, sine.
- No Linear branch or Hadamard combination remains. The encoder returns 32
  coordinates directly. Kernel MLP input is 32; hidden/output widths are kept.
- If the native MLP input is already 32, the same input weights are retained.
  Otherwise the existing Xavier-initialized input replacement path is used.
- Never redraw G to match a new shape or slice the first 32 concatenated
  features: the latter could select only Chebyshev features.
- Insufficient source basis sizes are rejected rather than padded or expanded.

The unchanged `base_configuration` plus the new condition reconstructs the
model. `model_configuration.direct_input32_basis` records the resolved rule.
Ablation checkpoint loading (not native model-only checkpoint reconstruction)
checks that rule and restores all weights with strict shape validation.
Native `model.get_config()` alone is not a complete ablation reconstruction
contract: it does not describe the branch-free topology.

This is a fixed-32 comparison. The supplied ReacDiff (and CFD) Full config has
MLP input 16, so it is not MLP-input-matched to Full on those datasets. Total
parameters also need not match because Full contains learned branches.

Use a new result root, e.g. `results/ablations_input32_v1`, and select only the
new condition to run it. Never rename an old condition's results or rewrite
their commits. Keep the old source/commit for unfinished old-condition runs
and for historical aggregation. No new experiment ID or parent manifest is
needed: the distinct condition is already part of every run path and contract.

The new package's source hash changes. Unchanged old conditions are numerically
regression-tested, but that does not disable strict completed-run source/runtime
checks or certify automatic reuse of previous results under a new commit.
