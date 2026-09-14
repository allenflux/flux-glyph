# Balanced spatial repair

All three spatial-rejection trials restored Kaiti SC coverage to 21/21 but
raised iOS false font names. Their Android false-system-font counts were 30, 29
and 28, still above the limit 27; short correct names were 1,660, 1,662 and 1,665
against the required 1,669. None qualified for export. Aggregate CAL diagnosis traced most of that increase
to unknown Weibei SC being named Kaiti SC: 9 cases in R22, 10 in the previous
spatial-high trial, and 48 in rejection-low. These are reused CAL diagnostics,
not new labels or training examples. The existing rejection grid is preserved.

The next fixed experiment changes only the extra Kaiti CE coefficient from 1.0
to 0.05. The existing supervised Kaiti loss remains. Unknown CE weight 0.25,
false-system penalties 0.25/0.5/1.0, inherited R22 initializer, 2,400-step sampler,
zero-sum spatial residual, unknown-row adaptation, and all runtime settings stay
the same. Two actual Torch checks verify that the other repair loss terms are
identical and gradients remain finite.

`training/train_spatial_balanced.py` and `training/spatial_balanced_objective.py`
write a new grid under `artifacts/unified-font-v7/`. Trial order is balanced-low,
balanced-medium, balanced-high; each starts independently from the same R22
spatial expansion. Existing v5/v6 artifacts and source files remain immutable.

Qualification still requires all 46 original and seven additional CAL checks,
the Android false-system-font limit 27, full 37,834-tile ONNX parity, and 36 DEV
comparisons before delivery. No thresholds are relaxed. No test dataset is
opened. Training remains local, and no failed candidate is deployed.
