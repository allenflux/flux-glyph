# Spatial refinement qualification

The three balanced trials passed all 46 original CAL checks. Balanced-high
also met the Android false-system-font limit exactly (27), but correctly named
1,660 short known views against the unchanged requirement of 1,669 / 3,055.
None qualified for export. These diagnostics reuse CAL and are not a blind test.

This fixed grid preserves balanced-high's complete training objective, verified
TRAIN population and sampling, initializer, 2,400 steps, and runtime thresholds.
It changes learning rates to 1.2e-4, 1.6e-4 and 2.4e-4, with a cosine schedule
ending at one tenth of each initial rate. All three retain confusion weight 1.0,
extra unknown CE 0.25 and extra Kaiti CE 0.05. Each trial starts independently
from the same R22 spatial expansion. The first final checkpoint passing every
CAL check is eligible for the remaining qualification stages.

Training remains local. The only trainable tensors are the spatial zero-sum
residual and the existing unknown classifier row's deltas. Folding produces one
ordinary CNN with 25 font outputs and one size output; no platform routing,
font-name OCR, model voting or reference-image matching is added.

Qualification requires all 53 unchanged CAL checks plus the Android limit,
full 37,834-tile ONNX parity, 36 DEV comparisons and local delivery validation.
TEST remains sealed; these gates qualify a preview, not a blind-tested stable
release. Failed candidates are never activated.
