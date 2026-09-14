# Spatial rejection training

The completed spatial residual grid improved short-region recognition but did not
qualify for release. Its high-rate trial passed 52 of the 53 CAL checks, raised
short correct names from 1,607 to 1,680 of 3,055, and reduced short unknown false
names from 161 to 153 of 752. Kaiti SC coverage was 0.95238 against 0.97, and
Android false names in PingFang/SF Pro/Helvetica were 35 against the R22 limit 27.
The independent CAL audit confirmed these results. R22 remains deployed.

This follow-up starts each trial from the same R22 spatial expansion, not from
the selected CAL errors. It retains the exact native TRAIN sampler, teacher
cache, original losses, 2,400 steps and fixed runtime. CAL and DEV remain reused
evaluation sets, not blind evidence; TEST remains sealed. No evaluation pixels or
labels enter training.

The spatial residual still sums to zero over each group of four horizontal
child cells. The existing unknown classifier row also learns a weight and bias
delta; all 24 known classifier rows and inherited coarse parameters remain
unchanged. The folded result is one ordinary 3,810,234-parameter CNN with 25 font
outputs and its original size output. No platform selector, extra deployed
network, score correction, or OCR transcription is added.

Verified replay, focus and mobile TRAIN labels receive three additional losses:

- Unknown labels: 0.25 times mean cross-entropy to the existing unknown class.
- Kaiti SC labels: one additional mean cross-entropy term to the true class.
- All labels other than PingFang/SF Pro/Helvetica: grouped cross-entropy against
  those three incorrect classes, regardless of the screenshot's platform.

Pair rows retain their original pair-margin objective and receive none of these
extra terms. Known/unknown pairs can update the unknown weight delta; the class
bias cancels in that pair loss.

The fixed trial order uses confusion penalties 0.25, 0.5 and 1.0, each with
learning rate 0.00008, cosine decay to 0.000008, and the same 2,400-step sampler.
Only the final step is evaluated. The first trial passing all original 46 CAL
checks, seven R22/short-region checks, and the Android false-name guard may
proceed to export. All gates remain unchanged. Export must pass full CAL ONNX
parity; DEV must pass all 36 checks against R21/R22 before delivery and deployment.

Code: `training/train_spatial_rejection.py`,
`training/spatial_rejection_objective.py`,
`training/spatial_rejection_network.py`.
Evidence root: `artifacts/unified-font-v6/`.
