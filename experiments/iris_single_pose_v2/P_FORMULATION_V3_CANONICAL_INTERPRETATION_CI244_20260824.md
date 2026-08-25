# P Formulation V3 — Canonical Interpretation of CI244/V2

Date: 2026-08-24
Status: `COMPLETED__FIXED_HALF_EXTENT_FALSIFIED__P_GEOMETRY_INVARIANTS_HEALTHY`

## Authority

CI244/V2 corrected the CI237 master-NPZ apparatus false reject without changing the frozen 16-asset FIT panel or scientific geometry thresholds.

Completed label emitted by the frozen V3 prereg:

`P_V3_CORPUS_GEOMETRY_CLOSURE_FAIL`

This label is retained. Its causal interpretation is narrow: **the hardcoded `half_extent=0.54` assumption is false on the frozen corpus panel.**

## What passed

The corrected G1 opened real geometry metrics on all 16 frozen FIT sentinel assets.

Across all 16:

- canonical bbox center infinity norm max = `2.9802322387695312e-08`;
- canonical largest bbox extent min/median/max = `0.9999999403953552 / 1.0 / 1.0`;
- canonical max absolute coordinate max = `0.5`;
- raster-P projection p95 across-asset max = `6.715862355122226e-08`;
- sampled projection-error fraction >1e-3 across-asset max = `0.000244140625`;
- exact-depth analytic reconstruction p95 across-asset max = `3.998794184667531e-08`.

Therefore canonical gauge, triangle/barycentric P authority, camera basis/yaw convention, raster projection, and the analytic decomposition

`P = h*gx*right - h*gy*up + depth*forward`

are all supported at essentially numerical precision when `h` is the asset's actual acquisition scale.

## What failed

14/16 assets used `half_extent=0.54` and passed every frozen V3 gate.

Two assets were internally self-consistent but used a different constant half extent across all eight views:

- `asset_00aa1b666ba193851a498194`: `0.5570941257476807`;
- `asset_76313e4bd82b82fcd1659c70`: `0.6172158837318421`.

Those two assets still had near-zero raster-P projection and exact-depth reconstruction error. Their only fatal V3 condition was the fixed-0.54 equality gate.

Hence do **not** attribute CI244 to a broken P ontology, bad barycentrics, bad canonical normalization, raster drift, R/2 discretization, or learned depth. No optimizer ran.

## Product-contract consequence

Do not feed `camera.json half_extent` to IRIS. Product IRIS remains image-only.

The canonical geometry gauge is observable from the ordered RGBA sheet itself: canonical max bbox extent is one; yaw-0 horizontal occupancy exposes X extent, yaw-90 horizontal occupancy exposes Y extent, and image height exposes Z extent.

The next formulation is P-V4:

`h_sheet = 1 / (2*max(V0_x_occupancy, V2_x_occupancy, max_view_z_occupancy))`

from alpha only, followed by

`P = h_sheet*gx*right - h_sheet*gy*up + depth*forward`.

Manual native-alpha checks on the two CI244 fatal assets already gave observable scale within about `0.00186` and `0.00114` of the diagnostic camera scale, respectively. This is diagnostic evidence only; promotion requires the preregistered P-V4 optimizer-zero corpus closure against the unchanged 3D P p95 <=0.005 target.

## Current policy

- CI104 `P_GEOMETRY_SUFFICIENT` remains CLOSED/PASS.
- CI202 raw-XYZ learner remains historical diagnostic only.
- CI237 V1 remains preserved as apparatus false reject.
- CI244 V2 is the valid V3 real-corpus result and falsifies only fixed 0.54 framing scale.
- Training remains forbidden.
- Current gate: `P_FORMULATION_V4_OBSERVABLE_SCALE_CLOSURE_PREREG_20260824.md`.
