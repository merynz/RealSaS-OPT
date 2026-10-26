# IRIS Camera / XY Geometry Court — Preregistration V1

**Status:** sealed before execution.  
**Execution class:** GitHub-hosted CPU.  
**No Product / Truth139 / FIT8 authority.**

## Question

The current `CameraAwareViewBankVNextV1` represents a source patch position as the sum of separately learned image, camera and XY embeddings. Orthographic world-space ray/basepoint geometry contains multiplicative terms such as `u * half_extent * right + v * half_extent * up`.

The earlier cross-view court established that query-mediated fusion can reconcile views when the source token already contains a usable view/position representation. It did **not** establish that the current factorized camera+XY representation is the best inductive bias under held-out camera geometry.

This court asks:

> With the same query-mediated fusion mechanism, does deterministic orthographic backprojection improve held-out-camera correspondence over the current factorized camera+XY tokenization?

## Synthetic scene

Use a fixed 4x4 canonical lattice of 16 loci in the plane `z=0`, coordinates inside `[-0.45,0.45]^2`.

For every sample:

- assign an independent random bit to every `(view, canonical_locus)` pair;
- create four orthographic views;
- project each canonical locus to source `(u,v)` using the exact synthetic camera basis and half extent;
- shuffle all source tokens so ordering is not a correspondence shortcut;
- one canonical scene query per locus must recover the four view bits for that locus.

## Camera split

Training camera yaws are sampled from:

`{0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330}` degrees.

Training half extents are sampled from:

`{0.70, 0.90, 1.10, 1.30}`.

Evaluation uses disjoint yaws:

`{15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345}`

and disjoint half extents:

`{0.80, 1.00, 1.20}`.

No evaluation camera geometry appears in training.

## Arms

Both arms use the same:

- source payload bit projection;
- scene-query count and canonical query coordinates;
- 3-layer query-mediated Transformer decoder;
- hidden width / heads / optimizer / steps;
- source-token count.

### `CURRENT_FACTORIZED`

Source spatial contribution is current-like:

`camera_mlp(camera_basis + half_extent) + xy_mlp(u,v)`.

The camera feature contains right/up basis and half extent; no deterministic world-space multiplication is provided.

### `DETERMINISTIC_BACKPROJECTED`

Compute the zero-parameter canonical source-plane point:

`p = u * half_extent * right + v * half_extent * up`

for the centered orthographic synthetic camera.

Source spatial contribution is `world_xy_mlp(p_x,p_y)`. The view identity/basis itself is not required for correspondence once the canonical point is known.

The arm may have fewer trainable parameters than current, but not more.

### Controls

- `RAW_CORRESPONDENCE_POSITIVE_CONTROL`: gather the four bits by known canonical locus index; expected 1.0.
- `RANDOM_LABEL_FIREWALL`: current arm trained/evaluated against independent random target bits; expected chance.

## Seeds / budget

Seeds:

`731, 947, 1217, 1597, 2017`.

Fixed CPU training budget declared in the executable probe. No outcome-dependent extension.

## Gates

Court is valid only if:

- raw control mean/min >=0.995/0.99;
- random firewall mean in `[0.48,0.52]`;
- backprojected parameter count <= current parameter count.

### Keep current factorized contract

If `CURRENT_FACTORIZED` mean >=0.99 and min >=0.97, current geometry encoding is considered sufficient on this mechanism task. Do not add deterministic backprojection merely because it is available.

### Select deterministic backprojection treatment

Only if current fails the sufficiency gate and backprojection achieves mean >=0.99, min >=0.97, and paired mean gain >=0.05 with positive gain in at least 4/5 seeds.

Otherwise status is inconclusive and neither geometry contract is frozen by V1.

## Interpretation limits

This court tests camera/XY representation and held-out camera generalization only. It does not test real DINO features, occlusion, perspective cameras, alpha, real FIT8 morphology or full reference-scale IRIS.
