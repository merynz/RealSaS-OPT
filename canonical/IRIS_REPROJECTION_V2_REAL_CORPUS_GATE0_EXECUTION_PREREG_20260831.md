# RealSaS — IRIS Reprojection V2 Real-Corpus Gate-0 Execution Prereg V1

**Date:** 2026-08-31  
**Status:** `FROZEN_BEFORE_REAL_CORPUS_GATE0_OUTPUTS__NO_OPTIMIZER_AUTHORIZED`

This document binds the first real-corpus execution of the deterministic V2 Gate 0. It is measurement/preflight only. No learned V2 output may be opened by this gate.

## Population

Population is the exact S1 `TRAIN512` membership, not the new clean-C0 population.

Reason: V2 apparatus feasibility must not be confounded with the separate H0 clean-corpus intervention.

Required authority interlocks:

- membership raw SHA-256: `ae1edcaad47cf4867f11d51305eb57db06acb63ef13fc60ca3263c043ef8f71c`
- membership content SHA-256: `305bcc2efc863c035ad7bce6e0afb29f2354ecc57abd1746e7b15b0485f71d24`
- TRAIN512 set SHA-256: `1958fa5ed80430ac8ae8f9e66f8d94bc5bfe89c74b5553d13b891c87fdefb2a2`
- exact asset count: `512`

No DEV32 or held-out population may be opened.

## Source authority per asset

- native inference silhouette source: `renders/V{v}/cel_clean.png`, exact RGBA 1024x1024;
- camera: `renders/V{v}/camera.json`, contract `realsas.level_orthographic_z_orbit.v1`, yaw `45*v`;
- sampled truth only for Gate-0 containment measurement: `primary_geometry.npz` + `raster_authority.npz` barycentric authority.

Truth geometry is **measurement-only**. It is not exposed to the learned V2 inference path.

## Camera checks

For every asset/view:

- camera contract exact;
- yaw exact within `1e-6` degrees;
- orthonormal forward/right/screen_up basis;
- positive half extent;
- half extent equal across 8 views within `1e-8`;
- common world-Z row invariance is checked by the already sealed analytic unit suite and spot-checked on real cameras.

Any hard camera failure blocks Gate 0.

## Inference foreground definition

Use the exact Phase-I image rule already frozen for the corpus audit:

- if any alpha <255: foreground is `alpha >= 128`;
- otherwise foreground is complement of exact green RGB `(0,255,0)`.

No raster authority is used to construct the inference hull mask.

## Visual-hull padding sweep

Native pixel paddings:

`{0,1,2,4,8}`.

Square binary dilation is the conservative implementation for this preflight.

### Sampled truth containment

For each asset and each of 8 views, construct a deterministic union of:

1. up to `4096` uniformly sampled authoritative raster rows without replacement;
2. up to `2048` authoritative rows whose own-view raster pixels lie on the 4-neighbor foreground boundary.

Seeds derive only from `SHA256(asset_id | view | fixed_gate0_tag)`.

Reconstruct sampled world truth by barycentric authority. For each padding, a sampled world point is contained only if its reprojection lands inside the padded inference foreground in **all 8 views**.

No sampled truth miss is allowed for a padding candidate to be geometrically admissible.

### Volume/search reduction

The inference-available canonical bounding domain is the fixed cube:

`[-half_extent,+half_extent]^3`.

For each asset, draw exactly `32768` deterministic uniform world candidates from this cube using an asset-hash seed. Reuse the identical candidates for every padding.

For each padding report:

- active fraction;
- active count;
- normal-approx 95% binomial interval;
- implied dense-lattice active candidate counts for each spacing candidate.

This is explicitly a deterministic Monte-Carlo estimate, not an exact volume integral.

### Padding policy rule — frozen before outputs

1. Find the smallest padding with **zero sampled truth misses over all 512 assets**.
2. If none exists, Gate 0 blocks.
3. That smallest padding is the geometrically admissible padding candidate.
4. Hull pruning is classified `MATERIAL_COMPUTE_REDUCTION` only if population median active fraction `<= 0.75` (at least 25% median candidate reduction).
5. If geometrically admissible but median active fraction `>0.75`, the hull may remain a proof/domain object but may not be claimed as a useful compute-pruning mechanism without a new explicit policy.

No larger padding may be chosen merely because it looks smoother after outputs.

## Coarse spacing sweep

World spacings remain:

`{0.016, 0.008, 0.004}`.

No spacing is selected by this prereg.

For each spacing report the fixed-cube total lattice count and the Monte-Carlo-implied active candidate count under each padding.

Spacing selection requires observed-structure resolvability evidence below and a separate policy freeze after measurement. Learned output quality may not be used to select spacing.

## Observed thin-structure proxy

The objective is to detect when a coarse lattice can miss a visible structure before local refinement has any mode to refine.

For each inference foreground mask:

1. compute Euclidean distance transform in native pixels;
2. identify 3x3 local maxima inside foreground;
3. define local observed-support diameter proxy `2 * EDT` pixels;
4. convert to world units using exact `2*half_extent/1024`.

This is an **observed silhouette medial-thickness proxy**, not a claim of true 3D object thickness.

For each spacing s, stratify medial diameters by cells:

- `<1`
- `[1,2)`
- `[2,4)`
- `>=4`.

Report per-view/per-asset and population aggregates including min, p01, p05, p50 and counts.

If any product-relevant class remains sub-cell, later policy must either provide a pre-registered finer/adaptive search or explicitly route it to `RESOLUTION_UNSUPPORTED -> UNKNOWN`. Continuous refinement may not be cited as a cure for an absent coarse mode.

## Orientation/morphology diagnostics

Gate 0 does not claim 3D morphology reconstruction. It records per-view yaw and thickness strata so Gate 0.5/Gate 1 can later report failures by orientation relative to the orbit. Any learned orientation-specific diagnosis belongs to those later gates.

## Runtime/chunking

All candidate tests are chunked; raw eight-view descriptors are not involved in this real Gate-0 runner.

The runner must never allocate a full dense 3D descriptor volume.

## Output contract

Machine result must include:

- all source authority hashes;
- population IDs/set hash;
- hard failures;
- camera summary;
- hull containment x reduction frontier;
- frozen padding-policy decision;
- spacing candidate accounting;
- observed thickness proxy strata;
- runtime/progress;
- `scientific_optimizer_steps = 0`;
- `training_authorized = false`.

## Status rules

`PASS_REAL_GATE0_MEASUREMENT_COMPLETE__SPACING_POLICY_FREEZE_NEXT` requires:

- exact population/hash checks pass;
- zero hard camera/file/raster authority failures;
- at least one padding candidate has zero sampled truth misses.

It does **not** authorize optimizer step 1.

Any authority failure or no admissible padding returns fail-closed status.

## Next authority

After a PASS measurement:

1. freeze padding decision as dictated above;
2. freeze a spacing/resolution policy using only Gate-0 geometry/resolvability/compute evidence;
3. implement and seal DINO-S taps, native pyramid, canonical evidence learner, loss/runtime;
4. only then consider Gate 0.5 one-family optimizer authorization.
