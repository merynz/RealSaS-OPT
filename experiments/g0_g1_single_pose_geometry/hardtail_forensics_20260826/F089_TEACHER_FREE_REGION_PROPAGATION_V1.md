# f089 Teacher-Free Region Discovery + Planar Propagation — CPU Oracle V1

**Date:** 2026-08-26  
**Asset:** `asset_f089abadcd071194617d640b`  
**Execution:** assistant local CPU/container; no Colab; optimizer/training steps = 0  
**Status:** `EXPLORATORY_POSITIVE__NOT_FORMAL_GATE`

## Why this experiment exists

The preliminary f089 oracle showed that a huge nearly uniform plane dominates the asset's hard tail and that image-derived confident seeds plus planar propagation can collapse catastrophic depth error. However, that preliminary propagation rung was still given the plane membership from teacher triangle IDs.

This V1 removes that membership oracle. The inference-side treatment receives only:

- 8 `cel_clean` RGBA views at 512×512;
- the known orthographic 8-yaw camera contract;
- no teacher triangle IDs;
- no teacher visibility mask;
- no teacher geometry.

Teacher raster/geometry are used only after prediction for evaluation.

This is exploratory, not preregistered. The adaptive failure/correction lineage below is preserved intentionally.

## A. Detached quad geometry audit

The giant rectangle is not a plotting artifact. In `primary_geometry.npz` it is a disconnected mesh component:

- vertices: `591, 592, 593, 594` — exactly four vertices;
- faces: `776, 777` — exactly two triangles;
- disconnected component size: 4 vertices / 2 faces;
- each of the four vertices has total source skin weight `0.0`;
- face areas: `0.1488463677`, `0.1488463732`;
- unit normal: approximately `[0.99999036, -0.00257924, -0.00355237]`;
- canonical bounds:
  - x: `[-0.27675956, -0.27293128]`
  - y: `[-0.18348445, 0.18348445]`
  - z: `[-0.45830041, 0.35292885]`

This strongly establishes a detached, non-skinned rectangular component. Its semantic role is still **UNKNOWN**. The available geometry package has no material/UV/component-name authority, so this report does **not** label it junk. It may be an ancillary card/helper/static element or valid static geometry.

## B. Teacher-free region discovery

For each view independently:

1. take foreground pixels from RGBA alpha;
2. find the modal foreground RGB color;
3. keep pixels whose maximum per-channel RGB distance from that mode is <= 6;
4. apply one 3×3 binary closing;
5. keep the largest 8-connected component.

No triangle IDs are used.

Evaluation against teacher faces `776/777`:

| view | discovered-region IoU vs teacher quad |
|---|---:|
| V0 | 0.0000 — edge-on; not a usable region view |
| V1 | **0.9921** |
| V2 | **0.9833** |
| V3 | **0.8885** |
| V4 | 0.0000 — edge-on; not a usable region view |
| V5 | **0.9921** |
| V6 | **0.9146** |
| V7 | **0.9577** |

Thus the dominant planar region is recoverable to high overlap from raster appearance alone on the six non-edge-on views.

## C. First teacher-free attempt — preserved FAIL

The first visibility-free attempt used a `best-3` target-view aggregator: for each candidate depth, use the three lowest costs among the seven other views.

This failed on several views. The mechanism diagnosis is important: the most informative depth constraints for this plane are the edge-on V0/V4 views. A `best-3` rule systematically discards those harder-but-informative views because many oblique views can match the same uniform brown color at wrong depths.

Observed propagation P90 under this first treatment:

- V1: `0.00207`
- V2: `0.32777` **FAIL**
- V3: `0.34947` **FAIL**
- V5: `0.00169`
- V6: `0.00147` on the majority plane but contaminated tail remains
- V7: `0.48730` **FAIL**

This failure is retained as part of the experiment lineage. It is exactly the kind of apparently catastrophic result that would be misleading without mechanism localization.

## D. Corrected teacher-free treatment — all-view geometric consistency

Treatment:

- reference loci come from the image-discovered dominant region;
- 321 camera-forward depth candidates uniformly span `[-0.54, +0.54]`;
- every candidate is reprojected into **all seven** other known-camera views;
- per-view cost = RGB L1 + alpha/out-of-frame penalty;
- aggregate cost = mean across all seven views;
- confidence = cost gap to the best alternative at least `0.012` depth away;
- choose up to 80 spatially separated highest-confidence seeds;
- robust RANSAC fit of orthographic plane depth:
  `d = a*s_r + b*s_u + c`;
- propagate that image-derived plane over the discovered region.

Teacher visibility and teacher plane membership are not used in inference.

### Results

| ref | discovered sample actually teacher-plane | seed good <.005 | pointwise P90 | propagated P90 | teacher-plane-only propagated P90 |
|---|---:|---:|---:|---:|---:|
| V1 | 100.0% | **100.0%** | 0.002280 | **0.001113** | **0.001113** |
| V2 | 99.71% | **98.75%** | **0.359339** | **0.001253** | **0.001242** |
| V3 | 90.50% | **95.00%** | 0.007464 | **0.001460** | **0.001146** |
| V5 | 100.0% | **100.0%** | 0.002284 | **0.001059** | **0.001059** |
| V6 | 93.71% | **97.50%** | 0.046122 | **0.001526** | **0.001271** |
| V7 | 96.93% | **100.0%** | 0.002688 | **0.001345** | **0.001259** |

The headline witness is V2:

`pointwise P90 0.359339 -> teacher-free propagated P90 0.001253`

and V6:

`0.046122 -> 0.001526`

The earlier teacher-membership oracle therefore was not the sole reason propagation worked. For this asset, the dominant planar region can be found from the image itself and the correct plane can be reconstructed from image-only multiview seed evidence plus known cameras.

## E. Important residual / region-boundary caveat

V3 and V6 discovered regions contain ~9.5% and ~6.3% non-plane pixels respectively because the same dominant brown color also appears on nearby geometry. Consequently their whole-region P95 jumps catastrophically even though the teacher-plane-only P90 is ~0.0011–0.0013.

This localizes the next problem further:

> planar propagation itself works; **region ownership / surface-support segmentation** is now the remaining f089-specific weakness.

That is different from missing depth information.

## Interpretation

For f089:

1. The giant rectangle is a real detached two-triangle mesh component, not a visualization bug.
2. It is non-skinned in the source package, but its semantic validity remains unresolved.
3. The catastrophic pointwise depth tail is **not** evidence that depth is absent from the observations.
4. The plane region is discoverable from the images without teacher triangle membership.
5. Keeping all-view consistency is essential; aggressively selecting only the easiest views can erase the exact edge-on constraints that disambiguate depth.
6. Once confident image-derived seeds are retained, robust planar propagation collapses the hard tail by roughly two orders of magnitude on the dominant plane.
7. Remaining error is mostly region contamination / surface-support ownership, not plane-depth estimation.

This is direct evidence for a PatchMatch/ACMM/ACMMP-like **hypothesis-test-propagate** mechanism, but it does **not** authorize full PatchMatch or establish that all gauge-safe hard-tail assets share this failure mode.

## Scientific status

- optimizer steps: 0
- training: none
- architecture changed: false
- DEV/TUNE/CAL/EXTERNAL: unopened
- PatchMatch: not authorized by this exploratory witness alone
- teacher plane membership at inference: **removed**
- teacher visibility at inference: **removed**
- formal preregistered PASS: **not claimed**

## Next required experiment

Run the same discriminability ladder on a gauge-safe hard-tail asset that does not contain this detached giant-quad pathology. Preserve f089 as:

`CORPUS/ASSET-PATHOLOGY WITNESS + LOW-TEXTURE PROPAGATION POSITIVE CONTROL`

rather than treating it as the sole representative of normal product hard tail.
