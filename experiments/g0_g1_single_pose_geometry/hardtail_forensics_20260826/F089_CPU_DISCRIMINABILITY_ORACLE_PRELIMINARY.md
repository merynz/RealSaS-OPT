# f089 CPU Discriminability Oracle — Preliminary

Asset: `asset_f089abadcd071194617d640b`

Execution: local assistant CPU/container; no Colab, no optimizer/training steps.

## Inputs

- `primary_geometry.npz`
- 8 × `cel_clean_512.png`
- 8 × `raster_authority.npz`
- known orthographic camera contract, half extent = 0.54, yaw spacing = 45°

Teacher raster authority was used only for audit/evaluation and, in one ladder rung, exact truth-visibility selection. It is not assumed available at inference.

## 1. Whole-asset pointwise cross-view oracle

Deterministic sample: 500 visible raster loci per reference view = 4000 loci total.
Candidate depth sweep: 81 camera-forward depth hypotheses + exact truth for margin audit.
Cost: cross-view point RGB L1 + alpha penalty, averaged over teacher-visible target views.
Usable loci (>=2 visible target views): 3951/4000.

- truth uniquely beats alternatives >=0.005 away: 23.6649%
- depth argmin absolute error P50: 0.016398
- P75: 0.080594
- P90: 0.145373
- P95: 0.188703
- P99: 0.305087

Reference-view asymmetry is large: V0/V4 are much easier than side/diagonal views.

## 2. Dominant low-texture plane localization

Faces 776/777 form one huge near-planar rectangle:

- face areas: ~0.148846 each
- face normal: ~[0.9999904, -0.0025792, -0.0035524]
- vertices span approximately x=-0.274, y=[-0.1835, 0.1835], z=[-0.4583, 0.3529]

Fraction of visible raster rows belonging to faces 776/777:

- V0: 1.07%
- V1: 76.84%
- V2: 97.91%
- V3: 76.12%
- V4: 1.07%
- V5: 76.84%
- V6: 85.28%
- V7: 75.60%

On the whole-asset oracle sample, splitting by these two faces gives:

| region | n | median depth error | P90 depth error | unique-truth rate |
|---|---:|---:|---:|---:|
| non-plane | 1455 | 0.003061 | 0.033717 | 46.87% |
| huge plane | 2496 | 0.047548 | 0.174806 | 10.14% |

This is the main localized hard-tail mechanism in f089.

## 3. Normal/plane patch warp alone

A 5×5 exact-normal plane-warp NCC oracle was tested on deterministic subsets. It did not materially solve the uniform plane interior. It slightly helps some oblique views and hurts others; the pointwise ambiguity remains. Therefore surface-normal warping alone is not the missing mechanism here.

## 4. Planar propagation / confident-seed oracle

For the huge plane only:

- 1000 raster loci per problematic reference view
- 161 depth hypotheses
- image-only cross-view RGB+alpha candidate costs over all other views
- confidence = cost gap to the best depth hypothesis >=0.01 away
- top 60 confident seeds
- RANSAC fit of orthographic plane depth `d = a*s_r + b*s_u + c`
- plane membership (faces 776/777) is oracle-provided for this audit

Results:

| ref view | seed top60 good (<.005) | before P50 | before P90 | after P50 | after P90 | after P95 |
|---|---:|---:|---:|---:|---:|---:|
| V1 | 100.0% | 0.001025 | 0.002132 | 0.000226 | 0.000502 | 0.000572 |
| V2 | 98.33% | 0.001276 | 0.360643 | 0.001561 | 0.002743 | 0.002930 |
| V3 | 100.0% | 0.001027 | 0.002264 | 0.000247 | 0.000560 | 0.000599 |
| V5 | 100.0% | 0.000983 | 0.002327 | 0.000305 | 0.000731 | 0.000776 |
| V6 | 100.0% | 0.001253 | 0.289796 | 0.001488 | 0.002565 | 0.002844 |
| V7 | 100.0% | 0.001037 | 0.002306 | 0.000376 | 0.000776 | 0.000851 |

The catastrophic V2/V6 tails collapse from ~0.29–0.36 P90 to <0.003 P90 after robust planar propagation.

## Preliminary interpretation

For f089, the dominant hard tail is not evidence that camera-forward depth is absent from the observations. It is a **pointwise correspondence failure on a huge, nearly uniform planar surface**. Sparse cross-view loci are highly confident and correct; the missing operation is robust region/plane propagation rather than asking every interior pixel to independently choose depth.

This maps directly to PatchMatch/ACMM/ACMMP-style mechanisms: hypothesis-test-propagate, low-texture planar priors, geometric consistency, and robust view/seed selection. It does **not** yet prove the same diagnosis for every gauge-safe hard-tail asset.

## Caveats

1. The whole-asset visibility-selected rung uses teacher raster authority for truth visibility.
2. The planar-propagation rung uses image-derived candidate costs/confidence, but plane membership is supplied by teacher triangle IDs; plane/region discovery still needs its own observable test.
3. Only `cel_clean` has been tested in this local preliminary run.
4. This is an oracle/discriminability audit, not a shipping inference architecture.
