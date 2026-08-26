# IRIS P Hard-Tail Forensics — 2026-08-26

Status: `ACTIVE_RESEARCH_HISTORY__DO_NOT_DELETE`

This archive intentionally preserves the apparent catastrophe **and** the immediate causal diagnosis. Future readers must not see only the bad P95 numbers and conclude that the observation problem was inherently unsolved.

## 1. Scale ladder: apparent catastrophe

Frozen FIT-scale ladder (`P-V5/R256`, same architecture and 7168-update budget per rung):

| FIT families | aggregate P95 | cell median P95 | worst cell P95 |
|---:|---:|---:|---:|
| 32 | 0.29040165 | 0.17066082 | 0.51614741 |
| 128 | 0.21184100 | 0.09814133 | 0.51423088 |
| 512 | 0.19979690 | 0.06592983 | 0.51396067 |

Verdict: `SCALE_SIGNAL_BUT_NOT_STRONG`.

The fixed TRAIN_DIAG32 control under the same 7168-update budget worsened with the larger family rung, confirming that the 512 rung is also optimization-budget limited. However, that does **not** explain the stationary ~0.514 worst tail.

Repeated micro-P95 spikes of `28.4803` occurred at steps `2752`, `4768`, and `6240`.

## 2. Gauge localization: first diagnosis

Zero-gradient gauge audit scanned the 544 cached TRAIN assets × 2 styles. No optimizer step and no heldout split was opened.

Decision:

`TARGET_GAUGE_PATHOLOGY_ESTABLISHED__CORRECT_BEFORE_OPTIMIZATION_OR_PATCHMATCH`

Key counts:

- PROXY32 analytically impossible cells: `2/64`
- PROXY32 gauge-safe cells: `62/64`
- TRAIN512 analytically impossible cells: `6/1024`
- common asset across all repeated `28.4803` anomaly batches:
  `asset_90b5f2cc531696040da996a9`

Thus the hard tail was not one phenomenon. A small but severe target/gauge pathology contaminated the tail, while most proxy cells remained genuinely gauge-safe.

The canonical gauge artifacts were written to Drive under:
`IRIS_SINGLE_POSE_V2_P_V5_R256_HARDTAIL_GAUGE_LOCALIZATION_V3`.

## 3. f089: gauge-safe residual witness

`asset_f089abadcd071194617d640b` is gauge-safe:

- analytic screen P95 ≈ `7.3e-05`
- P-V5/R256 full-P P95:
  - cel ≈ `0.408874`
  - ink ≈ `0.448743`

Therefore its ~0.4 residual cannot be attributed to the analytic screen-plane gauge.

### Asset-structure caveat discovered immediately afterward

The asset contains a detached rectangular component:

- vertices `591..594`
- faces `776,777`
- exactly 4 vertices / 2 triangles
- disconnected from the rest of the mesh
- all four vertices have total source skin weight `0.0`
- each triangle area ≈ `0.148846`
- near-constant-x plane normal ≈ `[0.99999036, -0.00257924, -0.00355237]`

This is why some render views contain a huge brown rectangle. It is a real source-geometry component, not a plotting bug. Its semantic role is unresolved because the current geometry package has no material/UV/component-name authority.

**Do not use f089 as the sole representative of normal product hard tail.**

## 4. Preliminary CPU oracle: plane-membership-assisted

A local CPU discriminability audit showed:

- non-plane loci: median depth error `0.003061`, P90 `0.033717`
- huge-plane loci: median `0.047548`, P90 `0.174806`
- exact-normal 5×5 warp alone did not solve the plane interior
- with image-derived confident seeds but teacher-provided plane membership, planar propagation collapsed:
  - V2 P90 `0.360643 -> 0.002743`
  - V6 P90 `0.289796 -> 0.002565`

This established that depth information exists in sparse cross-view evidence, but left open whether the plane region itself could be discovered without teacher membership.

See:
`F089_CPU_DISCRIMINABILITY_ORACLE_PRELIMINARY.md`.

## 5. Teacher-free region/propagation oracle

The follow-up removed both teacher plane membership and teacher visibility from the inference treatment.

Image-only region discovery (dominant foreground RGB + connected component) achieved teacher-evaluated IoU:

- V1 `0.9921`
- V2 `0.9833`
- V3 `0.8885`
- V5 `0.9921`
- V6 `0.9146`
- V7 `0.9577`

V0/V4 are edge-on and are not usable as region masks, but they are crucial geometric constraints.

### Preserved exploratory FAIL: best-3 view selection

The first teacher-free attempt aggregated only the three easiest target views. It failed:

- V2 propagated P90 `0.32777`
- V3 `0.34947`
- V7 `0.48730`

Diagnosis: `best-3` systematically discards the hard edge-on V0/V4 views that actually localize plane depth. Easy same-color oblique views are not the informative views.

### Corrected exploratory treatment: all-view consistency

Using all seven other views for candidate consistency:

| ref | seed good <.005 | pointwise P90 | propagated P90 |
|---|---:|---:|---:|
| V1 | 100.00% | 0.002280 | **0.001113** |
| V2 | 98.75% | **0.359339** | **0.001253** |
| V3 | 95.00% | 0.007464 | **0.001460** |
| V5 | 100.00% | 0.002284 | **0.001059** |
| V6 | 97.50% | 0.046122 | **0.001526** |
| V7 | 100.00% | 0.002688 | **0.001345** |

This is not a preregistered formal PASS. It is a mechanism-localization result.

The remaining V3/V6 P95 tail is largely region contamination: same-colored non-plane geometry enters the simple image-only connected component. This moves the local problem from “missing depth” toward **surface-support/region ownership + propagation**.

See:
- `F089_TEACHER_FREE_REGION_PROPAGATION_V1.md`
- `F089_TEACHER_FREE_REGION_PROPAGATION_V1_METRICS.json`
- `f089_teacher_free_region_propagation_v1.py`

## 6. Current interpretation

The sequence is:

```text
apparent P catastrophe
  -> target/gauge pathology localized
  -> majority of proxy cells remain gauge-safe
  -> f089 chosen as gauge-safe residual witness
  -> detached giant quad discovered
  -> low-texture pointwise ambiguity localized
  -> teacher-membership propagation succeeds
  -> teacher membership removed
  -> naive best-view selection FAILS
  -> informative edge-on views identified
  -> all-view geometric consistency + robust propagation succeeds
```

What is established:

- some catastrophic P tails were corpus/target apparatus failures;
- at least one gauge-safe residual contains recoverable multiview depth information;
- low-texture pointwise regression can fail while sparse confident matches remain correct;
- propagation can recover the plane without teacher region membership;
- aggressive “easy-view” selection can destroy the relevant geometry;
- f089 also has an unusual detached non-skinned quad, so external generalization is not claimed.

What is **not** established:

- all gauge-safe hard tails are planar;
- full PatchMatch is required;
- f089 is representative of normal character/object geometry;
- product P closure has been reached.

## 7. Next gate

Run the same discriminability ladder on a gauge-safe hard-tail asset without the detached giant-quad pathology, on open FIT evidence only.

Candidate from the frozen proxy tail:
`asset_ea593d044e14f20abe6d2818`

No architecture promotion and no PatchMatch authorization before this representative witness.
