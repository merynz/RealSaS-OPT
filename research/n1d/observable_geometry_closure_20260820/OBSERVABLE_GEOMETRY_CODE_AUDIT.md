# RealSaS IRIS — Observable Geometry Closure Code Audit

**Date:** 2026-08-20  
**Status:** `AUDIT_CLOSED__G1_TREATMENT_SELECTED__CAL16_TRUTH_CLOSED`

## Decision

The current IRIS geometry stack is not globally broken. The deterministic camera model, orthographic projection, rank-3 least-squares triangulation, visual-hull carrier construction, and set-valued feasible-world representation are structurally sound in the controlled corpus. The active weakness is earlier: raster evidence is converted into per-view proposal rankings mostly by descriptor score, while explicit multi-view geometric compatibility is used later to construct candidates but is not exposed strongly enough as proposal-level G evidence to the global correspondence solver.

Therefore the first closure treatment is **not** a larger pointmap backbone and is **not** an R retune. It is a one-factor G intervention: add deterministic multi-view reprojection consistency to proposal-level G while keeping V5 full-pair R bit/formula unchanged.

## Frozen executable source finding

The frozen front door contains:

1. shared multi-scale raster encoder;
2. global 2-pose × 8-view token fusion;
3. point / normal / visibility / uncertainty heads;
4. dense persistent descriptor head;
5. N1D same-view A↔B local correlation/transport;
6. raster-native visual hull / carrier construction;
7. global B-foreground top-4 descriptor proposal search;
8. known-camera rank-3 triangulation and all-view reprojection refinement;
9. set-valued feasible candidate support H;
10. compiler-side global correspondence / relational solve.

The geometry constructor already computes information analogous to modern geometry-grounded systems, but the V3-A/V5 proposal-level G unary remains descriptor-rank only.

## External architecture gap matrix

| System / mechanism | Useful mechanism | Current IRIS status | Closure role |
|---|---|---|---|
| DUSt3R / DPM / V-DPM | common/world-frame geometry; target-time-conditioned persistence | representation concept present; not paper-faithful | retain ontology; learned fusion is later treatment if needed |
| MASt3R | dense correspondence plus reciprocal matching evidence | dense descriptor present; reciprocal evidence absent from current G | prospective G2 if G1 passes |
| MV-TAP | explicit camera-ray embedding; time→view→space correspondence refinement | generic global fusion exists; tracked-proposal camera-aware refinement absent | learned fallback after deterministic G closure |
| MapAnything | use supplied camera/ray/pose priors directly | canonical camera metadata exists but proposal inference does not fully exploit it | architecture reference for learned closure |
| GGPT | matching/cycle/epipolar/triangulation/reprojection geometric authority | triangulation/reprojection exists in constructor; not exposed as proposal G reliability | **G1 treatment** |
| MVTracker / GGPT 3D refinement | refine in 3D after a reliable seed exists | absent | defer until correspondence is closed |

## Causal diagnostics on already-open e01 dev

All diagnostics use only frozen raster/model proposal evidence in forward scoring. Evaluator truth is opened only after scores exist and is used to label which proposal was oracle-best.

### Projection / triangulation parity

- independent projection implementation vs frozen source: max image-coordinate discrepancy ≈ `3.4e-5 px`;
- two-view triangulation round-trip median XYZ error ≈ `3.7e-8`, maximum ≈ `8.6e-8`.

This rejects a camera sign / scale / least-squares implementation bug as the main explanation.

### Proposal evidence

On the 79 rank-3 hard-tail view rows:

- descriptor G top-1: `18/79 = 22.78%`;
- frozen V5 descriptor-G + full-pair R: `26/79 = 32.91%`;
- exact candidate-pool-style refined reprojection selector: `33/79 = 41.77%`;
- final G1 solver replay (50% descriptor rank + 50% geometry rank, unchanged V5 R): `37/79 = 46.84%`.

On all 2,288 applicable open-dev rows:

- descriptor G: `490/2288 = 21.42%`;
- V5: `676/2288 = 29.55%`;
- G1 replay: `894/2288 = 39.07%`.

G1 hard-tail complementarity versus V5:

- G1 rescues 14 V5 misses;
- V5 has 3 correct rows missed by G1;
- 23 rows are correct in both.

### Reciprocal appearance diagnostic

A post-hoc MASt3R-like reverse-association diagnostic also carries independent signal:

- reverse-association rank: `24/79`;
- same-carrier reverse descriptor similarity: `28/79`.

The diagnostic union of V5 global R, explicit reprojection geometry, and reciprocal appearance reaches substantially above any single channel. This is **not** a deployable metric; it only establishes that multiple observation-native signals remain unfused.

## Scientific conclusion

The highest-information next step is to repair **G evidence exposure before increasing model capacity**.

Current bottleneck:

```text
raster / descriptor
    ↓
per-view top-k proposals
    ↓
[explicit geometry exists here but weakly influences proposal unary]
    ↓
global R solver
```

G1:

```text
raster / descriptor
    ↓
per-view top-k proposals
    ├── descriptor rank
    └── explicit multi-view reprojection / conditioning rank
             ↓
       geometry-grounded G
             ↓
     unchanged full-pair R
```

Only if this deterministic treatment fails prospectively should IRIS move to a learned camera-aware cross-view proposal fusion module. That module should combine MV-TAP-style camera-conditioned iterative correspondence with V-DPM-style target-time persistence, while retaining explicit geometry as deterministic authority. 3D learned refinement is deferred until correspondence selection is reliable.

## Prospective boundary

The G1 formula is frozen before semantic access to the canonical cal16 × e01 evaluation sidecars. `sealed21` and `external10` remain CLOSED. No G1 weight, threshold, proposal rule, R factor, or solver rule may change after cal16 truth opens.
