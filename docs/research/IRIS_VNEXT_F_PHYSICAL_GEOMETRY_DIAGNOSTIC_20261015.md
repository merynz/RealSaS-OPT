# IRIS-vNext F physical geometry diagnostic

**Date:** 2026-10-15  
**Status:** diagnostic architecture evidence — not a geometry-quality PASS, no final F freeze  
**Depends on:** `IRIS_VNEXT_F_PHYSICAL_ASSEMBLY_ORACLE_20261015.md`

---

## 1. Question

The physical assembly oracle showed that qualified local boundary/source-address incidence can reconstruct correct global topology on several FIT8 morphologies using completely fresh local triangulation.

That does **not** imply that the crude fresh triangulation has acceptable geometry.

This court asks:

> If topology is correct but each disk patch is filled by a simple center fan and each annulus by a simple zipper, how far does the resulting local surface drift from the intended source patch?

The result is intentionally used to separate **topology representation** from **geometry parameterization**.

---

## 2. Diagnostic metric

For sampled candidate triangles:

1. compute the candidate triangle centroid;
2. find its exact closest point among the source triangles belonging to the teacher patch that generated that candidate triangle;
3. report distance normalized by extraction-cell pitch;
4. compare the candidate triangle normal with the nearest source triangle normal using orientation-insensitive absolute angle.

This is a local oracle diagnostic, not a full symmetric Chamfer/Hausdorff surface proof.

---

## 3. Horse

Sampled candidate faces: `50,000` out of `103,954`.

Centroid distance / cell pitch:

```text
p50  = 0.00264
p95  = 0.05463
p99  = 0.09111
max  = 0.18281
mean = 0.01154
```

Absolute normal-angle error:

```text
p50  = 1.39 deg
p95  = 32.80 deg
p99  = 64.38 deg
max  = 89.98 deg
mean = 6.74 deg
```

Two degenerate candidate triangles were observed in the crude fan baseline.

Interpretation:

- topology closure does not imply a production-quality local parameterization;
- centroid position remains relatively close to the source at this pitch;
- tangent/normal quality can degrade badly in curved local patches.

---

## 4. T-Rex

Deterministic sample: `5,000` candidate faces out of `59,562`.

Centroid distance / cell pitch:

```text
p50  = 0.00249
p95  = 0.08049
p99  = 0.12963
max  = 0.30920
mean = 0.01660
```

Absolute normal-angle error:

```text
p50  = 1.49 deg
p95  = 55.39 deg
p99  = 84.28 deg
max  = 89.99 deg
mean = 10.44 deg
```

No degenerate candidate triangle was observed in this sample.

---

## 5. George

Deterministic sample: `5,000` candidate faces out of `155,130`.

Centroid distance / cell pitch:

```text
p50  = 0.01064
p95  = 0.09452
p99  = 0.13168
max  = 0.19043
mean = 0.02470
```

Absolute normal-angle error:

```text
p50  = 5.56 deg
p95  = 57.63 deg
p99  = 88.93 deg
max  = 90.00 deg
mean = 13.68 deg
```

No degenerate candidate triangle was observed in this sample.

---

## 6. Architecture consequence

The result argues **against** turning the residual topology language into an independent geometry generator.

The cleaner split is:

```text
G
  continuous geometry / distance / uncertainty field

F geometry core
  sparse high-resolution extraction
  bounded vertex deformation / configuration
  normals / geometry losses
  mature DMC/FlexiCubes-class local surface parameterization

F topology state
  qualified local sheet identity
  crossing multiplicity
  face strands / loops / open endpoints
  authored boundary / singular incidence
  sparse explicit residual where base capacity is insufficient
```

The topology residual should alter **incidence/capacity**, while geometry positions remain bound to the same G/F continuous scene evidence.

In particular:

```text
TOPOLOGY RESIDUAL != SECOND GEOMETRY MODEL
```

This preserves the strongest part of the InstantMesh -> StdGEN -> StdGEN++ extraction lineage while extending its local topology capacity for generic RealSaS morphology.

---

## 7. Relationship to T

This result also reinforces the F/T separation:

```text
F = high-fidelity explicit reconstruction surface
T = animation/runtime carrier retopology
```

T should not be asked to repair a geometrically crude F surface.

The correct sequence remains:

```text
G continuous field
  -> high-fidelity topology-capable F
  -> T dynamic/runtime retopology
  -> Compiler canonical M
  -> AXIS / MIRA / CAO
```

TESSA-derived topology reasoning belongs in the animation-carrier problem, not as a replacement for F's high-fidelity geometry parameterization.

---

## 8. Next geometry court

The crude fan/zipper candidate is now only a lower baseline.

Next matched oracle comparison should use the **same exact teacher local evidence** with:

```text
A. crude fan/zipper topology-only baseline
B. DMC/FlexiCubes-class base geometry
C. F-R base geometry + sparse topology residual
D. F-C explicit upper-capacity reference
```

Required geometry metrics should expand to:

- symmetric point-to-surface error;
- normal error;
- depth/silhouette error in the eight canonical views;
- false bridge / missing sheet count;
- self-intersection;
- thin-structure retention;
- residual-region-specific error;
- memory/query cost.

Only after those courts close should F be trained behind the expensive image-to-scene IRIS trunk.
