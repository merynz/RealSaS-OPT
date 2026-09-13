# Mage FIT2 Temporary Render Mesh Seal — P1_B2_G10

**Date:** 2026-09-13  
**Status:** `SEALED_TEMPORARY_RENDER_MESH__NOT_SCIENTIFIC_PASS__NOT_PRODUCT_PASS`  
**Treatment:** `P1_B2_G10`

## Purpose

This seal fixes the temporary mesh used by the product/render path while the fully qualified product mesh remains scientifically open.

Hard boundary:

- this is **not** a mesh scientific PASS;
- this is **not** PRODUCT_PASS;
- this is **not** an Arachne input;
- this is **not** permission to relax the frozen mesh policy;
- it replaces the old incomplete Living Compile/demo mesh as the explicit temporary render mesh;
- exact treatment identity is required; `latest/newest/current mesh` aliases are forbidden.

## Algorithm identity

`P1_B2_G10` is the first preregistered arm of `BASELINE-PRESERVING ADAPTIVE PATCH CDT`:

- coverage authority: exact sealed baseline CDT replay;
- local repair/recovery operator: adaptive support-derived constrained Delaunay triangulation;
- repair neighborhood: 1 baseline face-adjacency hop;
- boundary sampling stride: 2 px;
- interior sampling spacing: 10 px;
- generated points: only `IDENTITY_SURFACE_NODE` or `LOCAL_CONVEX_INTERPOLATION` over admitted S;
- unbound quality-Steiner generation: disabled;
- global adaptive reconstruction: forbidden;
- global uniform Steiner subdivision: forbidden;
- cross-component / UNKNOWN-relation / extrapolative bridges: forbidden;
- every accepted patch obeys monotonic coverage and exact-alpha containment.

Prereg authority: `canonical/FIT2_BASELINE_PRESERVING_ADAPTIVE_PATCH_CDT_PREREG_20260913.md`.

## Why this treatment is being sealed for temporary rendering

The treatment fails the frozen all-view product-mesh policy because five views retain raster triangle min-angle/max-aspect outliers. It nevertheless has high foreground coverage, exact 100% alpha precision in all eight views, no degenerate/duplicate/non-manifold topology in the recorded treatment, and is qualitatively/quantitatively far from the historical ~20–30% coverage demo failure.

| View | Recall | Precision | IoU | Largest hole | Min angle° | Max aspect | Full policy |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| V0 | 97.230% | 100.000% | 97.230% | 0.997% | 0.209181 | 279.419 | FAIL |
| V1 | 98.948% | 100.000% | 98.948% | 0.322% | 0.253990 | 250.000 | PASS |
| V2 | 97.679% | 100.000% | 97.679% | 1.373% | 0.136011 | 598.095 | FAIL |
| V3 | 98.339% | 100.000% | 98.339% | 0.645% | 0.194703 | 294.399 | FAIL |
| V4 | 97.343% | 100.000% | 97.343% | 1.001% | 0.040620 | 1584.926 | FAIL |
| V5 | 98.863% | 100.000% | 98.863% | 0.395% | 0.250198 | 249.355 | PASS |
| V6 | 97.502% | 100.000% | 97.502% | 1.374% | 0.259717 | 249.955 | PASS |
| V7 | 98.230% | 100.000% | 98.230% | 0.625% | 0.126289 | 476.922 | FAIL |

Range: recall `97.230%..98.948%`; mean `98.017%`; precision `100%` on all views. Three of eight views satisfy the entire frozen policy; five fail only recorded shape-quality invariants.

## Exact evidence

- Drive folder V1: `1cPzdUNGX_EYH0JrRHttwBPC3xZjjV6g8`
- result: `FIT2_BASELINE_PRESERVING_ADAPTIVE_PATCH_CDT_RESULT.json` / Drive id `1dyxQfF-bo5ptKMAjk9MKMrLIxbc-THa1`
- result SHA-256: `2c7c445376936acd160d12b2919974e8921c7229678f3f1ca5201e87d4cb8e4a`
- run log: `FIT2_BASELINE_PRESERVING_ADAPTIVE_PATCH_CDT_RUN.log` / Drive id `18V-jKdRbtHIM8ryQtpIeCObTop1h5-gI`

Per-view P1 mesh and candidate lineage hashes are frozen in the companion JSON seal.

## Product rule

For the current demo/productization work, reconstruct/use **exactly `P1_B2_G10`** from the sealed runner + exact FIT2 S/GSA inputs and bind the resulting exact per-view lineage into the product transaction. Do not silently substitute `P2_B1_G8`, `P2_B1_G6`, an older Living Compile mesh, or an unqualified regenerated mesh.

The future qualified product mesh may supersede this seal only through an explicit scientific closure/promotion transaction.
