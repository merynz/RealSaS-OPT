# RealSaS IRIS — Representation Authority Study V1 Prereg

**Date:** 2026-08-23  
**Status:** `FROZEN_BEFORE_NEW_CORPUS_REPRESENTATION_RESULTS`  
**Scope:** exact/legal observation representation and noise-tolerance ceiling; no IRIS optimizer steps, no joints/parents/skin/mechanics authority.

## 1. Question

For one neutral A-pose × 8 controlled views, what is the smallest observation-grounded representation that preserves persistent physical surface identity robustly enough for the rigging substrate?

The study separates:

1. **target authority** — continuous physical surface locus is the legal correspondence target;
2. **representation ceiling** — how much correspondence survives exact/noisy legal geometry;
3. **learner** — deferred; a failed learned descriptor cannot be called an information limit while the exact ceiling is high.

## 2. Ground-truth authority

The only correspondence authority is:

```text
foreground pixel
 -> triangle_id + barycentric_uv
 -> continuous canonical surface point P
```

`triangle_id` alone is never identity truth. Source vertex IDs, authored rig ownership, joints, parents, weights, Pose B and mechanics are forbidden as frontend correspondence authority.

Two pixels are the “same locus” only up to a preregistered physical surface tolerance induced by raster sampling. The evaluator uses exact continuous canonical geometry; it does not use source-rig identity.

## 3. Important E_surface correction

A fixed per-mesh Laplace–Beltrami eigenvector coordinate is **not** preregistered as canonical truth for arbitrary topology.

Reason:
- eigenvector sign/order and degenerate eigenspaces are not stable across remeshing;
- arbitrary assets do not share a template surface;
- forcing a per-mesh arbitrary basis would create a target-authority problem.

CSE is retained as a **learning principle**:

```text
same continuous physical locus across legal views -> descriptor positive
different physical locus -> descriptor negative / distance-aware negative
```

Therefore learned `E_surface`/`Z` is relation-supervised, not required to reproduce an arbitrary spectral teacher coordinate.

## 4. Exact/noise arms

### R0 — exact P
Common-frame continuous point only.

### R1 — exact P + N
Adds exact smooth normal under current corpus authority. This tests correspondence robustness only; S0-B2 separately establishes N's downstream Arachne value.

### R2 — noisy P
Independent controlled common-frame perturbation with normalized-object amplitudes:

`0, 0.0005, 0.0010, 0.0025, 0.0050, 0.0100`

These correspond to 0%, 0.05%, 0.10%, 0.25%, 0.50%, 1.00% of canonical max extent.

### R3 — noisy P + noisy N
Normal angular perturbation:

`0°, 5°, 10°, 20°, 40°`

No teacher rig information may orient/correct the perturbation.

### R4 — uncertainty-aware global relational address
Global pairwise relational compatibility derived from observation geometry, with soft confidence/uncertainty weights. Hard commitment to uncertain anchors is forbidden.

Historical hard-anchor noisy-R regression is treated as a falsified implementation, not as evidence against R-as-address.

## 5. Existing evidence reused, not rerun

- D1 coarse descriptor objective: directionally supported versus matched control.
- G2 same-view reciprocal/cycle rank: promoted deterministic primitive.
- D2 fine descriptor: global ranking authority falsified; local retained-top-k signal preserved.
- GFDR R: exact rank-3 global address ceiling supported under its historical oracle-granted within-pose carrier scope; noisy hard-anchor formulation not promoted.
- S0-B2: P core supported; direct N currently supported for Arachne/deformation, Geppetto joint-locus neutral.

These are prior constraints on interpretation, not new-corpus qualification.

## 6. Evaluation surface

Use family-disjoint deterministic samples from the new 3993-family corpus after Gate 1–3 closure.

Required stratification:
- source provider;
- split;
- capability class;
- component-count/complexity bucket;
- observation coverage quality.

Do not open CAL/DEV/EXTERNAL panels beyond their existing authority unless a separate authorization/prereg explicitly permits it. The representation study should use FIT/TUNE/open research surfaces by default.

View pairs:
- adjacent yaw;
- skip-one yaw;
- opposite/low-overlap pair where legal.

## 7. Metrics

For every arm/noise level:

- surface-locus top-1 success;
- top-k containment (`k=4,8`);
- physical localization error in canonical units;
- reciprocal disagreement;
- 3-view cycle inconsistency;
- family median and p90/p95 tail;
- source-stratified metrics;
- symmetry/repeated-structure hard-tail;
- coverage-conditioned metrics.

Mean-only promotion is forbidden.

## 8. Set-valued rule

If several candidates are physically indistinguishable within the evaluator tolerance or symmetry makes a singleton unsupported, the representation may retain a set/top-k.

A method is penalized for:
- excluding the true physical locus;
- forcing a false singleton;
- creating product-consequential inconsistency.

It is not penalized merely for preserving genuine ambiguity.

## 9. Promotion logic

### P/P+N ceiling supported
If exact R0/R1 gives near-ceiling top-k containment and failures are confined to raster discretization / legal visibility ambiguity, common-frame geometry remains a sufficient identity substrate.

### R needed
Promote uncertainty-aware R only if it improves a reproducible noisy-P hard tail across families without regressing low-noise geometry.

### Learned descriptor needed
A learned query-conditioned local descriptor/matcher is justified only for the residual tail after:
`coarse/global -> reciprocal/cycle -> geometry -> local rerank`.

D2 fine score may never become global top-1 authority.

### Information-limit claim
Forbidden in this study. A low learned score is not an information-limit result.

## 10. Deliverables

1. `REPRESENTATION_AUTHORITY_RESULT_V1.json`
2. per-arm/noise aggregate and family/source tables
3. hard-tail asset/view-pair manifest
4. uncertainty-aware R ablation
5. exact/noise tolerance envelope for IRIS engineering
6. decision:
   - `P_GEOMETRY_SUFFICIENT`
   - `P_PLUS_R_REQUIRED`
   - `LEGAL_REPRESENTATION_STILL_INSUFFICIENT__PROCEED_TO_SOI2`
   - or `APPARATUS_TARGET_REOPEN_REQUIRED`

## 11. Downstream handoff

This study does not itself qualify the product. Its output becomes the frontend engineering specification for:

```text
predicted P/N/persistence/U
 -> SurfaceBuilder
 -> exact-vs-predicted Geppetto/Arachne consequence
```

The decisive final IRIS criterion remains downstream product consequence, not pixel metric alone.
