# RealSaS S0-B2 — Derived Normal Falsification Report V1

**Date:** 2026-08-22  
**Status:** `S0_B2_COMPLETE__DIRECT_N_CURRENTLY_SUPPORTED_FOR_ARACHNE__GEPPETTO_NEUTRAL`  
**Authority:** fresh family-disjoint research falsification; not product qualification.

## Question

Can a stronger deterministic normal derived from admitted geometry close the downstream gap to direct/exact normal information?

This is a head-assignment question. S0-B already established that normal information is useful; S0-B2 asks whether that information must currently remain a direct IRIS output candidate or can be moved entirely into deterministic SurfaceBuilder derivation.

## Frozen source and split

Source bundle: `IRIS_M5_V18_76_PRODUCT_CORE_CORPUS_001.zip`  
Frozen package SHA-256: `78be9007cd4e0df1deb64e2258dd090a11c54a55de649896c74ed3ed24253897`.

The bundle contributes 792 train rows, disjoint from the prior S0-B bundle-000 pilot panel.

SHA-ordered split:

- train: 600;
- selection: 96;
- qualification: 96.

Frozen ID hashes:

- train: `ab09eccc375abac499e742ab26a22771969015dcaaf7f2eb1a8615ee188d4ac5`;
- selection: `fbd2db34ed98b6e30ee10f89d2be20198e9c3fa4a80202ce64ff7da49e9aa0f2`;
- qualification: `bd64de6914a1e00b6342617a6b689a7245a27185e121ee92fbf62f531dd334ce`.

The method/probe prereg was committed before qualification was opened: `S0_B2_PREREG_V1.md`.

## Deterministic normal treatment

The selected deterministic treatment uses:

1. 12-neighbour common-frame geometry;
2. distance-weighted covariance `exp(-(d / median(d))^2)`;
3. minimum-eigenvalue eigenvector as unsigned local normal;
4. 40-neighbour local point-mass asymmetry for sign orientation, choosing the direction away from the weighted local mass centroid;
5. unit normalization.

Selection-phase alternatives included naive PCA and explicit view-plane/Delaunay constructions. The selected density-oriented weighted-PCA treatment was the strongest tested deterministic option. View-plane triangulation did not improve the selection normal target and was not promoted.

The deterministic treatment uses no exact normals, rig truth, weights, Pose B or mechanics truth.

### Normal fidelity

On fresh 96-family qualification:

- row-median angular error, median across families: **24.09°**;
- row-p90 angular error, median across families: **111.78°**;
- row-p95 angular error, median across families: **150.38°**;
- >90° orientation-error fraction: **16.77%**;
- mean per-family angular error: **44.34°**.

This is materially better than the S0-B naive derivation (~32° median and ~26% >90°), but still far from exact normal orientation.

## Frozen primary arms

Every arm receives identical P-derived graph descriptors and ordered-view XY/support slots. Only the 3D normal slot changes:

```text
B0  P + graph + provenance/support; N = 0
B1  B0 + strongest selected deterministic N_det(P)
B2  B0 + direct/exact N
```

## Arachne information probe

Frozen probe: 57D point/joint pair feature slots -> MLP `96 -> 96 -> 1`, softmax over GT product-core skeleton controls; seed 1862, AdamW `2e-3`, WD `1e-4`, 6 epochs, BEST by selection soft-label CE.

### Qualification

| Arm | CE ↓ | weight MAE ↓ | top1 ↑ | GT mass in predicted top4 ↑ | deform mean ↓ | deform family-p95 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 1.169798 | 0.038841 | 0.683573 | 0.944923 | 0.00184629 | 0.00472406 |
| B1 derived N | 1.157670 | 0.038377 | 0.688395 | 0.945576 | 0.00181863 | 0.00468050 |
| B2 exact N | **1.134800** | **0.037561** | **0.698690** | **0.948100** | **0.00178366** | **0.00454483** |

### Paired causal statistics

Derived N is useful versus no N:

- B0 -> B1 CE delta: **−0.012128**, 95% bootstrap CI `[-0.01692, -0.00785]`, 77.1% families improved, Wilcoxon `p≈3.29e-8`;
- B0 -> B1 deformation mean delta: **−2.77e−5**, CI `[-3.89e−5, -1.69e−5]`, 79.2% improved, `p≈1.20e-9`.

But exact N retains a material advantage over the stronger deterministic normal:

- B1 -> B2 CE delta: **−0.022871**, 95% CI `[-0.02735, -0.01875]`, 91.7% families improved, Wilcoxon `p≈1.02e-15`;
- B1 -> B2 deformation mean delta: **−3.50e−5**, 95% CI `[-4.40e−5, -2.64e−5]`, 86.5% improved, `p≈1.28e-12`.

For context, the full B0 -> B2 CE gain is `−0.034998`; the stronger deterministic normal recovers only part of that gap.

### Arachne interpretation

`N_det(P)` is genuinely useful, but under the strongest tested deterministic treatment it **does not close the exact-N downstream gap**. For skinning/deformation, direct normal information therefore remains justified as the current best IRIS output assignment.

## Oracle-count Geppetto information probe

Frozen information-isolation probe: 36D point slots -> 48D query-attention set decoder with 52 queries; GT joint count supplied only as experimental control; seed 1862, AdamW `1.5e-3`, WD `1e-4`, 6 epochs, BEST by selection matched joint mean.

### Qualification

| Arm | joint mean ↓ | family-p95 ↓ | joint-p95 mean ↓ | PCK@0.05 ↑ | PCK@0.08 ↑ |
|---|---:|---:|---:|---:|---:|
| B0 | 0.133121 | 0.213529 | 0.250393 | 0.139127 | 0.318864 |
| B1 derived N | **0.133051** | **0.212611** | 0.254202 | 0.141214 | 0.322898 |
| B2 exact N | 0.133185 | 0.213749 | 0.251691 | **0.146186** | **0.323447** |

Paired joint-mean deltas are effectively null:

- B0 -> B1: `−0.000070`, bootstrap CI crosses zero, Wilcoxon `p≈0.945`;
- B1 -> B2: `+0.000134`, bootstrap CI crosses zero, Wilcoxon `p≈0.913`;
- B0 -> B2: `+0.000063`, bootstrap CI crosses zero, Wilcoxon `p≈0.792`.

### Geppetto interpretation

For this oracle-count joint-locus probe, normal information is not a meaningful bottleneck once P + graph + ordered-view provenance are present. This does **not** negate the Arachne result; it says the need for direct N is currently downstream-task-specific.

## Decision

### SUPPORTED

- `P` remains the core geometric primitive.
- Deterministically derived local normal information is useful for skinning.
- The stronger deterministic treatment materially improves over the naive S0-B derivation but still leaves a reproducible skinning/deformation gap to exact N.
- Direct/exact N provides no meaningful joint-locus advantage in the current oracle-count Geppetto information probe.

### CURRENT BEST ESTIMATE — REVISIONABLE

**Keep `N` as a direct IRIS output candidate for the current RiggingSurface contract.**

Reason: Arachne/deformation requires information not recovered by the strongest deterministic normal treatment tested here. Geppetto does not require N strongly, but downstream sufficiency is conjunctive: a field can be justified by Arachne even if Geppetto is neutral.

This is **not an immutable architectural truth**. The canonical contract explicitly states that field/head assignments are evidence-derived current estimates and may change if a stronger future deterministic surface reconstruction closes this gap.

### KEEP DETERMINISTIC-FIRST

- adjacency/local graph;
- curvature/local differential descriptors;
- provenance bookkeeping;
- sheet/component construction until evidence says otherwise.

### STILL UNTESTED / DEFERRED

- calibrated `U` under predicted/noisy geometry;
- set-valued `H` under genuine ambiguity;
- artist-domain inconsistency;
- full product Geppetto count/topology.

## Execution consequence

S0 output-head search stops here unless later evidence exposes a major missing quantity.

Current estimated frontend contract:

```text
IRIS
  P              direct learned core
  N              direct learned current-best-estimate
  V/support      preserve/estimate as observation support; head status may evolve
  U/risk         current predictive-risk candidate; calibration later
  persistence    mandatory capability; explicit Z optional
  provenance     preserved bookkeeping
       ↓
SurfaceBuilder
  deterministic graph/neighbourhood/sheet/reprojection/cycle
       ↓
Geppetto -> Arachne -> Compiler
```

Research attention now returns to the IRIS frontend descriptor/correspondence path rather than continuing to expand output heads.