# RealSaS-OPT — Current State

**Date:** 2026-08-22  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `S0_B_PILOT_COMPLETE__S0_B2_DERIVED_NORMAL_FALSIFICATION_NEXT__G1_BASELINE_PRESERVED_UNSTARTED`

## Read this first

This file is the single continuation authority for a new session. Older N1D/GFDR reports remain historical evidence but do **not** define the active product roadmap.

## Canonical product architecture

```text
ONE neutral pose × 8 ordered views
 -> IRIS: observation-grounded rigging-sufficient geometry
 -> deterministic SurfaceBuilder: canonical RiggingSurface
 -> Geppetto: clean skeleton/hierarchy proposal
 -> Arachne: skinning/weight proposal
 -> Compiler: canonicalize/verify/repair/export
 -> editable puppet + runtime animation
```

Shipping target remains `1 pose × 8 views`. Pose B is not a shipping, G1 inference or target-construction dependency.

## Canonical problem definition — FROZEN

Authority:
`canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`

Canonical scientific question:

> From one neutral 8-view raster observation, recover the **minimal observation-grounded geometric substrate sufficient for downstream skeleton generation, skinning and compiler verification**.

The target is the functional 8-view observational equivalent of the geometric shape input consumed by 3D auto-riggers. It is **not** exact hidden-mesh reconstruction and it is not hidden mechanical ontology recovery.

The final IRIS head list is explicitly an **S0 experimental result**.

### Head-derivation rule

A learned head is promoted only when:
1. downstream rigging requires the information;
2. deterministic derivation from already admitted evidence is insufficient;
3. matched family-disjoint ablation shows material downstream/safety benefit;
4. the quantity is observable from `A×8`, or uncertainty/set-valued representation can safely express non-identifiability.

Therefore:

`HEADS = minimal downstream-sufficient observable quantities`.

### Working geometric object

Current candidate representation is a common/object-frame oriented surfel evidence field:
- `P`: position evidence;
- `N`: local orientation/normal information;
- `V`: view-wise visibility/support;
- `U`: geometric risk/uncertainty;
- provenance/source-view support;
- required cross-view correspondence/persistence capability;
- optional retained geometric hypotheses where singleton geometry is unjustified.

`Z` remains optional as an explicit learned correspondence embedding. Correspondence itself is not optional.

Normals, visibility, adjacency, curvature and other quantities are **not automatically separate neural heads**. S0 tests direct/learned versus deterministic derivation.

### Deterministic SurfaceBuilder

A non-learned SurfaceBuilder is an explicit boundary between raw IRIS evidence and Geppetto/Arachne. Candidate duties: surfel fusion, duplicate merge, provenance/support, reprojection/cycle checks, local neighborhood/adjacency, sheet/component separation and stable derived local geometry. It is not a fourth learned model.

IRIS does **not** own authored mechanical owner identity, parent/topology prediction, skeleton, weights or mandatory GFDR.

## S0 — ACTIVE FRONTIER

Authorities:
- `experiments/s0_rigging_substrate/S0_PLAN_AND_PREREG_V1.md`
- `experiments/s0_rigging_substrate/S0_A_INFORMATION_DERIVABILITY_MATRIX.json`
- `experiments/s0_rigging_substrate/S0_A_CORPUS_BINDING_REPORT.md`
- `experiments/s0_rigging_substrate/S0_B_PILOT_REPORT_V1.md`
- `experiments/s0_rigging_substrate/S0_B_PILOT_RESULT_V1.json`

Scientific question:

> What is the minimum observation-grounded geometric information that must be delivered from `A×8` so downstream skeleton and skinning systems can recover a clean, editable, functionally valid rig?

### S0-A — DATA PLANE PASS

The corpus binding remains:

```text
V19.14 observation rows: 1408
M5 V18.76 rows:          3456
exact sample-id join:    1408
observation-only:           0
joined split:            1024 train / 384 validation
```

All 1408 joined records have privileged teacher arrays + manifest present, all 1408 M5 row validations pass and each row has 512 surface points. Product-core control count across the joined set is 6..61, median 21. Most rows carry 16 standardized deformation probes.

Important: M5 manifest reports `component_contract_status = pending_surface_connectivity_owner`; surface connectivity is not imported as hidden authority.

## S0-B — OPEN-DEVELOPMENT PILOT COMPLETE

Canonical result:
`experiments/s0_rigging_substrate/S0_B_PILOT_REPORT_V1.md`

### Scope / truth discipline

The pilot used only `IRIS_M5_V18_76_PRODUCT_CORE_CORPUS_000.zip`:
- 790 rows;
- 790 unique families;
- 512 surface points per row;
- deterministic SHA-ordered split: 600 train / 95 selection / 95 qualification;
- official 384-row M5 validation remained closed.

This result is **open-development evidence, not confirmatory authority**, because the exact probe preregistration was not committed to GitHub before the 95-row pilot qualification was opened. Do not relabel it as confirmatory later.

### Tested substrate arms

```text
A0  P only
A1  P + N_derived(P), deterministic kNN/PCA k=12
A2  P + exact/direct N
A3  A2 + deterministic local graph descriptors
A4  A3 + exact view visibility/support + provenance XY
```

A5/A6/A7 were not tested because exact GT substrate does not instantiate genuine prediction uncertainty/multimodality and no geometry-only failure justified semantic appearance features.

### ArachneProbe result — SUPPORTED

Probe: selected substrate + **GT product skeleton/parents** -> dense skinning -> exact deformation probes.

Qualification summary:

| Arm | CE ↓ | top1 ↑ | deform mean ↓ | family-p95 deform ↓ |
|---|---:|---:|---:|---:|
| A0 | 1.318255 | 0.614453 | 0.002124 | 0.004661 |
| A1 | 1.307283 | 0.615049 | 0.002107 | 0.004562 |
| A2 | 1.269899 | 0.636431 | 0.002050 | 0.004493 |
| A3 | 1.266380 | 0.640892 | 0.002038 | 0.004510 |
| A4 | 1.281625 | 0.642681 | 0.002037 | 0.004554 |

Key causal evidence:
- A0 -> A2 exact normals: CE delta about `-0.04836`; deformation mean delta about `-7.44e-5`; both paired bootstrap intervals exclude zero.
- A2 -> A3 deterministic graph: small additional positive effect.
- A3 -> A4 support/provenance: no meaningful fixed-skeleton deformation gain and worse CE/hard-tail.

Interpretation:
- `P` alone carries substantial skinning information;
- **normal information is causally useful**;
- the tested naive deterministic normal derivation does not recover the full exact-N benefit;
- graph/neighborhood information remains deterministic-first;
- no learned faces/adjacency head is justified.

### Deterministic normal audit

For `N_derived(P)` using 12-NN PCA on pilot qualification:
- oriented median angular error `32.08°`;
- oriented p90 `154.18°`;
- >90° orientation error on `26.44%` of points;
- sign-invariant median error `23.13°`.

This falsifies **that specific naive derivation**, not all deterministic derivations from `P + view provenance`.

### Geppetto probe result — PARTIAL / CONTROLLED

The first unconstrained 48-query set decoder learned joint-locus coverage but its existence/count head destabilized; count MAE remained too large. It is marked:

`INVALID_FOR_HEAD_DECISION`

No product-Geppetto conclusion may be drawn from it.

A second **oracle-count joint-locus probe** used GT joint count only to isolate whether substrate fields carry joint-location information. It is a research information probe, not product inference.

Qualification:

| Arm | joint mean ↓ | family-p95 ↓ | PCK@0.08 ↑ |
|---|---:|---:|---:|
| A0 | 0.128674 | 0.210421 | 0.358516 |
| A1 | 0.128534 | 0.214812 | 0.361194 |
| A2 | 0.126855 | 0.215006 | 0.362233 |
| A3 | 0.125293 | 0.206513 | 0.370615 |
| A4 | 0.118838 | 0.205213 | 0.405811 |

Paired A0 -> A4 joint-mean delta is about `-0.00984` with bootstrap interval excluding zero. A3 -> A4 is also positive.

Interpretation:
- exact normals carry a small reproducible joint-locus benefit;
- deterministic local graph helps;
- **view support/provenance carries additional Geppetto-side information even when exact P/N are present**.

This supports preserving view/support/provenance in the RiggingSurface contract, but does not imply a separate learned V/provenance head; these may remain direct observation bookkeeping or deterministic reconstruction.

### Oracle-matched joint -> skin diagnostic

With oracle count, Hungarian semantic matching and GT parents retained:
- A0 deformation mean `0.003522`;
- A4 deformation mean `0.003319`;
- paired delta about `-0.000203`, bootstrap interval excludes zero.

This is diagnostic only, but shows that richer observation-grounded geometry can improve a joint->skin deformation chain.

## Current S0 decisions

### SUPPORTED

- `P` = **core primitive**.
- **Normal information is required/useful downstream.**
- correspondence/persistence remains required capability.
- view support/provenance should be preserved for downstream Geppetto-side use.
- local graph/neighborhood structure remains **SurfaceBuilder / deterministic-first**.
- exact hidden mesh faces/source topology are not required by current evidence.

### NOT YET FROZEN

- `N` as a **separate learned IRIS head**. Current naive `N_derived(P)` fails to close the exact-N downstream gap, but the strongest derivation from `P + exact view provenance/correspondence` has not yet been tested.
- full Geppetto joint count/topology generation.
- calibrated `U` and set-valued `H` requirements under predicted/noisy/artist-domain geometry.

## NEXT EXECUTABLE GATE — S0-B2

Before S0-C, run one stronger falsification:

> Can the exact-N downstream benefit be recovered deterministically from `P + view support/provenance/correspondence`, or does `N` need to remain a direct learned IRIS output?

Requirements:
1. Use a **fresh open-development family panel not used by S0-B pilot qualification**; do not reuse the opened 95 rows as confirmatory evidence.
2. Freeze strongest deterministic normal/surface derivation before opening its qualification panel.
3. Compare at minimum:
   - `P`;
   - `P + best deterministic N(P, provenance)`;
   - `P + direct/exact N`;
   - each with the same deterministic graph policy.
4. Reuse fixed Arachne information probe and oracle-count joint-locus information probe only after code/method parity is frozen.
5. If the direct-N gap survives materially, promote `N` as required learned IRIS output in S0-C.
6. If the gap closes, move normals into SurfaceBuilder and simplify the final IRIS head contract.

After S0-B2, freeze `RIGGING_SURFACE_CONTRACT_V1.json` in S0-C.

## G0 / G1 — PRESERVED, UNCHANGED

G0 remains frozen. G1 remains fully preregistered and **training has not started**.

Frozen G1 path:

```text
A×8
 -> retained SharedImageEncoder
 -> retained 8-view GlobalMultiViewFusion
 -> retained DenseFusionDecoder
 -> retained geometry head
 -> P / N / V / U
```

No S0 result retroactively mutates G1. G1 is still the smallest frozen baseline/causal surgery. It will be interpreted later against the S0-C substrate contract.

Canonical G1 facts remain:
- split 177 fit / 16 tune / 16 cal / 29 dev = 238;
- 24 epochs, batch 1, AdamW LR `5e-5`, WD `1e-4`, grad clip `2.0`;
- N1D BEST migration `141/141` destination tensors loaded;
- descriptor/camera-residual modules dormant/frozen;
- executable preflight PASS on family 10178;
- optimizer steps `0`;
- cal/sealed/external remain closed under their frozen authorization points.

## After S0-C / G1 routing

- coherence/correspondence gap -> **G1.5** reciprocal/cycle/high-recall persistence + SurfaceBuilder;
- camera/view ambiguity -> **G2** camera/ray-aware fusion;
- representation inadequacy/multimodality -> **G3** direct/richer common-frame geometry;
- grounding/calibration weakness -> **G4**;
- residual local hard-tail -> **G5**;
- stylized drawing/domain gap -> **G6**;
- final IRIS qualification -> **G7**;
- then R0/R1 Geppetto, R2 Arachne, R3 joint rig quality, C0 compiler rebind, P0 product gate.

## Non-negotiable process discipline

- GitHub is the canonical continuation surface.
- `CURRENT_STATE.md` must be updated at every closed gate or material architecture decision.
- No silent component deletion; preserve lineage/provenance.
- Every confirmatory prereg is frozen before optimizer steps/truth opening.
- Keep official validation/sealed/external panels closed until their frozen authorization point.
- Heavy corpus/cache/checkpoint data stays in Drive; GitHub stores code, hashes, manifests and compact results.
