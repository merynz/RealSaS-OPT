# RealSaS-OPT — Current State

**Date:** 2026-08-22  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `S0_CURRENT_ESTIMATE_RECORDED__FRONTEND_DESCRIPTOR_RESUMED__G1_BASELINE_PRESERVED_UNSTARTED`

## Read this first

This file is the single continuation authority. Detailed historical metrics/provenance remain preserved in the referenced canonical experiment files; this file summarizes the active state and does not supersede their frozen evidence.

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

Pose B is not a shipping or G1 dependency.

## Canonical problem definition

Authority: `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`

> From one neutral 8-view raster observation, recover the **minimal observation-grounded geometric substrate sufficient for downstream skeleton generation, skinning and compiler verification**.

Important authority rule now frozen at the top of that contract:

> substrate fields, neural-head assignments and SurfaceBuilder responsibilities are **evidence-derived current best estimates, not immutable truths**. They retain explicit evidence status and may be revised by stronger controlled experiments.

Canonical means “single current source of truth for what we believe and why,” not “guaranteed permanently correct.”

## S0 — OUTPUT/SUBSTRATE SEARCH PAUSED AFTER B2

Authorities:

- `experiments/s0_rigging_substrate/S0_A_INFORMATION_DERIVABILITY_MATRIX.json`
- `experiments/s0_rigging_substrate/S0_A_CORPUS_BINDING_REPORT.md`
- `experiments/s0_rigging_substrate/S0_B_PILOT_REPORT_V1.md`
- `experiments/s0_rigging_substrate/S0_B2_PREREG_V1.md`
- `experiments/s0_rigging_substrate/S0_B2_DERIVED_NORMAL_FALSIFICATION_REPORT_V1.md`
- `experiments/s0_rigging_substrate/S0_B2_RESULT_V1.json`
- `experiments/s0_rigging_substrate/RIGGING_SURFACE_CONTRACT_V1.json`

### S0-A data plane

The exact observation/rig join remains:

```text
V19.14 observation rows: 1408
M5 V18.76 rows:          3456
exact sample-id join:    1408
observation-only:           0
joined split:            1024 train / 384 validation
```

Every joined row has 512 surface points plus product-core skeleton/hierarchy, dense weights and deformation probes. Surface connectivity is not imported as settled hidden authority.

### S0-B prior pilot

Open-development bundle-000 pilot established:

- `P` carries substantial rigging signal;
- normal information is downstream-useful;
- naive `N_derived(P)` was weaker than exact N;
- deterministic local graph adds small value;
- view support/provenance adds Geppetto-side joint-locus information;
- exact source faces are not required by current evidence.

### S0-B2 fresh falsification — COMPLETE

Fresh source: bundle `001`, all train rows, disjoint from prior pilot qualification.  
Split: `600 train / 96 selection / 96 qualification`; prereg committed before qualification open.

Primary arms all received identical `P + deterministic graph + ordered-view XY/support`; only N changed:

```text
B0  N = zero
B1  strongest selected deterministic N
B2  direct/exact N
```

Strongest selected deterministic N:

- 12-NN distance-weighted covariance;
- minimum-eigenvector local plane;
- 40-NN local-density sign orientation.

Fresh qualification normal fidelity:

```text
row-median median angle   24.09 deg
row-p90 median angle     111.78 deg
>90 deg sign/error rate   16.77%
```

This materially improves the old naive ~32 deg / ~26% treatment but remains imperfect.

#### Arachne information probe

| arm | CE ↓ | top1 ↑ | deform mean ↓ |
|---|---:|---:|---:|
| B0 | 1.169798 | 0.683573 | 0.00184629 |
| B1 deterministic N | 1.157670 | 0.688395 | 0.00181863 |
| B2 exact N | **1.134800** | **0.698690** | **0.00178366** |

B1 -> B2:

- CE delta `-0.022871`, 95% bootstrap CI `[-0.02735, -0.01875]`, 91.7% families improve;
- deformation delta `-3.50e-5`, CI `[-4.40e-5, -2.64e-5]`, 86.5% improve.

**Interpretation:** stronger deterministic normals recover part of the normal benefit but do not close the exact-N skinning/deformation gap.

#### Oracle-count Geppetto information probe

Joint mean:

```text
B0  0.133121
B1  0.133051
B2  0.133185
```

B1 -> B2 joint-mean CI crosses zero and Wilcoxon `p≈0.913`.

**Interpretation:** normal information is not a meaningful joint-locus bottleneck in this controlled Geppetto probe once P + graph + ordered-view provenance are present.

### Current evidence-derived RiggingSurface estimate

Authority: `experiments/s0_rigging_substrate/RIGGING_SURFACE_CONTRACT_V1.json`

```text
IRIS
  P              direct learned core                     SUPPORTED
  N              direct learned current-best-estimate    SUPPORTED for Arachne; revisionable
  V/support      preserve observation support             head status not final
  U/risk         predictive-risk candidate                final requirement/calibration deferred
  persistence    mandatory capability                     explicit Z optional
  provenance     preserved bookkeeping                    SUPPORTED
       ↓
SurfaceBuilder
  local graph / neighbourhood     DETERMINISTIC-FIRST
  curvature/local geometry        DETERMINISTIC-FIRST
  sheet/component logic           DETERMINISTIC-FIRST
  reprojection/cycle              DETERMINISTIC-FIRST
  exact source faces              NOT REQUIRED by current evidence
```

`U`, set-valued `H`, artist-domain ambiguity and full product Geppetto count/topology remain future tests, but **output-head expansion is no longer the active research priority**. Reopen only if frontend/predicted-geometry evidence exposes a missing substrate quantity.

## ACTIVE FRONTIER — IRIS FRONTEND DESCRIPTOR / CORRESPONDENCE

We now return to the frontend line that was active before the S0 detour.

Preservation authority: `experiments/g0_g1_single_pose_geometry/PRESERVATION_LEDGER.json`.

Current retained lineage:

- **D1 coarse descriptor** — real descriptor signal; reserve as coarse/global persistence evidence / optional `Z`.
- **D2 fine spatial descriptor** — strong local precision gain but unsafe as global rank authority; reserve for local refinement.
- **D3 matcher idea** — reserve as learned/local matcher if a staged correspondence system still has a hard tail.
- **reciprocal/cycle primitive** — previously positive deterministic correspondence primitive; should be combined with descriptor evidence rather than discarded.

Current architectural lesson:

```text
DO NOT:
D2 fine score -> global top1 authority

DO:
coarse/high-recall candidate generation
  -> reciprocal/cycle consistency
  -> fine local descriptor scoring/refinement
  -> retain ambiguity when top1 is unsupported
  -> common-frame P/N geometry
```

### Next descriptor experiment

The next frontend experiment should be a **single-pose A×8 staged correspondence test**, not a return to Pose-A/Pose-B mechanics:

```text
F0  D1 coarse/global descriptor only
F1  F0 + reciprocal/cycle consistency
F2  F1 + D2 fine-local reranking inside retained top-k only
F3  learned D3-style matcher only if F2 leaves a reproducible hard tail
```

Primary metrics must include top1, top-k containment/recall, regret/localization distance, reciprocal disagreement, family hard-tail and common-frame geometry consequence. No mean-only promotion and no early singleton collapse.

This is now the active research direction.

## G0 / G1 — PRESERVED, UNCHANGED

G0 remains frozen. G1 remains fully preregistered and **training has not started**.

Frozen G1 path:

```text
A×8
 -> retained SharedImageEncoder
 -> retained 8-view GlobalMultiViewFusion
 -> retained DenseFusionDecoder
 -> geometry heads
 -> P / N / V / U
```

Canonical G1 facts remain:

- split `177 fit / 16 tune / 16 cal / 29 dev = 238`;
- 24 epochs, batch 1;
- AdamW LR `5e-5`, WD `1e-4`, grad clip `2.0`;
- N1D BEST migration `141/141` destination tensors loaded;
- descriptor/camera-residual modules dormant/frozen in the frozen baseline;
- executable preflight PASS on family 10178;
- optimizer steps `0`;
- cal/sealed/external remain closed under their frozen authorization points.

No S0 result retroactively changes the frozen G1 baseline. The frontend descriptor work is an explicit research continuation; any intervention into the G1 model itself must remain separately preregistered from the frozen baseline.

## Non-negotiable process discipline

- GitHub is the canonical continuation surface.
- No silent component deletion; preserve lineage/provenance.
- Every field/head statement carries evidence status and is revisionable under stronger evidence.
- Every confirmatory prereg is frozen before optimizer steps/truth opening.
- Keep official validation/sealed/external panels closed until their frozen authorization point.
- Heavy corpus/cache/checkpoint data stays in Drive; GitHub stores code, hashes, manifests and compact results.
