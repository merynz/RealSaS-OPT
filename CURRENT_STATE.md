# RealSaS-OPT — Current State

**Date:** 2026-08-22  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `S0_A_DATA_PLANE_PASS__S0_B_PROBE_IMPLEMENTATION_NEXT__G1_BASELINE_PRESERVED_UNSTARTED`

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

Shipping target remains `1 pose × 8 views`. Pose B may remain historical/research evidence but is not a shipping, G1 inference or target-construction dependency.

## Canonical problem definition — FROZEN

Authority:
`canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`

Canonical scientific question:

> From one neutral 8-view raster observation, recover the **minimal observation-grounded geometric substrate sufficient for downstream skeleton generation, skinning and compiler verification**.

The target is the functional 8-view observational equivalent of the geometric shape input consumed by 3D auto-riggers. It is **not** exact hidden-mesh reconstruction and it is not hidden mechanical ontology recovery.

The final IRIS head list is explicitly an **S0 experimental result**. The current `P/N/V/U + persistence/provenance` factorization remains the frozen G1 baseline candidate, not an assumption that every field requires an independent learned head or that no richer geometric state will be needed.

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
- `N`: local orientation/normal evidence;
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

Scientific question:

> What is the minimum observation-grounded geometric information that must be delivered from `A×8` so fixed downstream skeleton and skinning probes can recover a clean, editable, functionally valid rig?

### S0-A — DATA PLANE PASS

Initial field inventory is frozen. Current provisional conclusions:
- `P` is the clearest learned primitive candidate;
- `N` has strong downstream precedent but must beat `N_derived(P)` before becoming a required learned head;
- exact source mesh faces/topology are rejected as required IRIS product targets; locality is first represented by a canonical derived surface graph;
- `V/support`, provenance, graph structure, silhouette/boundary and local differential geometry have strong deterministic-derivation candidates;
- current G1 `log_sigma` is predictive risk, not yet calibrated `U`;
- correspondence/persistence is required, explicit `Z` is optional;
- set-valued geometry `H` remains a safety candidate for genuinely multimodal observations.

### S0-A corpus binding — PASS

The leading S0-B corpus is now directly provenance-bound:

```text
V19.14 observation rows: 1408
M5 V18.76 rows:          3456
exact sample-id join:    1408
observation-only:           0
joined split:            1024 train / 384 validation
```

All 1408 joined records have privileged teacher arrays + manifest present in the source-availability audit. All 1408 joined M5 row validations pass and every joined row has `512` surface points.

Joined product-core control counts: min `6`, max `61`, median `21`.  
Joined detail-control counts: min `6`, max `71`, median `28`.  
Most rows carry `16` standardized deformation probes (`1395/1408`; the remainder carry 12 or 14).

A real M5 row was downloaded and SHA-verified against its manifest. Its payload co-locates:

```text
surface_points [512,3]
surface_normals [512,3]
view_point_xy01 [8,512,2]
view_point_visibility [8,512]
product_core_node_xyz [J,3]
product_core_parent_index [J]
product_core_role_index [J]
product_core_surface_skin_weights [512,J]
product_core_support_mask [512,J]
deformation_probes
```

This is sufficient to implement fixed GeppettoProbe, ArachneProbe and JointProbe without building a new corpus first.

Important open field: M5 manifest reports `component_contract_status = pending_surface_connectivity_owner`. Surface connectivity is therefore **not** imported as a settled hidden authority; S0 must test deterministic SurfaceBuilder graph versus any later learned connectivity treatment.

### S0-B — NEXT EXECUTABLE GATE

Implement fixed-capacity research probes rather than product Geppetto/Arachne:
- `S0JoinedRow` loader with strict substrate/teacher separation;
- `GeppettoProbe`: selected substrate arm -> product-core skeleton/hierarchy;
- `ArachneProbe`: selected substrate arm + **GT product-core skeleton** -> dense skinning;
- `JointProbe`: predicted skeleton -> predicted skinning -> frozen deformation-probe evaluation.

Initial matched ablation ladder remains:

```text
A0  P only
A1  P + N_derived_from_P
A2  P + N_direct/exact
A3  A2 + deterministic surface graph
A4  A3 + V/support + provenance
A5  A4 + explicit ambiguity/hypothesis state where needed
A6  A5 + occupancy/thickness evidence if available
A7  A6 + appearance/semantic latent only if geometry-only arms leave a reproducible gap
```

First S0-B execution is **tiny open-development parity/smoke only**. Confirmatory field-ablation results are forbidden until the split, optimizer, probe capacity, evaluator and numerical non-inferiority margins are frozen.

### S0-C

Freeze `RIGGING_SURFACE_CONTRACT_V1.json`: minimal learned primitives, deterministic SurfaceBuilder outputs, ambiguity/uncertainty rules and downstream sufficiency metrics.

## Preserved research lessons

These remain active because they concern geometric observation quality rather than mandatory mechanics:
- high-recall correspondence before irreversible collapse;
- reciprocal/cycle consistency;
- explicit view/camera geometry when needed;
- common/object-frame geometry;
- geometric grounding/reprojection;
- visibility/support;
- uncertainty and provenance;
- set-valued ambiguity preservation;
- relational reasoning translated to static geometry.

Motion-specific outputs remain reserve diagnostics, not core IRIS authority:
- `p_active`, `log_amp`, motion `dir`, `ΔP`;
- GFDR/differential mechanics;
- Pose-B response;
- deformation Jacobians.

## D2 closure — preserved diagnostic

D2 fine spatial remains sealed as `D2_NOT_SUFFICIENT__PROCEED_TO_D3_MATCHER` under its original prereg lineage. D1/D2/D3 mechanisms are `RESERVE_CALLABLE`, not active G1 authority. D2 improved local precision but worsened the hard tail, so G0/G1 explicitly retain median + p90/p95 evaluation.

## G0 — FROZEN

Canonical file:
`experiments/g0_g1_single_pose_geometry/G0_CONTRACT_FREEZE.json`

G0 freezes:
- input exactly `A×8`;
- G1 baseline SurfaceEvidence `P/N/V/U` semantics plus required persistence/provenance;
- sample metrics and family mean/median/p90/p95 aggregation;
- no mean-only promotion;
- no final product thresholds at this stage; G7 owns product qualification.

G0/G1 are not retroactively rewritten by S0. S0 decides what the eventual product substrate must contain.

## G1 — ENTRY FROZEN, TRAINING NOT STARTED

Scientific question:

> With Pose B and all mechanical/cross-pose objectives removed, can the retained IRIS multiview core produce a usable common-frame surface from `A×8` alone?

Active frozen path:

```text
A×8
 -> retained SharedImageEncoder
 -> retained 8-view GlobalMultiViewFusion
 -> retained DenseFusionDecoder
 -> retained geometry head
 -> P / N / V / U
```

No MV-TAP/ray-aware fusion, direct pointmap reformulation, DPM/GGPT, D3 matcher, larger backbone or resolution increase is allowed before the frozen G1 baseline result.

### Exact split/sample freeze

`G1_SPLIT_FREEZE.json` remains:
- fit: `177`
- tune: `16`
- cal: `16`
- dev: `29`
- total: `238`
- source split SHA-256: `450cc8ce3137f7072a2db8613ce2178fe88328c99b4eb8312516e7e175bbc4f4`

G1 uses one unique Pose-A sample per family. `e00` is the target-source episode for all 238 families.

### Truth boundary

Canonical G1 cache construction reads Pose-A fields only:
`family_id, camera_center, camera_half_extent, surface_points_A, surface_normals_A, surface_xy_A, surface_visibility_A`.

The GPU target contains only:
`P_A, N_A, XY_A, V_A, direct_obs_A`.

Pose-B fields, scene flow, differential/mechanics targets, authored rig truth and `carrier_id_TRAINING_ONLY` are excluded by the loader boundary.

### Warm-start / passive preservation

Canonical N1D `BEST.pt` migration remains:
- source tensors: `170`
- destination tensors: `141`
- loaded: `141/141 = 100%`
- missing: `0`
- shape mismatches: `0`
- archive-only: `29`, all `differential.*`.

Dormant `descriptor_z` and camera-residual modules remain frozen for provenance/migration. D1/D2 source remains recoverable for later local refinement use.

### Executable preflight

`G1_EXECUTABLE_PREFLIGHT.json` remains `PASS` on live corpus family `10178` with optimizer steps `0`; unit tests are `7/7 PASS`.

Warm-start metrics from this one family remain diagnostic only.

## Frozen G1 training contract

`experiments/g0_g1_single_pose_geometry/G1_PREREG_V1.json` remains unchanged:
- 24 epochs, batch size 1;
- AdamW, LR `5e-5`, weight decay `1e-4`, grad clip `2.0`;
- only `requires_grad=True` parameters enter optimizer;
- no augmentation and no hyperparameter sweep;
- tune16 selects BEST with hard-tail statistics;
- dev opens once after BEST freeze under the existing contract;
- cal/sealed/external remain closed at their frozen authorization points.

G1 does not define product PASS. It is a baseline/causal gate; G7 owns final IRIS qualification against the S0-C substrate contract.

## Current execution order

1. Implement **S0-B fixed probe interfaces/loaders** against the 1408 exact-join corpus.
2. Run a tiny open-development loader/forward/backward/deformation-evaluator parity smoke only.
3. Freeze S0-B family split, optimizer, fixed probe capacity, evaluator and numerical non-inferiority margins.
4. Run family-disjoint field ablations A0→A6; authorize A7 only if geometry-only arms leave a reproducible gap.
5. Freeze S0-C `RIGGING_SURFACE_CONTRACT_V1.json`.
6. Run/interpret the already-frozen G1 baseline against this downstream-sufficiency target without changing its prereg.
7. Route later geometry interventions from observed failure mode only:
   - coherence/correspondence gap -> **G1.5** reciprocal/cycle/high-recall persistence + SurfaceBuilder;
   - camera/view ambiguity -> **G2** camera/ray-aware fusion;
   - representation inadequacy/multimodality -> **G3** direct/richer common-frame geometry;
   - grounding/calibration weakness -> **G4**;
   - residual local hard-tail -> **G5**;
   - stylized drawing/domain gap -> **G6**;
   - final IRIS qualification -> **G7**.

## Non-negotiable process discipline

- GitHub is the canonical continuation surface.
- `CURRENT_STATE.md` must be updated at every closed gate or material architecture decision.
- No silent component deletion; preserve lineage/provenance.
- Every confirmatory prereg is frozen before optimizer steps/truth opening.
- Keep sealed/external panels closed until their frozen authorization point.
- Heavy corpus/cache/checkpoint data stays in Drive; GitHub stores code, hashes, manifests and compact results.
