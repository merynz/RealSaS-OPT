# RealSaS-OPT — Current State

**Date:** 2026-08-22  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `G1_ENTRY_FROZEN__A100_RUNALL_AUTHORIZED_NOT_STARTED`

## Read this first

This file is the single continuation authority for a new session. Older N1D/GFDR reports remain historical evidence but do **not** define the active product roadmap.

## Canonical product architecture

```text
ONE neutral pose × 8 ordered views
 -> IRIS: P/N/V/U + required surface correspondence/persistence semantics
 -> Geppetto: clean skeleton/hierarchy proposal
 -> Arachne: skinning/weight proposal
 -> Compiler: canonicalize/verify/repair/export
 -> editable puppet + runtime animation
```

Shipping target remains `1 pose × 8 views`. Pose B may remain historical/research evidence but is not a G1 inference or target-construction dependency.

### IRIS boundary

Required semantic output/capability:
- `P`: common/object-frame position or equivalent surface geometry;
- `N`: local surface orientation/normal evidence;
- `V`: per-view visibility/observational support;
- `U`: calibrated geometric uncertainty;
- geometric surface correspondence/persistence strong enough to form coherent multiview surface hypotheses;
- provenance/source-view support.

`Z` is optional as an explicit learned correspondence embedding. Correspondence itself is **not optional**.

IRIS does **not** own authored mechanical owner identity, parent/topology prediction, skeleton, weights or mandatory GFDR.

## D2 closure — preserved diagnostic

D2 fine spatial remains sealed as `D2_NOT_SUFFICIENT__PROCEED_TO_D3_MATCHER` under its original prereg lineage. D1/D2/D3 mechanisms are `RESERVE_CALLABLE`, not active G1 authority. D2 improved local precision but worsened the hard tail, so G0/G1 explicitly retain median + p90/p95 evaluation.

## G0 — FROZEN

Canonical file:
`experiments/g0_g1_single_pose_geometry/G0_CONTRACT_FREEZE.json`

G0 now freezes:
- input exactly `A×8`;
- SurfaceEvidenceSet `P/N/V/U` semantics plus required persistence/provenance;
- sample metrics and family mean/median/p90/p95 aggregation;
- no mean-only promotion;
- no final product thresholds at this stage; G7 owns product qualification.

## G1 — ENTRY FROZEN, TRAINING NOT STARTED

Scientific question:

> With Pose B and all mechanical/cross-pose objectives removed, can the retained IRIS multiview core produce a usable common-frame surface from `A×8` alone?

Active path is still the smallest causal surgery:

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

`G1_SPLIT_FREEZE.json` is frozen from the canonical N1D family split:
- fit: `177`
- tune: `16`
- cal: `16`
- dev: `29`
- total: `238`
- source split SHA-256: `450cc8ce3137f7072a2db8613ce2178fe88328c99b4eb8312516e7e175bbc4f4`

G1 uses exactly one unique Pose-A sample per family. `e00` is the target-source episode for all 238 families; the episode index confirms `e00` exists for every family. Pose-A rasters are the shared family-level `poseA/00_V0.png..07_V7.png` set.

### Truth boundary

Canonical G1 cache construction reads **Pose-A fields only** from the observation sidecar:
`family_id, camera_center, camera_half_extent, surface_points_A, surface_normals_A, surface_xy_A, surface_visibility_A`.

The GPU target contains only:
`P_A, N_A, XY_A, V_A, direct_obs_A`.

Pose-B fields, scene flow, differential/mechanics targets, authored rig truth and `carrier_id_TRAINING_ONLY` are excluded by the loader boundary.

On real family `10178/e00`, the direct A-only sidecar projector reproduced canonical `P_A/N_A/XY_A/V_A/direct_obs_A` bit-for-bit.

### Warm-start / passive preservation

Canonical N1D `BEST.pt` migration remains:
- source tensors: `170`
- destination tensors: `141`
- loaded: `141/141 = 100%`
- missing: `0`
- shape mismatches: `0`
- archive-only: `29`, all `differential.*`

Dormant `descriptor_z` and camera-residual modules remain in the state dict for provenance/migration but are frozen (`requires_grad=False`) and have zero G1 loss authority. D1/D2 source remains recoverable for later G5 use.

### Executable preflight

`G1_EXECUTABLE_PREFLIGHT.json` is `PASS` on live corpus family `10178` with optimizer steps `0`:
- all 8 Pose-A PNG hashes match the family manifest;
- G1 input shape: `[1,8,4,128,128]`;
- no forbidden truth leakage;
- active loss gradient reaches encoder/fusion/decoder/geometry;
- descriptor gradient: `0`;
- camera-residual gradient: `0`;
- unit tests: `7/7 PASS`.

Warm-start metrics from this one family are **diagnostic only**, not a G1 result: point mean `~0.02247`, point p95 `~0.04927`, cross-view point spread p95 `~0.03427`, normal median `~22.75°`, normal p95 `~57.67°`, visibility Brier `~0.1654`, uncertainty-error Spearman `~0.50`.

## Frozen G1 training contract

Canonical prereg:
`experiments/g0_g1_single_pose_geometry/G1_PREREG_V1.json`

- 24 epochs, batch size 1;
- AdamW, LR `5e-5`, weight decay `1e-4`, grad clip `2.0`;
- only `requires_grad=True` parameters enter optimizer;
- no augmentation and no hyperparameter sweep;
- tune16/e00 selects checkpoint with explicit hard-tail family statistics;
- dev stays closed until BEST is frozen, then opens exactly once;
- cal stays closed throughout the G1 baseline;
- sealed21 and external10 stay closed.

G1 does **not** define a product PASS threshold. It is a baseline/causal gate; G7 owns final product qualification.

## NEXT EXECUTABLE STEP

1. Build the hash-pinned A100 G1 Run-All directly from `G1_PREREG_V1.json` and `G1_FREEZE_MANIFEST.json`.
2. Run the frozen baseline without architecture/hyperparameter changes.
3. Freeze BEST on tune16.
4. Open dev29 once and run the preregistered geometry metrics + causal diagnostics.
5. Route the next intervention from the observed failure mode only.

After G1 result:
- coherent geometry but camera/view ambiguity -> **G2** camera/ray-aware cross-view fusion;
- weak common-frame geometry -> **G3** direct common-frame pointmap/surface formulation;
- geometry adequate but grounding/safety weak -> **G4** explicit geometric grounding/refinement;
- geometry strong with residual localization hard-tail -> **G5**, where D2/D3 reserve components may return locally.

## Non-negotiable process discipline

- GitHub is the canonical continuation surface.
- `CURRENT_STATE.md` must be updated at every closed gate or material architecture decision.
- No silent component deletion; preserve lineage/provenance.
- Every prereg is frozen before optimizer steps/truth opening.
- Keep cal/sealed/external panels closed until their frozen authorization point.
- Heavy corpus/cache/checkpoint data stays in Drive; GitHub stores code, hashes, manifests and compact results.
