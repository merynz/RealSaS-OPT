# RealSaS-OPT — Current State

**Date:** 2026-08-27  
**Active branch:** `g0-g1/single-pose-geometry`  
**Status:** `E0_THREE_ARM_OBSERVABLE_GEOMETRY_APPARATUS_READY__CALIBRATION8_NEXT__SEALED_CLOSED`

## Read this first

This file is the single continuation authority.

The active question is the **single-pose observable common-frame geometry hard tail**. Before changing the raster representation, E0 must distinguish:

1. information lost because A×8 exposes only an observable surface subset;
2. information lost by deterministic cross-view persistence / SurfaceBuilder;
3. downstream consumer OOD caused by closed-mesh sampling/normalization expectations.

These are separate causal questions and must not be collapsed into one pretrained-consumer score.

## Frozen prior evidence

P-V5/R256 uses analytic screen-plane coordinates from raster XY + known camera and learns only camera-forward depth.

Frozen FIT scale ladder, same architecture/objective:

| FIT families | aggregate P95 | cell median P95 | worst cell P95 |
|---:|---:|---:|---:|
| 32 | 0.29040165 | 0.17066082 | 0.51614741 |
| 128 | 0.21184100 | 0.09814133 | 0.51423088 |
| 512 | 0.19979690 | 0.06592983 | 0.51396067 |

Frozen rung512 checkpoint SHA-256:
`a92fa18c46975578f2705ee3baa426cce573d4a767d42490e7599ed905f6e48e`.

Gauge localization V3 established a small number of target/gauge pathologies but did not explain the representative hard tail.

## D1 / D2→D5 localization — COMPLETE

Canonical report:
`experiments/g0_g1_single_pose_geometry/hardtail_forensics_20260826/P_HARDTAIL_ORACLE_D2_D5_20260826.md`

Representative gauge-safe witnesses:

- `f089 = asset_f089abadcd071194617d640b`: low-texture giant-plane positive control;
- `ea593 = asset_ea593d044e14f20abe6d2818`: severe foreshortening/correspondence hard tail;
- `662ed = asset_662ed7f1e328bd85959157cf`: thin/multisurface/repeated-appearance hard tail.

D2 frozen learned-feature reprojection:

| asset | matched direct P95 | best D2 P95 |
|---|---:|---:|
| f089 | 0.416423 | **0.123106** |
| ea593 | **0.471771** | 0.574262 |
| 662ed | **0.271590** | 0.304407 |

D3 exact-normal plane warp did not close ea593/662ed. D4 teacher visibility/source-view selection did not close them. D5 coarse-to-fine feature consistency pruned truth too early:

- ea593: s8 top64 `0.618` -> after s4 top16 `0.253`;
- 662ed: s8 top64 `0.840` -> after s4 top16 `0.338`.

D2b tiny 3-asset learned pair metric was negative on ea593/662ed and is too small to prove the representation itself lacks information.

**Strongest current localization:** representative gauge-safe failures are bottlenecked **before or at cross-view correspondence evidence/ranking**. Search cannot recover truth that the evidence has already ranked away. This is not an information-theoretic impossibility claim and not yet proof that a pretrained prior is required.

## E0 — CURRENT EXECUTABLE GATE

Authority directory:
`experiments/g0_g1_single_pose_geometry/e0_observable_geometry_20260827/`

Key documents:

- `E0_OBSERVABLE_GEOMETRY_PREREG_V1.md`
- `E0_DOWNSTREAM_ADAPTER_AND_OOD_CONTRACT_V1.md`
- `PRETRAINED_CONSUMER_CONTAMINATION_AUDIT_V1.md`
- `E0_MEMBERSHIP_V1.json`

E0 removes the raster learner entirely and now has **three geometry arms**:

```text
E0-0  FULL-MESH CANONICAL CEILING
      deterministic area-uniform full source surface
      same RealSaS canonical/object frame
      matched point budget
      NO external-consumer bbox normalization

E0-a  OBSERVABLE + ORACLE PERSISTENCE
      exact A×8 visible common-frame P
      P-only anchor selection
      teacher (triangle,bary) attached only after anchor freeze
      -> perfect physical persistence upper bound

E0-b  OBSERVABLE + DETERMINISTIC PERSISTENCE
      same exact visible P / same anchor budget as E0-a
      P-derived local normals
      known ortho geometry
      common-frame proximity
      reciprocal reprojection/cycle
      raster provenance
      NO teacher identity in matching API
```

Causal interpretation:

```text
E0-0 -> E0-a = observable coverage / supported-surface gap
E0-a -> E0-b = deterministic persistence / SurfaceBuilder gap
```

The primary E0 information gate stays in the **same canonical coordinate gauge**. RigAnything/TokenRig-specific centering/scaling belongs only to the later consumer-OOD diagnostic.

The observable anchor sampler also does not preserve naïve raster view-frequency weighting: it takes a fixed capped candidate budget from each view and then performs common-frame farthest-point sampling.

## E0 downstream decision

Primary decision uses the S0 from-scratch matched-probe discipline, not a pretrained rigging model:

```text
scratch E0-0 adequate?
  NO  -> downstream apparatus invalid; stop.
  YES -> compare E0-a.

E0-a non-inferior to E0-0?
  NO  -> observable-only substrate loses downstream-required information;
         revise substrate/SurfaceBuilder contract before correspondence training.
  YES -> compare E0-b.

E0-b non-inferior to E0-a?
  NO  -> deterministic persistence is the blocker even with exact P.
  YES -> observable geometry + deterministic persistence are information-sufficient;
         advance to C1/C2/C3 correspondence intervention.
```

Only after the primary scratch decision may pretrained RigAnything/TokenRig be used as a **consumer-OOD diagnostic**.

### External-pretrain contamination rule

RealSaS holdout cleanliness and external-pretrain cleanliness are separate statuses.

- RigAnything reports RigNet + 9,686 curated Objaverse training shapes, but the audited public release does not expose authoritative membership IDs: Objaverse-origin RealSaS assets are `PRETRAIN_CONTAMINATION_UNKNOWN` unless exact inclusion/exclusion is proven.
- Recommended TokenRig/SkinTokens checkpoint reports ArticulationXL2 + VRoid Hub + ModelsResource training; overlapping RealSaS source lineages cannot be treated as pristine external-pretrain holdout evidence without exact exclusion provenance.
- The RealSaS `EXTERNAL_HOLDOUT` seal is **not globally revoked**; the extra restriction applies only to claims involving an external pretrained consumer.

## E0 apparatus state

Local/source tests after the physical-persistence and full-mesh-ceiling corrections:

`11/11 PASS`

Firewalls:

- E0-b API contains no teacher triangle/bary/faces input;
- E0-a source `(triangle,bary)` identity is attached only after P-only anchor selection;
- `camera.json` is not consumed;
- Pose B / rig / weights / parents / mechanics are absent from geometry construction;
- E0 geometry-stage optimizer steps = `0`;
- full-mesh E0-0 remains canonical-frame, not consumer-normalized.

No E0 scientific outcome has been opened yet.

## Frozen population / next action

Population is inherited prospectively from the already-frozen FIT scale ladder:

- 512 FIT train-order authority for later fixed downstream probes;
- historical disjoint 8-asset FIT calibration panel;
- frozen family-disjoint `FIT_PROXY32` qualification panel;
- TUNE/CAL/DEV/EXTERNAL closed.

**NEXT EXECUTABLE STEP:** run **E0 calibration-8 geometry only** and inspect E0-0/E0-a/E0-b distribution/persistence diagnostics. The calibration execution package must not contain an executable `qualification_proxy32` path.

After calibration geometry:

1. freeze the scratch downstream probe and non-inferiority margins;
2. run the matched scratch E0-0/E0-a/E0-b calibration procedure;
3. only then authorize a separate proxy32 E0 qualification package.

## AFTER E0 ONLY — frozen three-arm correspondence intervention

Exactly three matched arms:

```text
C1  scratch + current objective
    -> current baseline

C2  scratch + tail-aware correspondence objective
    -> cheap control for objective/curriculum mismatch

C3  pretrained prior transplant
    -> RealSaS-native decoder with pretrained visual/multiview prior
       (MapAnything-family prior is a candidate)
       NOT the full MapAnything camera/ray/product wrapper
```

The scientific question is whether hard-tail correspondence evidence/ranking improves. Only if a representation demonstrates correct truth ranking/top-k containment may multiview search / fusion / propagation / PatchMatch-style mechanisms be promoted.

## Authorization state

`GAUGE_LOCALIZATION = COMPLETE`

`D2_D5_CORRESPONDENCE_LOCALIZATION = COMPLETE`

`E0_THREE_ARM_GEOMETRY_APPARATUS = LOCAL_TEST_PASS_11_OF_11`

`E0_CALIBRATION8_GEOMETRY = NEXT`

`E0_PROXY32_QUALIFICATION = CLOSED`

`E0_SCIENTIFIC_OUTCOME = NOT_YET_OPENED`

`C1_SCRATCH_CURRENT = NOT_YET_EXECUTED`

`C2_SCRATCH_TAIL_AWARE = NOT_YET_EXECUTED`

`C3_PRETRAINED_PRIOR_TRANSPLANT = NOT_YET_EXECUTED`

`MAPANYTHING_FULL_WRAPPER = NOT_AUTHORIZED`

`FULL_PATCHMATCH = NOT_AUTHORIZED`

`LONGER_P_TRAINING = NOT_NEXT`

`TUNE_CAL_DEV_EXTERNAL = CLOSED`

`PRODUCT_SUBSTRATE_CLOSURE = NOT_CLAIMED`
