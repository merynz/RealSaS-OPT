# RealSaS E0 — Observable Geometry Sufficiency / Persistence Prereg V1

**Date:** 2026-08-27  
**Branch:** `g0-g1/single-pose-geometry`  
**Status:** `PREREG_DRAFT_IMPLEMENTATION_PREFLIGHT__NO_OUTCOME_OPENED`  
**Scientific optimizer steps at E0 geometry stage:** `0`

## Why E0 exists

S0 established which fields in an exact observation-equivalent substrate help fixed rigging probes. E0 asks a narrower prerequisite that S0 did not close:

> Is the surface geometry that is actually supported by one neutral native-1024 eight-view observation sufficient, and how much does realistic deterministic persistence lose relative to oracle persistence when the underlying visible `P` is exact?

E0 is not a raster learner experiment and is not a MapAnything experiment. IRIS prediction is removed from the causal graph.

## Frozen causal comparison

Both observable arms receive the **same exact view-local visible common-frame `P`** reconstructed from native `raster_authority.npz` + `primary_geometry.npz`. The only treatment variable between E0-a and E0-b is cross-view persistence authority.

### E0-a — oracle persistence upper bound

A canonical physical anchor is a teacher surface point identified by source `(triangle_id, barycentric_uv)`. Anchor selection itself is P-only; teacher identity is attached only after the anchor rows are frozen. The exact continuous projection is then tested against target-view raster authority for same-surface visibility support. Teacher provenance is permitted in this arm because E0-a is an upper bound.

Interpretation: perfect physical persistence is granted; never-visible mesh completion is still forbidden.

### E0-b — deterministic SurfaceBuilder persistence

Teacher triangle/bary identity, exact normals, rig truth and camera JSON are forbidden from scoring/merging. Correspondence may use only:

- exact common-frame `P` (E0 premise);
- normals derived deterministically from view-local visible `P` neighborhoods;
- known frozen orthographic view geometry;
- source/target raster pixel provenance;
- common-frame proximity;
- normal continuity;
- reciprocal reprojection / cycle consistency.

Teacher surface provenance may be opened only after E0-b has emitted matches, to evaluate them.

## Shared observable surface construction

- Native authority: `1024×1024`, views `V0..V7`.
- No `camera.json` consumption. Camera convention is the frozen P-V5/R256 known-yaw contract: yaw `0,45,...,315`, screen-up `+Z`, fixed render half-extent `0.54`.
- No Pose B.
- No joints, parents, skin weights, owner IDs, mechanics/GFDR or compiler IDs enter E0 surface construction.
- Never-visible mesh vertices/faces are not emitted as RiggingSurface evidence.
- The same deterministic P-only anchor sampler is used for E0-a and E0-b so differences cannot be caused by different point budgets.

## Frozen population

Membership is inherited byte-semantically from the pre-existing FIT scale ladder, before E0 outcomes:

- `probe_train_512`: frozen FIT train-order 512;
- `calibration_anchor8`: historical frozen 8-asset FIT panel, disjoint from the 512 training set;
- `qualification_proxy32`: frozen family-disjoint `FIT_PROXY32`, including the previously diagnosed hard-tail witnesses;
- TUNE/CAL/DEV/EXTERNAL remain sealed.

No post-result family selection is permitted.

## Geometry/persistence metrics

E0-b is compared with E0-a per anchor/target-view pair:

- support/correspondence precision and recall;
- false-match rate;
- missed-visible-support rate;
- reciprocal reprojection residual in native pixels;
- common-frame match error;
- normal-continuity statistics;
- view-support count distribution;
- anchor coverage and abstention rate.

Hard-tail reporting is family-wise; no mean-only promotion.

## Downstream sufficiency phase — three geometry arms, two consumer questions

E0 freezes **three geometry arms** so coverage/distribution loss cannot be confused with persistence loss:

- `E0-0`: full/closed source-mesh surface through a consumer-native sampler/normalizer — downstream distribution ceiling only;
- `E0-a`: A×8 observable visible union with oracle physical persistence;
- `E0-b`: the same observable visible union with deterministic SurfaceBuilder persistence.

Thus:

```text
E0-0 -> E0-a = observable coverage / sampling / normalization gap
E0-a -> E0-b = persistence / SurfaceBuilder gap
```

The **primary information-sufficiency gate** reuses the S0 from-scratch matched-probe discipline: identical capacity, initialization rule, optimizer budget, point budget, split and evaluator across all three geometry arms. It exposes:

1. Geppetto information-isolation probe;
2. Arachne probe with GT product skeleton fixed;
3. joint/deformation diagnostic where valid.

Pretrained RigAnything/TokenRig results are a separate **consumer-OOD diagnostic**, not the primary definition of substrate sufficiency. A stock-pretrained failure can be caused by closed-mesh sampling/coverage/normalization expectations or pretrained-data overlap. Observable-only finetuning/adaptation may diagnose OOD only after a legal/technical training path and contamination-clean evaluation membership are established.

Exact adapter and contamination rules are frozen in `E0_DOWNSTREAM_ADAPTER_AND_OOD_CONTRACT_V1.md` and `PRETRAINED_CONSUMER_CONTAMINATION_AUDIT_V1.md`.

Numerical non-inferiority margins must be frozen on `calibration_anchor8` before `qualification_proxy32` is opened for E0 outcome inspection.

## Decision tree

```text
E0-0 full-mesh ceiling adequate under the fixed scratch probe?
  NO  -> downstream probe/target apparatus is not qualified; stop.
  YES -> compare E0-a.

E0-a non-inferior to E0-0?
  NO  -> observable-only coverage/sampling/normalization loses downstream-required information;
         substrate/SurfaceBuilder contract requires revision before correspondence training.
  YES -> compare E0-b.

E0-b non-inferior to E0-a?
  NO  -> deterministic persistence is the blocker even with exact P.
  YES -> observable geometry + deterministic persistence are information-sufficient under E0;
         advance to the frozen three-arm raster correspondence intervention.

Only after this primary decision may stock-vs-observable-adapted pretrained consumers be used to diagnose consumer OOD.
```

## Next intervention after E0 only

Three arms, matched except for the stated treatment:

1. `C1 scratch + current objective` — frozen current baseline;
2. `C2 scratch + tail-aware correspondence objective` — cheap control for loss/curriculum mismatch;
3. `C3 pretrained prior transplant` — RealSaS-native decoder with pretrained visual/multiview prior (MapAnything-family prior candidate), not the full MapAnything camera/ray/product wrapper.

Only if a representation demonstrates correct correspondence ranking/containment on frozen hard tails may multiview search, fusion, propagation or PatchMatch-style mechanisms be promoted.

## Firewalls

- `E0 learner optimizer steps = 0`.
- `FULL_PATCHMATCH = NOT_AUTHORIZED` during E0.
- `MAPANYTHING_FULL_WRAPPER = NOT_AUTHORIZED`.
- E0-b must pass an implementation audit proving teacher identity is evaluation-only.
- Qualification data must not tune thresholds.
- Pretrained external-consumer results may not redefine the canonical substrate contract without the scratch sufficiency gate.
- Failed arms remain archived; no silent promotion/deletion.
