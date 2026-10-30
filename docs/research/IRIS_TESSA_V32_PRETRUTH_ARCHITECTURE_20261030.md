# IRIS/TESSA V3.2 — frozen pre-Truth139 architecture

Date: 2026-10-30  
Status: `FROZEN_PRETRUTH_ARCHITECTURE_V3_2__TRAINING_BLOCKED_ON_TARGET_READINESS`  
Scope: first Truth139 model architecture and staged training boundary

## 0. Change discipline

V3.2 is a delta over the latest qualified V3.1 lineage, not a rewrite.

**Invariant:** every V3.1 responsibility, learned head, evidence family, authority boundary and proven correction remains unchanged unless this document explicitly changes it. Omission from a diagram is not permission to delete or demote a head.

V3.1 remains historical rollback/comparison evidence. The first Truth139 model run uses one V3.2 candidate; no architecture-selection A/B is required before that run.

## 1. Explicit V3.2 changes

Only five architecture boundaries change:

1. **Geometry supervision** — exact source-surface distance/orientation becomes geometry authority; global sign/occupancy is exact only where a sign-validity court passes; normals are explicitly supervised; F* becomes source-consistency auxiliary only.
2. **Observation fusion** — add patch-aligned raw/local RGB evidence. V3.1 consumed DINO final + projected DINO L2 + raw alpha + cameras but no direct RGB patch values. V3.2 token form is `FINAL384 + L2_64 + ALPHA64 + RGB64`.
3. **Geometry-first TESSA** — topology is downstream of a qualified Geometry surface `G`, but TESSA retains direct bounded access to the same Dense3D scene memory `Z`. Reduced 17-D/GSA state cannot replace `Z`.
4. **Mechanical topology before compactness** — mechanically admissible topology is selected before triangle-count/representation cost optimization.
5. **Final Compiler mesh** — canonical product `M` is `WATERTIGHT_REQUIRED`; this output requirement does not imply that every raw source mesh has a valid global SDF.

Everything else remains the latest V3.1 contract.

## 2. Product/input boundary

Eight views are the current high-information research/training apparatus, not a permanent product UX assumption.

The downstream architecture must remain compatible with:

`1-view or 2-view user input -> observation normalization/evidence construction -> same V3.2 downstream family`.

No head may define its semantics as “the user always supplies eight views.”

## 3. Observation contract

Preserved V3.1 inputs:

- frozen DINOv2-S/14 final patch features: 384-D,
- DINO layer-2 projected to 64-D,
- raw alpha patch evidence projected to 64-D,
- camera-aware multi-view fusion and typed coordinates.

New V3.2 input:

- raw/local RGB patches aligned to the DINO S/14 grid, projected to 64-D.

For each 14x14 RGB patch the direct source contains 588 RGB scalars before projection. The fused patch token is:

`[DINO_FINAL384, DINO_L2_64, RAW_ALPHA64, RAW_RGB64]`

for width 576 before the Dense3D observation projection.

DINO(RGB) is semantic/structural evidence; it is not a substitute for source photometry.

## 4. Shared-scene responsibility graph

`Z` is persistent Dense3D shared scene memory. It is evidence state, not a mesh and not a second discrete authority.

```text
RGB(A) observations + cameras
          |
          +-- DINO final384
          +-- DINO L2 -> 64
          +-- raw alpha -> 64
          +-- raw RGB -> 64
          |
          v
+--------------------------------------+
|              Dense3D Z               |
|       persistent shared scene        |
+--------------------------------------+
       |          |          |         |
       v          v          v         v
   Geometry    Graphics   Response   Relations
       |          |          |         |
 surface field   RGBA/     non-rigid   typed
 + normals       boundary   response   evidence
 + validity      graphics   + energy
       |
       v
 qualified Geometry surface/evidence G
       |
       | mandatory G + direct Z access
       v
+--------------------------------------+
|                TESSA                 |
| topology proposal / structuralizer   |
| Z + G + admitted mechanical/present. |
+--------------------------------------+
       |
       v
    Compiler
       |
       v
 canonical watertight-required puppet M
```

Geometry, Graphics, Response and Relations remain first-class learned/evidence responsibilities. TESSA is downstream of Geometry semantically while still sharing the full scene memory.

## 5. Geometry contract

### 5.1 Exact surface authority

Truth139 raw source meshes are exact **surface** evidence even when open or non-manifold. V3.2 therefore separates universally measurable surface truth from global sign truth.

Always usable after exact-source geometry admission/rebinding:

- exact surface points and triangle+barycentric addresses,
- truncated unsigned distance to the exact source surface,
- oriented source-surface normals where orientation is qualified,
- connected-component/local topology evidence,
- local two-sided normal-offset probes.

### 5.2 Global sign/occupancy validity

Global inside/outside truth is admitted only for a component/subject that passes, at minimum:

- zero boundary edges for the signed component,
- manifold edge incidence,
- finite nondegenerate geometry,
- consistent orientation/winding,
- independent parity or winding agreement.

For sign-qualified geometry:

- exact inside/outside occupancy is supervised,
- exact signed truncated distance/TSDF is supervised.

For open/non-manifold/sign-ambiguous geometry:

- global sign/occupancy target is `UNKNOWN/ABSTAIN`,
- unsigned distance + normals + local surface evidence remain trainable,
- F* may enforce observation consistency but may not fabricate a 3D sign label.

### 5.3 Geometry outputs/objective

Geometry output family:

- truncated surface distance / signed TSDF where sign-valid,
- occupancy/sign prediction with explicit target validity,
- oriented normal prediction in the qualified near-surface band,
- existing bounded local delta where contract-compatible.

Objective family:

`L_G = L_surface_distance + lambda_sign L_sign(valid_only) + lambda_occ L_occ(valid_only) + lambda_n L_normal + lambda_delta L_delta + lambda_src L_Fstar_aux`

Sign/occupancy and normal metrics remain independent; low scalar field MAE alone cannot hide sign failure. Gradient-to-normal/eikonal regularization is not silently introduced by V3.2.

### 5.4 Watertight output

`Compiler canonical M -> WATERTIGHT_REQUIRED`.

This is stronger than the raw corpus condition. An open source mesh is not relabeled as fake watertight truth. Learned field, TESSA proposal and Compiler closure must instead produce a qualified final substrate while remaining bounded by exact surface, source-view, component and mechanical evidence.

Any future derived closure teacher requires its own displacement/source-view/component courts.

## 6. Graphics

`GRAPHICS` is a first-class learned product head.

It is not:

- auxiliary only to Geometry,
- auxiliary only to Relations,
- reducible to visibility/occlusion metadata,
- synonymous with CAA.

Its primary responsibility is drawable surface appearance across view and motion. Raw RGB is therefore part of V3.2 observation input.

Preserved V3.1 outputs include the learned RGBA/boundary path. Truth139 may bind richer typed graphics truth: surface-addressed RGBA, provenance/support, visibility/occlusion, K-hit and drawable-affinity evidence.

Observed source appearance remains provenance-preserving; learned unobserved completion is never relabeled as direct source authority.

`CAA` remains only a legacy Graphics subcomponent/policy name. This freeze does not silently rename existing code.

## 7. Response / mechanics

Response remains continuous evidence, not a semantic mechanical-class taxonomy.

The latest V3.1 correction is preserved. V3.2 must not return to the failed Gaussian mechanics likelihood. Use the latest qualified fixed-scale/fixed-df Student-t location-only treatment over existing nonnegative/log1p response targets, with no subject-specific weighting and no truth-row clipping/deletion.

Response may condition TESSA and Compiler qualification; it does not mint topology or canonical mesh authority.

## 8. Relations

Existing relation/evidence responsibilities and gradient-ownership/firewall rules are preserved.

Truth139 drawable-affinity / primitive-relation evidence may extend admitted relation targets without global cross-subject semantic owner IDs.

Graphics and Geometry may help Relations, but neither exists merely to serve it.

## 9. Geometry -> TESSA dependency

Semantic order:

`observations -> Z -> Geometry -> qualified G -> TESSA -> Compiler M`.

Geometry defines where the supported 3D surface/evidence domain is. TESSA decides how that domain should be discretized/connectively represented.

TESSA must not produce canonical topology before qualified `G` exists.

At the same time, this bottleneck is forbidden:

`Z -> reduced 17-D/GSA summary -> second scene reconstruction -> TESSA`.

TESSA receives both:

1. direct bounded scene/query access to Dense3D `Z`, and
2. Geometry-derived surface evidence `G`: xyz, normal, distance/sign validity, local query state/evidence and validity.

For the first Truth139 run, TESSA conditioning is detached from the qualified IRIS checkpoint; TESSA loss may not move already-qualified Geometry.

## 10. TESSA compact + mechanical teacher policy

The existing TESSA differentiable objective deliberately uses grammar/token/coordinate teacher forcing while decoded-mesh quality is tested independently. V3.2 preserves that separation rather than injecting an unqualified differentiable motion loss through the autoregressive decoder.

Mechanical suitability and compactness are encoded first in **teacher topology construction**, then independently rechecked on free-running output.

### 10.1 Teacher construction

For each TRAIN subject:

1. start from exact admitted rest mesh + rig/skin + admitted motion/deformation evidence,
2. build a deterministic bounded simplification ladder including exact source topology as fallback,
3. preserve/transfer component, skin/weight and attachment bindings through explicit typed mappings,
4. reject static geometry/source failures,
5. reject mechanical failures: inversion, motion-induced cracks/unsupported holes, self-intersection/fold-through, strain/Jacobian tails, attachment separation, and motion-view silhouette drift attributable to connectivity,
6. among all static+mechanical PASS candidates select the lowest representation cost,
7. if no simplified candidate passes, use exact source topology fallback.

Thus the target is **mechanically admissible first, compact second**.

### 10.2 TESSA training / free-running qualification

TESSA trains with the stable token/coordinate objective against the selected teacher topology. Teacher topology is target-only and never student conditioning.

Free-running generated topology must independently pass the relevant static and mechanical courts. Teacher-forced loss alone is never a topology PASS.

The historical `PASS_FULL_SOURCE_TOPOLOGY_SEQUENCE` teacher is only a fallback member of the V3.2 teacher ladder, not the compact target policy by itself.

## 11. First Truth139 training chronology

One V3.2 architecture, staged execution:

### Phase A — IRIS scene/geometry foundation

Train observation fusion, Dense3D `Z`, protected query paths and Geometry against surface-distance/normal truth plus validity-masked exact sign/occupancy and F* auxiliary.

### Phase B — IRIS evidence completion

Train/continue Graphics, Response and admitted Relations while protecting Geometry from negative transfer using established private-path and gradient-ownership rules. Raw RGB is live in the observation path. Preserve the Student-t Response correction.

### Phase C — Geometry qualification/freeze

Seal an IRIS checkpoint whose Geometry passes required surface, normal, sign-validity-aware, source-view and output-preparation courts. TESSA is not enabled before this gate.

### Phase D — TESSA structuralization

Build detached TESSA conditioning from frozen IRIS `Z + G` plus admitted response/presentation evidence. Train against the mechanically-qualified compact teacher topology.

### Phase E — Compiler consequence qualification

Free-running TESSA proposal -> Compiler qualification/repair -> canonical `M`. Final `M` must pass watertight, static and mechanical consequence courts.

## 12. Existing Truth139 V1 implementation compatibility

PR #292 is useful scaffolding but is **not** the final V3.2 trainer unchanged.

Reusable:

- Dense3D carrier and protected query/private paths,
- staged IRIS foundation then evidence training,
- fixed-scale Student-t Response correction,
- direct dense scene + predicted surface TESSA bridge concept,
- detached IRIS conditioning for TESSA,
- teacher topology physically separated from student conditioning,
- free-running decoded-mesh court separated from teacher-forced loss,
- leakage-connected-component TRAIN/UNSEEN split.

Required upgrades before V3.2 training:

- feature cache/input adds raw RGB: `final/layer2/alpha/rgb`,
- F*-primary geometry supervision becomes exact surface + validity-masked sign/occupancy + normals + F*-auxiliary,
- IRIS geometry output schema exposes V3.2 distance/sign-validity/occupancy/normal outputs,
- TESSA surface-source schema binds V3.2 Geometry evidence,
- exact-full-source topology teacher becomes the mechanically-qualified simplification-ladder teacher,
- current mechanics-abstain Truth139 shape bridge becomes mechanically qualified,
- Compiler consequence court enforces final watertight-required output.

These are implementation/data-readiness tasks under a frozen architecture, not reasons to reopen architecture selection.

## 13. Truth139 requirements before training authorization

Required target/data families:

- exact source surface geometry/topology facts and validity,
- distance-to-surface truth,
- sign/occupancy-validity court and exact sign labels where valid,
- oriented surface-normal truth,
- source/local RGB(A) observation evidence and Graphics truth,
- Response/mechanics truth with validity,
- selected admitted relation truth,
- rig/skin/motion/deformation evidence for topology-teacher mechanical qualification,
- mechanically-qualified compact TESSA teacher manifests,
- **lineage-safe leakage-connected-component TRAIN/UNSEEN split with fixed TRAIN-internal validation rows**,
- Knight kept outside Truth139 as independent regression lineage.

A missing truth family is a data-readiness blocker. It does not authorize fallback to F*-authority, the failed Gaussian mechanics objective, source-only full topology as the compact target, or removal of a V3.2 head.

## 14. Platform authority

When targets and split are sealed, execute one V3.2 candidate through:

`Experiment -> Attempt -> EngineRelease -> Artifact Registry/CAS`.

Platform owns source/data/checkpoint identity, staged gate state, telemetry, failure attribution and reproducibility. It is not an architecture arm-selection mechanism here.

## 15. Frozen non-claims

This freeze does not itself:

- create the Truth139 split,
- admit unfinished target bytes,
- alter a currently running materialization worker,
- mint trained weights, ModelRelease or ProductRevision,
- claim FIT139/LOFO/unseen success,
- claim every source mesh is watertight,
- fabricate global sign for open/non-manifold truth,
- allow TESSA to bypass Geometry qualification,
- rename legacy CAA code in place,
- erase any unmentioned V3.1 responsibility.

When required Truth139 targets and the split are sealed, the next model step is one Platform-bound V3.2 Run-All training notebook implementing Phases A-E.