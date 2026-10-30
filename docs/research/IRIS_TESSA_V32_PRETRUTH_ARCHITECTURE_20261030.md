# IRIS/TESSA V3.2 — frozen pre-Truth139 architecture

Date: 2026-10-30
Status: `FROZEN_PRETRUTH_ARCHITECTURE_V3_2__TRAINING_BLOCKED_ON_TARGET_READINESS`
Scope: first Truth139 model architecture and staged training boundary

## 0. Change discipline

V3.2 is a delta over the latest qualified V3.1 lineage, not a rewrite.

**Invariant:** every V3.1 responsibility, learned head, evidence family, authority
boundary and proven correction remains unchanged unless this document explicitly
changes it. Omission from a diagram is not permission to delete a head.

V3.1 remains historical rollback/comparison evidence. The first Truth139 model run
uses one V3.2 candidate; it is not preceded by architecture-selection A/B arms.

## 1. V3.1 -> V3.2: only the explicit changes

1. **3D geometry supervision is upgraded.**
   - exact source-surface distance/orientation replaces F* as geometry authority;
   - exact global sign/occupancy is used only where the source component/subject
     passes a sign-validity court;
   - oriented surface normals are explicit supervision;
   - F* is retained only as observation/source-consistency auxiliary.
2. **Raw/local RGB becomes an observation input.**
   - V3.1 saw DINO features + raw alpha + cameras, but no direct RGB patch values;
   - V3.2 fusion is `FINAL384 + L2_64 + ALPHA64 + RGB64`.
3. **TESSA is geometry-after but not scene-bottlenecked.**
   - TESSA runs only after a qualified Geometry surface exists;
   - TESSA simultaneously retains direct bounded access to the same Dense3D `Z`;
   - a reduced 17-D/GSA handoff cannot replace the shared scene.
4. **TESSA compactness is subordinate to mechanical validity.**
   - mechanically suitable compact topology is the teacher/qualification goal;
   - triangle count cannot rescue a mechanically bad mesh.
5. **Final product mesh is watertight-required.**
   - this is an output/Compiler requirement;
   - it does not authorize pretending every raw corpus mesh has a valid global SDF.

Everything else remains the latest V3.1 contract.

## 2. Product/input boundary

Eight views remain the current research/training apparatus and high-information
regression setup. They are not a permanent product UX assumption.

The downstream architecture must remain compatible with future normalization:

`1-view or 2-view user input -> observation normalization/evidence construction -> same V3.2 downstream family`.

No head may define its semantics as "the user always supplies eight views".

## 3. Observation contract

Preserved V3.1 inputs:

- frozen DINOv2-S/14 final patch features: 384-D,
- DINO layer-2 projected to 64-D,
- raw alpha patch evidence projected to 64-D,
- camera-aware multi-view fusion and typed coordinates.

New V3.2 input:

- raw/local RGB patches aligned to the DINO S/14 grid, projected to 64-D.

For each 14x14 RGB patch the direct source contains 588 scalar RGB values before
projection. The fused patch token is:

`[DINO_FINAL384, DINO_L2_64, RAW_ALPHA64, RAW_RGB64]`

for width 576 before the Dense3D observation projection.

DINO(RGB) is semantic/structural evidence; it is not treated as a substitute for
source photometry.

## 4. Shared scene and responsibility graph

`Z` is persistent Dense3D scene memory. It is shared evidence state, not a mesh
and not a second discrete authority.

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

Graphics, Response and Relations remain first-class responsibilities. They are
not hidden inside TESSA and are not removed by the geometry-first dependency.

## 5. Geometry contract

### 5.1 Exact surface authority

The raw Truth139 source mesh is exact **surface** evidence even when it is open or
non-manifold. Therefore V3.2 separates what is always measurable from what needs
a stronger validity court.

Always-valid after the existing exact-source geometry rebinding court:

- exact source surface points / triangle+barycentric addresses,
- truncated **unsigned** distance to the exact source surface,
- oriented source-surface normals where orientation is qualified,
- connected-component and local topology evidence,
- local two-sided normal-offset probes.

### 5.2 Global sign/occupancy authority

Global inside/outside truth is admitted only for a component/subject that passes a
sign-validity court, at minimum:

- closed/watertight boundary (`boundary_edge_count == 0`),
- manifold edge incidence,
- finite nondegenerate geometry,
- consistent orientation/winding sufficient for an independent parity/winding
  court to agree.

For sign-qualified geometry:

- exact inside/outside occupancy is supervised,
- exact signed truncated distance/TSDF is supervised.

For open/non-manifold/sign-ambiguous source geometry:

- global sign/occupancy target is `UNKNOWN/ABSTAIN`,
- unsigned distance + normals + local surface evidence remain trainable,
- F* may contribute source consistency but may **not** fabricate a 3D sign label.

This preserves scientific truth while still training a sign/occupancy head from
all genuinely qualified subjects/components.

### 5.3 Geometry outputs/objective

Geometry output family:

- truncated surface distance / signed TSDF where sign-valid,
- occupancy/sign prediction with explicit target validity,
- oriented normal prediction in the qualified surface band,
- existing bounded local delta where contract-compatible.

Objective family:

`L_G = L_surface_distance + lambda_sign L_sign(valid_only) + lambda_occ L_occ(valid_only) + lambda_n L_normal + lambda_delta L_delta + lambda_src L_Fstar_aux`

Low field MAE alone cannot hide sign failure; sign/occupancy and normal metrics
remain separate.

Gradient-to-normal/eikonal regularization is **not** silently added by V3.2.

### 5.4 Watertight output requirement

The product requirement is stronger than the raw corpus condition:

`Compiler canonical M -> WATERTIGHT_REQUIRED`.

A raw open source mesh does not become fake watertight truth. Instead the learned
field/TESSA proposal/Compiler closure must produce a qualified watertight final
mechanical substrate while remaining bounded by exact source-surface, source-view,
component and mechanical evidence.

Any future learned/deterministic closure teacher is a separate derived truth with
its own displacement/source-view/component courts; it may not be silently minted
from raw winding.

## 6. Graphics

`GRAPHICS` is a first-class learned product head.

It is not:

- auxiliary only to Geometry,
- auxiliary only to Relations,
- reducible to visibility/occlusion metadata,
- synonymous with CAA.

Its primary responsibility is drawable surface appearance across view/motion.
Raw RGB is therefore part of the V3.2 observation contract.

Preserved learned outputs include the V3.1 RGBA/boundary path. Truth139 may bind
richer typed graphics evidence (surface-addressed RGBA, provenance/support,
visibility/occlusion, K-hit, drawable affinity) without demoting Graphics to an
auxiliary head.

Observed source appearance remains provenance-preserving; learned unobserved fill
must never be mislabeled as direct source observation.

`CAA` is only a legacy graphics subcomponent/policy name. It is not the top-level
Graphics architecture name. No silent code rename is performed by this freeze.

## 7. Response / mechanics

Response remains continuous evidence, not a semantic mechanical-class taxonomy.

The latest V3.1 correction is preserved: the first V3.2 run must not return to the
failed Gaussian mechanics likelihood. Use the latest qualified fixed-scale/fixed-df
Student-t response treatment on the existing nonnegative/log1p target coordinates,
with no subject-specific weighting and no deletion/clipping of truth rows.

Response may condition TESSA and Compiler qualification; it does not mint topology
or canonical mesh authority.

## 8. Relations

Existing relation/evidence responsibilities and gradient-ownership/firewall rules
are preserved.

Truth139 drawable-affinity / primitive-relation evidence may extend the admitted
relation targets without global cross-subject semantic owner IDs.

Graphics/Geometry may help Relations, but neither head exists merely to serve it.

## 9. Geometry -> TESSA dependency

Semantic order:

`observations -> Z -> Geometry -> qualified G -> TESSA -> Compiler M`.

Geometry defines the supported 3D surface/evidence domain. TESSA decides how that
domain should be discretized/connectively represented.

TESSA must not produce the canonical topology before qualified `G` exists.

At the same time this bottleneck is forbidden:

`Z -> reduced 17-D/GSA summary -> second scene reconstruction -> TESSA`.

TESSA receives both:

1. direct bounded scene/query access to Dense3D `Z`, and
2. Geometry-derived surface evidence `G` (xyz, normal, distance/sign validity,
   local query state/evidence and validity).

This matches the useful part of the existing Truth139 bridge: scene and surface
conditioning are detached/frozen before TESSA training, while TESSA sees dense
scene evidence rather than teacher topology.

## 10. TESSA compact + mechanical teacher policy

The existing TESSA differentiable objective intentionally uses token/coordinate
teacher forcing and leaves decoded mesh quality to independent courts. V3.2 keeps
that stable separation; it does **not** introduce an unqualified differentiable
motion loss through an autoregressive mesh decoder.

Therefore compactness and mechanical suitability are encoded first in the
**teacher topology construction**, then independently re-checked on free-running
TESSA output.

### 10.1 Qualified topology-teacher construction

For each TRAIN subject:

1. start from the exact admitted source rest mesh + rig/skin + admitted motion or
   deformation-response evidence;
2. construct a deterministic bounded simplification ladder, including the exact
   source topology as the no-simplification fallback;
3. for every candidate, preserve/transfer component, skin/weight and attachment
   bindings through an explicit typed mapping;
4. reject candidates that fail static geometry/source courts;
5. reject candidates that fail required mechanical courts (inversion, cracks,
   fold-through/self-intersection, strain/Jacobian tails, attachment separation,
   motion-view silhouette drift where attributable to connectivity);
6. among mechanically admissible candidates, select the smallest representation
   that passes all required static and mechanical gates;
7. if no simplified candidate passes, select the exact source topology fallback.

The selected teacher is therefore **mechanically admissible first, compact
second**. Triangle count never overrides a failure.

### 10.2 TESSA training and free-running court

TESSA trains with the existing grammar/token/coordinate objective against the
qualified teacher topology. Teacher topology is target-only and never student
conditioning.

Free-running generated topology must then pass the same relevant static and
mechanical qualification family. Teacher-forced token loss alone is never a
TESSA PASS.

The old `PASS_FULL_SOURCE_TOPOLOGY_SEQUENCE` teacher is acceptable only as the
fallback member of the ladder; it is not by itself the V3.2 compact teacher policy.

## 11. First Truth139 training chronology

One architecture candidate, staged training:

### Phase A — IRIS scene/geometry foundation

Train observation fusion, Dense3D `Z`, protected query paths and Geometry against
surface-distance/normal truth plus validity-masked exact sign/occupancy and F*
auxiliary.

### Phase B — IRIS evidence completion

Continue with Graphics, Response and admitted Relations while protecting qualified
Geometry from negative transfer using the established private-path and gradient
ownership rules. Raw RGB is live in the observation path.

Use the latest Student-t Response correction; do not restore the old Gaussian.

### Phase C — Geometry qualification/freeze

Before TESSA is enabled, seal an IRIS checkpoint whose Geometry passes the
required surface, normal, sign-validity-aware, source-view and output-preparation
courts. This is a dependency gate, not an architecture A/B.

### Phase D — TESSA structuralization

Build/cache detached TESSA conditioning from the frozen IRIS `Z` + Geometry `G`
plus admitted response/presentation evidence. Train TESSA against the mechanically
qualified compact teacher topology. TESSA loss must not move the already-qualified
IRIS Geometry in the first Truth139 run.

### Phase E — Compiler consequence qualification

Free-running TESSA proposal -> Compiler qualification/repair -> canonical `M`.
Final `M` must satisfy the watertight-required product mesh contract plus static
and mechanical consequence courts.

## 12. Current Truth139 V1 execution compatibility audit

The existing PR #292 V1 implementation is useful scaffolding but is **not** the
final V3.2 trainer unchanged.

Reusable concepts:

- Dense3D carrier and protected IRIS query/private-path design,
- staged geometry-foundation then evidence training,
- Student-t mechanics correction,
- Truth139 TESSA bridge semantics: direct dense scene source + predicted surface
  source, detached before TESSA,
- teacher topology kept physically separate from student conditioning,
- decoded-mesh courts independent from teacher-forced loss.

Required V3.2 upgrades before training:

- feature cache/input must add raw RGB (`final/layer2/alpha/rgb` rather than only
  `final/layer2/alpha`),
- current F*-primary `geometry_supervision_v1` must be replaced by V3.2 exact
  surface + validity-masked sign/occupancy + normals + F*-auxiliary supervision,
- current IRIS geometry head must expose distance/sign-validity/occupancy/normal
  outputs required by V3.2,
- current TESSA surface-source schema must bind the V3.2 Geometry evidence,
- current exact-full-source topology teacher must become the mechanically
  qualified simplification-ladder policy above,
- current Truth139 shape bridge that explicitly `mechanics_abstained=True` is not
  sufficient for V3.2 mechanical topology qualification,
- final Compiler consequence court must enforce the new watertight-required output
  contract.

These are implementation/data-readiness tasks under the frozen architecture; they
are not permission to reopen the architecture choices.

## 13. Truth139 requirements before training authorization

Required target/data families:

- exact source surface geometry + topology facts and validity,
- distance-to-surface truth,
- sign/occupancy-validity court and exact sign labels where valid,
- oriented surface-normal truth,
- source/local RGB(A) observation evidence and graphics truth,
- response/mechanics truth with validity,
- admitted relation evidence selected by the trainer,
- rig/skin/motion/deformation evidence needed for topology-teacher mechanical
  qualification,
- mechanically qualified compact TESSA teacher manifests,
- lineage-safe subject/family-disjoint TRAIN/VAL/UNSEEN split,
- Knight outside Truth139 as independent regression lineage.

A missing truth family is a data-readiness blocker. It does not authorize silent
fallback to F*-authority, old Gaussian mechanics, source-only full topology as the
compact target, or removal of a V3.2 head.

## 14. Platform authority

Once the target families and split are sealed, execute one V3.2 candidate through
Platform:

`Experiment -> Attempt -> EngineRelease -> Artifact Registry/CAS`.

Platform owns source/data/checkpoint identity, staged gate state, telemetry,
failure attribution and reproducibility. It is not an arm-selection mechanism for
this freeze.

## 15. Frozen non-claims

This freeze does not itself:

- create the Truth139 split,
- admit unfinished target bytes,
- alter a currently running materialization worker,
- mint trained weights, ModelRelease or ProductRevision,
- claim FIT139/LOFO/unseen success,
- claim every source mesh is watertight,
- fabricate sign for open/non-manifold truth,
- allow TESSA to bypass Geometry qualification,
- rename legacy CAA code in place,
- erase any unmentioned V3.1 responsibility.

When the required Truth139 targets and split are sealed, the next model step is a
single Platform-bound V3.2 training notebook implementing Phases A-E above.
