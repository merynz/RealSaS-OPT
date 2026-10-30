# IRIS/TESSA V3.2 pre-Truth139 architecture freeze

Date: 2026-10-30
Status: FROZEN_PRETRUTH_ARCHITECTURE_CANDIDATE_V3_2
Scope: architecture/training-boundary freeze for the first Truth139 model run

## Change discipline

V3.2 is an amendment to the frozen V3.1 lineage, not a rewrite.

**Invariant:** every V3.1 responsibility, learned head, evidence family, authority
boundary and training input remains unchanged unless this document explicitly
amends it. Later V3.1 corrective evidence (for example the fixed-df Student-t
response likelihood) is preserved at its latest qualified revision and must not
silently regress to an older failed formulation.

Removing, renaming, collapsing or demoting an unmentioned head is itself an
architecture change and is forbidden without an explicit decision.

V3.1 remains the historical baseline/rollback. The first Truth139 training run
uses this single V3.2 candidate rather than spending the corpus on architecture
selection A/B arms.

## Explicit V3.2 changes only

Only these boundaries change:

1. **Exact 3D geometry authority.** Geometry supervision moves from legacy
   F*-only signed-field authority to qualified watertight 3D truth: truncated
   metric signed distance, inside/outside occupancy/sign and oriented normals.
   Legacy F* remains an auxiliary source-consistency signal.
2. **Raw/local RGB observation evidence.** V3.1 already consumed DINO final,
   projected DINO layer-2 and raw alpha, but did not expose raw/local RGB to the
   learned scene. V3.2 adds patch-aligned RGB evidence to observation fusion.
   The frozen first implementation form is `FINAL384 + L2_64 + ALPHA64 + RGB64`.
3. **Geometry-first TESSA dependency without a scene bottleneck.** TESSA retains
   direct access to the same Dense3D scene memory `Z`, but topology is not
   semantically independent of geometry. TESSA runs from `Z` plus qualified
   Geometry-derived surface evidence after the geometry surface is available.
   A reduced 17-D/GSA handoff cannot replace `Z`.
4. **Mechanical topology before compactness.** TESSA compactness and TESSA
   mechanical suitability are distinct objectives. Mechanical admissibility and
   motion quality precede triangle-count optimization.

Everything else remains the latest V3.1 contract unless separately amended.

## Product/input boundary

The current eight-view camera/input apparatus is a research and training
contract, not a permanent product UX assumption.

The learned scene representation and downstream heads must remain compatible
with future observation normalization that can reduce product input to two views
and ultimately one view while internally constructing the evidence required by
the same downstream model family.

No head may define its semantics as "the user always supplies eight views".
Eight-view remains the current high-information research apparatus and regression
reference.

## Observation contract

V3.1 foundation evidence is preserved:

- frozen DINOv2-S/14 final patch features (384-D),
- projected DINO layer-2 patch features (64-D),
- raw alpha patch evidence projected to 64-D,
- camera-aware multi-view fusion,
- typed coordinate conversion.

V3.2 adds one explicit observation family:

- raw/local RGB patches aligned to the DINO S/14 patch grid, projected to 64-D.

For a 14x14 RGB patch, the direct local source is 14x14x3 = 588 scalar values
before the learned RGB projection. The fused per-patch observation token is:

`[DINO_FINAL384, DINO_L2_64, RAW_ALPHA64, RAW_RGB64]`

for a total width of 576 before the Dense3D observation projection.

DINO features remain important semantic/structural evidence, but DINO(RGB) is
not treated as a substitute for source photometry. Raw RGB is required because
Graphics is a product head and exact/local color and texture evidence must be
available to the learned scene rather than reconstructed only from semantic
features.

## Shared scene and head map

`Z` is the persistent shared Dense3D scene memory. It is the common scene state,
not a mesh and not a second discrete authority.

```text
RGB(A) observations + cameras
          |
          +--> DINO final384
          +--> DINO layer2 -> 64
          +--> raw alpha -> 64
          +--> raw/local RGB -> 64
          |
          v
+-----------------------------------------+
|              Dense3D Z                  |
|       persistent shared scene memory    |
+-----------------------------------------+
       |          |          |          |
       |          |          |          |
       v          v          v          v
   Geometry    Graphics   Response    Relations
    decoder     decoder      head        head
       |          |          |          |
       |          |          |          |
 TSDF/sign      RGBA/     non-rigid   typed pair/
 occupancy      boundary   response    relation
 normals        graphics   + energy    evidence
 bounded delta             uncertainty   |
       |          |          |          |
       v          |          |          |
 qualified 3D    |          |          |
 surface G       |          |          |
       |          |          |          |
       +----------+----------+----------+
       |
       | mandatory: G + direct access to Z
       v
+-----------------------------------------+
|              TESSA                      |
| topology proposal / structuralizer      |
| reads Z + Geometry surface G + admitted |
| mechanical/presentation evidence        |
+-----------------------------------------+
          |
          v
      Compiler
          |
          v
   canonical puppet M
```

This diagram is dependency-aware. Geometry, Graphics, Response and Relations are
continuous evidence consumers of the shared scene. TESSA is downstream of the
qualified Geometry surface while still retaining direct scene access.

## Geometry -> TESSA dependency

The semantic order is:

`observations -> Z -> Geometry field -> qualified surface G -> TESSA -> Compiler M`

Geometry defines **where the surface is**. TESSA decides **how that surface is
discretized/connectively represented** for the canonical mesh proposal.

Therefore TESSA must not mint a final topology before a qualified Geometry
surface exists.

At the same time, the following historical bottleneck is forbidden:

`Z -> reduced 17-D/GSA node state -> second scene reconstruction -> TESSA`

TESSA receives both:

1. direct bounded access/query evidence from the same Dense3D `Z`, and
2. typed Geometry-derived surface evidence `G` (surface xyz, oriented normal,
   signed/occupancy evidence, local geometry/query evidence and validity).

The surface stream is an addressable Geometry result, not a replacement for the
scene stream. Direct `Z` access prevents the IRIS->TESSA handoff from discarding
scene evidence; Geometry-first ordering prevents topology from becoming detached
from the actual reconstructed surface.

## IRIS Geometry

Exact qualified watertight truth geometry is the supervision authority.

Required targets:

- truncated metric signed distance / TSDF in the qualified 3D domain,
- exact inside/outside occupancy/sign from qualified watertight 3D truth,
- oriented surface normals in a qualified near-surface band,
- the existing bounded local delta where still useful and contract-compatible.

Legacy eight-view F* is retained only as an auxiliary source-consistency signal.
It cannot override exact 3D sign, occupancy or normal truth.

Initial objective family:

`L_G = L_tsdf + lambda_occ L_occ + lambda_n L_normal + lambda_delta L_delta + lambda_src L_Fstar_aux`

The TSDF sign and occupancy head are independently measurable so a low scalar
field MAE cannot hide a sign/topology failure. Normal supervision is explicit;
it is not inferred only through rendered silhouette quality.

Gradient-to-normal/eikonal-style regularization is not silently enabled by this
freeze. It may be added later only as an explicit measured amendment.

The output remains a watertight-capable 3D mechanical substrate. Presentation
being 2D does not reduce the geometry contract to 2D.

## IRIS Graphics

`GRAPHICS` is a first-class learned product responsibility.

It is not:

- an auxiliary feature that exists only to help Geometry,
- an auxiliary feature that exists only to help Relations,
- reducible to visibility/occlusion metadata,
- synonymous with CAA.

Its primary responsibility is to infer/preserve drawable surface appearance from
the available observations so the compiled puppet can be rendered correctly
across view and motion. Raw/local RGB is therefore part of the learned
observation contract in V3.2.

Preserved graphics outputs/responsibilities from the V3.1 lineage include the
learned RGBA/boundary path and source/raster presentation evidence. Truth139 may
bind richer typed graphics truth (surface-addressed RGBA, support/provenance,
visibility/occlusion, K-hit and related evidence) without redefining Graphics as
an auxiliary head.

Observed source appearance remains provenance-preserving. Learned completion of
unobserved appearance is a Graphics responsibility and must not be mislabeled as
directly observed source authority.

### Graphics naming boundary

`GRAPHICS` is the top-level architecture responsibility.

Historical `CAA`-named code/contracts are legacy graphics subcomponents/policies,
not the architecture name for Graphics as a whole. This freeze does not silently
rename or rewrite those implementations; naming migration is a separate change.

## Response / mechanics

The response/mechanics responsibility is preserved, including the correction
learned from the completed V3.1 Snake evidence run.

The first Truth139 V3.2 run must not silently return to the failed Gaussian
mechanics likelihood. The preserved response objective is the latest qualified
fixed-df Student-t treatment over the existing nonnegative/log1p response target
coordinates, with no subject-specific weighting and no deletion/clipping of
truth rows.

Response remains continuous evidence, not a semantic mechanical-class taxonomy.
Its outputs/evidence may condition TESSA and Compiler qualification, but Response
does not mint topology or canonical mesh authority.

## Relations

Existing relation/evidence responsibilities and gradient-ownership rules are
preserved unless a separate amendment explicitly changes them.

Relations may read useful Graphics/Geometry evidence, but neither Graphics nor
Geometry exists merely to serve Relations. Relation-private training must not
silently rewrite protected continuous state through a bypass of the established
gradient firewall.

Truth139 drawable-affinity / primitive-relation evidence can extend the admitted
relation targets without introducing global cross-subject semantic owner IDs.

## TESSA topology

TESSA remains proposal-only. Compiler remains the sole canonical mesh authority.

### Mandatory inputs

TESSA requires:

- direct scene/query access to Dense3D `Z`,
- a qualified Geometry-derived surface `G`,
- typed validity for surface evidence,
- admitted mechanical/deformation evidence where available,
- preserved presentation/graphics evidence where contractually valid.

TESSA cannot use teacher topology at inference.

### Static / representation objective

- surface fidelity,
- manifold/topological validity,
- triangle quality,
- representation efficiency,
- compactness / triangle and vertex count.

### Mechanical objective

- triangle inversion under deformation,
- crack/hole exposure under motion,
- self-intersection / fold-through instability,
- strain/Jacobian tail instability,
- motion silhouette drift attributable to connectivity,
- attachment/deformation stability,
- local connectivity suitability under admitted rig/skin/motion evidence.

A low-triangle mesh is not mechanically valid merely because it reconstructs the
rest surface well.

Selection/qualification order is lexicographic:

1. mechanical admissibility,
2. motion quality envelope,
3. static geometry/source fidelity,
4. compactness / representation cost.

Triangle count is a late optimization objective/tie-breaker, not a proxy for
animation suitability.

## Training order for the first Truth139 run

The model family is one V3.2 candidate, but training is staged to respect causal
and semantic dependencies.

### Phase A — IRIS scene/geometry foundation

Train observation fusion, Dense3D scene state, Geometry and the protected
continuous query paths against exact 3D geometry truth. Existing geometry warmup
semantics are retained, but the target changes from F*-authority to exact 3D
TSDF/sign/normal authority with F* auxiliary.

### Phase B — IRIS evidence completion

Continue with Graphics, Response and admitted Relations/evidence objectives while
protecting Geometry from negative transfer according to the established private
path/gradient-ownership rules. Raw RGB participates from the observation input;
Graphics is trained as a direct product responsibility.

The fixed-df Student-t response likelihood is used; the older exploding Gaussian
mechanics objective is not restored.

### Phase C — Geometry surface freeze / qualification

Before TESSA topology training is enabled, mint a versioned IRIS checkpoint whose
Geometry surface passes the preregistered exact-3D surface gates required by the
Truth139 trainer (sign/occupancy, surface displacement, normals, watertightness
and source consistency where applicable).

This is a training dependency gate, not a separate architecture A/B experiment.

### Phase D — TESSA structuralization

Train TESSA from the qualified/frozen IRIS scene and Geometry surface. TESSA reads
both direct `Z` scene evidence and `G` surface evidence; the topology objective is
not allowed to move the already-qualified Geometry surface through an accidental
backpropagation path in the first Truth139 run.

TESSA then emits a topology proposal. Compiler qualification/repair remains the
only path to canonical `M`.

This staged order resolves the apparent contradiction between shared-scene
sibling access and geometry-first topology semantics.

## Preserved responsibilities

Unless explicitly amended above, V3.2 preserves the latest V3.1 lineage for:

- DINOv2-S/14 foundation identity,
- camera-aware multi-view fusion and typed coordinate conversion,
- Dense3D persistent scene carrier and query machinery,
- protected/private continuous paths,
- Graphics RGBA/boundary learning,
- Response/mechanics evidence and its latest qualified Student-t likelihood,
- Relations/evidence heads and gradient-ownership firewall,
- source fidelity and raster/presentation evidence,
- TESSA proposal-only authority,
- Compiler as sole canonical discrete mesh/rig/skin/graphics authority,
- fail-closed evidence admission and provenance,
- Knight as regression lineage rather than Truth139 training authority.

The new raw RGB path is the only observation-family addition. No unmentioned
responsibility is removed by omission from a diagram.

## Truth139 requirements before training authorization

Architecture can be frozen before all target bytes exist, but training must not
start until the Platform-bound training manifest can bind the required target
families.

Required for the V3.2 first run:

- exact qualified watertight 3D geometry suitable for TSDF/sign/occupancy queries,
- oriented surface-normal truth,
- source/local RGB(A) observation evidence and surface graphics truth,
- existing response/mechanics truth with validity,
- admitted relation evidence used by the selected trainer,
- exact topology/mesh truth and the mechanical/deformation evidence required to
  qualify TESSA proposals,
- lineage-safe subject/family-disjoint training/validation/unseen split,
- Knight kept outside Truth139 as an independent regression lineage.

A missing truth family is a data-readiness blocker, not permission to silently
fall back to the old V3.1 target or remove a V3.2 head.

## Platform/training authority

Once required Truth139 targets are admitted, the first 139-subject training run
uses this single V3.2 candidate through Platform authority.

Platform owns:

- Experiment/Attempt identity,
- EngineRelease/source binding,
- immutable Artifact Registry/CAS provenance,
- training/evaluation telemetry,
- staged gate state and failure attribution,
- reproducibility and final evidence binding.

Platform is not used here to choose among legacy architecture arms before the
first 139 training run.

V3.1 remains historical comparison/rollback evidence after the fact.

## Frozen non-claims

This architecture freeze:

- does not itself create the Truth139 split,
- does not admit unfinished Truth139 target families,
- does not alter a running materialization worker,
- does not mint trained weights, a ModelRelease or ProductRevision,
- does not claim FIT139/LOFO/unseen success,
- does not rename legacy CAA implementation code in place,
- does not allow TESSA to bypass Geometry surface qualification,
- does not erase any V3.1 responsibility except where explicitly amended above.

When the required Truth139 target families and split are sealed, the next model
step is one Platform-bound V3.2 training notebook implementing Phases A-D above.
