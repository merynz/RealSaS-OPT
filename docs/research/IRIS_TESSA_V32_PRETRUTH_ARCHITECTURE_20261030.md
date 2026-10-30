# IRIS/TESSA V3.2 / V4-pretruth architecture amendment

Date: 2026-10-30
Status: PRETRUTH_CANDIDATE__DIRECT_TRUTH139_TRAINING_ARCHITECTURE

## Change discipline

V3.2 is an amendment to the frozen V3.1 architecture, not a rewrite.

**Invariant:** every V3.1 responsibility, learned head, evidence family, authority
boundary and training input remains unchanged unless this document explicitly
amends it.

Removing, renaming, collapsing or demoting an unmentioned head is itself an
architecture change and is forbidden without an explicit decision.

V3.1 remains the historical baseline/rollback. The first Truth139 training run
should use this V3.2 candidate directly rather than spending the corpus on a set
of architecture-selection A/B arms.

## Explicit changes in V3.2

Only these boundaries change:

1. Geometry sign/normal authority moves from legacy F*-only supervision to exact
   qualified 3D truth: TSDF, inside/outside sign/occupancy and oriented normals.
   F* remains as an auxiliary source-consistency signal.
2. The shared-scene boundary is made explicit: Geometry, Graphics and TESSA are
   sibling consumers of the same Dense3D scene memory `Z`. A reduced 17-D/GSA
   handoff cannot replace `Z` for TESSA.
3. TESSA compactness and TESSA mechanical suitability are distinct objectives.
   Mechanical admissibility and motion quality precede compactness.

Everything else remains V3.1 unless separately amended later.

## Product/input boundary

The current eight-view camera/input apparatus is a research and training
contract, not a permanent product UX assumption.

The learned scene representation and decoder boundaries must remain compatible
with future observation normalization that can reduce product input to two views
and ultimately one view while internally constructing the evidence required by
the same downstream model family.

Therefore no head in this architecture may rely on "the user always supplies
eight views" as its semantic definition.

## Shared-scene architecture

`Z` is the persistent shared Dense3D scene memory.

Geometry, Graphics and TESSA are first-class sibling consumers of `Z`:

```text
observations + cameras
        |
        v
 observation / DINO / RGB-alpha fusion
        |
        v
+-----------------------------------+
|          shared Dense3D Z         |
|       persistent scene memory     |
+-----------------------------------+
        |            |            |
        |            |            |
        v            v            v
  IRIS Geometry   IRIS Graphics   TESSA Topology
    decoder          decoder          decoder
        |            |               |
 TSDF / sign      appearance /    connectivity /
 occupancy        surface graphics  mesh proposal
 normals          support/evidence  evidence
 bounded delta    visibility etc.       |
        |            |               |
        +------------+---------------+
                     |
                     v
              typed evidence R
                     |
                     v
                  Compiler
                     |
                     v
               canonical puppet M
```

Typed surface/query references may address locations in `Z`; they are addresses
and local evidence, not a replacement scene representation.

### Forbidden bottleneck

The historical pattern

`Z -> reduced 17-D/GSA node state -> second scene reconstruction -> TESSA`

is not the V3.2 information boundary.

TESSA must retain direct access to shared scene memory/query evidence rather than
being asked to reconstruct information already represented in `Z`.

## IRIS Geometry

Exact qualified watertight truth geometry becomes the supervision authority.

Targets:
- truncated metric signed distance / TSDF in the qualified 3D domain;
- exact inside/outside occupancy/sign from qualified watertight 3D truth;
- oriented surface normals in a qualified near-surface band;
- existing bounded local delta where still useful and contract-compatible.

Legacy eight-view F* is retained only as an auxiliary source-consistency signal.
It cannot override exact 3D sign, occupancy or normal truth.

Initial objective family:

`L_G = L_tsdf + lambda_occ L_occ + lambda_n L_normal + lambda_delta L_delta + lambda_src L_Fstar_aux`

Gradient-to-normal alignment may be added only after the exact target/domain
contract is measured and frozen. It is not silently enabled by this amendment.

The output remains a watertight-capable 3D geometry field suitable for extraction
of the mechanical substrate. Presentation being 2D does not reduce the geometry
contract to 2D.

## IRIS Graphics

Graphics/appearance remains a **first-class learned product head**.

It is not:
- an auxiliary feature that exists only to help geometry;
- an auxiliary feature that exists only to help relations;
- reducible to visibility/occlusion metadata;
- synonymous with CAA.

Its primary responsibility is to infer and preserve the character's drawable
surface graphics/appearance from the available observations so the compiled
puppet can be rendered correctly across views and motion.

The Graphics head may also contribute useful evidence to relations, visibility,
occlusion, topology qualification and presentation, but those cross-task benefits
are secondary to its own product responsibility.

Preserved V3.1/Truth139 graphics evidence families include, where admitted:
- source RGBA / source-addressed appearance evidence;
- surface-addressed graphics;
- support / provenance / confidence evidence;
- visibility and occlusion evidence;
- ordered ray / K-hit evidence;
- drawable-affinity / presentation-relevant evidence;
- graphics uncertainty where defined.

Observed source evidence remains provenance-preserving. Unobserved appearance
completion is a learned/compiled graphics responsibility; it must not be confused
with directly observed source authority.

### Naming boundary

`GRAPHICS` is the top-level architecture responsibility.

The historical `CAA` name is not the architecture name for Graphics and must not
be used as a synonym for the full head. Existing CAA-named code/contracts remain
legacy implementation subcomponents until a separate, explicit naming migration;
this amendment does not silently rewrite their behavior.

## TESSA Topology

TESSA remains proposal-only. Compiler remains the sole canonical mesh authority.

TESSA reads the same shared Dense3D `Z` as Geometry and Graphics, plus qualified
local/query evidence and admitted mechanical/deformation evidence where required.

Two objectives are distinct:

### Static / representation objective
- surface fidelity;
- manifold/topological validity;
- triangle quality;
- representation efficiency;
- compactness / triangle and vertex count.

### Mechanical objective
- triangle inversion under deformation;
- crack/hole exposure under motion;
- self-intersection / fold-through instability;
- strain/Jacobian tail instability;
- motion silhouette drift attributable to connectivity;
- attachment/deformation stability;
- local connectivity suitability under admitted rig/skin/motion evidence.

A low-triangle mesh is not mechanically valid merely because it reconstructs the
rest surface well.

Selection/qualification order is lexicographic:

1. mechanical admissibility;
2. motion quality envelope;
3. static geometry/source fidelity;
4. compactness / representation cost.

Triangle count is a late objective/tie-breaker, not a proxy for animation
suitability.

## Preserved responsibilities

Unless explicitly amended elsewhere, V3.2 preserves V3.1 responsibilities for:
- frozen observation/foundation features and local RGB/alpha evidence;
- camera-aware multi-view fusion and typed coordinate conversion;
- Dense3D persistent scene carrier and query machinery;
- Graphics/appearance learning;
- response/mechanics evidence and uncertainty treatment;
- relations/evidence heads and gradient-ownership rules;
- source fidelity and raster/presentation evidence;
- Compiler as sole canonical discrete authority;
- fail-closed evidence admission and provenance;
- Knight as regression lineage rather than Truth139 training authority.

No unmentioned responsibility is removed by omission from a diagram.

## Truth139 training intent

Once required Truth139 targets are materially admitted, the first 139-subject
training run should train this single V3.2 candidate through Platform authority.

Platform is used for:
- Experiment/Attempt identity;
- EngineRelease binding;
- immutable Artifact/CAS provenance;
- training/evaluation telemetry;
- failure attribution and reproducibility.

It is not being used here to choose among legacy architecture arms before the
first 139 training run.

V3.1 remains available as historical comparison/rollback evidence after the fact.

## Authority boundary

This amendment:
- does not create a train/validation/LOFO split;
- does not admit unfinished Truth139 target families by itself;
- does not alter the running Truth139 materialization worker;
- does not mint a ModelRelease or ProductRevision;
- does not promote V3.2 to product authority;
- does not rename or rewrite legacy CAA implementation code in place;
- does not erase any V3.1 head or responsibility except where explicitly amended above.

The next executable step after Truth139 readiness is a single Platform-bound V3.2
training/evaluation run with lineage-safe split and unseen evaluation defined
separately from this architecture amendment.
