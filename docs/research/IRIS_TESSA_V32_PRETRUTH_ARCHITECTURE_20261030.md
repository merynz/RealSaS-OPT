# IRIS/TESSA V3.2 / V4-pretruth architecture amendment

Date: 2026-10-30
Status: PRETRUTH_CANDIDATE__DO_NOT_PROMOTE_WITHOUT_TRUTH139_COURTS

## Purpose

Retain the proven V3.1 observation/shared-scene architecture while removing two
known conceptual weaknesses before Truth139 training:

1. Geometry sign/normal authority must come from exact qualified 3D truth, not
   only the legacy eight-view F* visual-hull surrogate.
2. TESSA compactness and TESSA mechanical suitability are separate objectives.

No current V3.1 checkpoint or training receipt is superseded by this amendment.

## Shared-scene boundary

The scene bottleneck is the Dense3D carrier `Z`. IRIS geometry and TESSA
topology are sibling decoders over the same scene, not a serial
IRIS-output -> TESSA-input pipeline.

```text
8-view observations + cameras
        |
        v
  shared Dense3D Z
     /         \
    /           \
IRIS geometry   TESSA topology
decoder         decoder
    |              |
TSDF/sign/      mesh proposal
normals         evidence
    \              /
     \            /
        Compiler
       canonical M
```

Forbidden handoff: reducing `Z` to the historical 17-D GSA node tensor and then
asking TESSA to reconstruct lost scene information.

Typed surface/query references may address locations in `Z`; they are addresses
and local evidence, not a replacement scene representation.

## IRIS geometry amendment

Exact qualified watertight truth mesh is geometry authority.

Targets:
- truncated metric signed distance / TSDF,
- exact inside/outside occupancy/sign,
- oriented surface normals in a qualified near-surface band,
- existing bounded local delta where still useful.

Legacy eight-view F* remains available only as source-consistency auxiliary.
It cannot override exact 3D sign or normal truth.

Initial objective family:

`L = L_tsdf + lambda_occ L_occ + lambda_n L_normal + lambda_src L_Fstar_aux`

Gradient-to-normal alignment may be added only after the exact target/domain
contract is measured and frozen; it is not silently enabled by this amendment.

## TESSA amendment

TESSA remains proposal-only. Compiler remains sole canonical mesh authority.

Two independent questions must be measured:

### Static/compact topology
- surface fidelity,
- manifold/quality courts,
- triangle/vertex count,
- representation efficiency.

### Mechanical topology
- triangle inversion under deformation,
- crack/hole exposure under motion,
- self-intersection under motion,
- strain/Jacobian tails,
- motion silhouette drift,
- attachment/deformation stability.

A mesh cannot be promoted because it is compact if it fails mechanical
admissibility. Candidate selection is lexicographic:

1. mechanical admissibility,
2. motion quality,
3. static surface fidelity,
4. compactness / triangle count.

Triangle count is a late tie-breaker, not a proxy for animation suitability.

## Required A/B after Truth139

Compare at matched IRIS scene and matched rig/weight/motion evidence:

A. exact-3D IRIS + deterministic field extraction,
B. exact-3D IRIS + TESSA proposal + Compiler qualification.

TESSA is justified only if B improves mechanical/topological quality at a useful
compactness budget without degrading source/geometry fidelity.

## Authority boundary

This amendment:
- does not create a train/validation/LOFO split,
- does not admit Truth139 derived targets by itself,
- does not alter the running materialization worker,
- does not promote V3.2/V4 to product authority,
- does not declare TESSA necessary,
- does not declare deterministic extraction sufficient.

Truth139 measurement decides those questions.
