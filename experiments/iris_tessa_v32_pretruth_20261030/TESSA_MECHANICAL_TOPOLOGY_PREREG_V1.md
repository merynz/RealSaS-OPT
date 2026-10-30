# TESSA Mechanical Topology Court V1

Date: 2026-10-30
Status: PREREGISTRATION_SOURCE__PLATFORM_EXECUTION_REQUIRED

## Question
Does a topology objective conditioned on deformation/mechanical evidence produce meshes that are more suitable for animation than a static/compact topology objective at comparable geometry fidelity?

## Fixed factors
- Same admitted IRIS shared scene and exact geometry truth.
- Same rig, skin/weight evidence and motion probe set for every arm.
- Same Compiler qualification/repair policy unless explicitly tested in a separate intervention.
- Same rendering/camera/source-consistency apparatus.

## Arms
M0 DETERMINISTIC_FIELD_EXTRACTION: no learned TESSA proposal; deterministic extraction from exact-3D IRIS field.
M1 TESSA_STATIC_COMPACT: learned topology proposal optimized/selected for topology sequence validity, static geometry fidelity and compactness only.
M2 TESSA_MECHANICAL: learned topology proposal receives/uses qualified deformation/mechanical evidence and is selected with motion-conditioned courts before compactness.

## Mechanical admissibility gates
A candidate is mechanically inadmissible if it exceeds preregistered limits for any required stratum:
- triangle inversion under deformation;
- motion-induced crack or unsupported-hole exposure;
- self-intersection / fold-through where the truth motion remains valid;
- strain/Jacobian tail instability;
- attachment separation or unstable local connectivity;
- motion-view silhouette drift attributable to topology.

Static surface fidelity and watertight/manifold validity remain mandatory.

## Selection rule
Lexicographic, fail-closed:
1. mechanical admissibility;
2. motion quality envelope across subjects/motions/views;
3. static geometry/source fidelity;
4. compactness (face/vertex count and representation cost).

Triangle count cannot rescue a mechanically failing candidate and cannot stand in for animation suitability.

## Required controls
- Same geometry with connectivity changed where possible, to isolate mechanical equivalence from geometric equivalence.
- AC-vs-BD diagonal/local remeshing controls on matched surface patches.
- Static-only control showing that the mechanical court does not reward arbitrary extra triangles.
- Motion-null control showing that a compactness winner is not mechanically promoted without deformation evidence.

## Interpretation
TESSA is justified as a mechanical structuralizer only if M2 improves the motion-conditioned envelope over M0/M1 without material static/source regression. If deterministic extraction is mechanically equivalent, TESSA remains optional or may be removed from the shipping path.

## Non-claims
No TESSA necessity, product activation, ModelRelease promotion or generalization claim follows from source presence or a green engineering workflow. Canonical execution requires Platform Experiment/Attempt, reviewed-main EngineRelease and immutable Artifact Registry/CAS evidence.
