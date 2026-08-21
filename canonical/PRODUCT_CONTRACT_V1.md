# RealSaS — Canonical IRIS → Geppetto → Arachne Product Contract V1

## North star

`ONE 8-VIEW CHARACTER SHEET -> EDITABLE, RIGGED, ANIMATABLE PUPPET`

The default user input is one neutral pose rendered/drawn in eight ordered views. Pose B is not a shipping dependency unless a future controlled end-to-end comparison proves a material final-quality advantage that cannot be recovered downstream.

## IRIS

Single responsibility:

`8-view raster observation -> safe, uncertainty-aware, rig-ready geometric surface substrate`

Required external semantics:
- `P`: common/object-frame position or equivalent surface geometry;
- `N`: local normal/orientation evidence;
- `V`: per-view visibility and observational support;
- `U`: calibrated geometric uncertainty/confidence;
- geometric surface correspondence/persistence across views;
- provenance/support metadata.

An explicit learned `Z` correspondence embedding is optional. The capability to associate observations into coherent surface hypotheses is required, whether represented explicitly or implicitly through common-frame geometry.

IRIS is not responsible for authored mechanical owner identity, source-rig exactness, skeleton topology, parents, skinning weights or mandatory GFDR.

## Geppetto

`IRIS geometry -> clean editable skeleton/hierarchy proposal`

Owns joint/control locations and skeleton structure/hierarchy. Existing M4-lineage authored skeleton/hierarchy supervision maps here. Exact source-rig recovery is not required when a cleaner functionally equivalent rig exists.

## Arachne

`IRIS geometry + Geppetto skeleton -> editable skinning weights`

Owns dense/sparse skinning. Existing M5-lineage dense weighting/deformation supervision maps here.

Geppetto and Arachne are responsibility boundaries, not a permanent requirement for two physical checkpoints; they may later share an encoder or be jointly trained if evidence supports it while typed interfaces/evaluation remain separable.

## Compiler

Canonical product authority after learned proposals. Owns IDs/graph normalization, structural validity, hierarchy/weight sanity, cleanup, deterministic postprocess, deformation probes, verification, repair/reselection, fail-closed behavior and editable puppet export.

## Quality contract

IRIS is geometry-first. Evaluation must cover position/surface accuracy, normals, silhouette reprojection, cross-view consistency, coverage, local geometric fidelity, uncertainty calibration, artist-domain robustness and downstream rig-readiness. Owner partition, parent accuracy, authored-rig exactness and GFDR accuracy are not IRIS promotion gates.

A visually plausible surface is insufficient if it destroys geometry needed for downstream rigging. This is tested through geometry-sensitive metrics and ultimately controlled `GT geometry -> downstream` versus `IRIS geometry -> downstream` quality comparisons, not by returning mechanical ontology to IRIS.

## Research roadmap

```text
D2 diagnostic closure
 -> G0 contract/evaluator freeze
 -> G1 single-pose multiview geometry baseline
 -> G2 camera-aware cross-view fusion
 -> G3 direct common-frame geometry
 -> G4 explicit geometry grounding/refinement
 -> G5 geometry-aware local refinement IF needed
 -> G6 artist-domain robustness
 -> G7 IRIS product qualification
 -> R0 corpus rebind
 -> R1 Geppetto
 -> R2 Arachne
 -> R3 joint rig quality
 -> C0 Compiler restoration/rebind
 -> P0 end-to-end product gate
```

The old D1→D2→D3→D4 descriptor ladder is not automatically the product roadmap. D1/D2/D3 components remain preserved and may be reactivated where the geometry line demonstrates need.

## Change control

An explicit contract revision is required before:
- making Pose B mandatory for shipping;
- returning hidden mechanical-owner identity to IRIS;
- making GFDR mandatory for IRIS qualification;
- removing typed boundaries or editability;
- replacing final functional-quality gates with teacher-rig exactness;
- moving canonical final authority entirely into opaque neural output.
