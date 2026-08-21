# RealSaS — Canonical IRIS → Geppetto → Arachne Product Contract V1

## North star

`ONE 8-VIEW CHARACTER SHEET -> EDITABLE, RIGGED, ANIMATABLE PUPPET`

The default user input is one neutral pose rendered/drawn in eight ordered views. Pose B is not a shipping dependency unless a future controlled end-to-end comparison proves a material final-quality advantage that cannot be recovered downstream.

## Canonical problem-definition authority

The exact IRIS scientific problem is frozen in:

`canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`

The key rule is:

> IRIS must recover the **minimal observation-grounded geometric substrate sufficient for downstream rigging**. The final learned head list is an experimental result of S0, not an assumption.

The current `P/N/V/U + persistence/provenance` factorization is the frozen G1 baseline candidate, not yet a proof that every item needs an independent learned head or that no additional geometric field is required.

## IRIS

Single responsibility:

`8-view raster observation -> safe, uncertainty-aware, rigging-sufficient observable geometric substrate`

Current candidate external semantics:
- `P`: common/object-frame position or equivalent surface geometry;
- `N`: local normal/orientation evidence;
- `V`: per-view visibility and observational support;
- `U`: geometric uncertainty/risk, calibrated at the dedicated calibration gate;
- geometric surface correspondence/persistence across views;
- provenance/support metadata.

An explicit learned `Z` correspondence embedding is optional. The capability to associate observations into coherent surface hypotheses is required, whether represented explicitly or implicitly through common-frame geometry.

S0 determines whether normals, visibility/support or other quantities require independent learned heads, can be deterministically derived, or whether richer geometric state such as ambiguity hypotheses is needed.

IRIS is not responsible for authored mechanical owner identity, source-rig exactness, skeleton topology, parents, skinning weights or mandatory GFDR.

## SurfaceBuilder

A deterministic geometric canonicalization layer sits between raw IRIS evidence and the learned rigging stages. It is **not** a fourth learned model.

Candidate responsibilities include surfel fusion, provenance/support bookkeeping, reprojection/cycle checks, local adjacency/neighborhood construction, sheet/component separation and stable derived differential geometry. S0 decides which of these are required.

## Geppetto

`RiggingSurface -> clean editable skeleton/hierarchy proposal`

Owns joint/control locations and skeleton structure/hierarchy. Existing M4-lineage authored skeleton/hierarchy supervision maps here. Exact source-rig recovery is not required when a cleaner functionally equivalent rig exists.

## Arachne

`RiggingSurface + Geppetto skeleton -> editable skinning weights`

Owns dense/sparse skinning. Existing M5-lineage dense weighting/deformation supervision maps here.

Geppetto and Arachne are responsibility boundaries, not a permanent requirement for two physical checkpoints; they may later share an encoder or be jointly trained if evidence supports it while typed interfaces/evaluation remain separable.

## Compiler

Canonical product authority after learned proposals. Owns IDs/graph normalization, structural validity, hierarchy/weight sanity, cleanup, deterministic postprocess, deformation probes, verification, repair/reselection, fail-closed behavior and editable puppet export.

## Quality contract

IRIS is geometry-first. Evaluation must cover position/surface accuracy, normals where required, silhouette reprojection, cross-view consistency, coverage, local geometric fidelity, uncertainty/risk calibration, artist-domain robustness and downstream rig-readiness. Owner partition, parent accuracy, authored-rig exactness and GFDR accuracy are not IRIS promotion gates.

A visually plausible surface is insufficient if it destroys geometry needed for downstream rigging. This is tested through geometry-sensitive metrics and controlled `GT observation-equivalent substrate -> downstream` versus `IRIS substrate -> downstream` quality comparisons, not by returning mechanical ontology to IRIS.

## Research roadmap

```text
D2 diagnostic closure
 -> G0 contract/evaluator freeze
 -> S0 observable rigging-substrate definition + sufficiency ablation
      S0-A information/derivability inventory
      S0-B fixed downstream probe ablation
      S0-C RiggingSurface contract freeze
 -> G1 preserved single-pose multiview geometry baseline
 -> G1.5 geometry coherence interventions IF needed
 -> G2 camera-aware cross-view fusion IF needed
 -> G3 direct/richer common-frame geometry IF needed
 -> G4 explicit geometry grounding + calibration
 -> G5 geometry-aware local refinement IF needed
 -> G6 artist-domain robustness
 -> G7 IRIS product qualification against S0 contract
 -> R1 Geppetto
 -> R2 Arachne
 -> R3 joint rig quality
 -> C0 Compiler restoration/rebind
 -> P0 end-to-end product gate
```

G1 remains historically and scientifically frozen; S0 does not rewrite its architecture or preregistration. S0 tells us what downstream sufficiency target G1 and later geometry systems must ultimately meet.

The old D1→D2→D3→D4 descriptor ladder is not automatically the product roadmap. D1/D2/D3 components remain preserved and may be reactivated where the geometry line demonstrates need. High-recall correspondence, reciprocal/cycle consistency, explicit camera geometry, grounding, provenance and ambiguity preservation remain active architecture lessons even though motion-specific GFDR outputs are no longer core IRIS targets.

## Change control

An explicit contract revision is required before:
- making Pose B mandatory for shipping;
- returning hidden mechanical-owner identity to IRIS;
- making GFDR mandatory for IRIS qualification;
- declaring a fixed final IRIS head list without S0 sufficiency evidence;
- removing typed responsibility boundaries or editability;
- replacing final functional-quality gates with teacher-rig/source-mesh exactness;
- moving canonical final authority entirely into opaque neural output.
