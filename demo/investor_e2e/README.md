# RealSaS — Investor Single-Specimen Learned E2E Demo

**Branch:** `demo/investor-single-specimen-e2e`  
**Base main commit:** `bba22f25313b32aa4f8c2fbe418e3f2590198f38`  
**Authority class:** `EXPERIMENTAL_SIDE_BRANCH__NOT_MAIN_CONTINUATION_AUTHORITY`

## Goal

Produce one honest end-to-end architecture/integration proof:

```text
8 canonical neutral RGBA views + fixed known camera contract
 -> IRIS demo-fit perception
 -> ObservationEvidenceIR
 -> GeometricSubstrateAssembler / RiggingSurfaceIR
 -> Geppetto demo-fit model
 -> SkeletonProposalIR
 -> Compiler.qualify_skeleton
 -> QualifiedSkeletonIR
 -> Arachne demo-fit model
 -> SkinProposalIR
 -> Compiler.qualify_skin
 -> QualifiedSkinIR
 -> generic demo mesh/discretization + qualified mesh-skin binding
 -> CanonicalPuppetGraph.v2
 -> generic deformation/playback proof harness
 -> recorded animation artifact
```

## Claim firewall

- `GENERALIZATION_CLAIM = FALSE`
- `SINGLE_SPECIMEN_MEMORIZATION_ALLOWED = TRUE`
- `IMAGE_ONLY_FINAL_INFERENCE_REQUIRED = TRUE`
- `IRIS_REQUIRED = TRUE`
- `GEPPETTO_REQUIRED = TRUE`
- `ARACHNE_REQUIRED = TRUE`
- `COMPILER_QUALIFICATION_REQUIRED = TRUE`
- `MANUAL_OUTPUT_INJECTION = FORBIDDEN`
- `SPECIMEN_SPECIFIC_CODE = FORBIDDEN`
- `SPECIMEN_SELECTION_BEFORE_ARCHITECTURE_READY = FORBIDDEN`

A model may memorize through learned parameters. Python/source code may not contain specimen IDs, known joint coordinates, known topology, known weight tables, output lookup tables, or specimen-specific thresholds.

## Main-branch authority firewall

`main` currently states that IRIS V2 and Geppetto/Arachne learned optimizer steps are not authorized. This demo branch does **not** revise that scientific authority.

The user explicitly authorized, on 2026-09-01, single-specimen fitting/optimizer execution for the investor demo **only after the generic demo architecture is implemented and sealed READY on this experimental branch**. Any result remains experimental side-branch evidence unless separately reconciled and explicitly promoted to `main` under `canonical/BRANCH_AUTHORITY_V1.md`.

## Specimen firewall

No real demo specimen is selected or named until `DEMO_ARCHITECTURE_READY_V1 = PASS`.

Before that gate, implementation and tests may use only generic/synthetic fixtures and generic teacher-adapter schemas.

After READY, specimen binding is data/config only. Selecting a specimen may not require a Python code change.

## Current base audit

At the base commit:

- Compiler typed `SkeletonProposalIR -> QualifiedSkeletonIR` execution exists.
- Compiler typed `SkinProposalIR -> QualifiedSkinIR` execution exists.
- MWB-0/MWB-1 typed mesh/mesh-skin lineage exists, but a useful generic demo discretizer is not yet implemented.
- Current Geppetto R6 code contains teacher/evaluator projection, not a learned Geppetto model.
- Current Arachne has no learned codec/predictor implementation.
- Current IRIS V2 has deterministic Gate0 apparatus but not the complete learned V2 predictor; older controlled IRIS scaffolds are historical implementation evidence only.
- Current native runtime source remains historical external byte authority; therefore this branch must implement a generic proof/playback harness first and must not pretend it is the native runtime.

## Working order

1. Freeze scope, acceptance contracts, architecture readiness, risks, and ledger.
2. Implement generic interfaces/adapters and zero-specimen static firewalls.
3. Implement IRIS demo-fit model against the current external depth/evidence contract.
4. Implement Geppetto conditioning + learned proposal model.
5. Implement Arachne teacher field target + learned influence predictor.
6. Implement generic view-local mesh/discretization and mesh-skin path without hidden geometry authority.
7. Implement generic deformation/probe/playback harness.
8. Close `DEMO_ARCHITECTURE_READY_V1` on synthetic fixtures, with zero optimizer steps.
9. Only then select a specimen under the frozen envelope.
10. Fit IRIS, Geppetto, Arachne to that specimen.
11. Run teacher-firewalled final inference from images only.
12. Export investor-facing evidence and animation.
