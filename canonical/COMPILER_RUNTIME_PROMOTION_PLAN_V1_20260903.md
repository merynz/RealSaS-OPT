# Compiler / Runtime Promotion Plan V1 — 2026-09-03

**Base:** `2b5d467186839401ab30f9566015d9e9d49a2a06`  
**Branch:** `restoration/compiler-runtime-promotion-v1-20260903`

## Principle

The old Compiler is not restored as a competing system. Selected historical production mechanisms are promoted behind the current `realsas_compiler_core` authority.

## Historical findings applied

The v0.5 source contains full proof, orchestrator, export, reference-runtime, mesh, weight, deformation, rig and native-runtime trees. However, the old `realsas_orchestrator/pipeline.py` is a 5k-line monolithic coordinator coupled to superseded front-brain/truth packages; wholesale restoration would recreate the ownership problem the current architecture solved.

Therefore:

- native C++ runtime: exact-source promotion;
- motion/deformation proof + failure signatures: semantic rebind;
- bounded attribution/repair loop: semantic rebind;
- export/runtime package: rebind to current `CanonicalPuppetGraphV3` + exact `ProductProofBundleIR`;
- CDT/BBW-KKT/ARAP/XPBD: source-diff selection before promotion;
- old monolithic pipeline/front-brain/truth routes: no promotion.

## Gates

Every stage must preserve current behavioral source gates and complete-E2E. Native runtime additionally requires CMake build + ABI CTest, then a real package-to-native consumer gate.
