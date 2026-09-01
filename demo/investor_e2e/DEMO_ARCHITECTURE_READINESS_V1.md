# RealSaS — Demo Architecture Readiness V1

**Status:** `OPEN__SPECIMEN_SELECTION_FORBIDDEN`

`DEMO_ARCHITECTURE_READY_V1` must PASS before any real specimen is selected, named in code/config, or used for optimizer execution.

## Base audit — main @ bba22f25313b32aa4f8c2fbe418e3f2590198f38

### Existing executable authority to reuse

- `compiler/realsas_compiler_core/types.py`
  - `ObservationEvidenceIR`
  - `RiggingSurfaceIR`
  - `SkeletonProposalIR`
  - `QualifiedSkeletonIR`
  - `SkinProposalIR`
  - `QualifiedSkinIR`
  - typed mesh / mesh-skin IRs
  - `CanonicalPuppetGraphV2`
- `compiler/realsas_compiler_core/surface.py`
  - analytic observation-depth -> common-frame surface path
- `compiler/realsas_compiler_core/rig.py`
  - proposal lineage/support checks
  - global graph optimization
  - root/parent qualification
  - Compiler-minted canonical IDs
- `compiler/realsas_compiler_core/skin.py`
  - lineage/reference/simplex legality
  - bounded correction
- `compiler/realsas_compiler_core/mesh_binding.py`
  - typed identity support bindings and exact skin copy baseline
- `compiler/realsas_compiler_core/product.py`
  - `CanonicalPuppetGraph.v2` assembly and exact-state proof binding
- `experiments/geppetto_arachne_r6_20260901/teacher_projection_v1.py`
  - generic anonymous deform-control teacher/evaluator projection only

### Material implementation gaps for this demo

1. **IRIS demo learned producer — MISSING**
   - current V2 has deterministic Gate0 apparatus but no complete learned predictor;
   - older `iris_controlled_v1` is historical scaffold and emits an obsolete direct-P/N style boundary;
   - demo implementation must emit current depth/support evidence and keep `P` analytic.

2. **Geppetto learned producer — MISSING**
   - current R6 has teacher/evaluator projection, no predictor;
   - required functional obligations: global shape condition, variable/existence control evidence, continuous joint positions, root scores, parent-edge scores.

3. **Arachne learned producer — MISSING**
   - no current `SkinFieldCodec` or predictor;
   - for the single-specimen demo, a direct generic geometry+skeleton-conditioned influence-field predictor is allowed, but it must emit `SkinProposalIR` and pass the existing Compiler skin qualifier.

4. **Useful generic demo mesh/discretization — MISSING**
   - MWB-1 identity-subset qualification exists;
   - no generic product-useful triangulator is current main authority;
   - branch needs an experimental view-local support-bound triangulation path with zero hidden geometry authority.

5. **Generic deformation/probe/playback harness — MISSING**
   - native C++17 runtime remains historical external byte authority;
   - branch needs a generic learned-product consumer that applies skinning/preset transforms and exports visual animation evidence without claiming native-runtime promotion.

6. **Fit/eval/checkpoint/inference firewall harness — MISSING**
   - must be generic before specimen selection.

## READY gate checklist

All items below must be `[x]` before changing status to PASS.

### Generic code surface

- [ ] package `demo/investor_e2e/realsas_demo_e2e/` imports without optional specimen data
- [ ] IRIS demo model class implemented
- [ ] Geppetto conditioning adapter implemented
- [ ] Geppetto demo model class implemented
- [ ] Arachne teacher target adapter implemented
- [ ] Arachne conditioning adapter implemented
- [ ] Arachne demo model class implemented
- [ ] generic mesh candidate/discretization path implemented
- [ ] generic deformation + playback harness implemented

### Contract integration

- [ ] IRIS emits current external depth/evidence contract
- [ ] analytic Compiler surface path consumes IRIS result
- [ ] Geppetto emits existing `SkeletonProposalIR`
- [ ] existing Compiler skeleton qualification consumes Geppetto result
- [ ] Arachne emits existing `SkinProposalIR`
- [ ] existing Compiler skin qualification consumes Arachne result
- [ ] mesh and mesh-skin types pass existing Compiler validators
- [ ] `CanonicalPuppetGraph.v2` assembles from the generic path

### Fit / evaluation infrastructure

- [ ] all V1 acceptance metrics implemented in code
- [ ] teacher truth loaders are explicitly training/evaluation-only
- [ ] checkpoint save/load includes model/config/source hashes
- [ ] run manifest binds exact input/checkpoint/code/output hashes
- [ ] deterministic seed policy implemented

### Final-inference firewall

- [ ] fresh-process final inference entrypoint implemented
- [ ] teacher paths are absent from final-inference CLI/schema
- [ ] source rig/mesh paths are absent from final-inference CLI/schema
- [ ] no specimen ID is accepted as a model input
- [ ] static scan rejects forbidden specimen-specific constants/patterns in demo model code
- [ ] runtime assertion records `teacher_objects_reachable = false`

### Zero-specimen tests

- [ ] synthetic observation fixture exercises IRIS output schema
- [ ] synthetic surface fixture exercises Geppetto -> Compiler qualification
- [ ] synthetic qualified skeleton fixture exercises Arachne -> Compiler qualification
- [ ] synthetic mesh fixture exercises mesh/skin binding
- [ ] synthetic qualified product exercises deformation/playback
- [ ] test suite passes with no real specimen files present

### Change-control

- [ ] no optimizer step executed before READY
- [ ] no real specimen selected or named before READY
- [ ] code review/audit finds no specimen-specific branches, coordinates, topology, weights, or thresholds

## PASS semantics

`DEMO_ARCHITECTURE_READY_V1 = PASS` means only:

> The generic branch-local implementation and its firewalls are ready to accept a subsequently selected specimen for an explicitly authorized single-specimen memorization fit.

It does not mean any model has learned, any specimen has passed, or generalization exists.
