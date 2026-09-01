# RealSaS — Demo Architecture Readiness V1

**Status:** `OPEN__SPECIMEN_SELECTION_FORBIDDEN__FULL_PRODUCT_V3_REQUIRED`

`DEMO_ARCHITECTURE_READY_V1` must PASS before any real specimen is selected, named in code/config, or used for optimizer execution.

Binding architecture contract:

- `demo/investor_e2e/FULL_PRODUCT_SINGLE_SPECIMEN_FIT_CONTRACT_V1.md`
- current product topology `canonical/SYSTEM_ARCHITECTURE_V3_20260901.md` from architecture PR #19

The demo does not authorize a reduced model architecture. The only difference from generic RealSaS is that all learned components are fitted on one specimen.

## Existing executable authority to reuse

- current Compiler typed IRs and qualification paths;
- `ObservationEvidenceIR -> GeometricSubstrateAssembler/RiggingSurfaceIR` analytic P boundary;
- Compiler skeleton graph optimizer and canonical `J:*` minting;
- Compiler skin legality/simplex/bounded-repair authority;
- typed mesh / mesh-skin / `CanonicalPuppetGraph.v2` product assembly;
- exact-state proof binding;
- R6 anonymous skeleton teacher projection;
- SHA-verified historical v0.5 deformation/proof/runtime authority through an isolated typed bridge/promotion boundary.

## Historical artifacts explicitly NOT sufficient for READY

The following may remain for research history but cannot satisfy the final architecture gate:

- `IrisDemoV1` simple U-Net/Transformer scaffold;
- `GeppettoDemoV1` fixed-query scaffold;
- `ArachneDemoV1` point-joint MLP scaffold;
- direct demo Arachne path without SkinFieldCodec;
- fixed 48/64/72 product query/control caps;
- old G0.1 27D surface+interior `ConsumerTokenV1` path;
- branch-local toy deformation/playback replacing recovered historical runtime authority.

## Full-product implementation gaps before READY

### 1. IRIS Reprojection-Centered V2-A learned apparatus

Must implement the current V3 responsibilities, not a direct dense depth U-Net shortcut:

- native 1024 8-view input contract;
- frozen DINOv2-S/14 multi-level descriptor path;
- learned native high-resolution shared spatial pyramid;
- visual-hull candidate-domain construction;
- canonical world q-lattice;
- exact camera reprojection into all views;
- descriptor + learned-feature sampling;
- robust view evidence aggregation preserving validity;
- compact 3D evidence field;
- world-space regularization;
- supported/ambiguous mode extraction;
- local continuous refinement;
- exact rendering to forward depth/support/uncertainty;
- no hidden geometry/skeleton/weight authority.

### 2. Geppetto R6 predictor

Must consume only a deterministic adapter over admitted `RiggingSurfaceIR S` and provide:

- template-free variable control count;
- continuous control loci + uncertainty;
- root evidence;
- directed parent/edge evidence;
- topology/mechanical relation inside contextual/generative state;
- endogenous stop/unsupported behavior;
- anonymous `SkeletonProposalIR` output;
- no fixed product K;
- no source-rig identity/tails/teacher columns as inference features.

### 3. SkinFieldCodec

Before final Arachne predictor readiness:

- generic dynamic-N/dynamic-J compact per-control skin-field codec;
- encoder is teacher/training-only;
- decoder reconstructs dense influence fields over current S;
- one-family synthetic/reconstruction capacity gate;
- heterogeneous synthetic structural gate independent of a real demo specimen;
- deformation-sensitive ceiling under generic probes;
- no teacher identity as product input.

### 4. Arachne R6 predictor

Must consume only deterministic legal information from `S + QualifiedSkeletonIR G` and provide:

- surface contextual encoding;
- skeleton/tree contextual encoding;
- explicit point-control / bone-segment geometry;
- cross-entity interaction;
- prediction of the codec's per-control latent field representation;
- dense decoded influence proposal + uncertainty;
- weight reconstruction and deformation-sensitive optimization objectives;
- dynamic current control count;
- `SkinProposalIR` only; Compiler retains final legal/simplex authority.

### 5. Mesh / exact product / runtime bridge

- current typed mesh path must produce a legal editable support-bound mesh candidate;
- mesh skin binding must be typed and lineage-checked;
- `CanonicalPuppetGraph.v2` must assemble;
- exact-state proof must bind current product hash;
- recovered historical deformation/runtime authority must consume current typed state through the verified adapter; no replacement toy runtime is accepted.

### 6. Fit/eval/checkpoint/firewall infrastructure

All generic before specimen selection.

## READY gate checklist

All items below must be `[x]` before changing status to PASS.

### Full-product neural architecture

- [ ] IRIS V2-A source-complete with all current V3 functional blocks
- [ ] Geppetto R6 conditioning adapter source-complete
- [ ] Geppetto R6 variable-cardinality model source-complete
- [ ] Geppetto has no fixed product control-count cap
- [ ] SkinFieldCodec source-complete
- [ ] SkinFieldCodec reconstruction + deformation synthetic gates implemented
- [ ] Arachne R6 conditioning adapter source-complete
- [ ] Arachne R6 predictor source-complete
- [ ] Arachne is dynamic in current surface/joint count

### Compiler/product integration

- [ ] IRIS emits current depth/support/uncertainty evidence contract
- [ ] analytic substrate assembler consumes IRIS result
- [ ] Geppetto emits existing `SkeletonProposalIR`
- [ ] existing Compiler skeleton qualification consumes Geppetto result
- [ ] Arachne emits existing `SkinProposalIR`
- [ ] existing Compiler skin qualification consumes Arachne result
- [ ] typed mesh and mesh-skin validation passes
- [ ] `CanonicalPuppetGraph.v2` assembles
- [ ] exact-state proof path is bound
- [ ] historical/current deformation-runtime bridge consumes the exact product state

### Fit-capacity gates without a real specimen

Synthetic fixtures may test representational capacity but cannot contain the later selected demo asset.

- [ ] IRIS synthetic multi-view depth/reprojection fit reaches its hard contract
- [ ] Geppetto variable-cardinality synthetic forest/tree fit reaches exact count/loci/root/parent evidence contract
- [ ] Geppetto learned stop is demonstrated without oracle K at inference
- [ ] SkinFieldCodec reconstructs synthetic dense sparse fields under hard reconstruction/deformation ceiling
- [ ] Arachne predicts synthetic dynamic-J skin fields and passes deformation-sensitive ceiling
- [ ] end-to-end synthetic typed product reaches proof/runtime without teacher objects crossing the inference boundary

### Fit / evaluation infrastructure

- [ ] anonymous R6 teacher projection loader is training/evaluation-only
- [ ] M5/Master dense skin truth adapter is training/evaluation-only
- [ ] checkpoint metadata binds architecture/config/source/truth hashes
- [ ] deterministic seed and resume policy implemented
- [ ] each fit stage has hard PASS/FAIL metrics; loss decrease alone is insufficient
- [ ] final checkpoint selection is based only on the one-specimen fit contract after specimen selection

### Final-inference firewall

- [ ] fresh-process image-only entrypoint implemented
- [ ] input schema accepts 8 RGBA views + exact cameras + frozen checkpoints + generic config only
- [ ] teacher paths absent from inference schema
- [ ] source rig/skin/source mesh truth absent from inference schema
- [ ] specimen ID absent as prediction key
- [ ] static scan rejects known joint coordinates/topology/weights/specimen branches
- [ ] per-specimen threshold overrides forbidden
- [ ] runtime receipt proves teacher/truth objects unreachable

### Change control

- [ ] no real demo specimen selected before READY
- [ ] no optimizer has consumed a real demo specimen before READY
- [ ] reduced V1/V2 architecture-smoke runners are hard-disabled
- [ ] source freeze commit/tree recorded at READY
- [ ] after specimen selection, model/inference source changes invalidate READY and require a new pre-specimen audit

## PASS semantics

`DEMO_ARCHITECTURE_READY_V1 = PASS` means only:

> The full current RealSaS product architecture has a specimen-agnostic, executable, capacity-tested implementation and strict truth firewall, ready to be fitted on one subsequently selected full-truth specimen.

It does not mean generalization exists, and it does not mean the selected specimen has yet been fitted.