# BUSTED — V6 typed Rig/Skin producer-worker integration

**Reviewed product goal:** Input → Shape → Mesh → Rig v2 → Skin v2 → Mechanics v1 → {Graphics, Motion v2} → Proof → ProductRevision.  
**Source of two V2 CAS executors:** #297 pinned commit `dd3089957e3126d93db4ad2804e30511cfdca348`.  
**Current prerequisite:** #373 V6 runtime source and safe checkpoint loaders. This PR is temporarily stacked on #373 to keep imports/test scope honest; it must be rebased directly onto updated `main`, retargeted and fully retested after #373 merges.

## What actually changes

- Import exact original `rig_executor_v2.py` blob `f9a101dcd236ebad3aaed02dbcf8995f66a3caab`.
- Import exact original `skin_executor_v2.py` blob `edd744d51c0f5a92dd1b9231e939244a9c7f614a`.
- Extend existing Platform Python worker dispatch with **explicit** `RealSaS.ProductRigInference.v2` and `RealSaS.ProductSkinInference.v2` only. Keep V1 handlers and Mechanics CAS rest executor unchanged.
- Add actual canonical CAS input decoding/lineage tests, no ModelRelease rejection, missing/mislabeled parent artifacts, rehashed-but-wrong RigProgram→SkinCode parent, wrong MIRA checkpoint, and actual async worker-route tests.
- Dedicated worker CI installs pinned Python/Torch/Temporal and tests old+new workers. Current registered Go capability descriptors remain unchanged (v1 Rig, v1 Skin).

## The real V2 producer/consumer contract

`product.mesh` yields CanonicalMesh v1 and RiggingSurface v1. AXIS V6 Rig worker consumes **both** plus exactly one registered AXIS V6 ModelRelease, runs one causal program, seals:
- CanonicalRig v2 (deform structure + control program)
- RigProgramProposal v1 and corresponding qualification
- SkinCodeProposal v1 with model and original RigProgram proposal hashes, carrier bindings and exact MIRA codec checkpoint hash

The MIRA V6 Skin worker must consume **the same** CanonicalMesh v1, CanonicalRig v2, SkinCodeProposal v1 plus exactly one registered MIRA V6 ModelRelease and exact checkpoint bytes. It never reconstructs SkinCode, re-runs AXIS, selects a different codec, or substitutes a v1 Rig/Skin. It validates SkinCode-to-rig and carrier parents, checkpoint architecture/hash, decodes token-conditioned influences, bounds simplex qualification, and emits CanonicalSkin v2.

Mechanics v1 then consumes the exact Mesh/Rig/Skin typed triple. No product capability change is authorized until its source-level and runtime/ModelRelease gates are mechanically closed.

## What our tests establish—and do NOT establish

The tests execute *real current canonical parsers and worker entrypoints* against synthetic CAS JSON; the absence of model releases is a hard error, not an invitation to fabricate fake learned output. The worker dispatch is exercised through the actual async activity and marked as a synthetic probe only.

The original AXIS/MIRA package and grammar tests remain mandatory. **No test here claims successful learned inference, real checkpoint admission, source-verified genericity, 6D grip, Graphics/CAA integration, .rss editing, 60fps, ProductRevision or model quality.** These require qualified exact artifacts, source interventions and product end-to-end proof, not a fallback model.

## Next integration gates

1. Merge #373 after all exact-head CI and review. Rebase this short-lived branch to new main; require broad current-mainline/compiler/Go/CAA checks in addition to dedicated CI. **No bulk merging #297.**
2. Only after real output/ModelRelease proof, make one versioned Go product descriptor transition from Rig/Skin v1 to v2 as a separate reviewed change. The operator must refuse incompatible checkpoint channels and missing typed parents.
3. Register Mechanics, then separate Graphics and Motion sibling roles, with proof-carrying exact ProductRevision. Runtime must render from actual CanonicalMotion v2 controls and Mechanics deformation, preserving Graphics draw order and source appearance; old baked Stage40/43 render is not sufficient.
4. Complete edit→requalification→native .rss export/reopen and cold compile/warm playback tests before calling BUSTED available.
