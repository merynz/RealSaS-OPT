# RealSaS / BUSTED — Control Motion V2 mainline integration slice

**Date:** 2026-11-07  
**Owner:** canonical Compiler Motion runtime, not training or Graphics  
**Source:** exact code/test blob copies of V6 #297 `dd3089957e3126d93db4ad2804e30511cfdca348`.

## Scope and proof
After [#363](https://github.com/merynz/RealSaS-OPT/pull/363) typed Rig/Skin contracts and [#364](https://github.com/merynz/RealSaS-OPT/pull/364) CanonicalMechanics/FABRIK+LBS reference, bring forward the **existing** control-space Motion V2 contract and sampling semantics. No animation generator or planner is built by this PR.

- `control_motion_ir_v1.py` qualifies a proposed Motion clip's named control tracks, frame times, normalized quaternions and lineage against Rig/Mechanics.
- `canonical_motion_ir_v2.py` seals an editable canonical Motion clip over **animator controls**, not bone/joint tracks.
- `control_motion_runtime_v2.py` samples translation/scale and quaternion slerp, then calls the existing `solve_control_pose_v1` Mechanics solver.
- Original V6 Motion tests prove a clip can manipulate a direct ROOT control or drive IK+poles; new tests check exact loop-endpoint behavior, negative-time repeat, nonloop clamping, sign-equivalent quaternions, forged track lineage and unknown control fail-closed.

**Source policy:** The imported three modules and two V6 test files must preserve their original Git blobs. New negative tests and CI are mainline integration artifacts.

**Not included:** Motion worker/CAS/Go ProductRevision wiring, model inference, automatic motion generation, clip blending/retarget, 6D multigoal research promotion, editable UI or Graphics. Those are separate product integration cuts and proof gates.

**Merge order:** PR #363 → PR #364 → this Motion V2 slice. An earlier stacked PR base is temporary; rebase to actual main after preceding squash merge before accepting a PR review. No merge before all required CI succeeds.
