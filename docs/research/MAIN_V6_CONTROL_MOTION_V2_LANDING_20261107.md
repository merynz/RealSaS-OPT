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

## Exact-main CI admission audit (2026-11-07)

- #363 and #364 are already merged to current `main` at `ffcbaa9ca68a31bda0c07511fcb886556941c1a0`; this PR is directly based on that commit and its review diff contains exactly the eight files enumerated above.
- The three original Motion source modules and two original V6 tests still match pinned V6 #297 Git blobs **5/5**; preservation is a migration proof, **not** interactive/product readiness.
- On previous head `43a32cbfc8ecf9f377a49eb08e75840ae122da5b`, the dedicated Motion CI and three other observed runs passed, but no `current-mainline-ci` was visible among commit-associated PR runs. The absence of the required broad check is a verification gap, **not** implied success.
- This documentation checkpoint deliberately creates a new PR synchronization event against the already updated main. **Do not merge** until the new exact-head dedicated Motion CI, Platform/Go if invoked, current-mainline Compiler regressions and other required affected checks are actually observed successful.
- Product capability execution remains blocked on typed V2 producer/worker/Go routing and independent source/behavioral evidence; a Motion V2 code merge does not mint a ProductRevision.
