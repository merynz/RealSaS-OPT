# BUSTED — ROOT/FK live Mechanics consumer correctness

**Source audit:** [#353](https://github.com/merynz/RealSaS-OPT/pull/353) exact head `a9f3efd225eed7d08e418c2ebfd303da3b0cb613`.  
**Main parent:** `173664070e4e2b81d4aa0e79d8c8d4248dbbfe9e` after Motion V2 and Mechanics CAS worker integration.  
**Owner:** Canonical Mechanics control execution; no model, Graphics, proof, site or corpus ownership.

## Real runtime bug addressed

The previous `solve_control_pose_v1` directly transformed only `target_canonical_joint_ids` for ROOT/GLOBAL/FK/SOCKET_CONTROL. Descendant joints stayed in rest locations unless another operator moved them. This is incorrect when an animator drags or rotates a parent whose skin weights are distributed over descendants. A user sees a broken chain or unexpected deformation even if the static canonical Rig/Skin contract passes.

**Repair:** import the existing research solver file **unchanged**, blob `6b08101d4beab65e9261253b10c023d31c266251`. It builds declared joint ancestry, rejects unknown parents/cycles, applies direct controls shallow-to-deep to the whole declared subtree, and allows explicit child FK to override only its own subtree. It neither synthesizes rig controls nor infers missing hierarchy, DOF or constraints.

## Actual producer → consumer courts

The new mainline tests use the real production-facing Compiler APIs:

1. Synthetic CanonicalMesh v1 + CanonicalRig v2 (including parent/child and sibling) + CanonicalSkin v2 are bound by hash and rest-qualified through `mint_canonical_mechanics_v1`.
2. An authored ROOT translation executes through `solve_control_pose_v1`, moves every descendant and yields matching full-vertex `deform_mesh_lbs_v1` output — not just the root joint.
3. A child FK control declared **before** ROOT in the control list rotates its child subtree while preserving an unrelated sibling, proving hierarchy not source-order precedence.
4. Control-space `CanonicalMotionIR.v2` is qualified, minted, sampled and consumed by the **same Mechanics solver**. At t=0.5 the whole joint and skinned mesh system must move by the interpolated amount.
5. Unknown controls, missing joint parents and parent cycles fail closed. Existing V6 Mechanics, Motion, CAS-worker regression suite remains mandatory.

All fixtures are **synthetic for deterministic engineering regression**. Their PASS does not claim unseen 3D mechanics quality, source Blender parity, a trained AXIS/MIRA model, Graphics draw-order correctness or a shippable 60fps runtime.

## Deliberate scope and next consumer

- No research multigoal/secondary/attachment modules copied from #353.
- No platform product registration, ProductRevision, `.rss` writer/reader modification or model promotion.
- After tests and full CI: explicitly bind this Mechanics implementation into the live control override preview with typed source/version and behavioral diagnostics, then test real source interventions and a native Graphics/materialization consumer before product qualification.
- Preserve `#353` research evidence and tests; avoid an uncontrolled merge of its 50-file stack.

**UI-facing requirement:** dragging the character root must move the feet, limbs, skin and attached controls consistently; this slice enforces the first mechanical invariant, not all constraints/contact/grip semantics.
