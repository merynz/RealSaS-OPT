# RealSaS Mage FIT1 Demo — Continuation Checkpoint V2 — 2026-09-14

This file supersedes `CONTINUATION_CHECKPOINT_20260914.md` for continuation after the second physical closure run. Read the older checkpoint for full immutable input hashes and P1/V5 per-view lineages; use this V2 file for the latest product state and next actions.

## Branch and claim discipline

- Repository: `merynz/RealSaS-OPT`
- Product branch: `product/mage-fit1-demo-fulfillment-20260914`
- Previous checkpoint commit: `40317a0887c8c8295de1b99af851d404e93cc127`.
- `PRODUCT_PASS = FALSE` until qualified directional motion bake + motion-frame continuity/raster + ProductProofBundle/runtime/export hash interlock all close.
- `UNSEEN/GENERALIZATION = FALSE`.
- FIT2/Arachne training remains deferred; bounded demo line is sealed FIT1/V5 + current P1/P1Q product mesh.
- Never force historical hashes, never repair V5 weights for rigid accessories, never infer semantic ownership from filename/source-index equality, never use rejected nearest-owner completion.

## Stable upstream physical authorities

These were physically re-materialized after a runtime reset and reproduced their prior exact hashes:

- P1 current-authority manifest: `8a2586370907d37930710b1ba599708c0d43551f21c366b97c94a029daf622a6` — 8/8 exact historical numeric projection proof, current GSA lineage retained.
- V5 direct -> current P1 manifest: `491b82b7c503307b84a5a9adef43ba0702f43ff8b0f0ca43d84d51ee72d0c17d` — direct decoder query on exact current P1 vertices; no optimizer/backward/source-skin-row transfer.
- Shared semantic role binding physical hash prefix: `0ecfefdda71f…`.
- Exact source-owner raster source-quality manifest prefix: `4df8d42a…`; two independent runs produced byte-identical artifact SHA sets.

See the V1 checkpoint for all exact input hashes, camera/observation hashes, P1 per-view mesh lineages and V5 per-view mesh-skin lineages.

## P1Q current-authority closure — PHYSICAL PASS

Second current-authority P1Q run closed all 8 views under the frozen face-subset policy.

Exact removal sequence:

- V0: 3 — `[311, 1318, 4081]`; recall `0.9721433468049483`; precision `1.0`; min angle `0.2585708658022066°`; max aspect `249.12088852443355`.
- V1: 0 — identity path; recall `0.9894795368589644`; precision `1.0`.
- V2: 4 — `[807, 2350, 3192, 3336]`; recall `0.9767551883718063`; precision `1.0`; min angle `0.2503746706444417°`; max aspect `247.5117476350834`.
- V3: 2 — `[1016, 5553]`; recall `0.983376805914073`; precision `1.0`; min angle `0.25984302361032885°`; max aspect `248.3008032672835`.
- V4: 3 — `[90, 2531, 4934]`; recall `0.973267`; precision `1.0`.
- V5: 0 — identity path; recall `0.988632`; precision `1.0`.
- V6: 0 — identity path; recall `0.975021`; precision `1.0`.
- V7: 6 — `[962, 963, 964, 1138, 2393, 3572]`; recall `0.981964`; precision `1.0`.

Total removed faces: `18` (`3 / 0 / 4 / 2 / 3 / 0 / 0 / 6`).

Current P1Q materialization manifest prefix from this run: `1d3423b6d45f…`; 38 artifacts, 8/8 PASS. The exact full bytes were local-runtime artifacts and were lost in the later runtime reset; do not invent the remaining hash suffix. Re-materialize from the exact authorities if bytes are required.

## Corrected typed component assembly — PHYSICAL PASS

Second corrected assembly used only:

- current P1Q,
- unchanged P1Q-derived V5 direct binding,
- exact source-owner raster authority,
- shared semantic role binding.

Semantics:

- BODY = `DEFORMABLE_COMPONENT`; measured multi-joint support, 14 active canonical joints in the assembly audit.
- BOOK = `RIGID_BONE_ATTACHMENT` -> left hand slot.
- STAFF/WAND = `RIGID_BONE_ATTACHMENT` -> right hand slot.
- HAT = `RIGID_BONE_ATTACHMENT` -> head.
- CAPE = `RIGID_BONE_ATTACHMENT` -> chest.
- Rigid components do NOT claim V5 skin as their attachment truth; no fake one-hot weight repair.

Physical assembly hash prefix: `9312392ad37a…`.
Physical assembly seal SHA prefix: `c2682c6de198…`.

These bytes were also local-runtime artifacts and were lost after reset; re-materialize rather than guessing full suffixes.

## Rest-frame composed raster acceptance — PHYSICAL SEALED PASS

Composition under test: P1Q/V5 continuous deformable underlay + exact rigid source-owner foreground.

All 8 views passed. Physical sealed result:

- worst-view recall: `0.995651` (~99.5651%)
- worst-view precision: `1.0` (100%)
- worst-view IoU: `0.995651` (~99.5651%)
- false-positive pixels at the measured rest gate: `0`

Manifest prefix: `5716726b…`.
Seal prefix: `cf29a8da…`.

This closes the requested rest-pose acceptance goal far above the intended ~98% worst-view target. It does NOT substitute for motion-frame proof.

## Underlay binding status — BLOCKED, fail-closed

An 8-view continuity-underlay binding rerun stopped on V0 because reconstructed appearance lineage did not equal the saved qualified P1Q appearance lineage.

Interpretation:

- mesh/skin was not the blocker;
- the wrapper was incorrectly reconstructing appearance instead of consuming the qualified P1Q retained-face appearance authority;
- do not weaken the hash check or replace the expected hash;
- fix the wrapper/producer to consume the exact saved qualified P1Q appearance authority when re-materializing.

Because local P1Q bytes were lost after runtime reset, this underlay binding will need re-materialized P1Q appearance artifacts before final motion-exposed seam proof.

## Motion audit — IMPORTANT CURRENT BLOCKER/DECISION

The existing `fc16e49021980…` articulated motion state (32 effective tracks) is real joint-based motion and passed diagnostic LBS identity/loop/dynamic checks, but it contains a small authored root `translation_xy` bob.

The typed current directional evaluator is deliberately `ROTATION_ONLY_CURRENT_PRESET_V1` and rejects any nonzero translation with `DIRECTIONAL_EVALUATOR_TRANSLATION_UNITS_NOT_QUALIFIED` because no typed directional translation-unit contract exists yet.

Therefore:

- diagnostic direct LBS is NOT final product motion authority;
- the old retracted mechanical-XY/raster shortcut must not be used;
- final motion must go through `DirectionalJointViewBindingSetIR -> DirectionalMotionEvaluator -> QualifiedDirectionalMotionBakeProviderV1 -> QualificationOwnedMotionBakeIR`;
- for the bounded demo, remove only the unqualified root translation bob while preserving articulated root/spine/head/arms/legs rotations, bilateral gait phase and counter-swing;
- this is a proof-scope correction, not a return to global warp.

The rotation-only source patch was prepared locally before a runtime reset but had not yet been committed. Reproduce and test it before claiming a new motion-state hash.

## Draw order/runtime

Current runtime/export code supports per-frame/per-view draw order and requires each view's draw order to be an exact permutation of that frame's mesh set. Runtime-v2 consumes qualification-owned motion bakes and forbids export-time solver/model replay.

The final Mage product still needs current artifact rebinding after the rotation-only qualified bake.

## Runtime reset note

The local runtime that produced the second P1Q/assembly/rest-raster closure was reset. Google Drive search after reset did not locate the final P1Q seal, assembly hash artifacts, or rest-raster seal by their exact names/hash prefixes. The immutable input authorities and exact source snapshots remain available, so missing bytes must be re-materialized; never pretend the local outputs survived.

## Immediate continuation order

1. Commit/test the rotation-only qualified articulated preset patch: all joint-track `translation_xy == (0,0)`; preserve articulated rotations and existing clip IDs.
2. Add/keep a regression invariant that all current Mage preset translations are zero and the typed directional evaluator's translation firewall remains fail-closed for nonzero translations.
3. Pass self-hosted External Render Continuity Contract CI.
4. Re-materialize current P1 -> V5 -> P1Q -> corrected assembly/rest state only as needed to recover bytes for the typed directional product state. Do not recompute if a trusted exact artifact becomes available.
5. Fix continuity-underlay producer to consume saved qualified P1Q appearance rather than reconstructing a drifted appearance lineage.
6. Qualify current `DirectionalJointViewBindingSetIR` on all V0–V7 views.
7. Produce qualification-owned idle/run motion bakes with the rotation-only evaluator/provider; bind directional binding hash, evaluator policy hash, proof plan hash and product state hash.
8. Run motion-frame geometry quality + attachment-interface seam continuity + composed raster behavior. Natural outer silhouettes are not attachment seams.
9. Bind exact per-frame/per-view draw order.
10. Build ProductProofBundle -> runtime-v2 -> export hash interlock.
11. Only if every current gate passes set `PRODUCT_PASS = TRUE`; otherwise remain fail-closed with the exact blocker.

## New-chat instruction

Say: **"Read `experiments/mage_demo_fit1_v5_p1/CONTINUATION_CHECKPOINT_20260914_V2.md` on `product/mage-fit1-demo-fulfillment-20260914` and continue exactly from there."**
