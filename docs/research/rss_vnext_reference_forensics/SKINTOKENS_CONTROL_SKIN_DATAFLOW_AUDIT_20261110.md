# RSS.vNext: SkinTokens genealogy, shared carrier and full-control authority

**2026-11-10. Status: code-inspected, research ablation implemented and 40 CPU tests passed. NOT final architecture freeze or real-character training qualification.**

## Source code, exact pin and ancestry

Reference: https://github.com/VAST-AI-Research/SkinTokens/tree/273b691d35989d71cd17ff2895fdc735097b92d1 (MIT licensed). SkinTokens credits UniRig, Qwen3, Michelangelo, 3DShape2VecSet and FSQ in its README. Do not treat the README's GRPO claims as a runnable source trainer.

Inspected code:

| Path | Function and learning contract |
|---|---|
| src/model/tokenrig.py:63-152,195-279,424-665 | Michelangelo mesh point/normal encoder; causal Qwen3-0.6B configured rig vocabulary; shape-token prefix; skeleton then per-joint FSQ skin token sequence; skin decoder queries point geometry |
| src/model/skin_vae_model.py:26-154 | Perceiver point latent compression and dense/uniform skin teacher input, including exact 3D points and normals |
| src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py:25-111,183-305 | Per-joint conditioned FSQ-CVAE, 512-wide eight-layer encoders; 1024-wide 16-layer query decoder |
| src/model/skin_vae/autoencoders/autoencoder_kl_tripo2.py:22-250 | Tripo2 point-query cross-attention encoder and decoder |
| src/model/skin_vae/autoencoders/FSQ.py:42-180 | Straight-through finite scalar quantization, mixed-radix token indexes |
| src/tokenizer/tokenizer_part.py:35-190 | Skeleton branching, coordinate discretization and typed token grammar |
| src/data/dataset.py and src/model/spec.py | Source asset mesh/rig/weights and dense sampled skin evidence |

**Critical reproducibility limit:** TokenRig.training_step and SkinVAEModel.training_step explicitly raise NotImplementedError. No runnable GRPO trainer was independently verified in the pinned public code tree. Source decoder architecture and training targets can be inspected; full reported training recipe CANNOT be claimed reproduced.

## Exactly who can see what in the unified neural graph

~~~text
2 USER OBSERVATIONS: FRONT_RGBA + BACK_RGBA (+ calibrated camera)
    |
DINO / image embeddings (original source tokens retained)
    |
Shared scene transformer -> Z / H(x), 3D tri-plane features (no private mesh copy)
    |        |                    |                     |
    |        |                    + K-way 2D presentation / Graphics evidence
    |        + FlexiCubes SDF + 21 cell params + source-motion training gradients
    |                           |
    +---------> structural-memory cross attention (reads raw H,
                            semantic evidence, region relation evidence,
                            optional extracted FC surface H(x))
                            |
                  structural token grammar + exact indexed Mesh proposal
                            |
               mesh vertices/faces/components, per-vertex H/XYZ/normals
                            |
                 RigProgram causal decoder (cross-reads structural memory
                      AND exact mesh H, NOT a reduced 7D carrier)
                            |
               joint/DOF, controls, effectors, anchors, constraints,
               SKIN_CODE tokens in a shared causal sequence
                            |
            per-joint predicted FSQ T tokens (at previous causal position)
                            |
         spatial skin query: EXACT mesh vertex [H(x), XYZ, normals]
            attends all J x T predicted joint-code tokens
                            |
              per-index normalized skin weights W[V,J]
                            |
               learned deformation-quality probes / loss
                            |
Compiler: typed graph/lineage qualification and deterministic execution
(FABRIK, pole, ROOT/FK, constraint execution, canonical LBS);
Graphics drawables, visibility, editable runtime/export require separate proof.
~~~

**One shared carrier is an INFORMATION-ACCESS promise, not a promise that all token-to-token decisions are differentiable.** Structural topology and rig grammar are discrete; motion- and behavior-conditioned supervision must train their probabilities. Geometric, skin, semantic and FC differentiable losses reach shared H where the code graph connects them. No teacher skin, original rig, hidden RGB render or source meshes enter student inference.

### Implemented research ablation

- models/rss_vnext/skin_query_field.py: per-vertex cross attention to *all* predicted joint FSQ code positions, joint masking, full H+XYZ+normal features, bounded chunked exact-index output.
- models/rss_vnext/unified_character_model.py: RSSvNextSkinCoupler selects dot-product baseline or point-query cross-attention. The latter does not change topology ownership, causal FSQ grammar or joint IDs.
- models/rss_vnext/contracts.py: skin_decoder_mode = DOT_PRODUCT_BASELINE (unchanged default) or POINT_QUERY_CROSS_ATTENTION. Same [B,V,J] output. SELECT using held-out source-consequence evidence, not speculative architecture taste.
- tests/models/test_rss_vnext_skin_query_field.py: chunk-invariant forward and gradient, invalid joints cannot receive weight, source query-index integrity, image→H→mesh→RigProgram→skin gradient, first-skin-code causal no-leak.
- CI 40 passed: https://github.com/merynz/RealSaS-OPT/actions/runs/42888538575

This **does not** assert full SkinTokens CVAE layer-wise parity or that SkinTokens pretrained weights can be inserted into the RSS codec.

## Kavya as capability benchmark, NOT a reverse-engineered source

Public https://www.kavya.io/ advertises editable 2D meshes/rigs, IK, skins/attachments, physics/secondary motion, FFD, transform/path constraints and animation. No internal source/trainer or deterministic solver code was verified. It is unwarranted to claim which stages inside Kavya are learned or procedural.

## RSS model vs Compiler owner, exact current evidence

| Responsibility | Learned proposal | Qualified deterministic owner | Current gap |
|---|---|---|---|
| Mesh topology and deformation regions | source-conditioned structural tokens + indexed geometry/affordances | Compiler binds approved geometry, rejects failed source/pose/CAAs | real source multi-pose topology generalization not measured |
| Skeleton, rest frames, DOFs, IK controls, targets, sockets | typed causal RigProgram candidate from mesh and shared H | Compiler validates parent/target, identity, frame, resources | source skeleton truth does NOT automatically teach a full control authoring program |
| Dense joint skin | FSQ token logits + exact per-vertex spatial code decoding | Compiler verifies skin identity; runtime LBS | query decoder vs dot baseline not yet compared on admitted corpus |
| ROOT, FK, IK, pole, relative joint transforms | learner selects valid controls/chains | mechanics_control_solver_v1.py already implements FABRIK, pole, subtree root/FK, LBS | full coupled hips/feet/secondary motion not product-qualified |
| Other constraints | learner proposes operator and binding | rig_constraint_schema_v1.py currently implements IK, AIM, FOLLOW, COPY_TRANSFORM, STRETCH | learned vocabulary additionally contains LIMIT_ROTATION, LIMIT_TRANSLATION, PATH; UNEXECUTABLE without new typed schema and tested solver; reject, never silent fallback |
| Grip, attachments, clicks and secondary motion | propose source-supported surface, anchor, control + motion region | qualification + binding + runtime surface selection + temporal simulator | no verified product-wide surface-select and secondary-motion event bridge |
| Graphics/2D view edits | appearance/region/source rays and visibility proposals | ray compositing/drawable qualification/CAA/export | no full two-view learned mesh→drawable→runtime interlock proved |

**Compiler must solve declared mathematics, NOT hallucinate semantics.** It can run deterministic IK, DOF/constraint evaluation, lineage and source support checks. It cannot invent unseen anatomical topology, select unsupported attachment winners, recover source appearance from novel-view hallucinations, or certify an unsupported constraint by its name.

## Gaps before final freeze

1. Run exact same held-out 2-source character through skin dot baseline versus spatial point decoder; compare skin error, leak across disconnected components, held-out multi-pose surface/graphics metrics, resources and unseen morphological transfer.
2. Per-vertex ablate (H+XYZ), (+indexed normals), (+valid component restrictions), token pooling versus full J×T cross attention. Spatial skin fields can bleed across close independent components unless component evidence/constraints are provided.
3. Derive honest controllable RigProgram training targets or behavioral rewards from *qualified source* rigs/animations and intervention simulations. Do not assume the original asset has IK/pole/socket authoring truth.
4. Close discrepancy between learned constraint vocabulary and executable operators. PATH requires an actual path object and stable corresponding runtime ABI; joint limits require properly specified frames/bounds.
5. Test canonical mesh/rig/skin/graphics ProductRevision lineage and product live editor consumption. Existing Compiler unit tests are not product promotion.
6. Run behavioral proof: drag sword tip, move body support chains and plant feet, rotate 45°, repeat, preserve attachments, Graphics and export. Source-relative pose and source identity gates must PASS independently.
7. Only after architecture/code capability gates are closed, derive the final Truth Foundry notebook supervision families (qualified field masks, confidence, role/region, topology, mechanical control, skin, Graphics). Do not create truth targets that source builder cannot generate.

**Next gate:** validate neural-vs-Compiler capability vocabulary and run existing deterministic solver test suites in the same CI, then create real source holdout ablations. Keep architecture research branch unmerged until audited.
