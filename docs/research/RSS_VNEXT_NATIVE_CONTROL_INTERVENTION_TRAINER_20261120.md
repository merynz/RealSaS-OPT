# RSS.vNext — Source-native one-control intervention and M* response trainer (2026-11-20)

**Status:** optional *teacher-forced* behavioral loss integrated in the active product-seeded trainer; exact source receipts and synthetic negative controls. **NOT** demonstrated automatic wing rigging, correct cape sockets, physically accurate cloth, inferred material properties, learned unseen control semantics, or ProductProof.

## Why StdGEN + TailorRig still need BUSTED learning

**StdGEN** (CVPR 2025) reconstructs layered 3D surfaces (body/garment/hair) with shared SDF/semantic/RGB evidence. It does not by itself define a dynamic socket or an auxiliary wing/cape rig. **TailorRig** (2026) learns structure for auxiliary bone chains, and suppresses chain-to-bone weight bleeding for garment-aware humanoid rigging. Its approach is conditioned on a 3D mesh, a template skeleton and category labels. It neither establishes arbitrary wing anatomy from two images nor independently teaches the source-observed motion of a source-specific cape pin.

**BUSTED already has:** one H(x), single TESSA final indexed M*, AXIS causal chain/PART/CONSTRAINT/ANCHOR program, native face attachment loss, MIRA chain-aware skin and exact-vertex posed-surface loss, source-certified OPEN pins and declared XPBD domain/compiler, source-native M* body/garment clearance and posed 2D Graphics. **The major missing proof:** the student rig reproduces SOURCE-VERIFIED *intervention-specific* motion without dragging inactive anatomy or skinning everything to the nearest torso joint.

## A generic intervention consequence, not another head

On an independently recorded **single control event** from the native rig, observe the before/after source-native poses. For example:
- a **wing-branch-only** control moves a wing's native M* vertices while unaffected torso vertices remain stationary;
- a **host-only** control can move an attached region while independently verified other regions remain unchanged. This is a *kinematic response witness*, not a source pin/cloth dynamics oracle.

Qualified source-asset fields (loss-only) are:
- `source_control_event_sha256` = independently authored source control event, exactly one changed control, recorded by a trusted source ledger; `source_scope` is `SOURCE_HOST_ONLY` or `SOURCE_BRANCH_ONLY`;
- `baseline_pose_index` and later `intervention_pose_index`, indexing the **same** qualified `source_candidate_posed [1,P,V,3]`, `source_pose_mask [1,P,V]`, and `source_pose_matrices [1,P,J,4,4]` used by AXIS/MIRA;
- `changed_skin_joint_matrix_mask [J]`, checked against exact source joint matrix differences. World-joint motion alone is not evidence that **one** artist control was actuated; the independent trusted ledger must separately certify that;
- `responding_vertex_ids [K]` and `protected_vertex_ids [Q]` in the **same final TESSA M*** address namespace, with native face/barycentric transport. They must not overlap, be student-nearest predictions, or be guessed from names/SLR classes;
- source landmark checks: measured responding vertices actually move, source protected vertices remain stationary within a declared normalized tolerance, and both poses have qualified per-vertex support.

Trusted launcher receipts must SHA-pin **all** arrays and control-event identity, original source RGBA and generated-view checkpoint identity, dataset rights, exact M* topology address, native pose transport receipt, and source-author-verified control witness. UNKNOWN does not become a zero-motion negative label.

### Objective

With `\Delta X = X_{after}-X_{before}`, scale `s` as independently source-normalized canonical character span, and sets `D` (responding), `Q` (protected):

```text
L_driven    = SmoothL1(DeltaStudent[D]/s, DeltaSource[D]/s)
L_protected = SmoothL1(DeltaStudent[Q]/s, DeltaSource[Q]/s)
L_contrast  = SmoothL1(mean(DeltaStudent[D]-DeltaStudent[Q])/s,
                       mean(DeltaSource[D]-DeltaSource[Q])/s)
L_bleed     = mean((DeltaStudent[Q]/s)^2)
L_control   = L_driven + L_protected + L_contrast + .25 L_bleed
```

Only **continuous** M*, AXIS/MIRA and shared-H differentiable paths receive these gradients. Discrete causal tokens, face IDs and FSQ argmax still require their separately qualified losses. The fixed 0.25 is a declared research candidate coefficient, not a measured material/solver constant.

Under a matching source, this loss vanishes. If the wing does not unfold, `L_driven` detects it; if the body is dragged, `L_bleed` and `L_protected` detect it. This distinguishes functional articulation from merely fitting a plausible rest-pose mesh.

## Trainer integration and source handoff

- New source type: `SourceNativeControlInterventionTeacherV1` in `models/rss_vnext/product_native_control_intervention_v1.py`.
- Enabled only if `product_native_control_intervention_weight>0`, product seeded mesh-matched AXIS+MIRA, source native skin and posed-surface objectives, and immutable independent receipt registry are present.
- Default 0; missing qualifying event in a sparsely qualified corpus stays UNKNOWN. Same M* rest/posed, no control teacher ever enters customer inference H(x).
- New eighth separate gradient family `CONTROL_INTERVENTION`; never hide cross-task conflict inside Geometry or Mechanics. Source reception and registered source family appear in `source_supervision_requirements_v1()`.
- Checkpoint objective SHA and trusted registry digest pin teacher provenance; stage must be native posed or full shared-H training. No synthetic optimizer-quality or unseen-generalization promotion.

## Compiler/runtime proof still absent

The neural program's predicted ANCHOR/CHAIN/PART semantics still require the Compiler's exact source-face socket proof and runnable Mechanics constraints. For an open cape, only source-native pin/hem/host-only intervention plus Compiler M*->canonical replay can authorize an XPBD surface domain. For a wing, distinguish independently articulable spars, MIRA-weighted membrane, source-qualified branching PART and optional declared membrane physics.

A source-native pose response is NOT contact, pin, collision or XPBD-stiffness truth. The source-native moving first-hit M* graphics loss does NOT prove the end-user's post-Physics RGB/alpha/visibility or authored texture. Final admission needs held-out **unseen morphology**, independently repeated single-control interventions, source-independent save/edit/reopen and native runtime 2D/physics parity.

### Research references

- [StdGEN (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/He_StdGEN_Semantic-Decomposed_3D_Character_Generation_from_Single_Images_CVPR_2025_paper.html) — layered semantic 3D reconstruction, not learned sockets.
- [TailorRig arXiv 2611.02902v2](https://arxiv.org/abs/2611.02902) — human-garment auxiliary chains, hierarchical chain skin and actual deformation eval; not universal wing/cloth motion.
- [Automated Accessory Rigs for Layered 2D Character Illustrations (UIST 2021)](https://jingyi.me/research/autorigaccessories/) — practical attachment-point, coincident-boundary, overlap and occlusion constraints. These do not supply BUSTED's 3D pin truth but motivate keeping visual and mechanical relations distinct.
- BUSTED `docs/rss_vnext/seeded_attachment_product_contract.md`, `docs/rss_vnext/source_qualified_open_surface_pin_xpbd_contract.md`, `docs/rss_vnext/causal_axis_mechanical_part_teacher_contract.md`.

**Status is an executable research candidate and CI source-safety test, not real-source feature performance.**
