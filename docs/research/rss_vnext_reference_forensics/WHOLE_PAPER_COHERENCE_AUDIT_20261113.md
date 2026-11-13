# RSS.vNext — Whole-paper training & representation coherence audit (2026-11-13)

**Status:** research-only, architecture NOT frozen, no trained/unseen/product PASS. **Source base:** `merynz/RealSaS-OPT@40aa72d10a9ddc575be9e490f77ae0a65b8b3498`; feature research branch `feat/stdgen-foundation-source-graphics-20261113` / PR #484. This file does not authorize model/schema/runtime modification.

## 0. Question and evidence grades

**Question:** How can the two-original-view, one-checkpoint RSS.vNext student learn geometry + semantics + graphics + topology + articulation + skin as a causally coherent, editable puppet, without importing task-specific assumptions that only worked in another method?

Labels: **PAPER** means explicitly asserted/equated in published methodology; **SOURCE** means found in released code (not necessarily trained); **RSS** means present in our source; **HYPOTHESIS** needs an ablation; **BLOCKED** needs runtime/training proof. Paper diagrams, a module name, a checkpoint receipt, and trained behaviour are different evidence grades.

Two original FRONT/BACK RGBA and cameras are the only observed product conditions; manufactured 6/8 views are **PREDICTED** hypotheses, not new source truth. Source asset meshes/poses may be independent *training teachers* and must never enter student inference.

## 1. The papers are coherent training systems, not interchangeable blocks

### StdGEN (CVPR 2025) — geometry/semantic/appearance, STAGED over one scene

- **PAPER:** one-photo canonicalization -> generated six-view RGB/normal -> ViT/Transformer triplane S-LRM; a common 3D field decodes geometry/density, color, semantic probabilities. Semantic-equivalent extraction changes which surfaces are exposed, not merely labels a finished mesh. Their four semantic materials are *not* universal mechanical parts.
- **Training order:** pre-trained InstantNeRF plus attention LoRA/semantic decoder; Stage 1 NeRF image/mask/semantic on exterior; Stage 2 outside-in selectively peeled RGB/semantic/mask under the same field; Stage 3 mesh-based multi-layer high-resolution training with semantic-conditioned SDF/FlexiCubes; then iterative geometry and color refinement. The semantic-equivalent SDF follows Eq. 3, Stage 2 transmittance Eqs. 6–9, Stage 3 Eq. 10. **Not** one all-at-once loss with arbitrary scalar weights.
- **RSS:** `models/rss_vnext/stdgen_slrm_training_driver_v1.py`, `stdgen_joint_slrm_render_v1.py`, `stdgen_flexicubes_mesh_render_v1.py` encode the staged losses/mesh rendering, but pretrained training + GPU real gradient + full source corpus + end-to-end refinement are not proven. PR #484 adds *separate* native InstantNeRF candidate and GPU/appearance candidates, **not** an admitted full product foundation.
- **Blocking mismatch:** published ViT-B/16 + 16x1024 S-LRM + 80-channel triplanes vs RSS DINOv2-S/14 (384 tokens) + 6x512 IRIS + 96-channel triplanes. These are not tensor-compatible checkpoints. A learned transfer, full-blown donor initialization, or exact architectural change requires a separate controlled engineering decision and tests; silently `load_state_dict(strict=False)` is forbidden.
- **SOURCE caveat:** public StdGEN inference/render/refine is inspectable; its complete original trainer is not established in the examined release. StdGEN++ distinct trainer is not source-verified; distinguish literature-only detail from code.

### AniGen (SIGGRAPH/TOG 2026) — shape/skeleton/skin over SHARED DOMAIN but DISTINCT SUPPORTS

- **PAPER:** `S^3` has surface shape voxels `V`, *interior bone-intersecting/dilated* skeleton voxels `V_sk`, and dual skin fields: per-surface-vertex and per-joint features. Skeleton field votes local joint/parent offsets; confidence decays at nearest-joint Voronoi ambiguity. Joint-number-independent SkinAE maps joint embeddings and vertex embeddings to pairwise skin weights, normalized across joints.
- **Training dependency:** train SkinAE on source mesh/skin first, **freeze it**; train sparse occupancy and structured latent autoencoders with geometry/skin/joint confidence objectives (SkinAE embeds teacher skin); then Stage I dual-stream sparse structure flow (surface vs skeleton voxels), Stage II structured latent flow (geometry+surface skin branch, skeleton branch). Bidirectional communication is inside sparse Transformer stages; shape and skeleton are not forced to use identical voxel sets. Learned flows act over separately noised latents and image conditions.
- **SOURCE:** released `anigen/models/structured_latent_vae/anigen_base.py` exposes geometry/skin/skeleton in interacting blocks with typed cross attention; `skin_models.py` encodes relative joint/parent features and tree-aware skin; `trainers/vae/anigen_skin_ae.py` samples face-area-weighted barycentric GT skin; `trainers/flow_matching/anigen_sparse_flow_matching.py` optimizes shape and skeleton velocity terms. Published pipeline also includes geometry/parent/skin postprocessing: these corrections are **not** proof that upstream neural predictions were correct.
- **RSS comparison:** current IRIS `H(x)` and TESSA/AXIS cross-read common memory, but it remains **UNPROVEN** that the student provides a sufficiently dense independent interior skeleton support, or that confidence is weighted by correct source-ambiguous topology rather than generic uncertainty. Changing carrier to AniGen's sparse two-support system is **not approved**; instrument and ablate memory access first.

### SkinTokens (ACM TOG 2026) — rigging becomes a code problem, then causal prediction

- **PAPER:** first learn an FSQ-CVAE conditioned on mesh geometry to encode intrinsically sparse per-bone vertex weights, with nested dropout, positive-support importance sampling, BCE + Dice + small MSE; then train autoregressive TokenRig over **whole skeleton first, corresponding per-bone skin codes after**; then optional GRPO on decoded rig rewards (joint coverage, bone containment, skin coverage/sparsity, random-pose deformation smoothness).
- **PAPER reference capacity:** paper states FSQ levels `[8,8,8,5,5,5]` = 64,000 codes, up to 32 skin tokens and Qwen3-0.6B AR design. Those are *reference-specific*, not mandates for RSS. A token codec must be learned *before* its IDs become stable teacher labels for an AR model.
- **SOURCE:** `VAST-AI-Research/SkinTokens/src/model/tokenrig.py` implements causal grammar/generation but its published `training_step` raises `NotImplementedError`; `src/model/skin_vae_model.py` likewise. Described GRPO cannot be treated as independently runnable/open training code. No copying undeclared model code/weights into BUSTED.
- **RSS:** `models/mira/skin_fsq_codec_v6.py` has an FSQ encoder/decoder; `models/mira/product_training_v6.py` already implements a source-qualified **codec-first** objective. AXIS product teacher training already requires a pinned codec hash. Current product research config uses `(5,5,5,5)`, four codes per joint, vocab 625, and has both expected soft codes in training and discrete runtime decode. This is a *different representation*, not evidence of upstream SkinTokens parity.
- **BLOCKED:** measured sparse active-influence reconstruction (not just all-vertex MSE), held-out morphological sparsity, KL/BCE/Dice ablation, exact generated token vs hard decode parity, rig–skin random-pose deformation. GRPO/volume containment must not penalize valid open/disconnected/unknown-solid characters using invented inside/outside ground truth.

### ArtMesh (2026) — TWO-STATE rigid articulation/mesh-topology mechanics, NOT character skinning

- **PAPER:** multi-view RGB-D and part maps for **two articulation states**; jointly reconstruct per-state mesh appearance, depth and part logits; **harden part IDs**, restricted Delaunay remesh *within rigid part boundaries* without cross-part triangles, then fix topology and fit per-part revolute/prismatic motion. Bidirectional vertex-wise same-part surface consistency plus pixel-wise rendered appearance use analytic inverse transforms. Their later articulation phase holds mesh fixed.
- **Critical source difference:** state-to-state two-view motion GT and part masks exist in its experiment; RSS's two input images are FRONT/BACK of a rest character, NOT two observed temporal motion states. Therefore ArtMesh loss may only be a source-qualified *training-time dynamic consequence* with independently supplied posed states, never an inference-time teacher.
- **Not equivalent:** ArtMesh's removal of cross-part triangles assumes piecewise rigid motion and known labels. For skinned non-rigid limbs, deleting every triangle across a semantic boundary can destroy bending continuity. A semantic part is neither an obligatory drawable nor automatically a CUT_BOUNDARY. Their material/raster model is not a BUSTED CAA provenance authority.
- **SOURCE limitation:** project site currently advertises code as 'coming soon'; paper/HTML is inspectable, code-derived implementation parity is **NOT verified**.

### Additional independent controls

- **Rigel3D (May 2026):** couples distinct surface-SLat and interior skeleton-SLat in a rig-aware autoencoder, teacher-forced AR skeleton generation cross-attending both token supports, and point/bone attention skin decoder; then two-stage SLat generation. Confirms AniGen's *separate supports with explicit cross communication* design is not an isolated idea; it is a competing formulation, NOT proof that RSS should replace TESSA with it. Paper claims training on ~225K rigged examples, which does not set RSS corpus requirements.
- **RigMo (Jan 2026):** uses **observed temporal mesh sequences** to self-supervise explicit Gaussian bones, skin, SE(3) and a separate motion latent. Useful for deformation-consequence supervision/rig-motion cycle checks. Not a two-static-view texture-faithful puppet compiler, nor evidence that Gaussian bones can represent every editable DCC joint/control and draw-order constraint.

## 2. RSS.vNext source-to-model mapping (current source, not future promise)

| Concern | Existing RSS asset | Research obligation |
|---|---|---|
| Observed conditions | `unified_character_model.py::RSSvNextFrozenDINOv2`, `encode_observations` (2 RGBA only) | maintain two-view source firewall and absolute pixel evidence |
| Shared source 3D | `models/iris/vnext/scene_core_v1.py` triplanes + queryable `H(x)` | quantify deep internal skeleton-access vs surface-only training |
| Early source semantics / presentation roles | `early_semantic_field.py`, `presentation_role_head`, `semantic_structure.py` | keep material/region/part/drawable/attachment contracts disjoint |
| Topology / deformation awareness | `models/tessa/v31`, `motion_aware_topology_learning.py` | prove discrete token consequence training uses source-matched source skin and pose masks |
| Joint program | `models/axis/rig_program_model_v6.py` grammar includes joints, frames, controls, chains, constraints, anchors, skin codes | joint count/parent labels are not full IK/control/attachment semantics |
| Skin codec and decoder | `models/mira/skin_fsq_codec_v6.py`, `models/rss_vnext/skin_query_field.py` | pretrained codec quality; hard/expected-code equivalence; dense vertex identity; joint mask |
| Product authority | `inference_pipeline_v1.py`, `inference_evidence_export_v1.py`, Compiler typed authority | one checkpoint producer, embedded executable MIRA child; no claimed canonical asset from neural hypothesis |
| Appearance source | source first-hit rays, CAA, PR #484 GPU 3D layer and generated-view color candidate | real source pixels override generated RGB; remesh invalidates color/skin address |

**Potential version drift to check, not a defect verdict:** older standalone AXIS/MIRA training/release documents may refer to separate workers and 47D RiggingSurface, while the intended RSS.vNext product producer is one unified checkpoint. Resolve *which training module and release boundary is actually live* before invoking a separate checkpoint contract as product architecture.

## 3. Decisions — explicit status

**KEEP (source-justified):** two observed views; source/prediction/teacher authority separation; spatially queryable common H; source roles distinct from body mechanics and drawables; TESSA discrete structural proposal plus dynamic source-relative candidate court; AXIS causal structural program; MIRA pretrained codec dependency; bounded deterministic Compiler.

**HYPOTHESIS (experiment, no code freeze):**
1. Augment internal skeleton voxel support or cross-query to interior H; compare to present scene carrier with fixed parameter/compute budget and *same* source teacher.
2. Compare existing MIRA FSQ dot-product vs point-query attention with real sparsity, hard-token and deformation courts; do not replace trained codec merely because another publication uses a larger codebook.
3. Compare StdGEN pretrained triplanes as frozen donor/proposal vs RSS trained-from-scratch scene, with explicit width transfer and gradient ownership; strict official weight load and visible/hidden source verification first.
4. Evaluate motion-consistency for TESSA candidate ranking and AXIS/MIRA skin deformation without leaking teacher pose in forward. ArtMesh's rigid-part cut policy is a separate ablation only when independently proven safe.
5. Compare jointly trained core vs phased frozen-codec/scene warm starts under equal supervision. A *single released checkpoint* is compatible with staged training; it does not imply training all losses on step one.

**REJECT WITHOUT NEW PROOF:** classifying semantic labels as joints/drawables; cut-all-label-boundaries; accepting generated views/depth as observed 3D labels; using mean-z instead of first-hit; silent remesh without correspondence to surface/skin/attachments/appearance; inventing hidden source pixels; attaching noncommercial NVIDIA-proprietary code to commercial product; claiming tests of mock GPU or synthetic teacher equal native/learned evidence.

## 4. Bounded evidence plan — before new neural architecture edits

1. **Checkpoint provenance court:** native InstantNeRF keys/params/config exact vs released ckpt; local real load/forward/backward; no license-incompatible runtime. Then compare raw triplane projections and whether any principled adapter into RSS H preserves RGB/geometry/semantic training value. No `strict=False` promotion.
2. **Codec-first court:** source-real MIRA weight reconstruction (active influence Dice/BCE, zero-weight precision, compactness); supervise AXIS only with *frozen, reproducible* code IDs. Compare expected-code vs discrete-code whole-pose/reopen consequences; never assign scientific success from shape tests.
3. **Internal-support court:** sampled source bone paths vs current `H(x)` queries, geometry/rig localization and gradients; same two-view student observations, source interior labels only through teacher loss. Compare separate interior memory and the existing carrier under identical budget.
4. **Mesh motion + renderer court:** exact teacher skin protocol for topology candidate comparisons; native CUDA raster gradients, source z-buffer parity, seam/strain/foldover/collision/occlusion consequences; no source pixel or appearance authority after remesh without explicit transfer.
5. **Training scheduler freeze after results:** freeze/unfreeze targets, independent gradients, loss masks, teacher forcing/free-run consistency, checkpoint ownership/resume, max VRAM, per-head eligibility. No 10k+ corpus/unseen campaign discussion in this research pass.

### Cross-head measurement (required, not just task accuracy)

For each teacher family, report `grad(norm,H)` and per-expert gradient conflicts; track whether source semantic, geometry, pose and skin losses can change the *same* scene evidence without destroying identity. Use architectural baselines that differ by exactly one mechanism. Record runtime memory/latency and fail-closed unknown coverage. Strong static mesh quality alone never proves editable puppet.

## 5. References (primary; supplementary source-code evidence where released)

- StdGEN CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/papers/He_StdGEN_Semantic-Decomposed_3D_Character_Generation_from_Single_Images_CVPR_2025_paper.pdf ; inspected upstream https://github.com/hyz317/StdGEN/tree/f1b0c822689dc2539c774bb73090a67c66e2e766
- AniGen 2026: https://arxiv.org/abs/2604.08746 ; released implementation https://github.com/VAST-AI-Research/AniGen/tree/c49db3d6b466537a02ccf2286688903d77af7e4f
- SkinTokens 2026: https://arxiv.org/abs/2602.04805 ; public implementation https://github.com/VAST-AI-Research/SkinTokens/tree/273b691d35989d71cd17ff2895fdc735097b92d1
- ArtMesh 2026: https://arxiv.org/html/2605.16582v1 ; project https://artmesh2026.github.io/artmesh-site/
- Rigel3D 2026: https://arxiv.org/abs/2605.13129
- RigMo 2026: https://arxiv.org/abs/2601.06378
- Existing RSS research index: `docs/research/rss_vnext_reference_forensics/README.md` and `RSS_VNEXT_PAPER_TO_COMPILER_DECISIONS_20261112.md`.

**Decision authority:** this is a source-backed candidate decision register, not an endorsement of blending all source models, a training data specification, an architecture freeze, or a product qualification.
