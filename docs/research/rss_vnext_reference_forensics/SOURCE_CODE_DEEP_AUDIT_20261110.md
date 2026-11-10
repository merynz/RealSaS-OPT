# RSS.vNext upstream source deep audit — 2026-11-10

**State:** intermediate source inspection; no approved architecture freeze. Evidence comes from pinned GitHub trees and line-specific original code, not architectural depictions from papers.

## Completeness of public source trees

Pinned commits are in README. The GitHub recursive tree API returned **untruncated** trees:
- `hyz317/StdGEN`: 82 blobs. Inference entrypoints `infer_canonicalize.py`, `infer_multiview.py`, `infer_slrm.py`, `infer_refine.py`; `slrm/utils/train_util.py` merely supplies instantiation utility; no S-LRM model training entrypoint found in this published tree. No separate StdGEN++ tree was verified.
- `VAST-AI-Research/SkinTokens`: 70 blobs; model classes, tokenizer, Blender parser and dataset source shipped; the two principal `training_step` methods have explicit `NotImplementedError`, so full supervised/GRPO train code is **not source-available in these classes**.
- `VAST-AI-Research/AniGen`: 232 blobs; five JSON training config files, `train.py`, generic training base, flow matching and VAE losses, pipeline and data loaders. The complete upstream raw corpus generation pipeline is NOT established by these assets; README directs readers to TRELLIS preprocessing and a sample dataset.

## StdGEN encoder and view constructor: actual neural calls

`slrm/models/encoder/dino_wrapper.py:22-59`
- Actual inference config uses **facebook/dino-vitb16**, 768-dimensional features; this is NOT DINOv2-L. `DinoWrapper` has a **16→768→768 camera MLP**, injecting camera embedding through the custom ViT `adaln_input`, not merely appending a view token after the encoder. Camera provenance and coordinate convention matter.
- `slrm/models/decoder/transformer.py:21-65`: `BasicTransformerBlock` does (1) image-token cross-attention, (2) triplane-token self-attention, (3) MLP. Residuals around each. Config: 16 layers, 16 heads, width 1024, 32→64 triplane resolution, 80 channels.
- `slrm/models/renderer/synthesizer_mesh.py:18-110`: independent output MLPs on **3 × 80** locally sampled triplane features: scalar SDF, 3D deformation, RGB, 4-way semantic; an additional 8-corner feature MLP emits extraction weights. Shared spatial representation does not imply shared final loss/authority.
- `slrm/models/lrm_mesh.py:144-303`: semantics modify the signed field per layer by min/max combinations before FlexiCubes. Hair/cloth handling is hard-coded, not component-generic. A fallback for all-positive/all-negative SDF explicitly edits SDF and **detaches gradients** for affected examples. This can hide a degeneracy from the optimizer; an RSS-compatible alternative must report invalid/reject rather than silently label reconstructed sign as source truth.
- `infer_multiview.py:42-102,165-224,264-277`: six generated view hypotheses from a single observed image; normal and color prompt embeddings and 40 denoising steps. Only `num_views=6` is supported by this exact public runner. `multiview/pipeline_multiclass.py` uses image encoder, VAE image latents and conditional U-Net denoising. Do not misread six generated views as six source views.
- `blender/blender_lrm_script.py:990-1085,1230-1325`: named-VRM material grouping, three render-level variants, RGB/alpha/depth/normal/semantic; unsupported materials raise. No generic attachment or skeletal ownership annotation proven.
- **Exact source boundary for StdGEN++:** `arXiv:2601.07660` describes Dual-Branch S-LRM, but that code and its full trainer are not verified in public repositories. The 2025 82-file repo must NOT serve as claimed implementation of this 2026 architecture.

## SkinTokens causal rig / skin generation: actual implementation

`src/model/tokenrig.py:63-118,195-279,424-524,536-591`
- A Michelangelo mesh encoder reads `vertices + normals` (6 per sampled point). Encoded mesh latents are projected into the Qwen hidden dimension via `Linear + RMSNorm`, and are prepended as **prefix embeddings** to the autoregressive sequence at generation time. There is NO explicit per-layer cross-attention to a persistent RSS-style 3D carrier in this implementation.
- `prepare_llm_tokens` places all skeleton grammar tokens, then all per-joint FSQ skin tokens, then EOS; grammar mask constrains section vocabulary and required number of skin tokens. A skeleton-EOS token acts as a switch before final-generation EOS.
- Skin decoding: predicted joint hierarchy determines joint count J; decode J×tokens_per_skin FSQ code indices, concatenate per-joint results to form dense [vertices,J] skin influence table. Incorrect skeleton count disrupts decoding; this is a coupled bottleneck and needs explicit invalid-state handling in RSS.
- `TokenRig.generate` asserts **no batch**. The pretrained frozen skin VAE gets separate condition latents from 6D vertex/normal samples for weight decoding; those are distinct from the Michelangelo→Qwen prefix, potentially a representation consistency issue when integrating.
- `src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py:26-108,247-303`: default codec architecture: `width_encoder=512`, `num_layers_encoder=8`, `width_decoder=1024`, `num_layers_decoder=16`, `num_attention_heads=8`, `latent_channels=64`. The actual checkpoint/config overrides need inspection before claiming these as published trained hyperparameters. Input point conditions embed XYZ; encoder quantizes with FSQ and the decoder combines `z` with learned condition latents to reconstruct influence values at sampled points.
- **Trainer evidence gap is definitive for checked classes**, not necessarily all code/assets elsewhere: `src/model/tokenrig.py:151` and `src/model/skin_vae_model.py:140` define `training_step` as `raise NotImplementedError()`. A Lightning base/data loader exists, but full model losses and GRPO/reward implementers were not in the inspected full-tree file listing.
- Training data must not be equated to source truth unless original rig/skin correspondence is qualified; TokenRig reads native rig data and transforms/samples it.

## AniGen cross-attention: code SUPPORT versus RELEASED CONFIG USE

**Major discrepancy requiring RSS caution:** `anigen/models/structured_latent_vae/anigen_base.py:90-257` defines three streams (geo/skl/skin) and three toggles `skin_cross_from_geo`, `skl_cross_from_geo`, `skin_skl_cross`; the code supports `SparseTransformerMultiContextCrossBlock` for exchange on each repeated decoder/encoder block. HOWEVER, `configs/slat_dae.json:46-48` (encoder) and `:100-102` (decoder) set all three flags to **false**. Therefore the published slat_dae config DOES NOT train the apparent fully cross-coupled implementation. Do not cite cross-attention as experimentally proven by that particular config.

Distinct mechanism: `anigen/models/anigen_sparse_structure_flow.py:200-252,390-486` with `configs/ss_flow_duet.json` **does enable** bidirectional shape↔skeleton stream adapters. Per-block **learned linear** exchange when tensor domains align; optional learned **cross-attention every four blocks** when global token layout requires it. The code's use of `cross_adapter_every` does not imply active cross-attention in the default 16³ SS config.

`anigen/models/anigen_structured_latent_flow.py:503-548` loops through three streams every block; geometry block attends image condition, skin block attends skeleton and receives a geometry adapter, skeleton block attends skin. This is a distinct **three-stream coupled SLat flow** in active `configs/slat_flow_auto.json` (24 blocks, 16 heads; geo width 1024, skin/skl widths 512), even though the trained VAE's `slat_dae` flags disabled optional VAE-level cross exchange. **Do not conflate the flow model with its decoder.**

`anigen/models/structured_latent_vae/skin_models.py:83-201`: `TreeTransformerSkinDecoder` computes per-vertex and per-joint latents, parent-aware joint features, optional vertex→joint cross blocks, then [B,V,J] logits from `einsum` and a learned temperature followed by softmax. Crucially the actually released `configs/slat_dae.json` sets `vert_cross_blocks_num=0`: its vertex pathway uses feed-forward blocks, not the optional vertex cross-attention. This is another source/config distinction.

`anigen/trainers/flow_matching/anigen_sparse_flow_matching.py:103-184`: two separately noised sparse feature fields, MSE of velocity per geometry and skeleton streams, configurable joint-count conditioning/dropout. The VAE's `anigen/trainers/vae/anigen_slat_mesh_vae.py:251-539` has a richer code-level training contract:
- Ground-truth meshes are rendered for mask/depth/normals and compared to generated mesh renders.
- Optional TSDF regularization, silhouette L1, depth L1/SmoothL1, normal L1+SSIM+LPIPS, optional color.
- Skeleton joint/parent losses formed through nearest-neighbor matching to GT joints and parent positions, with confidence weighting and **potentially ambiguous assignment**.
- Skin predictions reconstructed against GT weights, GT mesh→pred mesh correspondence via optional BVH barycentric or nearest vertex, KL and latent feature L2. `is_bad_skin` cases explicitly suppress true skin loss using zero-gradient terms; useful example of **head-specific eligibility**, not whole-asset rejection.
- Losses `lambda_joints=10`, `lambda_parents=5`, `lambda_skin_kl=10`, etc. in `configs/slat_dae.json` are specific to upstream data; no RSS retuning or training validation claimed.

`anigen/pipelines/anigen_image_to_3d.py:135-170` conditions sparse structure with estimated **normal** features and structured latent with source **RGB** features. The normal is an auxiliary prediction from DSINE, not a separately observed second camera. `:332-495` repairs parents, simplifies mesh, transfers weights, smooths/filter influences and generates **100 synthetic renders of its own generated mesh** to bake texture. None are evidence of additional source observations.

## RSS.vNext's existing deformation tools—do not reinvent or overclaim

`models/tessa/v1/mechanical_objective_v1.py:29-159` already provides a **differentiable same-face-set** geometric surrogate:
- rest and posed vertex positions; per-face 3×2 deformation Jacobians
- 2 singular values, area ratio, condition number, max edge stretch and posed normal
- weighted thresholds and optional source-relative Jacobian/normal supervision; orientation is compared with **source pose**, not blindly to rest-face winding (real joints can rotate >90°)
- gradient propagates through continuous vertex positions with a **fixed face indexing tensor**, not through a discrete proposed topology sequence.

`compiler/realsas_compiler_core/mesh/deformation_stress_v2.py` is a **G3 local-frame numerical micro-stress court**, using independent ±10°-class joint frame perturbations, explicitly NOT a full motion capability proof. `tools/audit_knight_production_g3_motion_court_v1.py` and Stage35 playbacks add stronger actual-clip evidence. Qualification requires exactly-bound source mesh, skeleton, and skin, with evidence hashes and fail-closed authority checks in `compiler/realsas_compiler_core/qualified_mesh_v2.py`.

**Implication:** RSS should reuse `triangle_deformation_metrics_v1` and `mechanical_consequence_loss_v1` as *starting continuous losses*, while maintaining hard G3/G3B/Stage35 judges and explicitly researching discrete topology training (candidate ranking / relaxed connectivity / structural teacher and posed consequences). Claiming these solve animation-ready topology would be false.

## Exact model/Compiler ownership risks exposed by source

1. A source-specific named material tag (StdGEN) is not generic region/attachment identity.
2. A nearest-neighbor joint hierarchy repair (AniGen) can create new relationships not observed in source; control lowering can validate/fail or construct standard solver components ONLY from supported evidence, not silently reparent.
3. An FSQ-coded skin sequence (SkinTokens) depends on earlier decoded joint count; the runtime must handle invalid or uncertain skeletons without propagating arbitrary weights as certainty.
4. Re-simplifying or UV-splitting a mesh changes vertex addressing; exact correspondence, skin transfer proof and invalidate-on-drift must be explicit.
5. Three views must never be represented by three **observed** view masks if only front/back were given. Internally 6/8 constructed views are **predicted hypotheses** with provenance, uncertainty and source reprojection gates.
6. Shared-character state cannot be defined solely as writing several outputs to one Python dictionary or serializing with the same filename. Need specific trainable fusion and explicit preserve/query paths.

## Not yet verified

- StdGEN++ full distinct code/trainer, lineage and licence: **missing public source**.
- Source S-LRM reconstruction trainer: **not in verified 82 files**.
- TokenRig supervised and GRPO train/reward routines: **not in verified principal published model classes**.
- Exact runnable checkpoints / GPU throughput / optimizer reproducibility for external sources.
- RSS two-view to internally generated 6/8 view mechanism and no-leak evidence.
- Unified RSS tensor shapes, block width and cross-attention placements, activation budget.
- Generic semantic/attachment head learning, discrete topology gradient path and full-control Compiler lowering.
- End-to-end source-correct editable puppet quality; 8k eligible dataset not available.
