# RSS.vNext reference archaeology — source-derived call paths, trainer access and ancestry

**Review state: partial source audit, not approved freeze.** All links below point to public implementation sources, not paper diagrams. Pin each upstream to the commit identified in [README.md](README.md). Link to source paths rather than copying third-party implementation.

## A. StdGEN actual inference DAG (2025 code, **not verified StdGEN++ code**)

```text
infer_canonicalize.py
  -> canonicalize/pipeline_canonicalize.py::CanonicalizationPipeline
  -> image encoder/image projection and UNet3DConditionModel over latent video frames
  -> canonical character image / A-pose candidate
infer_multiview.py
  -> SingleImageData(total_views=6)
  -> multiview/pipeline_multiclass.py::StableUnCLIPImg2ImgPipeline
  -> CLIP-like image embedding, VAE image latents, normal/color prompt embeddings
  -> conditional UNet denoising (40 inference steps in script)
  -> six predicted multiview RGB and normal images, three semantic appearance levels
infer_slrm.py
  -> slrm/models/lrm_mesh.py::MeshSLRM.forward_planes
  -> slrm/models/encoder/dino_wrapper.py (ViT-B image features D=768)
  -> slrm/models/decoder/transformer.py::TriplaneTransformer
  -> 16 x {cross-attention(image tokens) -> self-attention -> MLP}, D=1024, 16 heads
  -> deconv triplane 3*80*64*64
  -> slrm/models/renderer/synthesizer_mesh.py::TriplaneSynthesizer/OSGDecoder
  -> SDF / deformation / cube extraction weights / RGB / 4-way semantic
  -> slrm/models/lrm_mesh.py::get_sdf_deformation_prediction
  -> semantic-conditioned selected-level SDF, FlexiCubes geometry
infer_refine.py
  -> refine/mesh_refine.py (iterative mesh/appearance refinement)
  -> exported mesh layers
```

**Actual trust boundary:** source `SingleImageData` starts with one image; six predicted images are a distinct generative product. RSS two source observations do not turn any extra sampled image into ground truth. The 2025 tokenizer/semantic decoder hard-codes four visual material classes; specifically `blender/blender_lrm_script.py::parse_material` uses named VRM material conventions and raises on unknown names. Dataset renderer exports RGB/alpha/depth/normal/semantic and l=0,1,2 layer removal passes. This is a source-specific weak semantic annotation pipeline, not a generic morphological part interpreter.

**Lineage:** `README.md` acknowledges InstantMesh, Unique3D, Era3D, CharacterGen. `slrm/models/decoder/transformer.py` records earlier 2023 authorship; `synthesizer_mesh.py` explicitly credits EG3D and includes NVIDIA-proprietary license notice. `blender/blender_lrm_script.py` names Shap-E, Point-E and Zero123 source inspirations. Import license obligations remain review-blocking.

**Unreleased/missing:** repository provides `infer_*` scripts, public render data-list and Blender script; no verified full S-LRM or multiview training entrypoint found in public root/subtrees. No reliable way to review unpublished StdGEN++ dual-branch trainer code. Freeze must use an independent source/proof label for any feature inferred only from the 2026 paper.

Source anchors:
- https://github.com/hyz317/StdGEN/blob/main/infer_multiview.py
- https://github.com/hyz317/StdGEN/blob/main/multiview/pipeline_multiclass.py
- https://github.com/hyz317/StdGEN/blob/main/canonicalize/pipeline_canonicalize.py
- https://github.com/hyz317/StdGEN/blob/main/configs/mesh-slrm-infer.yaml
- https://github.com/hyz317/StdGEN/blob/main/slrm/models/decoder/transformer.py
- https://github.com/hyz317/StdGEN/blob/main/slrm/models/renderer/synthesizer_mesh.py
- https://github.com/hyz317/StdGEN/blob/main/slrm/models/lrm_mesh.py
- https://github.com/hyz317/StdGEN/blob/main/blender/blender_lrm_script.py

## B. SkinTokens actual graph

```text
src/data/dataset.py::RigDataset
  -> LazyAsset.load, transform.apply, TokenizeInput(joints,parents)
  -> src/tokenizer/parse.py + src/tokenizer/spec.py
  -> skeleton hierarchy token IDs
  -> sampled mesh vertices + normals
src/model/tokenrig.py::encode_mesh_cond
  -> src/model/parse_encoder.py::get_mesh_encoder
  -> src/model/michelangelo/* (Michelangelo shape encoder)
  -> learned output_proj to Qwen hidden size
src/model/skin_vae_model.py
  -> pretrained SkinFSQCVAEModel encoder -> per-bone FSQ skin tokens
src/model/tokenrig.py::prepare_llm_tokens
  -> [skeleton tokens][per-bone skin codes][EOS] with section masks
TokenRig (AutoModelForCausalLM, Qwen3-0.6B config)
  -> grammar-constrained autoregressive tokens
  -> skeleton detokenize + frozen skin VAE decode
  -> predicted dense mesh influences
```

**Source-grounded distinction:** despite high-level descriptions of a joint sequence, code physically puts *all skeleton tokens first, then skin tokens* in `prepare_llm_tokens`; do not imply token-by-token skeleton/skin alternation without checking actual sequence layout. At inference, skeleton and skin remain one conditioned causal generation sequence.

**Trainer gap proven in published classes:** `src/model/tokenrig.py::TokenRig.training_step` and `src/model/skin_vae_model.py::SkinVAEModel.training_step` each raise `NotImplementedError`. `src/model/spec.py::ModelSpec` supplies a Lightning base and data preparation; this is **not** equivalent to publication of full supervised and GRPO trainer code. Published reward definitions from README are claims about training design, not verified executable reward implementation in checked paths.

**Lineage:** UniRig predecessor; Qwen3 language model backbone; Michelangelo 3D point encoder; FSQ discrete skin autoencoder; GRPO reinforcement learning claim in README. Separate licensing/provenance checks required for source transplantation.

Source anchors:
- https://github.com/VAST-AI-Research/SkinTokens/blob/main/src/data/dataset.py
- https://github.com/VAST-AI-Research/SkinTokens/blob/main/src/model/tokenrig.py
- https://github.com/VAST-AI-Research/SkinTokens/blob/main/src/model/skin_vae_model.py
- https://github.com/VAST-AI-Research/SkinTokens/blob/main/src/model/parse_encoder.py
- https://github.com/VAST-AI-Research/SkinTokens/blob/main/src/tokenizer/spec.py

## C. AniGen actual graph and training

```text
anigen/pipelines/anigen_image_to_3d.py::from_pretrained
  -> local DINOv2 ViT-L/14-reg, DSINE normal network
  -> image + predicted normal conditioning
SS flow: anigen/models/anigen_sparse_structure_flow.py
  -> two shape/skeleton streams, 24 transformer blocks, 16 heads, width 1024
  -> learned linear information transfer for aligned token streams
  -> cross-attention adapters at configurable indices when global token counts differ
  -> sparse structure coordinates for shape and skeleton
SLat flow: anigen/models/anigen_structured_latent_flow.py
  -> 24 blocks, 16 heads; geometry 1024, vertex-skin 512, skeleton 512
  -> three-field structured latent generation
VAE: anigen/models/structured_latent_vae/anigen_base.py
  -> geometry, skeleton, skin streams in the SAME repeated processing loop
  -> configurable skin <- skeleton, skin <- geometry, skeleton <- geometry cross-attention
  -> anigen/models/structured_latent_vae/anigen_decoder.py
  -> geometry/skeleton/skin fields -> mesh / joint graph / dense influence reconstruction
Postprocess: anigen/pipelines/anigen_image_to_3d.py::run
  -> repair_skeleton_parents (may reparent to nearest exterior joint)
  -> simplify/fill holes; transfer skin barycentrically
  -> filter/smooth skin weights
  -> 100 generated mesh render views for texture baking
  -> export
```

**Trainer/dataset specific code:**
- `anigen/trainers/flow_matching/anigen_sparse_flow_matching.py::training_losses`: independent noise samples for shape and skeleton sparse latents, timestep-dependent targets, shape MSE + skeleton MSE, independently controlled joint-count dropout.
- `anigen/trainers/vae/anigen_skin_ae.py`: triangle area-weighted barycentric skin samples, skin KL divergence, optional L1 and L2.
- `anigen/trainers/basic.py`: DDP, optimizer, EMA, checkpoint load/save, batch splitting and gradient controls.
- `configs/ss_flow_duet.json`, `configs/slat_flow_auto.json`, `configs/anigen_skin_ae.json`: concrete parameter/memory/trainer choices; reference hyperparameters **not** automatically suitable for RSS.
- `anigen/datasets/anigen_sparse_feat2skeleton.py`: `voxels/*_skeleton.ply`, sparse features NPZ, renders and camera transforms, metadata qualification using `is_bad_skeleton`/`is_bad_skin`.
- `README.md` directs users to TRELLIS preprocessing and released sample dataset; a complete source-corpus truth builder is not proven solely by publicly available dataset loaders.

**Postprocess caveat:** source `repair_skeleton_parents` may silently change model topology for visualization compatibility. RSS Compiler cannot treat this as source truth or qualified structural correctness. The 100 rendered texture views are **model self-renders**, not new observed source photographs.

**Lineage:** AniGen acknowledges Microsoft's TRELLIS; includes separately licensed NVIDIA-derived cuBVH and FlexiCubes. `THIRD_PARTY_LICENSES.md` flags a non-commercial/research-only NVIDIA component; hard commercial-use barrier to copying that component without replacement or independent license clearance.

Source anchors:
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/pipelines/anigen_image_to_3d.py
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/models/anigen_sparse_structure_flow.py
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/models/anigen_structured_latent_flow.py
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/models/structured_latent_vae/anigen_base.py
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/models/structured_latent_vae/skin_models.py
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/trainers/flow_matching/anigen_sparse_flow_matching.py
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/trainers/vae/anigen_skin_ae.py
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/datasets/anigen_sparse_feat2skeleton.py
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/utils/skin_utils.py

## D. Implications for RSS.vNext before code freeze (hypotheses, NOT decisions)

1. **Two source views and auxiliary hypotheses:** Both front/back go into the observed token bank; learned multi-view construction may create 6/8 *predicted* view tokens or RGB/normal images. Explicit independent-source authority and ablations required. A hallucinated back is not equivalent to supplied back.
2. **Shared memory without blind compression:** Compare explicit typed carrier + point-queryable latent features against blockwise cross-attention from AniGen and causal rig/skin state from SkinTokens. Prove that each downstream module accesses necessary context, not merely a 7D or 17D detached reduction.
3. **Structural meaning vs semantic class:** Generic regions, parts, attachments and drawables must not use StdGEN's VRM material-name ontology as mechanical labels. Structural awareness should be learned from geometry, animation, source contacts, boundaries and intervention consequences.
4. **Deformation-aware rest mesh:** G3/G3B and source pose-truth can rank/qualify candidates, but stress response does not backpropagate through a hard face-token decision. Freeze only after defining continuous/discrete target paths and adversarial deformation court.
5. **Control program lowering:** Neural expert predicts joint frames, parent graph, effectors, control affordances and qualified attachment candidates; Compiler builds *only standard solver primitives supported by declared evidence*. Clear fail/unknown behavior. No arbitrary reparent/weight repair copied from AniGen.
6. **Training availability:** StdGEN++ complete code unavailable/unknown; StdGEN and SkinTokens trainer code gaps must be filled by our independently derived, tested training specification rather than invented upstream claims.

**Exit condition:** Do not mark this as complete before the module-by-module remaining audit, capacity/cross-attention A/B plan, legality tests, and exact source/teacher leakage proof are closed.
