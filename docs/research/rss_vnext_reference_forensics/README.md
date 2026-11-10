# RSS.vNext — External Source-Code Forensics (IN PROGRESS)

**Scope:** Read-only upstream code archaeology and architecture candidate evidence. **NOT AN ARCHITECTURE FREEZE, NOT TRAINING ADMISSION, NOT A MODEL IMPLEMENTATION, NOT A TRUTH SCHEMA.**
**BUSTED product input contract:** exactly **two original views (FRONT, BACK)**. Internal auxiliary 6/8-view construction is a legitimate *hypothesis-producing* operation, never an additional observed source or privileged teacher input.
**Repo base:** `merynz/RealSaS-OPT@9f9f53674777e961488262182260f039ea06ed77`. This review branch changes neither canonical main nor live Truth Foundry / corpus jobs.

## 1. Pinned upstream provenance

| Evidence family | Public upstream source | Observed commit | Code evidence status |
| --- | --- | --- | --- |
| StdGEN (CVPR 2025) | [hyz317/StdGEN](https://github.com/hyz317/StdGEN) | `f1b0c822689dc2539c774bb73090a67c66e2e766` | Inference, model and Blender rendering source visible; full trainer **not identified in public tree** |
| StdGEN++ (2026 extension) | Extended study `arXiv:2601.07660`; repository **not confirmed** | **UNVERIFIED** | Distinct dual-branch implementation/trainer **NOT source-verified**. Do **not** relabel StdGEN code as StdGEN++. |
| SkinTokens / TokenRig | [VAST-AI-Research/SkinTokens](https://github.com/VAST-AI-Research/SkinTokens) | `273b691d35989d71cd17ff2895fdc735097b92d1` | Inference/tokenizer/dataset/FSQ VAE sources available. Published `TokenRig.training_step` and `SkinVAEModel.training_step` raise `NotImplementedError`: complete training recipe **not available from these classes** |
| AniGen | [VAST-AI-Research/AniGen](https://github.com/VAST-AI-Research/AniGen) | `c49db3d6b466537a02ccf2286688903d77af7e4f` | Pipeline, dense/sparse models, datasets, trainer configs and loss routines available; run/weight/reproduction still **unverified** |

**Source tree/code ownership note:** GitHub files can contain third-party licensing different from repository root. StdGEN's `slrm/models/renderer/synthesizer_mesh.py` starts with `LicenseRef-NvidiaProprietary` and cites EG3D; do not transplant it without an independent license review. StdGEN credits InstantMesh, Unique3D, Era3D, CharacterGen; its Blender normalizing/rendering code cites Shap-E/Point-E/Zero123 lineage. AniGen `THIRD_PARTY_LICENSES.md` includes a **non-commercial NVIDIA source license** for one cuBVH component. Research the *mechanism*, not blindly paste code into commercial BUSTED.

## 2. StdGEN actual code path, separated from StdGEN++

1. `infer_canonicalize.py` — image-to-canonical pose generation.
2. `infer_multiview.py` — `SingleImageData.total_views=6`; diffusion produces multiview hypotheses. Six input views **are not six source observations**.
3. `infer_slrm.py` → `slrm/models/lrm_mesh.py::MeshSLRM` — `DinoWrapper` image features → `TriplaneTransformer` → `TriplaneSynthesizer`.
4. `configs/mesh-slrm-infer.yaml` actual config: ViT-B image feature **768**, triplane Transformer **16 layers, 16 heads, width 1024**, low-res **32**, high-res **64**, triplane feature width **80**, LoRA rank **128**.
5. `slrm/models/decoder/transformer.py::BasicTransformerBlock`: per-block **cross-attention → self-attention → MLP**; learned triplane query positions are deconvolved to high-res planes.
6. `slrm/models/renderer/synthesizer_mesh.py::OSGDecoder`: independently parameterized shallow MLPs for SDF, RGB, deformation, cube extraction weights, and **4-channel semantic probabilities**; all query same sampled triplane features.
7. `slrm/models/lrm_mesh.py::get_sdf_deformation_prediction`: **semantics alter level-dependent SDF extraction**, with distinct hair/cloth/body policies and differentiable FlexiCubes extraction. **This is not a generic component ontology.**
8. `infer_refine.py`/ `refine/*`: subsequent layer refinement is a separate inference process, not proof that the S-LRM generates editable mechanics.
9. `blender/blender_lrm_script.py`: render channels for RGB/alpha/depth/normal/semantic and three appearance configurations (all; no hair; no hair/cloth). The truth builder's `parse_material` uses naming rules such as `Hair`, `Face`, `CLOTH`, `SKIN`, `EYE`, colors mapped to red/blue/cyan/green. It **raises on unfamiliar materials**. This weak supervised semantic mapping **must not become universal RSS source truth**.

**Missing evidence before borrowing design:** exact S-LRM trainer and losses; source/target view sampling schedule; gradient through semantic layers; data admission/rejection protocol; augmentation and teacher-only source access. Public release documents inference code and rendering scripts, not a proven complete trainer. Distinct StdGEN++ dual-branch implementation remains unverified.

## 3. SkinTokens actual code path

1. `src/data/dataset.py::RigDataset`: loaded rig assets, transforms, sampled vertices/normals, tokenized real joints/parents; variable-sized per-asset collections become collated batches.
2. `src/tokenizer/spec.py`, `src/tokenizer/parse.py`: token grammar encodes joint hierarchy and coordinates. This is **rig structure**, not a full DCC constraint program.
3. `src/model/tokenrig.py::encode_mesh_cond`: mesh point position + normal → Michelangelo mesh encoder → latent projection.
4. `src/model/skin_vae_model.py` and `src/model/skin_vae/autoencoders/*`: FSQ skin codec for compact per-joint skin tokens; `TokenRig` loads frozen pretrained VAE.
5. `src/model/tokenrig.py::TokenRig`: Qwen3-0.6B-config causal language model, combined vocabulary **skeleton tokens + skin code tokens + EOS**; logits masking constrains valid sections/sequence lengths. One causal rig state, not independent skin reconstruction as a completely unrelated rig.
6. `prepare_llm_tokens` explicitly concatenates skeleton and per-joint skin tokens; the mesh condition is embedded as neural context. Decoder recovers dense per-vertex influences.
7. **Training-source gap:** `TokenRig.training_step` and `SkinVAEModel.training_step` are explicit `NotImplementedError` in published class code; descriptions of RL/GRPO and full optimization from README cannot be treated as an audited runnable training implementation. Inspect all shipped configs and any external training release separately.

**Transferable:** coupled causal skeleton/skin representation, explicit grammar, variable joint counts, FSQ representation, mesh conditioning and validity masks.
**Not proven transferable:** authoring handles, multi-chain IK control program, attachment ownership, editable 2D drawable graph, arbitrary-source 2-view conditioning.

## 4. AniGen actual code path

1. `anigen/pipelines/anigen_image_to_3d.py` constructs **DINOv2 ViT-L/14-register** image features and DSINE-derived normal estimates. The public pipeline starts from a single observed image.
2. First flow stage: `anigen/models/anigen_sparse_structure_flow.py::AniGenSparseStructureFlowModel`, default reference config `configs/ss_flow_duet.json`: **24 blocks, 16 heads, shape and skeleton streams width 1024**. Bidirectional adapters exist. When stream token counts differ, actual cross-attention adapters are placed at configured intervals (default `cross_adapter_every=4`); when aligned, learned linear transfer can be used. This precise conditional matters.
3. Second stage: `anigen/models/anigen_structured_latent_flow.py::AniGenSLatFlowModel`, reference config `configs/slat_flow_auto.json`: **24 blocks, 16 heads**, geometry width **1024**, vertex-skin and skeleton widths **512** each, structured sparse latent generation.
4. Shared latent decode: `anigen/models/structured_latent_vae/anigen_base.py::AniGenSparseTransformerBase` carries geometry, skeleton, skin streams in the **same blocks**, with configurable `skin_skl_cross`, `skl_cross_from_geo`, `skin_cross_from_geo` cross-attention. This is stronger evidence for a shared neural graph than merely joining artifacts with hashes.
5. `anigen/models/structured_latent_vae/skin_models.py`: skeleton-conditioned relative positional feature encoder and `TreeTransformerSkinDecoder`; variable-joint conditioning.
6. `anigen/trainers/flow_matching/anigen_sparse_flow_matching.py`: independently noised structure/skeleton latents, timestep target velocities, `MSE + MSE_skl`, CFG/dropout options. Trainer config includes AdamW, EMA, save/restart, mixed precision, adaptive gradient clipping.
7. `anigen/trainers/vae/anigen_skin_ae.py`: surface point samples are **area-weighted face selections with barycentric skin interpolation**; decoder trained with skin KL, optional L1/L2. `anigen/datasets/anigen_sparse_feat2skeleton.py` uses precomputed mesh/skeleton voxels, sparse image features, metadata filtering (`is_bad_skeleton`, `is_bad_skin`, max voxels) and render cameras.
8. **Crucial postprocessing distinction:** `anigen/pipelines/anigen_image_to_3d.py::run` invokes `repair_skeleton_parents`, mesh simplification/hole filling, barycentric weight transfer, weight filtering/smoothing, 100-view renders from **predicted generated mesh** for texture baking. `repair_skeleton_parents` can reparent a cyclic joint to a nearby outside joint. These are actual algorithmic mutations and cannot automatically be treated as proof of learned skeleton accuracy, true observed views or semantics-preserving Compiler edits.

**Transferable:** coupled geometry/skeleton/skin feature fields and cross-attention, sparse latent structure, posed/skin teacher preparation and masked variable-joint training.
**Not automatically transferable:** 2D source-appearance authority, animation-ready topology under severe nonrigid edits, complete controls or source-grounded attachment/drawable decisions.

## 5. Open RSS.vNext research decisions — do not freeze yet

- **Exactly two observed images:** FRONT/BACK in every product-mode student forward. Optional internally constructed 6/8 views must carry `PREDICTED_VIEW` authority, provenance, per-view confidence and original-source masks. Compare 2→direct 3D against 2→6 and 2→8, including 2-view consistency and hallucination sensitivity. No label/teacher extra-view leakage.
- **One neural character state:** preserve original source tensors and immutable evidence sidecar; shared scene features, structural states and mechanics/skin memory must not be replaced by anonymous low-dimensional bridges. Evaluate exact cross-attention, gradient ownership, memory and throughput rather than treating a Python class wrapper as genuine fusion.
- **Semantic/attachment:** source-observed region vs inferred region vs mechanical component vs attachment candidate vs drawable are different typed authorities. Variable cardinality, unknown/abstain and consequence evidence required.
- **Animation-ready mesh:** compare learned structural proposals with source-qualified rest/posed outcomes and G3/G3B; dynamic measurement/teacher-relative quality is not a discrete topology gradient. Inspect global and local G3 source-motion authority, signed normal/foldover witness correction, collision/strain/seam/visibility consequences.
- **RigProgram division:** model learns joint graph, DOF, frames, control affordance, effectors and attachment candidates; Compiler may lower sufficiently supported generic control graph to known FK/IK/constraints, but must not repair unsupported parentage, invent learned intent or alter source truth to turn red to green. Experiment on completeness + execution behavior.
- **Compiler non-bottleneck:** bounded, deterministic, typed legality/solver/materialization; explicit resource, repair-budget and abstention gates. Evaluate interaction p95, exact surface lineage/remesh invalidation, and editable reopen/serialize/reopen.
- **Policy/licensing:** no blind upstream code incorporation, particularly proprietary/non-commercial nested dependencies. Record each provenance and independent implementation choice.

## 6. Freeze admission checklist

The following must all be closed by source-linked evidence and code tests before architecture freeze / final supervision inventory:

- [ ] Complete source-call graphs for canonicalization, internal multiview construction, shared representation, rendering and all heads.
- [ ] Locate StdGEN++ distinct code or record verified **NOT PUBLIC** and delimit which conclusions remain literature-only.
- [ ] Enumerate exact upstream released trainer objectives, optimizer/scheduler, teacher inputs, sampling, augmentation, data filters and checkpoint formats; mark missing artifacts explicitly.
- [ ] Inspect AniGen full VAE decoder/structure/skin contracts and postprocess consequences; SkinTokens tokenizer/codec/decode/inference and train-source gaps.
- [ ] Compare source ancestry/third-party licenses and commercial suitability.
- [ ] Model graph agreed: layer widths, block order, cross-attention placements, teacher-forcing vs free-run paths, two-view input and optional constructed views.
- [ ] Mechanical/foldover loss + structure grammar + control program/Compiler lowering owners proven by adversarial tests; no teacher data leak.
- [ ] Common-carrier typed API, evidence retention, remesh correspondence and exact dependency invalidation.
- [ ] CPU shape/forward/backprop, gradient ownership, serialization, strict checkpoint resume, two-view no-leak, resource budget and fail-closed tests.
- [ ] **Only after architectural source/code freeze**: derive normative Truth Foundry target families and masks, notebook/materialization schema and head-wise eligible counts.

**Status:** STARTED; this file is an evidence index and unresolved-work map, not a completed line-by-line upstream source audit. No code implementation or training readiness assertion is authorized by this file.
