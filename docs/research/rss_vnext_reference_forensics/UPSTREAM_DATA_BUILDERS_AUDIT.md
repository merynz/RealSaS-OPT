# Reference data-builder archaeology (source inspection, not RSS truth schema)

**Purpose:** Investigate how upstream systems create their source-conditioned training inputs and teacher targets. No final RSS notebook columns or supervision families are frozen here. Verified code links are against public repository tree, not paper claims.

## StdGEN (published code) material-specific render builder

Sources:
- https://github.com/hyz317/StdGEN/blob/main/blender/blender_lrm_script.py
- https://github.com/hyz317/StdGEN/blob/main/blender/distributed_uniform_lrm.py

Asset inputs are primarily VRM-oriented. Mesh, camera and semantic render output are produced by Blender:
- Imported file → normalized scene → camera angles → rendered RGB, alpha, depth EXR, normal EXR.
- Material-name parser groups into hair / cloth / face / body; colors assign classes; render-level modifications remove hair and cloth at successive levels; semantically colored images are emitted.
- Semantic extraction does **not** establish arbitrary limb, joint, accessory, socket, attachment or draw-order truth, even if the image looks reasonable. Materials without recognized names can fail; facial hair/accessory conventions can be misgrouped.
- `distributed_uniform_lrm.py::check_already_rendered` uses existence of a single `02419_semantic.png` to skip processing; this is a **weak completion witness**, not an integrity/lineage/rights proof.
- `data/train_list.json` is a list, not a complete published source-corpus teacher generator. No main reconstruction trainer/source admission available from the 82-blob public tree.

## SkinTokens — Blender rig/skin source reader, not source qualification

Sources:
- https://github.com/VAST-AI-Research/SkinTokens/blob/main/src/rig_package/parser/bpy.py
- https://github.com/VAST-AI-Research/SkinTokens/blob/main/src/data/dataset.py
- https://github.com/VAST-AI-Research/SkinTokens/blob/main/src/model/skin_vae_model.py

`BpyParser.load` reads a native armature, rest local matrices, pose bases, parent graph, object-level mesh names, vertices/faces, normals and named vertex-group weights. A separate transform stage and sampling groups construct model examples. Tokenizer converts joints/parents to discrete skeleton grammar. The codec receives sampled XYZ/normals and per-joint dense/sparse influence samples.

This is richer than a rendered image, but model-ready arrays don't prove anatomy, canonical camera, licensing, benign pose, exact mesh correspondence or a valid generic controller. All those must remain separately qualified by RSS source courts.

## AniGen — model trainer consumes prebuilt artifacts

Sources:
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/datasets/anigen_sparse_feat2skeleton.py
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/datasets/anigen_structured_latent.py
- https://github.com/VAST-AI-Research/AniGen/blob/main/anigen/datasets/components.py

Training dataset expects pre-existing `metadata.csv`, camera `transforms.json`, `voxels/{sha}.ply` and `{sha}_skeleton.ply`, extracted DINO features NPZ, structured latent and native skeleton/skin/mesh. Metadata filters include aesthetic threshold, voxel limit, `is_bad_skeleton` and `is_bad_skin`; the skin VAE and reconstruction objective can selectively suppress skin loss when skin teacher is bad. Model config `slat_dae.json` masks via these filters/targets.

The README explicitly points to **TRELLIS preprocessing** for own-data preparation. Therefore AniGen loaders + config are not evidence that its entire raw-corpus truth builder is published in AniGen repo.

## TRELLIS — verified ancestor materialization DAG

Sources:
- https://github.com/microsoft/TRELLIS/tree/main/dataset_toolkits
- https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/build_metadata.py
- https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/render.py
- https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/render_cond.py
- https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/voxelize.py
- https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/extract_feature.py
- https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/encode_latent.py
- https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/encode_ss_latent.py

Concrete parent workflow:
```text
download + source metadata (sha256 key)
    → build_metadata.py incremental CSV merge of sharded status records
    → render.py (default 150 Hammersley-sphere camera views)
    → render_cond.py (default 24 conditioning renders)
    → voxelize.py (Open3D normalized mesh surface voxels at 1/64 unit grid)
    → extract_feature.py (camera/view renders + point/voxel features → NPZ)
    → encode_latent.py (feature NPZ → pretrained sparse-structured latents)
    → encode_ss_latent.py (voxel PLY → sparse-structure latent NPZ)
    → metadata status columns for eligible data loader
```

Code audit subtleties:
- `build_metadata.py` joins many sharded success CSV files by `sha256`; its flags such as `rendered`, `cond_rendered`, `voxelized`, feature and latent existence are **pipeline completion flags**. They do NOT validate that the asset is a character or prove rights/source identity beyond the key.
- `render.py` and `render_cond.py` have **different camera counts and conditioning purposes**, so a high view count in offline dataset creation says nothing about number of user inputs at inference. RSS product interface remains exactly two observed views.
- `voxelize.py` takes the rendered normalized mesh, clips to [-0.5, 0.5] and creates 1/64 voxels. This can lose thin parts, sub-voxel gaps, open surfaces and disconnected semantic ownership; the voxelized target alone must never replace source mesh topology/attachment data.
- `extract_feature.py` stores patch tokens for sampled surface voxels in NPZ with linked camera transforms. A frozen backbone feature is learned teacher-derived representation, not source truth by default.
- Incremental success logs and existence checks are useful for scalable materialization; RSS should add **hash verification, parse validation, per-field qualification, rights, lineage/dedup and exact source identity** rather than treating status==True as admission.
- Source asset can be non-character; TRELLIS object/generation corpora are not equivalent to a humanoid/non-humanoid character corpus. Aesthetic-score filtering is not a character/rigging suitability criterion.

## Why this matters before neural freeze

This ancestors' true data path reveals implicit model assumptions:
- StdGEN field classes are driven by VRM labels and material layer removal.
- SkinTokens receives real rig/vertex-group teachers, not full anatomical affordances or editing controls.
- AniGen receives partially normalized, feature-encoded structured latents and selective skeleton/skin masks, plus generated normals from input images.
- TRELLIS can process hundreds of camera renders offline but only conditions on the images supplied at inference.

RSS design must therefore specify which inputs are original **two observed images**, which are predictive/differentiable *hypotheses*, which are source-qualified teacher-side geometric/kinematic truths, and which are downstream policy/Compiler decisions. The model's supervision schema is only safe to finalize after the actual neural graph is frozen.

**Current status:** source ancestry audited at entrypoint/data-loader level, not yet verified as a runnable end-to-end upstream data job. The parent toolkits are informative examples, not a license to import training data or bypass admission.
