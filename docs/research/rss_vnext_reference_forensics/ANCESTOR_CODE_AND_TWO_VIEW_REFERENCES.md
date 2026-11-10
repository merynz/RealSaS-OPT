# RSS.vNext ancestor archaeology: UniRig and TRELLIS source mechanisms

**Status:** source-linked supplementary audit, **no model freeze**. These are verified ancestry-related sources (not substitutes for direct StdGEN++ code). Read them to close absent trainer/2-view references without claiming a successor implementation shipped the same code.

## UniRig → SkinTokens actual predecessor train path

Source: https://github.com/VAST-AI-Research/UniRig/blob/main/src/model/unirig_ar.py

- `UniRigAR.training_step`, lines 86–125, **is implemented**, unlike `TokenRig.training_step` in SkinTokens' released code.
- Michelangelo 3D point latent prefixes + skeleton grammar token embeddings enter one autoregressive Transformer as `inputs_embeds`; output logits select positions beyond the conditioning prefix.
- Token supervision: next-token shift, padding attention mask, standard cross-entropy and a continuous-distance-aware `dis_loss` for quantized joint-coordinate tokens. This provides an auditable **predecessor objective** for skeletal causal syntax; SkinTokens' combined skeleton+FSQ objective and GRPO are **not** thereby proven or source-recovered.

Source: https://github.com/VAST-AI-Research/UniRig/blob/main/src/model/unirig_skin.py

- `UniRigSkin` predicts skin separately from skeleton; uses bone-to-point cross-attention (bone feature encoding, mesh encoder, weight predictor).
- `UniRigSkin.training_step`, lines 444+, **is implemented**. Targets: variable-joint GT influence weights; optional **pose-matrix-conditioned LBS** reconstructs GT- and predicted-weight posed vertices. Loss includes posed vertex MSE, joint-weighted deformation MSE, skin L1/L2/SmoothL1, BCE and influence IoU metrics.
- This is highly relevant *teacher-side training evidence* for RSS skin/deformation branch, but a loss on weights under a given GT skeleton does not train the structure of an animation-ready topology, nor establish full effectors or control programs.
- SkinTokens successor shifted coupling to a single causal sequence; its trainer methods remain absent in inspected successor classes. Avoid presenting predecessor training code as the successor's published training recipe.

Source: https://github.com/VAST-AI-Research/SkinTokens/blob/main/src/rig_package/parser/bpy.py

- `BpyParser.load` explicitly imports Blender armature pose bones, joint names, actual parent links, rest-pose local matrices and pose basis, plus mesh object names and vertex-group skin. This supplies a concrete native rig truth builder *when* source assets, transforms and rights are qualified.
- Native `matrix_world`, `matrix_local`, `matrix_basis` are separate coordinate-frame authorities, **not interchangeable**. Using raw extraction without canonical-frame normalization will contaminate teacher labels. Mesh names / vertex groups are not proof of an abstract rig control program.

## TRELLIS → AniGen actual multi-image precedent

Source: https://github.com/microsoft/TRELLIS/blob/main/trellis/pipelines/trellis_image_to_3d.py

- `encode_image` lines 119–144: list of images → 518×518 source image tensors → DINOv2 features, each independently encoded.
- `inject_sampler_multi_image` lines 286–341 provides two **tuning-free** inference strategies:
  - `stochastic`: across flow denoising steps, choose the conditioning image index round-robin. For FRONT/BACK this alternates which observed view influences each denoising step.
  - `multidiffusion`: evaluate model at same latent/timestep conditioned on every input image and **average the predictions**, then apply negative conditioning CFG as configured.
- `run_multi_image` lines 343–376 runs both sparse-structure and structured-latent samplers using those multi-image condition strategies and decodes output.
- The README explicitly calls the multi-image technique **tuning-free without training a special multi-image model** and warns it may not yield optimal results on every image. Therefore it is a *baseline hypothesis*, not evidence of built-in anatomical FRONT/BACK camera awareness or exact two-view consistency.
- RSS should compare this with a **trained, camera-aware two-view joint encoder** and with 2→6/8 predicted-view approaches. Annotate actual front/back orientation and camera frame; naive averaging of view conditions may blur or contradict hidden surfaces.
- Critically TRELLIS's multiple images are source-supplied; StdGEN's 6 views are model-generated from one. Do NOT merge the provenance/authority meanings.

## More detailed genealogy

- SkinTokens → UniRig (skeleton token predecessor) → Qwen3-family causal decoding / Michelangelo shape encoding. Michelangelo itself uses a shape-image-text-aligned latent prior and transformer-based geometry decoding; we only borrow its idea of mesh feature encoders/attention, not claim native authoring controls.
- AniGen → TRELLIS structured latent/sparse representation, with its own coupled shape+skeleton and skin field extensions. TRELLIS provides a separate source-grounded multi-image inference control strategy missing from AniGen's single-image `AnigenImageTo3DPipeline.run` interface.
- StdGEN → CharacterGen / InstantMesh / Unique3D / Era3D and earlier EG3D, Shap-E, Point-E rendering utilities; explicit author credits do not prove exact copied fragments without file-level diff. Nested proprietary licenses must be cleared for commercial code import.

## Implications for code-freeze decisions, without premature endorsement

1. For the two genuine source images choose a **camera-aware** fused backbone by evidence, benchmarking TRELLIS-style alternation/averaging, direct joint cross-attention and optional StdGEN-style 6/8 synthetic views. A repeated latent conditioning vector is *not* proof of full shared memory.
2. For coupled rig/skin, keep the predecessor's quantized-joint CE + posed LBS weight supervision as candidates, while the successor's unified causal + FSQ coding is architecture evidence; do not invent publicly unreleased SkinTokens GRPO reward or schedule.
3. Do not permit source/teacher rig input to the student inference graph. It may supervise only when lineage/pose/frame checks pass, with per-target eligibility masks designed **after** architecture freeze.
4. Both strong upstream demos postprocess or re-interpret source representations. RSS requires explicit provenance, exact mesh/joint/skin identity and deterministic qualified lowering to avoid hiding model deficiencies.
