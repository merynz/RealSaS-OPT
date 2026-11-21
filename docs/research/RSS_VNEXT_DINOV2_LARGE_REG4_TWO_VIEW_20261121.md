# RSS.vNext — Frozen DINOv2-L/14 Reg4, two source views

**Implementation:** `models/rss_vnext/unified_character_model.py` and
`models/rss_vnext/experts/iris/vnext/contracts_v1.py`.

## Decision

- Replace the former frozen 384-D DINOv2-S/14 product encoder with the
  **pretrained** `facebook/dinov2-with-registers-large`: width **1024**, 24
  layers, 16 attention heads, patch **14**, **4 registers**.
- Only **FRONT/BACK** count as real source observations. No camera metadata is
  fabricated for these observations. Six diffusion-constructed views are
  conditioning evidence, not new source truth.
- Source 1024×1024 RGBA stays immutable for Graphics, original source,
  source-ray and teacher-evidence checks. The frozen ViT sees a deterministic
  **448×448** white-matted, antialiased representation. Unlike earlier 14/1024
  mismatch, the ViT **never** consumes an invalid patch grid.
- DINOv2 outputs `[CLS, REG0..REG3, PATCH...]`. Only the **1024 patch tokens**
  per source view enter IRIS. Register/CLS tokens do **not** masquerade as
  spatial patches. XY grid = 32×32. Pretrained position embeddings are
  interpolated internally by Hugging Face for the 448px input.
- Same frozen adapter is used in training and inference; inference images
  are not resized by a separate approximate path. The model only accepts
  square RGBA observations; upstream source canonicalization must produce
  square 1024 masters.
- Shared IRIS input token width is now 1024; existing tiny test encoder
  remains **test-only** at 32. AXIS/TESSA/MIRA/Compiler semantic authorities
  do not change.
- Stage A/B native diffusion image size is **1024** by default; its grid
  alignment is **64** (8×8 latent factor). The vision model's independent 448
  grid no longer constrains diffusion to multiples of 14.
- Old Small checkpoints fail via distinct
  `FROZEN_LOCAL_DINOV2_L14_REG4_448_V1` backend identity and strict tensor
  shape checks. The production worker now resolves
  `REALSAS_RSS_DINOV2_LARGE_REGISTERS_SOURCE` or the deployment-local
  `foundations/dinov2/dinov2_with_registers_large` directory.

## Preconditions and remaining gates

1. Acquire the official pretrained **model weights and config** into the
   deployment-local directory; do not substitute random weights or small
   encoders. The model loader does not download from the internet.
2. Record and pin a verified provider revision and actual source-file
   digests in the source/checkpoint manifest; the local loader checks
   architecture and the enclosing release pins exact checkpoint bytes.
3. Run production-scale GPU preflight for memory/throughput at 1024 diffusion
   and 448 vision, then source-qualified real supervision training.
4. Test feature quality/held-out character reconstruction and gradients.
   This code change is **not** proof of trained unseen quality, successful
   source-truth teacher formation or a release-qualified ProductRevision.
5. Do not rebrand DINOv2-L encoder features as a pretrained 3D generative
   prior. StdGEN still has a specialized learned 3D decoder and diffusion.
   Our decoder/heads must be trained.

References: official DINOv2 model card, Hugging Face
`facebook/dinov2-with-registers-large`, StdGEN source
`hyz317/StdGEN/configs/mesh-slrm-infer.yaml` and `infer_slrm.py`.
