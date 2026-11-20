# BUSTED RSS.vNext — pretrained SDXL/MV-Adapter Stage A/B engineering closure (2026-11-20)

## Intent and release authority

The user selected **strong pretrained visual weights**, not weak scratch UNets, for the E2E subject preparation. Never run the subject E2E until all preflight gates are satisfied. This PR prepares pinned official SDXL and **trained** MV-Adapter I2MV weights for offline evaluation, preserving their real execution paths, without pretending that a one-image RGB checkpoint already performs BUSTED's two-source six-RGB+normal job.

### Confirmed upstream and upstream task boundaries

- SDXL Base: `stabilityai/stable-diffusion-xl-base-1.0`, model card license `creativeml-openrail++-m`; verify the particular pinned revision, LICENSE.md, permitted commercial uses, use restrictions and any distribution/attribution obligations before deployment.
- MV-Adapter: `huanngzh/mv-adapter` file `mvadapter_i2mv_sdxl.safetensors`, card `apache-2.0`; training/inference code `huanngzh/MV-Adapter` Apache-2.0. The *specific code + weight provenance and derived weight rights* need independent human review; card tags alone are not legal authorization.
- Official recipe `scripts/inference_i2mv_sdxl.py`: SDXL I2MV pipe, `ShiftSNRScheduler` (`interpolated`, 8.0), `init_custom_adapter(num_views=6)`, `load_custom_adapter(...mvadapter_i2mv_sdxl.safetensors)`; single-reference RGBA preprocessing; 6 orthographic equal-elevation cameras; per-view Plücker geometry; at inference height/width 768. Loader retains trained in-UNet adapter attention, not an external latent-row approximation.
- Actual released checkpoint produces 6 RGB images from **one reference image**. It does **NOT** provide Stage A arbitrary-pose **two-source canonical** weights, conditioned **two-reference** I2MV, or **normal-map trained** six views. None of these can be claimed by mapping tensor keys or duplicating one image.
- Our active bundle `realsas_denoiser_architecture_v1.py` is `PairedEpsilon4ChannelLatentRow.v1`. MV-Adapter has a different in-UNet decoupled multiview/reference attention and SDXL added conditioning. These weights MUST NOT be loaded as the old bundled Stage B. A versioned train/infer path is needed for integration.

## Work committed in this PR

- `models/rss_vnext/realsas_sdxl_mvadapter_pretrain_v1.py`: exact local SHA/content inventory verification, source code path pin, SDXL model config validation, adapter hash validation and license-receipt identity; official pretrained Stage B pipeline load via its **own** I2MV SDXL class; official six-view pretrained RGB sample function; SDXL Stage A prior loader with fresh LoRA. No remote model downloads at inference, no fake normal weights.
- `tests/models/test_rss_vnext_sdxl_mvadapter_pretrain_v1.py`: offline safety regressions for weight tampering, source modification, architectural mismatch, license spoofing and official `load_custom_adapter` invocation.
- References: [MV-Adapter official source](https://github.com/huanngzh/MV-Adapter), [MV-Adapter official weights](https://huggingface.co/huanngzh/mv-adapter), [SDXL base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).

## Next engineering closures before any subject E2E

| Gate | Exact requirement | Status |
| --- | --- | --- |
| G1 | Download approved exact SDXL snapshot + adapter file + official source; record real immutable revisions and byte SHA; human rights review | **OPEN: no pretrained bytes installed in worker** |
| G2 | Train **two-source Stage A canonicalization** against independent source-rendered canonical targets, preserving identity/camera/pose; pass held-out paired conditioning acceptance | **OPEN** |
| G3 | Adapt official SDXL MV-Adapter Stage B from 1 reference/6 RGB to **2 canonical references/6 RGB+6 normal targets** while preserving actual pretrained attention, not lowering to old latent-row policy; require real source supervision | **OPEN** |
| G4 | Define and implement a **new versioned Stage AB bundle/training ABI**, so both fitted stages can be loaded by `run_two_source_neural_construction_v1` without pretending the old `PairedEpsilon4ChannelLatentRow.v1` is compatible | **OPEN** |
| G5 | Use independently qualified source truth (original two RGBA, true canonical targets, rendered aligned RGB normals and known virtual cameras); no generated stage predictions as training labels | **existing source contracts; real corpus unqualified** |
| G6 | Freeze conditional stage weights/checkpoint hashes and source/vision encoder; verify full-resolution A100 forward+backward, AMP, VRAM, restart | **OPEN** |
| G7 | Validate IRIS→TESSA→AXIS→MIRA source-only teacher confinement, moving/native Graphics and Compiler on the actual selected subject without running test prematurely | **OPEN** |

Pretrained I2MV six-RGB reference generation is a *diagnostic*, not E2E qualification. No `ModelRelease`, `ProductRevision` or ProductPASS is created in this PR.

## Architecture choice

Prefer **SDXL Base + its native pretrained MV-Adapter SDXL I2MV adapter** over an untrained LoRA-only Stage B. Do not downgrade to SD2.1 solely for convenience. The task-specific two-image and normal branches remain real engineering and training work, not already-trained upstream capabilities.

No E2E training is started by this PR.
