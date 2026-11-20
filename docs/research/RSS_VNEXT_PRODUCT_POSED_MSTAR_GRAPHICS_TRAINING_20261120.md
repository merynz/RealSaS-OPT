# BUSTED RSS.vNext — Same-M* moving 2D training bridge (2026-11-20)

**Status:** optional, source-qualified posed M* raster objective in the product-seeded trainer. **NOT:** full GPU-parity, cloth physics learning, UV-atlas authority, product native 2D release or unseen-model generalization.

## Why this is a distinct training task

The existing S-LRM Stage3 renderer works on a FlexiCubes extracted surface. The product's mechanical/presentation mesh is the single **TESSA M***. Treating a separately extracted FC witness as the actual posed M* would create competing geometry/appearance authorities.

This bridge renders **the exact final-indexed M***:

- **Posed geometry** is computed from student AXIS/MIRA output and independently admitted source-native pose matrices, with same final M* faces and skin domain. Source matrices are training-only interventions, NOT inference inputs or clips predicted by the network.
- **Rest-surface attributes** are interpolated using visible posed triangle barycentrics. The same shared IRIS H(x) is queried at canonical/rest surface points to supply Graphics RGB and SLR. Querying a moving point in world coordinates would make material colors swim as joints move.
- Visible face, depth, normals, mask and antialias RGB are produced through the existing bounded CPU raster or compiled PyTorch3D CUDA backend. No second cosmetic mesh, output framesheet, independently learned drawable or global artistic 2D mesh is created.
- Training remains: two original RGBA → six generated conditioning views → H(x) → FC seed → TESSA M* → AXIS/MIRA → posed M* → 2D M* raster → qualified source RGB/alpha/SLR/normal/depth losses.

**Scope limit:** first-hit opaque surface only. Source role labels alone do not authorize material transparency or cloth physics. Hidden texture and back-side appearance remain separate Graphics source qualification.

## Source-native teacher and mandatory trusted registry

The NativeMstarPoseRasterTargetV1 type carries pose index, source-rendered cameras, width, height, RGB-on-white, alpha, per-pixel RGB and alpha masks, optionally role IDs, normals and metric depth, and individual masks. All shape, finite and known/unknown contracts are checked. Source cameras are 13-field native 3D rendering cameras, used only in the loss. Their hash plus every target/mask hash must be present in the training launcher's trusted registry.

Independent receipt binds:
- Source asset SHA, dataset manifest/rights lineage, source qualification key;
- Original two RGBA bytes and cameras ledger, pinned generator checkpoint and input binding;
- Exact final M* topology address; source-native face/barycentric transport and AXIS joint order; native pose transport SHA; native joint matrices and valid pose mask;
- Source rendered image height/width, source cameras and RGB/alpha/optional SLR/normal/depth bytes;
- Explicit guarantees: independent source renderer, generated six-view NOT used as teacher, opaque first-hit source z-buffer, and teacher not exposed to the model input.

A sparse/missing source family never becomes an invented negative. Enabled pose render refuses a batch with no qualified source pose or no target unless sparse-source admission is explicitly selected. The existing native pose transport validation must complete before this loss.

## Training and checkpoint contract

- The optional product_mstar_pose_raster_weight defaults to zero and is only valid under PRODUCT_SEEDED_IRIS_TESSA_V1 with matching AXIS/MIRA source skin and posed-surface losses.
- NativeMstarPoseRasterWeightsV1 controls RGB, alpha, role, normal and depth channel weights; enabled weights and source registry hash are frozen into training objective and checkpoint identity.
- A unique POSED_MSTAR_RENDER gradient family prevents incorrectly attributing this coupled screen-space loss to Graphics alone. It can influence IRIS, TESSA, AXIS, MIRA and their bridges through permitted differentiable paths; discrete face/FSQ choices remain non-differentiable.
- The source-conditioned curriculum identifies SOURCE_POSED_CONSEQUENCE / FULL_SHARED_H_FINETUNE as possible subphases. No optimization winner is claimed without real-corpus results.
- CPU regressions check canonical appearance under posed-depth translation, differentientiable gradients, source receipt tamper, lack of native poses, and masked labels. CUDA backend/throughput and source-realism require dedicated GPU qualification.

## Still open for final ProductPASS

1. Real cloth/body clearance, seams, pins, contact/collision and XPBD source properties or explicit runtime policies: geometry/SLR does not independently certify cloth physical behavior.
2. UV/texture atlas identity and native C++ post-XPBD renderer parity; rest-coordinate neural color field is not an editable UV atlas.
3. Two-original-input teacher-free M*, AXIS rig and skin on unseen morphology; source-native pose matrices may not become inference inputs.
4. Source-truth coverage and GPUs for 10k+ qualified assets; full resolution training and selected multi-objective optimizer have not been validated.
5. Source-backed product proof and edits: a synthetic training loss does not mint an admitted ProductRevision.

References: StdGEN (CVPR2025) released S-LRM inference code; TailorRig (2026) rig/skin/deformation concept, not a verified training-code transplant; the local first-party realsas_flexicubes_mesh_render_v1 and compiled PyTorch3D adapter; the single TESSA indexed M* and native-face pose transport contracts.
