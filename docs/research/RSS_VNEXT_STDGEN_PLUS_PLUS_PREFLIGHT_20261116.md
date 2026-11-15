# RSS.vNext / BUSTED — StdGEN++ Production Architecture Preflight

Date: 2026-11-16. Scope: model and trainer only, no dataset mutations.

## Primary reference and code provenance

- StdGEN 2025 source: https://github.com/hyz317/StdGEN (infer_canonicalize.py, infer_multiview.py, infer_refine.py, slrm/models/lrm_mesh.py).
- StdGEN++ TPAMI 2026 paper: https://arxiv.org/html/2601.07660.
- The original StdGEN repo currently contains the 2025 implementation, not an independently identifiable public implementation of StdGEN++. Do not claim a source review of an unpublished ++ trainer.
- BUSTED must retain TWO original image observations, two generated canonical references, six generated RGB+normal views, one shared H(x), static/dynamic mesh, rig/skin/motion and first-class editable 2D Graphics.

## Adoption decisions

1. Two-stage canonicalization remains, but the first stage has TWO original arbitrary-pose image inputs and TWO source-rendered canonical target images. Stage B conditions on BOTH canonical source renders; the single first canonical reference stays only as legacy baseline compatibility. A target camera is never an observed source camera.
2. StdGEN++ Fullbody/Facial dual LoRA suggests localized high-frequency capacity. For BUSTED this must be role-free and capable of focusing on claws, horns, wings or thin accessories as well as heads, and share ONE persistent 3D carrier and mesh/rig identity. Do not introduce a second full character authority.
3. StdGEN++ uses sparse coarse-to-fine FlexiCubes extraction targeting 256x256x384, not dense grid evaluation. Our models/rss_vnext/flexicubes_surface_evidence.py evaluates every high-resolution grid point in chunks, has a default high-res cube budget of 1.5M and a maximum axis resolution of 256. A new sparse-grid planner is required; increasing budgets alone is insufficient. Preserve exact global-to-local corner addresses, uncertainty and geometry sign qualification.
4. models/rss_vnext/semantic_geometry_prior.py ALREADY implements source-qualified semantic-equivalent signed fields. Preserve this implementation. Open/UDF or modular planar meshes do not admit a signed SDF surface equation; use explicit separate extraction.
5. StdGEN++ trains NeRF single-layer appearance/semantic before inside-out layer supervision, then differentiable mesh normal/depth/dev/hole losses. Require independent source-rendered truth per objective; unknown hidden pixels are not source-visible negative labels. Hole-filling is risky around intentional openings and must be conditioned on independently verified closed thin structures.
6. StdGEN++ video-diffusion texture decomposition creates separately editable appearance layers with cross-layer visual context and occluded-region inpainting. We have source RGB, inferred RGB, semantic role and UNKNOWN separation in compiler/rss_vnext/workers/rss_vnext_uv_texture_atlas_v1.py, but do not yet have this learned decomposition. It belongs to Graphics/Drawable as color and appearance evidence over the same mesh, never a second mechanics system.
7. StdGEN++ layer-by-layer refinement uses fixed inner surfaces for outer collision boundaries; an analogous mechanics-aware constraint is useful in XPBD/pose deformation, but static normal refinement cannot by itself prove animation, attachment, contact or retargetable weights.
8. Preflight risk: RSSvNextFrozenDINOv2 requires source H/W divisible by 14, whereas RSSvNextPinnedNeuralBundleV1 defaults to 512 and checks multiple of 8. This is a real training/inference input-preprocess ABI mismatch; resolve using one explicit shared resizing contract before GPU production run.

## Reproducibility / readiness gates

- Dataset Stage A and B -> frozen two-source encoder -> actual epsilon denoising loss -> backward, including source-view perturbation, stage masks, source/target camera provenance and checkpoint parity.
- Independently qualified signed SDF coarse proposal, sparse high-resolution active cube extraction and true pinned FlexiCubes compatibility at production scales; no false claim from a tiny sphere regression.
- Local-detail adapter with source-qualified region supervision, shared geometry and body/detail consistency. Benchmark gains vs global-only branch on held-out thin/complex shapes.
- Layered appearance reconstruction with separate source-visible, source-hidden and generated hypotheses; exact canonical face/UV and mechanical support bindings in native 2D Presentation.
- Teacher-free 2-view E2E unseen motion with geometry, topology, AXIS/MIRA LBS, retarget, Graphics pixel parity and edit/save/reopen.
- StdGEN++ holistic A-pose LPIPS only shifts 0.066 -> 0.064 vs StdGEN; do not confuse localized surface/detail/decomposition gains with a large universal holistic metric improvement.

## Explicit exclusions

Do not copy human-only facial labels or A-pose shape assumptions into a generic animal/fantasy 3D puppet. Do not promote generated hidden pixels into original truth. Do not silently remesh away persistent mechanical vertex/face indices. Generated high-resolution drawable texture remains bound to the canonical rigging surface through stable support mappings. 
