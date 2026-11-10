# FlexiCubes animation-source audit → RSS.vNext architecture (2026-11-10)

**State: training mechanism coded and the real NVIDIA extractor verified; not a real-character architecture science freeze.**

Sources:
- NVIDIA pinned repo: https://github.com/nv-tlabs/FlexiCubes/tree/4cc7d6c3d0cee83c011ce36721b81adff0dd7db6
- Shen et al. SIGGRAPH 2023 supplement §4.2: https://nv-tlabs.github.io/flexicubes_website/flexicubes_suppl.pdf
- NVIDIA project animation: https://research.nvidia.com/labs/toronto-ai/flexicubes/

## Exact source claims, with caveats

1. The supplement describes **per-subject** optimization on a coarse 32³ FlexiCubes grid, using source photometric supervision.
2. There is a **300-iteration rest/T-pose warmup** then sampling random camera and animation frame each iteration. This number belongs to their per-asset experiment; we have NOT adopted it as a universal RSS network training schedule.
3. Source skeleton, animation matrices and source reference skin weights are GIVEN through rigged assets (USD/RenderPeople). This is NOT image-to-rig inference.
4. After each topology/vertex update, the current FlexiCubes surface is re-skinned from a fixed source mesh by **k=10 nearest source vertices, inverse-distance normalized weights** (supplement Equation 3), followed by differentiable LBS and posed-image objectives.
5. NVIDIA did not explicitly optimize the skinning weights under changing vertex IDs; the supplement calls continuous skin parametrization an open future direction.
6. The pinned public repo has 21 files, including the extractor and generic photogrammetry examples, but **no runnable animation training recipe**. The RSS code below is an independent implementation of the documented method, not a copied NVIDIA animation trainer.
7. NVIDIA extraction uses per-cell β=12 edge weights, α=8 corner weights and γ=1 quad-split weight. During training a quad becomes four differentiable triangles; inference selects a two-triangle diagonal. Their mesh-topology identity is NOT literally identical across these regimes.

## Where the gradients flow in RSS.vNext

Production observation is always exactly two source RGBA views (FRONT and BACK), plus qualified camera metadata. Teacher source mesh/rig/skin/poses are permitted ONLY as training targets.

Training source-animation path:

    FRONT/BACK → one live scene Z/H
       → SDF + semantic role field + voxel deformation + per-cube β/α/γ
       → pinned actual FlexiCubes differentiable extracted surface F
       → teacher kNN re-skin of *current* extracted vertices
       → source-pose LBS for P admitted frames
       → source-relative posed-surface loss
       → gradients to F / field / cube weights / original Z/H

Structure path reads the SAME Z/H and FC local features:

    Z/H + semantic evidence + FC surface H(x)
       → TESSA V3.1 structural attention and topology grammar
       → indexed animation-ready candidate mesh
       → AXIS control graph + MIRA exact-index skin
       → source-relative posed-deformation/Graphics courts
       → qualified Compiler candidate only after hard proof

Important: the FlexiCubes pose loss does NOT directly differentiate a discrete TESSA triangle-token selection. Separate source motion preference on causal token log probabilities (models/rss_vnext/motion_aware_topology_learning.py) trains that decision. Both losses backpropagate to the shared carrier. The mesh handed to mechanical controls retains TESSA vertex/face identity. Temporary FC IDs are never silently promoted to product IDs.

## Tests that have actually passed

- Pinned NVIDIA CPU operator, multi-pose skin transfer, current-surface gradient and shared two-view optimizer with TESSA: https://github.com/merynz/RealSaS-OPT/actions/runs/42882354728 (8 passed).
- Baseline two-view neural model, masking/rig/mesh/structural routes: https://github.com/merynz/RealSaS-OPT/actions/runs/42882354820 (28 passed).
- New implementation: models/rss_vnext/flexicubes_animation_supervision.py and models/rss_vnext/training_engine.py.
- Synthetic-only test source reconstructed from a detached surface or analytic field. A PASS proves differentiability, shape and fail-closed checks. It does NOT prove real unseen character quality, triangle density improvement or an optimal architecture.

## Mandatory real-data experiments before freeze

1. Matched ablation on independently admissible characters: rest-only FC vs rest+sequence FC vs TESSA motion-ranked structuralization vs coupled FC+TESSA. Same source FRONT/BACK, qualified original rig/poses, train compute budget.
2. Judge held-out poses: source-relative silhouette/depth, surface distance, edge/stretch relative to source (not absolute rigidity), foldover, seams, attachments, graphics ownership and inference latency.
3. Verify training 4-triangle differentiable fan vs inference 2-triangle discrete quad difference, including address/correspondence drift; avoid training-only microtriangles hiding deployment artifacts.
4. Current first pilot accepts SINGLE_QUALIFIED_CLOSED_COMPONENT only. Global kNN transfer across close-but-independent shells (hair/cape/prop) is unsafe. Need component-aware source transfer or separate patch extraction when corpus supports it; do NOT fake open/modular global sign.
5. Rig matrices must be source-qualified rest-to-pose including inverse bind, same canonical frame, source skin weight identity and signed surface policy. Rights/lineage external admission gates. Do not import RenderPeople data on the assumption its original commercial use permits ours.
6. Prove unseen 2-view inference works WITHOUT privileged source rig, animation, source views or teacher mesh. CPU smoke and per-subject fitting are not evidence of generalization.
7. Measure activation VRAM, sequence length/token bottleneck, dynamic mesh count, per-step FPS and trained validation metrics before deciding the computational graph and freeze.

**Tentative placement:** FlexiCubes is a motion-aware *surface scaffold/teacher* on the one carrier. TESSA remains the structural index/mesh owner unless controlled ablations show a better qualified graph. No premature final freeze.
