# RSS.vNext — Research-grounded Compiler decisions (2026-11-12)

This is a **decision map**, not a model-readiness or product-authority certification. The one-checkpoint RSS.vNext unified model, and only that model, is the product producer. Historical IRIS/TESSA/AXIS/MIRA implementations are reusable as **subexperts/algorithms**, never independent mandatory product releases.

## Research references → engineering obligations

| Literature | Confirmed contribution | RSS.vNext model responsibility | Compiler/runtime responsibility | Explicit non-claim |
| --- | --- | --- | --- | --- |
| **StdGEN (CVPR 2025); StdGEN++ (TPAMI 2026)** | Joint geometry/colour/semantics; semantic multi-surface extraction and layered appearance | Semantic membership + presentation-role fields share H(x) with geometry and graphics | Calculate qualifying drawable separations with geometry/visibility/edit-consequence; preserve uncertainty and per-source appearance authority | An annotated semantic part is not a joint or a drawable by definition |
| **AniGen (SIGGRAPH 2026, arXiv:2604.08746)** | Coherent shape/skeleton/skin across one spatial field, explicit skeletal boundary uncertainty, joint-count-agnostic skin features | Shared 3D witness domain and typed AXIS/MIRA articulation affordances | Rebind rig/skin to exact candidate/canonical mesh identity; reject unsupported DOF and ambiguous anchors | Compiler cannot deduce skeleton motion solely from surface topology |
| **SkinTokens (ACM TOG 2026, DOI:10.1145/3859874)** | FSQ skin codes and skeleton parameters in one causal sequence | AXIS predicts joints plus FSQ SKIN_CODE IDs; MIRA learns true FSQ code meaning and vertex/joint interaction | Decode the **actual embedded MIRA submodule** from the exact unified checkpoint; verify joint sequence, token bounds and dense consequences | Soft FSQ expectation from training is NOT automatically equal to discrete token decoding |
| **ArtMesh (2026, arXiv:2605.16582)** | Part-aware mesh boundaries plus bidirectional vertex/pixel motion consistency | Topology + semantic boundaries + pose evidence | Check face ownership, deformation continuity and pose-conditioned source consequences before asserting editable mechanical parts | 1-frame part labels do not prove movable attachment |
| **NVIDIA FlexiCubes (SIGGRAPH 2023)** | Differentiable flexible iso-surface extraction with geometric/topological quality degrees of freedom | IRIS surface field, geometry/normal/signed evidence and FlexiCubes gradients | Surface extraction, exact coordinate binding, degenerate/nonmanifold checks and source verification | Extracting a surface doesn't certify source geometry or sign |
| **NVIDIA nvdiffrast (SIGGRAPH Asia 2020)** | Rasterization, attribute interpolation, filtering are separable high-performance primitives | Graphics color/opacity/appearance and relation predictions | Pose-deformed 3D mesh → projected triangles → depth/order/alpha compositing; source authority remains separate | Neural candidate RGB isn't source pixel ownership |
| **Adobe LayerDecomp (CVPR 2025)** | Layer separation and transparent-effects consistency for editable graphics | 2D presentation candidates with opacity/edge hypotheses | Treat art layers/draw order/effects separately from mechanics and geometry; qualify under renderer consequences | Semantics alone don't determine layer order or hidden textures |
| **NVIDIA FreeForm (CVPR 2026)** | Reduced-order simulation for deformable objects | Motion/deformability and secondary response priors if supervised | Numeric integrator/constraints and stable reduced motion state | No implicit claim of physical correctness from model motion logits |

## Concrete Compiler requirements

1. **One source identity / one checkpoint:** Two original observed RGBA + camera observations. Preserve neural triplanes, H(x), Semantic, Graphics, AXIS events, continuous skin and all source-ray status. Sidecars have exact CAS parent/child references.
2. **Topology vs. semantics:** TESSA tokens are decoded with the exact RSS.vNext checkpoint codec. Mesh computes normals, areas, vertex addresses and component ownership. A cut or attachment remains an independent relational prediction; no arbitrary slot argmax becomes a product drawable.
3. **Mechanics:** AXIS generates causal joint/control/constraint/anchor/SKIN_CODE grammar; MIRA supplies skin decoding weights in the **same unified checkpoint**. The model must export an on-demand executable decoder child rather than a made-up independent MIRA checkpoint hash. Rig qualification must bind that child to original checkpoint and computed mesh. Missing dynamic weights, unsupported operators or DOF → no executable rig.
4. **2D Graphics:** Graphics predictions, source pixel evidence and computed pose-local projection have separate provenance. Runtime draws the *posed* mesh, resolves layering with qualified visibility, preserves transparency and doesn't assign hidden-surface textures without evidence.
5. **Budget:** Read only needed CAS children. A failed topology compile must not force materializing H, decoder or N×N relation matrices. A passing candidate mesh is **not** CanonicalMesh or ProductRevision.
6. **Training:** PartVerse static face2label may supervise Semantic; motion/contact teacher supplies Mechanics. Teacher masks and source lineage required. Any accepted rig must be checked with pose/deformation consequences; source-unknown stays unknown.

## Implemented and open

- Existing: RSS model's own neutral carrier, Shape fan-out, canonical two-view source, TESSA codec replay, sparse relation, graphics/semantic readout, typed candidate-rig structural court.
- This change: export same-checkpoint **embedded MIRA codec + spatial decoder** as separate immutable CAS child. Decode with actual pinned weights and admit the true token codec binding; reject changed parent, architecture, config, state, forged parity/product authority.
- **P0 remains**: discrete SKIN_CODE predictions vs training-time expected-code weights must pass a measured parity/deformation court; joint hidden states must survive the producer boundary for a true runtime decoder; canonical geometry/rig/skin qualification, pose-conditioned graphics and Studio persistence are not closed.
- **No trained/unseen-quality claim** is made by synthetic E2E.

## Paper URLs

- https://openaccess.thecvf.com/content/CVPR2025/html/He_StdGEN_Semantic-Decomposed_3D_Character_Generation_from_Single_Images_CVPR_2025_paper.html
- https://arxiv.org/abs/2601.07660
- https://arxiv.org/abs/2604.08746
- https://doi.org/10.1145/3859874
- https://arxiv.org/abs/2605.16582
- https://research.nvidia.com/publication/2023-08_flexible-isosurface-extraction-gradient-based-mesh-optimization
- https://research.nvidia.com/publication/2020-11_modular-primitives-high-performance-differentiable-rendering
- https://research.adobe.com/publication/generative-image-layer-decomposition-with-visual-effects/
- https://research.nvidia.com/labs/sil/projects/freeform/
