# RSS.vNext — Interior-support ablation protocol (13 Nov 2026)

**Status:** code-level research instrument, not learned result, not shipping architecture, not a claim that the source-native rig data exists or passes admission. PR #484.

## Actual architectural question

The current product scene is IRIS triplanes `[B,3,96,64,64]` (default config: 32→64 plane upsample) with an arbitrary-point `H(x)` decoder. TESSA's current memory reads `query_grid_side=8`, i.e. 512 regular probe sites plus the raw plane tokens and relation/graphics context. Nothing proves it has lost skeletal interior information, yet nothing guarantees uniform probes plus surface conditioning recover all internal bone/attachment evidence. The question is **spatial support**, not whether triplanes are categorically incapable of 3D reasoning.

AniGen's released sparse-structure autoencoder / flow has distinct shape and bone support domains with bidirectional adapters, and its structured latent decoder exchanges geometry, skeleton and skin streams. Rigel3D's author-described surface and skeleton SLat supports further motivate a controlled comparison; its public project page currently says source code is coming soon, so do not claim reproducible Rigel3D source parity.

## Two falsifiable experimental arms

- **FIXED_VOLUME_SUPPORT**: deterministic farthest-point indices on the existing scene query grid, scaled to 70% of the canonical box to avoid the input boundary, then direct IRIS H(x) at `K` candidate support points.
- **LEARNED_INTERIOR_SUPPORT**: exactly the same seeds, same K and the same IRIS H(x), but predicts bounded displacements using content-conditioned student queries over the existing H. Final location projection initialized to zero, so the arms' coordinates/features are identical at initialization if weights are shared.
- **Both**: independent `interior_from_surface` and `surface_from_interior` multi-head attention, same output width and memory/token count. This is intentionally a lightweight AniGen/Rigel3D-*informed* ablation, not a clone of either sparse architecture. The learned arm has additional *active* position-head computation; an eventual model-quality comparison must report its parameter/compute delta and add an appropriately capacity-matched baseline.

No source joint position, bone path, source mesh, source rig, source animation, third observation or teacher latent enters either arm's `forward`. It reads `model.iris`, `state.scene_triplanes`, `state.query_xyz`, and `state.local_features` only. Therefore it does not leak source labels into student inference.

## Training-only source contract

`source_qualified_bone_support_loss_v1` consumes teacher bone-centerline samples, independently qualified masks and a **trusted launcher-provided** source-native registry row matching exact teacher tensor/mask hashes, source asset and dataset lineage. Missing/incomplete labels abstain. Pairwise symmetric Chamfer teaches **where to place support**; it does not teach joint count, graph, skin, ownership, socket or deformation. The code cannot manufacture a trusted registry.

This objective has no role in production forward. A learned candidate support location is not a skeleton or source truth.

## Required measurements for a real decision

1. **Blind source holdout**: same actual 2 input FRONT/BACK views, same source split, fixed teacher/native rig family and mask eligibility; prohibit using GT bone samples in forward, precompute only loss/metric and calculate from source-native qualified source.
2. **Component fairness**: same IRIS checkpoint, K, surface/interior attention network, optimizer/schedule, total training examples and seeds; report active parameter count, GPU VRAM, wall time and learned support head overhead. Include a capacity-matched fixed support MLP control; no credit for merely adding parameters.
3. **Source-based metrics**: bone path coverage at multiple radii, joint/parent program correctness, exact owner/face alignment, structure/skin hard-code deformation, socket attachment decoy negatives, and effect of new support on Graphics appearance/occlusion. A Chamfer improvement alone is not puppet readiness.
4. **Cross-gradient ownership**: verify non-zero gradients into original `scene_triplanes` and expert heads under qualified source rig and geometry objectives, and quantify gradient conflicts (cosine/PCGrad ablation only if needed).
5. **Representation sweep**: fixed regular samples, learned positions, and separately trained sparse surface/skeleton tokens under equal budget; only replace triplanes if these controlled results show the H(x) substrate itself is bottleneck.
6. **Compiler handoff**: any chosen support must serialize along with actual source/mesh/rig/skin binding and UNKNOWN, without inventing holder→held, detachability, artistic draw order, hidden source pixels or secondary-motion constants.

**Do not freeze** `RSSvNextConfig`, modify `RSSvNextSceneState`/CAS archive, repurpose TESSA logits, or deploy this research module before the real teacher ablation and product behaviour checks. Existing `main` is unchanged.

## Primary references

- AniGen: https://arxiv.org/abs/2604.08746 ; released implementation `VAST-AI-Research/AniGen@c49db3d6b466537a02ccf2286688903d77af7e4f`, `anigen/models/anigen_sparse_structure_flow.py`, `anigen/models/structured_latent_vae/anigen_base.py`
- Rigel3D: https://arxiv.org/abs/2605.13129 and https://nchatzis.github.io/Rigel3D/
- SkinTokens for downstream codec-first/frozen teacher dependency: https://doi.org/10.1145/3859874
- RSS: `models/iris/vnext/scene_core_v1.py`, `models/rss_vnext/unified_character_model.py`, `docs/research/rss_vnext_reference_forensics/MODEL_COMPILER_MECHANICS_PRESENTATION_EVIDENCE_FLOW_20261113.md`.
