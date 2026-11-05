# Corpus370 V9 — Generic Geometric Ownership Court and StdGEN++ Comparison

Date: 2026-11-05. Research is V9 private original-authored-motion source-to-soup rendering with no training or drawable truth authority. PR #327 draft.

## Paper-derived verified facts
- StdGEN CVPR 2025: https://arxiv.org/html/2411.05738
- StdGEN++ arXiv 2601.07660 / IEEE TPAMI 2026: https://arxiv.org/html/2601.07660
- StdGEN++ explicitly uses Dual-Branch S-LRM, semantic-aware surface extraction, layer-specific supervised rendering, coarse-to-fine geometry and video-diffusion facial texture layers.
- Anime3D-EX extends 10,811 Anime3D++ characters. Three main semantic geometry classes are minimally clothed body, clothing, hair; separate face skin/iris/lashes textures are provided.
- Original StdGEN Appendix H says VRoid-Hub conventions group tight-fitting garments into base body and outer clothing/pants/skirts into clothing. These are *annotation conventions*, not geometry-derived objective partitions.
- StdGEN++ layer evaluation excludes 8 of 109 source samples due to ambiguous/defective semantic ground truth. StdGEN acknowledges limited training category/style diversity for animals and generic 3D objects.
- Critically, StdGEN++ is NOT merely 2D semantic segmentation. It reconstructs real 3D geometry and inside-to-outside layers; its method and benchmarks are substantial. The paper does not demonstrate arbitrary morphology, ontology-free artist drawable-ownership identification.

## Geometric ownership is a partially identifiable inverse problem

Two source-face partitions may produce exactly the same original multi-view/source animation, yet respond differently to independent editing. Conversely, partitions may produce identical response under tested interventions yet differ in untested ones. Hence source geometry and natural authored motion alone do not uniquely reveal artist-intended layer/attachment identity.

Our potential moat is an ontology-independent *evidence system*: identify which group hypotheses are contradicted by original source geometry, appearance, occlusion and controlled editor intervention, and leave unresolved partitions as UNKNOWN. We do not claim to have proven that no competitor can do this, or that geometry removes all prior assumptions.

Important distinction:
1. Source geometry / face correspondence: V8 already measured 80 original authored pose examples with exact original rest linkage across 3 subjects; no new pose collapse.
2. Animated renderer authority: V9 re-imports actual original authored action/owner binding, evaluates poses to match SHA-sealed teacher, builds face soup with original material/UV and original per-face posed source vertices. Compare 3 poses from distinct actions x 4 original cameras [0,2,4,6] x all group interventions.
3. Original-vs-soup parity must qualify before comparing any group edit consequences. The 192x192, 32-sample bounded pilot is NOT native-resolution CAA.
4. Fine candidate is a reference edit proxy, NOT a canonical artist segment. Group count cannot be selected from self-reference alone.
5. Visibility, attachment, mechanical causal intervention, same/different drawable evidence and model-training authority are separate scientific courts and remain false.

## V9 outcomes that matter
- Source original vs posed triangle soup full-image RGBA parity per pose/view, and no-op identity.
- Occlusion & hidden-surface edit effects from isolate vs all-clear; count coarse vs fine edits only on parity-qualified pose/views.
- Record actual blind spots, pose-dependent overlaps, 8/16/32/64 not interchangeable; never rewrite group IDs based on source mesh/material names.
- A measured failure is not a success. Return fail-closed ABSTAIN for parity mismatch/empty occupancy, do not automatically relax tolerance.
- Separate prior-induced human semantic labeling mistakes from intrinsic geometry identifiability and source animation/renderer mechanics.

## Next
After V9, perform independently controlled positive/negative drawable interventions and split/merge edit equivalence classes, then connect only independently qualified geometric constraints to IRIS/TESSA with loss masks and original SHA parentage. No admission from this report.
