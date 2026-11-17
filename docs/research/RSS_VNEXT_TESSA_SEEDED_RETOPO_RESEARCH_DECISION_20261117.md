# RSS.vNext v1 — evidence-based TESSA seeded remeshing research decision

Date: 2026-11-17 | Type: **research / architecture qualification**, not trained-model performance or implemented native GPU path.
Scope: StdGEN++ semantic mesh extraction -> one canonical mechanically suitable indexed mesh -> AXIS/MIRA. No dataset operation or training in this PR.

## 1. Critical research finding

DO NOT train a new autoregressive `SPLIT/COLLAPSE/FLIP` teacher from a single source mesh triangulation as if its edit order were objective truth. The same underlying geometry, boundary, UV and mechanical behavior admit many valid triangulations and many legal edit paths. ExMesh and RADmesh demonstrate that real geometry learning/optimization can be coupled with **deterministic, carefully constrained discrete remeshing**, without an end-to-end edge-token CE predictor.

The actual publicly readable StdGEN 2025 clean-room reference **already does this**: `hyz317/StdGEN/refine/opt.py` and `refine/remesh.py` implement continuous optimization interleaved with `collapse_edges`, `split_edges`, `flip_edges`. `refine/mesh_refine.py` drives multi-view normal/mask refinement, with an inner/outer geometry anchor. StdGEN++ 2026 builds on layered extraction, not a published general *learned mesh edit teacher*. This is a substantial simplification relative to our earlier speculative TESSA edge-action training matrix.

## 2. Direct evidence and code status (do not conflate)

| Research | Reproducible observations from actual source | Applicability / limit |
| --- | --- | --- |
| [StdGEN++ (TPAMI 2026)](https://arxiv.org/html/2601.07660) | Layered semantic SDF and coarse-to-fine FlexiCubes, normal-guided refinement. Animated cloth results enabled by separate layers. | PRIMARY scene/extraction reference; NO verified public StdGEN++ training code, no learned rig/skin/mesh-edit labels |
| [StdGEN (CVPR 2025)](https://github.com/hyz317/StdGEN) | Code `refine/opt.py` + `refine/remesh.py`: edge-collapse, split, flip under normal/mask rendering loss, not a remesh edit CE model | DIRECT executable baseline; project has its own licensing/dependency review, do not blindly paste |
| [ExMesh (CVPR 2026)](https://arxiv.org/html/2606.07288) | Public `Fan-Treasure/ExMesh/scene/mesh_model.py` implements splitting/merging and link-condition checks; `train.py` couples normals/texture/silhouette/depth/regularizers with per-scene optimization | Important **per-asset optimization** evidence, NOT amortized feed-forward TESSA training. It maintains per-face UV seams through edits. Repo Apache-2.0 overall but vendored PGSR/2DGS derivatives can bear additional noncommercial notices; clean-room or audit |
| [RADmesh (ECCV 2026)](https://github.com/threedle/radmesh) | Public `radmesh/bkremeshlerps_remesh_with_attrs.py` interpolates attributes onto new vertices, tracks `new2old_fi` with **-1 for newly created/changed faces**, and applies barycentric reprojection. Remesh operations interleave with continuous deformation; original optimization state can be interpolated | Exact parent lineage is not always 1:1! Do not falsely stamp every remeshed face as an original source face. Parent support may be best-effort for newly created regions. Repo MIT, transitive deps separately reviewed |
| [Learned Localized Mesh Refinement (SIGGRAPH Asia 2026)](https://arxiv.org/abs/2609.31946) | Paper introduces a local-conditioned AR refinement tokenizer with many valid trajectories from one target and physics-aware prioritization | Very relevant FUTURE learned-TESSA alternative. **GitHub main tree currently exposes ONLY LICENSE**, no runnable network/trainer to verify. Cannot claim code parity or trainability |
| [Simplification of Articulated Meshes (2009)](https://doi.org/10.1111/j.1467-8659.2009.01374.x) | Source representative poses + skin, simplify geometry to preserve **deformed** shapes and joint weights, not just rest pose | Most direct fundamental objective for TESSA+AXIS/MIRA, but SOURCE pose/skin subset required |
| [Single Edge Collapse Quad-Dominant Mesh Reduction (TOG 2025)](https://doi.org/10.1145/3731143) | QEM attribute-aware collapse preserving joint influences and quad topology | Useful mechanical face-budget/edge retention metric, but no known source skin at *inference* before AXIS |
| [Adaptive Anisotropic Remeshing for Cloth Simulation (2012)](https://doi.org/10.1145/2366145.2366171) | Curvature/velocity-gradient-driven anisotropic cloth refinement | Optional XPBD/garment patch quality hint, **NOT** permission to change canonical topology every runtime frame |

The 2025 StdGEN and 2026 StdGEN++ codes are not identical. The 2026 Learned Localized paper's abstract claims a code URL, but the currently accessible repository contains only a license: label code availability **UNVERIFIED** rather than implement it sight-unseen.

## 3. V1 design: seeded structural **metric** + deterministic remesh

```text
IRIS H(x) / StdGEN++ high-res layers
        -> M0 [original indexed shells; many components allowed]
             + typed source-role / open-solid / uncertainty status
        -> TESSA local query(H at vertices, incident faces, boundary,
                            geometry curvature, surface role, predicted deformation affordance)
             -> CONTINUOUS outputs: permissible vertex offset,
                                     optional target anisotropic density/edge metric,
                                     structural confidence
        -> deterministic geometry-aware split/collapse/flip with exact budgets
             -> M* [ONE final indexed mesh, preserving shell boundaries]
        -> AXIS & MIRA query H at M* plus full component/face-context
        -> Compiler admission of M* + rig + skin after neural inference
```

**Boundary conditions:** All topology operations finish BEFORE AXIS/MIRA and before Compiler canonical admission, UV packing, attachment selection and XPBD compilation. A final mesh asset can contain disconnected garment and body shells. No deletion of cloth because it is invisible from the two observed views. No forcing open hems to watertight sign; no crossing semantic components or UV islands and no silent weld. Part role `outer_covering` never means flexible material.

TESSA remains a **learned** specialist (continuous structural density/metric, geometry-aware refinements, learned deformation affordance from H), not a second autoregressive full-mesh generator. Discrete operations themselves are a **deterministic geometric operator**. Without source-qualified demonstration that a learned discrete policy improves unseen characters at matched compute/face budgets, **DO NOT make token edit CE required for BUSTED v1**.

**Required parent mapping:** Every changed vertex has `source_mesh_hash`, support type (original vertex, original edge, original triangle, or unsupported), parent IDs + barycentric weights, normalized source-coordinate transform and operation log. Changed faces need their own lineage validity, not just vertex parents; RADmesh's `new2old_fi==-1` shows why. For collapsed/seam-crossing edits, parent interpolation may be invalid: ABSTAIN or preserve locked boundary. The current `tessa_seeded_surface_v1.py` only proves identity + one-to-four midpoint subdivision; it is **not** sufficient for full production collapse/flip/source identity.

**UV:** Allocate product UV atlas **after** TESSA topology is final (simpler and safer for v1). Source-native UV teacher remains separate, with remapping provenance. If optional final editing includes topology operations, the editor must replicate ExMesh seam-aware UV support; post-admission graphics cannot quietly remesh.

## 4. Source-qualified truth and loss design

There are **two different training questions**:
1. IRIS predicts a good *continuous 3D surface*; train with independently source-rendered RGB/alpha/semantic/depth/normal and qualified SDF/UDF, as in StdGEN++.
2. TESSA changes that surface's mesh structure while minimizing functional distortion at a fixed budget. **No unique source triangle order is an eligible general target.**

### Targets, masks, and gradient paths

| Family | Eligible teacher | Valid mask | Loss family | Gradient owner |
| --- | --- | --- | --- | --- |
| Rest 3D surface | Source-native rest mesh converted to separately sampled surface points/normals | Valid geometry, face/normal orientation, source coordinate alignment | symmetric area-weighted surface distance + normal consistency | IRIS + TESSA vertex refinement; discrete policy selects feasible edits under no-grad |
| Sheet/boundary topology | Source-native component shell ID, known true boundary loops/seams | Real open hem / seam, exclude unknown broken topology | boundary distance + no-cross-component / manifold hard court | Continuous TESSA geometry only; discrete boundary locks |
| Surface complexity | Source/reference multi-view geometric curvature and renderer error | Source visible pixels/face correspondences ONLY | bounded curvature/normal residual + quality regularizers | IRIS + TESSA vertex offset + optional local metric |
| Mechanical posed fidelity | Source-native exact rest mesh, bind joint hierarchy, weights, pose matrices, posed surface samples | **Only rigged, correctly reindexed source-native assets** | multi-pose posed-surface distance + normal/strain error relative to source | IRIS + TESSA rest/metric + AXIS/MIRA through differentiable LBS |
| Skin influence preservation | Qualified source joint weights on mesh and pose frames | Same joint order and qualified barycentric transfer | area-weighted skin weight & posed vertex residual | MIRA + TESSA metric (via consequence or teacher importance) |
| Source layering/collision | Matched inside/outside component surfaces with source normals and scales | Valid separated layers and surface-distance collision support | penetration / signed separation barrier | TESSA continuous geometry, later Mechanics validation |
| Triangle shape/complexity | Deterministic, data-independent physical/algorithmic constraint | Faces in current candidate topology, boundary, finite | aspect/sliver/area distortion, face budget penalty | Continuous mesh geometry (discrete operations not differentiable) |

**Source teacher contract**: source asset content hash, license and source geometry identity, canonical normalization, 3D component semantics/mesh, rest and source pose qualified, bone+weight indices, per-sample source mask, operator configuration, optional UV seam mapping and original image camera provenance. Generated Stage A/B views can be inputs but never independently measured source pose/depth/role truth.

**No body/cloth magic:** Cloth secondary motion material/compliance does NOT come from a static mesh label. The seeded TESSA trainer can constrain open, pin-eligible garment sheet topology; XPBD material remains runtime declaration/policy, not a learned material truth. No invented cloth simulation teacher.

### Initial objective for experiments (not paper-specified optimum)

For rest and pose samples normalized by source bbox diagonal `D` and qualified pose count `P`:

```text
L_seeded_geometry =
  1.00 * normalized_surface_distance(rest)
+ 0.20 * (1-normal_cosine)
+ 0.50 * boundary_coverage
+ 0.10 * triangle_aspect_and_sliver_penalty
+ 0.50 * qualified_layer_penetration
+ 0.01 * normalized_face_budget

L_seeded_mechanics =
  2.00 * mean_{p in qualified poses} normalized_posed_surface_distance(p)
+ 0.25 * mean_{p in qualified poses} triangle_inversion_or_strain_penalty(p)

L_seeded_total = L_seeded_geometry + eligible_pose_mask * L_seeded_mechanics
```

These **are versioned candidate weights**, NOT measured balanced gradients, NOT weights from the papers. A training phase must reject active terms without corresponding source-verified targets and log each term's qualified support and gradient norm before optimization. Do not claim a frozen hyperparameter is scientifically validated. Source pose samples need correct rest, joint and MIRA face address. Compare at matched face budgets to decide whether `2.0` or any coefficient is acceptable.

**Optional learned metric:** If a density head is actually needed for zero-per-asset-optimization inference, derive relative importance from qualified source rest-surface curvature and source-pose deformation error under locally perturbed *feasible* remesh candidates, with strict provenance. It is a **derived, nonunique candidate-ranking target**, NOT direct one-hot `edge edit truth`. The structural model can learn a continuous importance ranking. Explicit token edit AR and `t_edit_rank` are RESEARCH ONLY until source coverage + matched-compute gains justify them.

## 5. Validation gates on the SAME held-out source character

- **Ablations at identical face budgets:** raw StdGEN++ FC output; deterministic isotropic/curvature remesh; deterministic *deformation-aware* remesh; learned-metric-conditioned remesh. Keep GPU, cameras, native poses and other experts pinned. An optional AR learned-edit model may enter ONLY after it wins.
- Check open cloth hem not closed, no cross-layer welds, directed-attachment face support survives and no active bone/body crossing under sampled poses.
- Check rest silhouette, original and novel view normals, depth, hidden texture unknown designation, posed max/mean surface distance, inverted/sliver triangles and self-intersections.
- Check **one final mesh address:** AXIS/MIRA skin hard-FSQ decode, native final pose, XPBD patch and final Graphics UV all refer to exact compiler-admitted M*; no post-hoc visual mesh.
- Check timing: end-to-end 2-source inference, S-LRM+FC field cost, remesh cost, memory and hidden/unsupported region abstentions. Product must not imply **8 source views**.
- No pass from two-triangle tiny unit fixture or frozen Knight composite. Unseen morphologies and licensed source-native poses required.
- Assure any deformed-surface reduction never evaluates a source pose under a separately regressed geometry or a teacher-forced mesh that product inference cannot reproduce.

## 6. Research conclusion, project decision

**Use StdGEN++ for semantic layered surface reconstruction. Use conservative continuous optimization + deterministic localized remeshing for the BUSTED v1 mesh structuralizer, drawing on actual StdGEN, ExMesh and RADmesh code. Use deformation-aware errors from articulated simplification/pose supervision to learn TESSA's structural importance and refinements when verified source poses are available. Keep learned localized AR edge-action generation an optional future ablation.**

This changes the speculative prior plan: **NO mandatory per-edge action cross-entropy, no copying source face order as GT, no automatic merge of draft new loss weights into active trainer.** The existing `training_engine.py` still teaches legacy independent full-mesh topology CE; this requires a **versioned new seeded training ABI** and explicit removal from v1 training gate. This document is a research decision; it does not claim that model/Compiler/trainer migration is already complete.

A simplified static re-mesher without any trainable TESSA or pose-aware gradient is merely a **fallback**, not a completed learned specialist. Prove the structural head's learned contribution with the ablations above. Do not remove all TESSA expertise under the guise of modernization.
