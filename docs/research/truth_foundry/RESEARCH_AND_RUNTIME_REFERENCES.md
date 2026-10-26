# Primary-source research and runtime audit

Read on 2026-10-26. Findings describe the inspected envelope, not a claim that
another method can never generalize. Our broader genericity is also unproven.

## Spine: a useful behavioral reference

Official runtime: [EsotericSoftware/spine-runtimes](https://github.com/EsotericSoftware/spine-runtimes),
4.2 pinned `e7dc1435fa4a0083ab431f1b28e083c14a1f5c68`.
The live API documentation defaults to 4.3, so implementation findings below refer
to pinned 4.2 TypeScript code, not an unpinned blend of versions.

| Inspected file under the pinned repository | Finding | Consequence for our truth |
|---|---|---|
| `spine-ts/spine-core/src/attachments/Attachment.ts`, `VertexAttachment.computeWorldVertices` | unweighted local coordinates follow one frame; weighted vertices combine per-bone local coordinates and weights; deform has separate unweighted/weighted meanings | retain source-space evaluator and local differential; shared bone ID is not a primary relation label |
| `spine-ts/spine-core/src/Bone.ts`, `updateWorldTransformWith` | affine world frames include scale/shear/reflection; inheritance can suppress selected transform effects | measure full affine relative response, keep singularity and orientation masks |
| `spine-ts/spine-core/src/Slot.ts`, `setAttachment` | slot carries state/colour/active attachment and resets deformation state on relevant switches | separate active visibility/state change from mechanical constraint changes |
| `spine-ts/spine-core/src/TransformConstraint.ts` | local/world and relative/absolute modes; independently mixed transform coordinates | partial coordinate following and switching are measurable; hierarchy parent alone is insufficient |
| `spine-ts/spine-core/src/IkConstraint.ts` | target following includes bend/compress/stretch/softness and scale-dependent cases | record solver response/residual and valid intervention envelope rather than only a joint name |
| `spine-ts/spine-core/src/PathConstraint.ts` | target path and position/spacing rules guide constrained frames | include sliding/tangential freedoms and curve-relative response where authored |
| `spine-ts/spine-core/src/PhysicsConstraint.ts` | persistent offsets/velocities, reset/update state, fixed step and authored coefficients influence response | dynamic truth needs initial state, step and parameters; a static relation cannot stand in for it |
| `spine-ts/spine-core/src/Skeleton.ts`, `updateCache` | constraint and bone dependencies determine update ordering | causal probes must record evaluation schedule; measured co-motion is not graph identification |
| `spine-ts/spine-core/src/Animation.ts`, `DrawOrderTimeline` | discrete animation keys change slot order separately from deformation | source artistic layering is independent evidence, not physical depth |
| `spine-ts/spine-webgl/src/SkeletonRenderer.ts` and core `SkeletonClipping.ts` | iterates draw order, computes world vertices, applies blend/tint/clipping and emits triangles | clip/blend/alpha footprints need explicit semantics; unsupported laws abstain |

Official explanatory references:
[attachments](https://us.esotericsoftware.com/spine-attachments),
[world transforms](https://us.esotericsoftware.com/spine-runtime-skeletons),
[transform constraints](https://esotericsoftware.com/spine-transform-constraints),
[weights](https://en.esotericsoftware.com/spine-weights).
Spine's term attachment includes nonvisual paths and bounding/clipping shapes; it
does not mean a generic physical binding predicate. The reference distinguishes
render organization, coordinate following and dynamical behavior. That is the
useful transfer to a 3D-to-2D system.

The runtime has its own [license](https://github.com/EsotericSoftware/spine-runtimes/blob/e7dc1435fa4a0083ab431f1b28e083c14a1f5c68/LICENSE).
This change records observations and implements standard mathematical measurements
independently; it does not vendor, translate or integrate runtime source.

## Papers and inspected public implementations

| Work / primary source | Input and assumptions | Useful idea and boundary |
|---|---|---|
| [StdGEN++](https://arxiv.org/html/2601.07660v1) | canonical human A-pose; semantic body/hair/clothing and specialized face processing | layered geometry and collision reasoning are useful. Human taxonomy/canonicalization are narrower than our role-free target. Do not import them as universal truth prerequisites. |
| [Bunraku](https://arxiv.org/html/2607.27348v1), [project](https://bunraku-live2d.github.io/) | Live2D organ taxonomy (8 groups/32 subclasses), Cubism parameters and artist poses | joint layer/canvas supervision is relevant; its authoring domain and parameter law do not establish arbitrary 3D mechanics. |
| [PartField, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_PartField_Learning_3D_Feature_Fields_for_Part_Segmentation_and_Beyond_ICCV_2025_paper.html), [project](https://research.nvidia.com/labs/toronto-ai/partfield-release/) | sampled 3D geometry and heterogeneous part supervision; hierarchical continuous features | strong name-free geometric control/baseline. Feature clustering is not proof of causal binding, contact or a unique drawable partition. |
| [FlowBot3D, RSS 2022](https://www.roboticsproceedings.org/rss18/p018.html) | potential motion on articulated rigid-object surfaces, interaction/kinematics and task-specific contact assumptions | per-point response field separated from planner is highly relevant. It tests novel categories within that setting; retain action/magnitude semantics rather than treating normalized flow as arbitrary mechanics truth. |
| [Ditto, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_Ditto_Building_Digital_Twins_of_Articulated_Objects_From_Interaction_CVPR_2022_paper.html), [project](https://ut-austin-rpl.github.io/Ditto/) | geometry observed before/after interaction; explicit articulated motion labels in training code | paired interventions and joint geometry/motion representation are useful. Its second observed state is extra input; do not leak it into our static student protocol. |
| [APES, CVPR 2022](https://zhan-xu.github.io/parts/) | multiple posed sprite images and reconstruction through articulated pieces | pose reconstruction is a useful external control; equivalent sprite co-motion does not identify a directed physical mechanism. |
| [Where2Act, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Mo_Where2Act_From_Pixels_to_Actions_for_Articulated_3D_Objects_ICCV_2021_paper.html) | localized actions and measured/simulated action success on articulated objects | make intervention and action validity explicit; failed and sham actions are informative. It does not recover all hidden mechanisms from a single static view. |

Code audit pins (no code copied):

| Repository and exact commit | Files actually inspected | Important distinction |
|---|---|---|
| `hyz317/StdGEN` `f1b0c822689dc2539c774bb73090a67c66e2e766` | `infer_slrm.py`, `slrm/models/lrm_mesh.py`, `configs/mesh-slrm-infer.yaml` | explicit body/hair/cloth level filters. This public code is the StdGEN lineage; the ++ paper's DualBranch implementation was not independently code-verified. |
| `SparcAI-Inc/Bunraku` `b9cc79049a50e36b1ba10db6140edb166f500e4e` | entire tree: `readme.md` only | no model/runtime implementation was present at this pin. Architecture claims come from the paper/project, not an unavailable code audit. |
| `cad-results/PartField` `0d16aa0bb77900f231c7f147f520d121134eb720` | `README.md`, `partfield/dataloader.py` | shape sampling, clustering and optional close-vertex merge; such a merge cannot silently change our teacher topology/weights. |
| `r-pad/flowbot3d` `5189de209d4021da06c6350039f6b8206e5b38ff` | `flowbot3d/datasets/flow_dataset.py`, `scripts/generate_dataset.py` | kinematic hinge/slider displacement generation and global maximum-magnitude normalization. Preserve our amplitudes/physical units instead. |
| `UT-Austin-RPL/Ditto` `f69579f49bbfaadc665fee511ad960f5243e1c1a` | `src/datamodules/datasets/geo_art_dataset_v1.py` | joint type/index and before/after states are privileged teacher information in our setting. |

APES and Where2Act are paper/project comparisons here; no claim of inspecting their
source implementation is made. No single cited method supplies our entire truth
object. The useful synthesis is continuous geometric fields plus controlled
response measurements plus explicit compositing, with identifiability and
consumer proof kept separate.
