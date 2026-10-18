# IRIS vNext ↔ StdGEN++ Re-audit — 2026-10-18

Status: **ACTIVE DESIGN EVIDENCE / NOT AN AUTHORITY ARTIFACT**

Purpose: re-evaluate StdGEN++, public StdGEN code, InstantMesh ancestry and the current IRIS vNext architecture after the completed V8 experiment. The goal is not to copy a semantic ontology. The goal is to identify transferable training, representation, data and extraction disciplines while explicitly separating what is proven by public code, stated by papers, inferred for RealSaS, or unavailable.

## Evidence levels

Every statement in this note belongs to one of four classes:

- `CODE_EXACT`: directly verified in public source code.
- `PAPER_EXACT`: directly stated or derived from equations in the published/preprint paper.
- `REALSAS_INFERENCE`: our engineering translation; not a claim about StdGEN++ implementation.
- `UNAVAILABLE`: the exact implementation is not publicly inspectable.

Primary sources:

- StdGEN++ paper: https://arxiv.org/abs/2601.07660
- StdGEN CVPR 2025 paper: https://openaccess.thecvf.com/content/CVPR2025/html/He_StdGEN_Semantic-Decomposed_3D_Character_Generation_from_Single_Images_CVPR_2025_paper.html
- public StdGEN source: https://github.com/hyz317/StdGEN
- InstantMesh source/training ancestry: https://github.com/TencentARC/InstantMesh

As of 2026-10-18 the public StdGEN repository advertises inference code, dataset/rendering support and pretrained checkpoints. Open issues #3 and #15 still request the training code. Therefore exact StdGEN/StdGEN++ optimizer-loop behavior that is not present in source must not be presented as code-verified.

## 1. What V8 changed in how we should read StdGEN++

The V5/V6 failures and V8 result separate two questions that were previously mixed:

1. can the shared representation encode geometry/mechanics/graphics structure?
2. can downstream objectives safely keep rewriting that representation?

V8 answers the first question positively under the current Fit8 teacher definitions: after the M1 carrier was sealed, geometry/extraction stayed invariant while appearance, relations and topology objectives continued improving. Therefore the immediate problem is not evidence that IRIS needs a named body-part semantic head.

The second question is also answered: unrestricted downstream rewriting of the shared carrier is unsafe in our curriculum. V5 and V6 showed this directly. The stable-carrier lesson is now an architectural invariant until a dedicated falsification court proves otherwise.

This makes StdGEN++ most useful to RealSaS as a study of **specialization, decomposition-aware supervision, geometric continuity and staged representation use**, not as a vocabulary source.

## 2. Public StdGEN code: exact representation structure

### 2.1 LoRA-constrained transformer adaptation — `CODE_EXACT`

In both `slrm/models/lrm.py` and `slrm/models/lrm_mesh.py`, when `lora_rank > 0`:

- `lora.mark_only_lora_as_trainable(self.transformer)` is applied;
- transformer positional embedding and deconvolution parameters are explicitly re-enabled;
- the image encoder is frozen;
- the synthesizer remains trainable.

This is stronger than a vague claim that “the backbone is frozen”, but weaker than “everything except the new head is frozen”. The public model constrains adaptation in the transformer while still permitting trainable field decoders/synthesizer capacity.

### RealSaS translation — `REALSAS_INFERENCE`

V8's full post-M1 `scene_core` seal is an intentionally stricter baseline. If a future authoritative specialist target cannot be decoded from sealed H, the first eligible response is a specialist adapter/branch—not broad reopening of `scene_core`. A limited LoRA-style adapter is therefore a **capacity-court outcome**, not the default next step.

## 3. Public StdGEN code: shared feature, separated decoders

### 3.1 Same sampled triplane feature feeds multiple capabilities — `CODE_EXACT`

`slrm/models/renderer/synthesizer_mesh.py` defines independent decoder networks for:

- SDF;
- RGB;
- deformation;
- 21 FlexiCubes weights;
- four-way semantic prediction.

All are decoded from the same sampled triplane features. This is structurally close to IRIS's shared H plus specialized evidence heads, even though the semantic vocabulary and extraction semantics differ.

### 3.2 Semantics are structurally active, not merely a classifier — `CODE_EXACT`

`MeshSLRM.get_sdf_deformation_prediction()` consumes semantic probabilities when constructing equivalent SDFs for selected layers. In the public StdGEN implementation, semantics can therefore change which geometry is extracted. This means removing named semantics from RealSaS requires replacing their **functions**, not merely deleting a classifier output.

For RealSaS those functions are intended to be distributed across:

- category-free relation/boundary evidence;
- Compiler fragment/depth/contact graphs;
- topology conditioning;
- source/geometry provenance;
- optional future structural-continuity evidence.

## 4. StdGEN++ dual branch and specialization

### 4.1 Dedicated full-body and facial LoRA branches — `PAPER_EXACT`

StdGEN++ introduces two specialized LoRA branches: one for global/full-body structure and another for cropped facial reconstruction. The paper states that LoRA is inserted into attention linear projections and that the branches carry separate trainable LoRA parameters.

The important transferable idea is **specialization without forcing one monolithic representation update to solve all scales/tasks**.

### RealSaS translation — `REALSAS_INFERENCE`

Do not copy anatomical branch names. If future courts justify specialization, branch by evidence domain or scale, for example:

- visual/boundary specialist;
- relation specialist;
- structure/topology specialist;
- high-resolution local geometry specialist.

The sealed carrier remains shared authority unless an explicit carrier-falsification court is passed.

## 5. StdGEN++ training stages

### 5.1 Stage 1 — `PAPER_EXACT`

The paper initializes from pretrained InstantNeRF and trains newly added LoRA plus semantic decoder under image, mask and semantic supervision.

### 5.2 Stage 2 — `PAPER_EXACT`

It performs hierarchical multi-layer supervision from outside to inside. Semantic subsets are masked during rendering so internal color/semantic information receives 2D supervision rather than being left entirely occluded.

This is the key reason their hidden-layer learning is identifiable: the dataset supplies semantic decomposition renders. It is not equivalent to guessing amodal ownership from ordinary FIT8 3D geometry.

### 5.3 Stage 3 — `PAPER_EXACT`

The system switches to mesh/FlexiCubes representation while retaining image/mask/semantic supervision and adding geometric supervision including rendered depth, normals, FlexiCubes deviation and hole regularization.

### RealSaS lesson — `REALSAS_INFERENCE`

The transferable rule is not “use their three stages verbatim”. It is:

> A representation transition must not silently discard the supervision needed to preserve capabilities already established, and later capabilities should be structurally isolated where possible.

V8 already implements the isolation part after M1. The remaining RealSaS work is to build stronger independent courts and specialist optimization rather than reopening the carrier to recreate cumulative-loss retention.

## 6. High-resolution coarse-to-fine extraction

### 6.1 Sparse active region proposal — `PAPER_EXACT`

StdGEN++ scales semantic-aware extraction from StdGEN's lower-resolution dense grid to a high-resolution full-body grid by:

1. evaluating a low-resolution coarse SDF;
2. constructing/dilating a surface occupancy region;
3. upsampling that mask;
4. evaluating fine SDF/deformation/semantic predictions only in the active high-resolution domain.

The paper reports a full-body grid of `256 x 256 x 384` and a facial branch grid of `180^3`.

### RealSaS translation — `REALSAS_INFERENCE`

IRIS currently uses a much smaller dense training extraction grid. A coarse-to-fine active-grid path is a serious candidate for later geometry fidelity work, especially thin structures, but it is **not** the immediate response to boundary/cut teacher debt.

Eligibility sequence:

1. independent topology/surface court exists;
2. current dense resolution is shown to be the limiting factor;
3. active-grid proposal preserves canonical surface/non-regression;
4. only then adopt a sparse high-resolution extractor.

## 7. Hole/sign regularization

### 7.1 StdGEN++ hole loss — `PAPER_EXACT`

StdGEN++ explicitly targets topological fractures in thin semantic subsets. Its hole regularizer operates on sign-sensitive neighboring SDF vertices and encourages interior connectivity while retaining a surface sign transition.

### 7.2 Public StdGEN empty-surface handling — `CODE_EXACT`

The public mesh implementation also detects fully-positive/fully-negative SDF volumes. It modifies selected center/boundary values to force opposite signs, computes a regularization signal and detaches bad examples for the affected extraction path.

### RealSaS decision — `REALSAS_INFERENCE`

Do **not** copy forced-sign mutation as product/scientific authority. Our surface collapse diagnostics should remain fail-closed.

Potentially adopt later:

- pre-failure sign-viability telemetry;
- continuity-conditioned sign regularization;
- topology/hole regularization selected by authoritative role-free continuity evidence.

Only introduce this after the independent TESSA/surface court demonstrates the concrete failure. It must be ablated against geometry preservation.

## 8. Data and truth: the largest architectural difference

### 8.1 StdGEN/StdGEN++ data model — `PAPER_EXACT` + public repo

The public StdGEN repository supplies train/test lists and Blender rendering code for the underlying character assets, including semantic map generation. StdGEN++ further describes hierarchical semantic renderings, head-centric data and separated facial texture layers. Their training targets are intentionally constructed around a known decomposition vocabulary.

This gives them truth that current generic FIT8 assets do not provide automatically:

- layer-specific render targets;
- semantic subset masks;
- explicitly decomposed hidden/interior supervision;
- dedicated face-scale supervision.

### 8.2 RealSaS data model — `REALSAS_INFERENCE`

RealSaS deliberately refuses to elevate an unidentifiable proxy into truth. Therefore the correct analogue is not to manufacture `hair/body/cloth` labels or anonymous owner IDs and call them semantics.

Our admissible structural corpus should distinguish:

- authored facts;
- deterministic geometric/raster derivations;
- qualified derivations;
- diagnostic-only evidence;
- unknown/abstained targets.

The `BoundaryCausalityTeacherV1` contract is the first formalization of this rule.

## 9. InstantMesh ancestry: exact open training lessons

StdGEN publicly acknowledges InstantMesh ancestry. InstantMesh's training implementation is available and provides useful lower-level evidence about the reconstruction family.

### 9.1 Separate NeRF and mesh training configurations — `CODE_EXACT`

InstantMesh exposes separate training entrypoints/configurations for NeRF and mesh representations. The mesh training configuration initializes from a NeRF checkpoint rather than training an unrelated mesh model from scratch.

### 9.2 Mesh-stage supervision — `CODE_EXACT`

`src/model_mesh.py` builds targets from RGB, alpha, depth and normal data. Its mesh loss combines:

- RGB MSE;
- LPIPS;
- mask loss;
- depth loss;
- normal loss;
- SDF/FlexiCubes regularizers.

This independently supports the broader lesson that mesh-stage training is not merely a topology token objective; geometric/raster constraints remain active.

### 9.3 Dataset sampling — `CODE_EXACT`

The released InstantMesh mesh config uses Objaverse-derived rendered data, six input views and four target views sampled from a larger rendered view bank. This is not a prescription for Fit8; it shows that the ancestral reconstruction model is trained by target-view rendering courts, not by a single scalar geometry objective.

## 10. IRIS re-evaluation after the re-audit

### 10.1 Keep

- one shared scene carrier `H` established by geometry/mechanics;
- no named body-part semantic taxonomy in the inference contract;
- role-free graphics evidence;
- Compiler as discrete/global consistency authority;
- sole canonical mechanical mesh;
- M1 post-carrier seal as current invariant;
- separate specialist heads;
- fail-closed geometry/surface courts.

### 10.2 Fix now

- current boundary/cut truth must stop conflating owner equivalence with causal boundary truth;
- post-V8 quality policy must be wired to the real trainer/optimizer path;
- head families require separate optimizer/convergence telemetry and held-out courts;
- Compiler consumption must be traced all the way to runtime behavior;
- TESSA needs an independent topology court.

### 10.3 Investigate after courts

- evidence-specific LoRA/adapters;
- sparse coarse-to-fine high-resolution extraction;
- continuity-conditioned hole/sign regularization;
- higher-resolution local geometry specialist;
- richer TESSA witness summaries from authoritative boundary/component evidence.

### 10.4 Reject for now

- reopening the shared carrier merely because a downstream head is slow;
- fixed body/hair/cloth ontology as the default RealSaS decomposition;
- forced SDF sign mutation as scientific/product authority;
- inferred hidden/amodal owner truth from generic FIT8 geometry;
- last-step checkpoint promotion without held-out selection.

## 11. New finding from re-reading current IRIS code

`models/iris/vnext/graphics_truth_v1.py` currently defines adjacent-pair `cut_probability` as `1 - same_drawable_affinity`, where `same_drawable_affinity` is derived from anonymous drawable IDs.

That makes `cut` a tautological complement of the current owner proxy, not an independently identified causal boundary target. This is now a **known truth debt** and must not be interpreted as independent cut evidence.

Replacement direction:

- preserve the historical V8 target for provenance;
- introduce a new truth schema with explicit validity/abstention;
- use `BoundaryCausalityTeacherV1` for provable boundary causes;
- do not manufacture a drawable cut target when authority is absent;
- train/evaluate cut only on samples where its target is independently defensible.

## 12. Current design conclusion

The re-audit strengthens the post-V8 direction rather than overturning it.

StdGEN++'s strongest lessons for RealSaS are:

1. protect a useful pretrained/shared representation from indiscriminate rewriting;
2. specialize capabilities with constrained branches/adapters when justified;
3. keep geometry/raster supervision alive through mesh refinement;
4. make decomposition supervision identifiable through the data rather than hoping the network infers hidden structure from ambiguous evidence;
5. scale geometry with coarse-to-fine active computation when resolution is proven to be the bottleneck;
6. treat thin-structure topology as an explicit optimization problem.

RealSaS adds a different authority split:

> the neural model predicts probabilistic structural evidence; Compiler mints globally consistent drawable/order/contact structure only from admissible evidence.

That distinction remains the preferred architecture unless future courts falsify it.
