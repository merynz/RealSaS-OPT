# IRIS vNext ↔ StdGEN++ Re-audit — 2026-10-18

Status: **ACTIVE DESIGN EVIDENCE / NOT AN AUTHORITY ARTIFACT**

Purpose: re-evaluate StdGEN++, public StdGEN code, InstantMesh ancestry and the current IRIS vNext architecture after V8. The goal is not to copy a semantic ontology. The goal is to identify transferable training, representation, data and extraction disciplines while separating public-code facts, paper claims, RealSaS inferences and unavailable implementation details.

## Evidence levels

- `CODE_EXACT`: directly verified in public source code.
- `PAPER_EXACT`: directly stated or derived from the published/preprint paper.
- `REALSAS_INFERENCE`: our engineering translation; not a claim about StdGEN++ implementation.
- `UNAVAILABLE`: exact implementation is not publicly inspectable.

Primary sources:

- StdGEN++ paper: https://arxiv.org/abs/2601.07660
- StdGEN CVPR 2025 paper: https://openaccess.thecvf.com/content/CVPR2025/html/He_StdGEN_Semantic-Decomposed_3D_Character_Generation_from_Single_Images_CVPR_2025_paper.html
- public StdGEN source: https://github.com/hyz317/StdGEN
- InstantMesh source/training ancestry: https://github.com/TencentARC/InstantMesh

As of 2026-10-18, the public StdGEN repository exposes model/inference, dataset/rendering support and pretrained checkpoints, while open issues still request the training code. Exact optimizer-loop behavior absent from source must therefore not be described as code-verified.

## 1. What V8 changed in how we should read StdGEN++

V5/V6/V8 separate two questions:

1. can the shared representation encode geometry/mechanics/graphics structure?
2. can downstream objectives safely keep rewriting that representation?

V8 supports the first under current Fit8 teacher definitions: after M1, `scene_core` was sealed, geometry/extraction stayed invariant and downstream appearance/relation/topology objectives still learned.

V5/V6 answer the second negatively for our curriculum: unrestricted downstream rewriting of shared H is unsafe; cumulative losses alone did not guarantee geometry retention.

Therefore StdGEN++ is most useful to RealSaS as a study of **specialization, decomposition-aware supervision, geometric continuity and staged representation use**, not as a vocabulary source.

## 2. Public StdGEN model structure — `CODE_EXACT`

### LoRA-constrained adaptation

In `slrm/models/lrm.py` and `slrm/models/lrm_mesh.py`, when LoRA is enabled:

- `lora.mark_only_lora_as_trainable(self.transformer)` is applied;
- transformer positional embedding and deconvolution parameters are explicitly re-enabled;
- the image encoder is frozen;
- the synthesizer remains trainable.

This is stronger than “everything keeps training”, but weaker than “all downstream capacity except the new head is frozen”.

### Shared sampled feature, separated decoders

`slrm/models/renderer/synthesizer_mesh.py` defines separate decoders for:

- SDF;
- RGB;
- deformation;
- 21 FlexiCubes weights;
- four-way semantic prediction.

All decode from the same sampled triplane feature.

### Semantics are structurally active

The public mesh implementation consumes semantic probabilities while constructing equivalent SDFs for selected layers. Their semantic output therefore changes extraction behavior; it is not merely a reporting classifier.

### RealSaS translation — `REALSAS_INFERENCE`

Removing named semantics means replacing their **functions**, not merely deleting their label names. RealSaS distributes those functions across:

- role-free relation/boundary evidence;
- Compiler fragment/order/contact graphs;
- topology conditioning;
- source/geometry provenance;
- future structural-continuity evidence where authoritative.

V8's full post-M1 carrier seal is intentionally stricter than the public StdGEN LoRA configuration. If a future clean target is not decodable from sealed H, the first eligible response is the smallest specialist adapter/branch, not broad carrier reopening.

## 3. StdGEN++ dual specialization — `PAPER_EXACT`

StdGEN++ introduces dedicated full-body and facial LoRA branches. The paper describes LoRA inserted into attention linear projections with separate trainable branch parameters.

Transferable idea: **specialize scale/task capability without forcing one monolithic update path to solve everything**.

RealSaS should not copy anatomical branch names. If future courts justify specialization, branch by evidence domain or scale, e.g. boundary/visual, relations, topology, or local high-resolution geometry.

## 4. StdGEN++ training stages — `PAPER_EXACT`

### Stage 1

Initializes from pretrained InstantNeRF and trains newly added LoRA plus semantic decoder under image, mask and semantic supervision.

### Stage 2

Uses hierarchical multi-layer supervision from outside to inside. Semantic subsets are masked during rendering so internal color/semantic information receives direct 2D supervision.

Important data lesson: hidden-layer learning is identifiable because decomposition targets are explicitly produced by the dataset. This is not equivalent to guessing amodal ownership from generic FIT8 3D.

### Stage 3

Transitions to mesh/FlexiCubes while continuing image/mask/semantic supervision and adding geometric terms including depth, normal, FlexiCubes deviation and hole regularization.

### RealSaS lesson

The transferable rule is:

> Representation transitions must not silently discard supervision needed to preserve established capabilities, and later capabilities should be structurally isolated where possible.

For RealSaS, V8 already supplies post-M1 isolation. We should not reopen the carrier merely to recreate cumulative-loss retention.

## 5. Coarse-to-fine high-resolution extraction — `PAPER_EXACT`

StdGEN++ scales extraction by:

1. evaluating a low-resolution coarse SDF;
2. constructing/dilating a surface occupancy region;
3. upsampling the active mask;
4. evaluating fine SDF/deformation/semantic predictions only in the active high-resolution region.

The paper reports a full-body grid of `256 x 256 x 384` and a facial branch of `180^3`.

### RealSaS decision — `REALSAS_INFERENCE`

This is a serious later candidate for thin-structure/geometry fidelity, but not the immediate answer to current boundary/cut truth debt.

Eligibility:

1. independent topology/surface court exists;
2. current resolution is demonstrated to be the limiting factor;
3. sparse high-resolution extraction preserves canonical-surface/non-regression gates;
4. only then adopt it.

## 6. Hole/sign regularization

### StdGEN++ — `PAPER_EXACT`

StdGEN++ explicitly targets thin-structure topological fractures with a sign-sensitive neighboring-SDF regularizer that encourages interior connectivity while retaining a surface transition.

### Public StdGEN empty-field handling — `CODE_EXACT`

The public mesh implementation detects fully-positive/fully-negative SDF volumes and alters selected center/boundary values to force opposite signs while producing regularization and detaching affected bad examples.

### RealSaS decision — `REALSAS_INFERENCE`

Do **not** copy forced-sign mutation as scientific/product authority. IRIS surface collapse remains fail-closed.

Potential future transfers only after topology court evidence:

- pre-failure sign-viability telemetry;
- continuity-conditioned sign regularization;
- role-free hole/topology regularization with authoritative selectors.

## 7. Data/truth is the largest architectural difference

StdGEN/StdGEN++ deliberately construct training truth around decomposition. Their data pipeline provides semantic maps/layers; StdGEN++ describes hierarchical semantic renders, face-centric data and separated facial texture layers.

That supplies targets generic FIT8 assets do not automatically give us:

- layer-specific render targets;
- semantic subset masks;
- decomposed hidden/interior supervision;
- dedicated face-scale supervision.

RealSaS deliberately refuses to elevate unidentifiable proxies into truth. Its structural corpus therefore distinguishes:

- authored facts;
- deterministic derivations;
- qualified derivations;
- diagnostic-only evidence;
- unknown/abstained targets.

`BoundaryCausalityTeacherV1` and `Graphics Truth V2` formalize this direction.

## 8. InstantMesh ancestry — `CODE_EXACT`

StdGEN publicly acknowledges InstantMesh ancestry. InstantMesh training source is available and gives useful lower-level evidence.

### Separate NeRF and mesh training

InstantMesh provides separate NeRF and mesh training configurations. The mesh config initializes from a NeRF checkpoint.

### Mesh-stage supervision

`src/model_mesh.py` builds targets from RGB, alpha, depth and normal data and combines:

- RGB MSE;
- LPIPS;
- mask loss;
- depth loss;
- normal loss;
- SDF/FlexiCubes regularizers.

This independently supports the lesson that mesh-stage learning is not just a topology-token objective.

### Dataset sampling

The released mesh config uses Objaverse-derived renders, six input views and four target views sampled from a larger rendered view bank. This is not a prescription for Fit8; it demonstrates target-view reconstruction supervision in the ancestry.

## 9. Re-evaluation of current IRIS

### Keep

- one shared carrier H established by geometry/mechanics;
- no mandatory body-part semantic taxonomy;
- role-free graphics evidence;
- Compiler as discrete/global consistency authority;
- one canonical mechanical mesh;
- M1 post-carrier seal as baseline invariant;
- separate specialist heads;
- fail-closed geometry/surface courts.

### Fix now

- boundary/decomposition truth;
- validity-aware relation supervision;
- head-specific convergence telemetry and held-out courts;
- full Compiler→runtime consumption lineage;
- independent TESSA topology court.

### Investigate after courts

- evidence-specific LoRA/adapters;
- sparse high-resolution extraction;
- continuity-conditioned hole/sign regularization;
- local high-resolution geometry specialist;
- richer TESSA boundary/component summaries.

### Reject for now

- reopening shared H because a specialist is slow;
- fixed body/hair/cloth ontology as RealSaS default;
- forced SDF sign repair as authority;
- inferred hidden/amodal owner truth from generic FIT8 geometry;
- last-step checkpoint promotion without held-out selection.

## 10. Relation-truth correction from current IRIS source audit

Two historical producer paths exist and must not be conflated.

### `graphics_truth_v1.py` helper

For adjacent faces:

```text
same_drawable = anonymous owner equality
cut           = 1 - same_drawable
```

This helper's cut is a tautological complement of an anonymous-owner proxy.

### Active Fit8 shard builder used by V8

`build_fit8_vnext_shards_v1.py::_build_view_banks`, inherited by the active v2 builder for relation rows, writes:

```text
same = (owner_a == owner_b)
cut proxy = (owner_a != owner_b) OR (material_id_a != material_id_b)
```

Therefore V8 cut residuals measure decodability of an **owner-or-material boundary proxy**. They must not be interpreted as pure `1-same` or as authoritative drawable-cut learning.

Exact correction: `docs/research/IRIS_VNEXT_RELATION_TRUTH_AUDIT_20261018.md`.

### Replacement

Implemented source contracts:

- `models/iris/vnext/graphics_truth_v2.py`
- `models/iris/vnext/training_v2.py`
- `tests/iris/test_graphics_truth_v2.py`

V2 requires value + evidence authority + provenance + explicit `valid_for_training`. The loss consumes explicit validity masks and abstains per family/sample instead of inventing all-negative truth.

Historical V8 shards remain immutable. V2 needs a new producer/corpus with support statistics before any focused training experiment.

## 11. Design conclusion

The re-audit strengthens the post-V8 direction rather than overturning it.

The strongest transferable StdGEN++ lessons are:

1. protect a useful pretrained/shared representation from indiscriminate rewriting;
2. specialize capabilities with constrained branches/adapters when justified;
3. retain geometric/raster supervision through mesh refinement;
4. make decomposition supervision identifiable through data rather than hoping hidden structure emerges from ambiguous evidence;
5. scale geometry with coarse-to-fine active computation when resolution is proven limiting;
6. treat thin-structure topology as an explicit optimization problem.

RealSaS adds a different authority split:

> neural models predict probabilistic structural evidence; Compiler mints globally consistent fragment/order/contact structure only from admissible evidence.

That remains the preferred architecture unless future courts falsify it.
