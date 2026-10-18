# IRIS vNext ↔ StdGEN++ Re-audit — 2026-10-18

Status: **ACTIVE DESIGN EVIDENCE / NOT AN AUTHORITY ARTIFACT**

Purpose: re-evaluate StdGEN++, public StdGEN code, InstantMesh ancestry and the current IRIS vNext architecture after V8. The goal is not to copy a semantic ontology. The goal is to identify transferable training, representation, data and extraction disciplines while separating public-code facts, paper claims, RealSaS inferences and unavailable implementation details.

## Evidence levels

- `CODE_EXACT`: directly verified in public source code.
- `PAPER_EXACT`: directly stated or derived from the published/preprint paper.
- `REALSAS_INFERENCE`: our engineering translation; not a claim about StdGEN++ implementation.
- `UNAVAILABLE`: exact implementation is not publicly inspectable.

## Source pins used for this re-audit

- StdGEN++ paper: arXiv `2601.07660`.
- public StdGEN: `hyz317/StdGEN@f1b0c822689dc2539c774bb73090a67c66e2e766`.
- public InstantMesh: `TencentARC/InstantMesh@08822c5200a4bcc0e2c1a4573409ac10b9f6692c`.

As of 2026-10-18, the inspected public StdGEN repository exposes model/inference, rendering/data support and pretrained checkpoints, but not the exact full StdGEN/StdGEN++ training loop. Open issues still request training code. Exact optimizer-loop/curriculum implementation absent from source must therefore remain `PAPER_EXACT` or `UNAVAILABLE`, never mislabeled `CODE_EXACT`.

## 1. What V8 changed in how we should read StdGEN++

V5/V6/V8 separate two questions:

1. can the shared representation encode geometry/mechanics/graphics structure?
2. can downstream objectives safely keep rewriting that representation?

V8 supports the first under current Fit8 teacher definitions: after M1, `scene_core` was sealed, geometry/extraction stayed invariant and downstream appearance/relation/topology objectives still learned.

V5/V6 answer the second negatively for our curriculum: unrestricted downstream rewriting of shared H is unsafe; cumulative losses alone did not guarantee geometry retention.

Important optimizer nuance: V8 already had carrier sealing and stage-level specialist isolation, but historical V8 physically grouped mechanical/appearance/relation parameters together under one `evidence_heads` optimizer group. Post-V8 `optimization_v2.py` merely separates future specialist groups/LRs/telemetry; it is not the origin of the carrier-freeze idea.

Therefore StdGEN++ is most useful to RealSaS as a study of **specialization, decomposition-aware supervision, geometric continuity, data construction and staged representation use**, not as a vocabulary source.

## 2. Public StdGEN model structure — `CODE_EXACT`

### LoRA-constrained adaptation

In the public predecessor code, when LoRA mode is enabled:

- transformer LoRA parameters are marked trainable;
- transformer positional embedding and deconvolution are explicitly re-enabled;
- the image encoder is frozen;
- the synthesizer remains trainable.

This is stronger than “everything keeps training”, but weaker than “all shared representation parameters are frozen”.

### Shared sampled feature, separated decoders

The public mesh synthesizer defines separate decoders for:

- SDF;
- RGB;
- deformation;
- 21-D FlexiCubes weights;
- semantic probabilities.

All decode from the same sampled triplane feature.

### Semantics are structurally active

The public mesh implementation consumes semantic probabilities while constructing semantic-equivalent SDFs for selected layers. Semantic output therefore changes extraction behavior; it is not merely a reporting classifier.

### RealSaS translation — `REALSAS_INFERENCE`

Removing named semantics means replacing their **functions**, not just deleting label names. RealSaS distributes those functions across:

- role-free relation/boundary evidence;
- Compiler fragment/order/contact graphs;
- topology conditioning;
- source/geometry provenance;
- future structural-continuity evidence where authoritative.

V8's post-M1 carrier seal is intentionally stricter than the public StdGEN LoRA configuration. If a future clean target is not decodable from sealed H, the first eligible response is the smallest specialist adapter/branch, not broad carrier reopening.

## 3. StdGEN++ dual specialization — `PAPER_EXACT`

StdGEN++ introduces dedicated full-body and facial LoRA branches. The paper describes LoRA inserted into attention projections with separate trainable branch parameters.

Transferable idea: **specialize scale/task capability without forcing one monolithic update path to solve everything**.

RealSaS should not copy anatomical branch names. If future courts justify specialization, branch by evidence domain or scale, e.g. boundary/visual, relations, topology, or local high-resolution geometry.

Generic LoRA is **not** authorized today because V8 already shows substantial A/R/T decodability from frozen H and current weak families still have truth/conditioning/support debt.

## 4. StdGEN++ training stages — `PAPER_EXACT`

### Stage 1

Initializes from pretrained InstantNeRF and trains newly added LoRA plus semantic decoder under image/mask/semantic supervision.

### Stage 2

Uses hierarchical multi-layer supervision from outside to inside. Semantic subsets are masked during rendering so internal color/semantic information receives direct 2D supervision.

Important data lesson: hidden-layer learning is identifiable because decomposition targets are explicitly produced by the dataset. This is not equivalent to guessing amodal ownership from generic FIT8 3D.

### Stage 3

Transitions to mesh/FlexiCubes while continuing prior image/mask/semantic supervision and adding geometric terms including depth, normal, FlexiCubes deviation and hole regularization.

### RealSaS lesson

> Representation transitions must not silently discard supervision needed to preserve established capabilities, and later capabilities should be structurally isolated where possible.

For RealSaS, V8 already supplies post-M1 isolation. We should not reopen the carrier merely to recreate cumulative-loss retention.

## 5. Coarse-to-fine high-resolution extraction — `PAPER_EXACT`

StdGEN++ scales extraction by:

1. evaluating a coarse SDF;
2. constructing/dilating a surface occupancy region;
3. upsampling the active mask;
4. evaluating fine SDF/deformation/semantic predictions only in the active region.

The paper reports a full-body grid of `256 x 256 x 384` and a facial branch of `180^3`.

### RealSaS decision — `REALSAS_INFERENCE`

This is a serious later candidate for thin-structure/geometry fidelity, but not the immediate answer to boundary/cut truth debt.

Eligibility:

1. independent topology/surface court exists;
2. current resolution is demonstrated to be limiting;
3. sparse high-resolution extraction preserves canonical-surface/non-regression gates;
4. only then adopt it.

## 6. Hole/sign regularization

### StdGEN++ — `PAPER_EXACT`

StdGEN++ explicitly targets thin-structure topological fractures with a sign-sensitive neighboring-SDF regularizer that encourages interior connectivity while retaining a surface transition.

### Public StdGEN empty-field handling — `CODE_EXACT`

The public mesh implementation detects fully-positive/fully-negative SDF volumes and mutates selected values to force opposite signs while producing regularization and detaching affected bad examples.

### RealSaS decision — `REALSAS_INFERENCE`

Do **not** copy forced-sign mutation as scientific/product authority. IRIS surface collapse remains fail-closed.

Potential transfers only after topology court evidence:

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

RealSaS deliberately refuses to elevate unidentifiable proxies into truth. Its structural corpus distinguishes:

- authored facts;
- deterministic derivations;
- qualified derivations;
- diagnostic-only evidence;
- unknown/abstained targets.

`BoundaryCausalityTeacherV1` and Graphics Truth V2 formalize this direction.

## 8. InstantMesh ancestry — `CODE_EXACT`

InstantMesh public training source provides a useful lower-level cross-check:

- separate NeRF and mesh training configurations exist;
- mesh training initializes from a NeRF checkpoint;
- mesh-stage supervision includes RGB, LPIPS, mask, depth, normal and SDF/FlexiCubes regularization.

This independently supports the lesson that mesh-stage learning is not just a topology-token objective.

## 9. Re-evaluation of current IRIS after the latest post-V8 source work

### Keep

- one shared carrier H established by geometry/mechanics;
- no mandatory body-part semantic taxonomy;
- role-free graphics evidence;
- Compiler as discrete/global consistency authority;
- one canonical mechanical mesh;
- M1 post-carrier seal as baseline invariant;
- fail-closed geometry/surface courts.

### Now implemented / green at source level

- exact post-M1 carrier guard (`quality_policy_v1.py`);
- explicit Graphics Truth V2 authority/validity contract;
- Relation Truth V2 sidecar/manifest producer;
- relation support/class-balance court eligibility;
- `GraphicsRelationHeadV2` with projected boundaries moved from camera-free static prediction to camera-conditioned view prediction;
- deterministic V1→V2 head migration semantics;
- head-separated specialist optimizer + gradient/update telemetry (`optimization_v2.py`).

These are source-contract achievements, **not** scientific PASS.

### Fix next

- materialize/seal actual Relation Truth V2 corpus;
- audit support/class balance/abstention per family;
- create a real V8 checkpoint→V2 migration receipt/checkpoint;
- run focused frozen-carrier specialist optimization;
- close Compiler→runtime consumption gaps;
- build independent TESSA topology court.

### Investigate only after courts

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

## 10. Relation-truth correction from current IRIS audit

Two historical producer paths exist and must not be conflated.

### `graphics_truth_v1.py` helper

```text
same_drawable = anonymous owner equality
cut           = 1 - same_drawable
```

### V8 Fit8 shard builder

```text
same      = (owner_a == owner_b)
cut proxy = (owner_a != owner_b) OR (material_id_a != material_id_b)
```

Therefore V8 cut residuals measure decodability of an **owner-or-material boundary proxy**, not authoritative drawable-cut learning.

### Replacement state

Source now exists for:

- explicit target authority/provenance/validity;
- versioned Truth V2 sidecar/manifest production;
- support/class statistics;
- conditioning-matched relation head;
- focused optimizer telemetry.

Still missing:

- sealed actual Fit8 Truth V2 corpus;
- real V8→V2 migrated checkpoint/receipt;
- focused training;
- held-out scientific PASS.

Current valid `occlusion_probability` rows are single-class positive support. This does not make the target false, but it does make binary discrimination court-ineligible until authoritative negative support exists. Do not fabricate negatives.

## 11. Design conclusion

The re-audit strengthens the post-V8 direction rather than overturning it.

The strongest transferable StdGEN++ lessons are:

1. protect a useful pretrained/shared representation from indiscriminate rewriting;
2. specialize capabilities with constrained branches/adapters only when justified;
3. retain geometric/raster supervision through representation transitions;
4. make decomposition supervision identifiable through data rather than hoping hidden structure emerges from ambiguous evidence;
5. scale geometry with coarse-to-fine active computation when resolution is proven limiting;
6. treat thin-structure topology as an explicit optimization problem.

RealSaS adds a different authority split:

> neural models predict probabilistic structural evidence; Compiler mints globally consistent fragment/order/contact structure only from admissible evidence.

That remains the preferred architecture unless future courts falsify it.
