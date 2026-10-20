# IRIS Shared Structural Evidence Plan — 2026-10-20

Status: **ACTIVE SOURCE-ONLY CONTINUATION / NO SCIENTIFIC RUN AUTHORIZED**

Long-term research branch: `research/iris-vnext-hybrid-tessa-v2-20261019`

This plan is the active continuation after the sealed `H-GEO-P3-SCALE-XFER` result. It does not replace the IRIS→TESSA V2 architectural split; it makes explicit the role-free geometry+graphics+mechanics hypothesis that the current source only partially implements.

## 1. Research thesis

RealSaS does not require a named semantic taxonomy (`arm`, `cape`, `hair`, ... ) to be the structural authority for 2D presentation.

The hypothesis is:

```text
8-view observations
        ↓
IRIS shared structural scene
        ↓
geometry + graphics + mechanics evidence
        ↓
role-free structuralization / topology reconstruction
        ↓
Compiler
        ↓
ONE persistent editable canonical asset
        ↓
3D mechanics → source-preserved 2D presentation
```

The scientifically relevant question is not "what semantic class is this region?" but:

- what remains connected / separate;
- what deforms together;
- what continues behind what;
- what is attached;
- what is visible under a camera and pose;
- what local draw-order relation is required;
- which source appearance support belongs to the persistent structural element.

Semantics may later be auxiliary metadata. They are not canonical structural authority.

## 2. What is already implemented / supported

### 2.1 Shared-scene evidence

IRIS V8 provides a sealed shared scene carrier. Post-M1 appearance, relation and topology heads can learn from that frozen carrier without rewriting geometry. This supports the existence of useful cross-modal evidence in H, but it is not a proof of jointly trained geometry+graphics scene learning.

### 2.2 Role-free graphics/runtime path

The repository already contains role-free relation/appearance evidence and Compiler/runtime consumption paths for fragment clustering, order edges and visibility behavior. Historical relation truth was partly proxy/misaligned; therefore relation-quality claims must use corrected authority only.

### 2.3 Relation abstraction correction

Pairwise front/order and global visibility are different problems. The current direction keeps:

- geometry-explicit, camera-conditioned pair evidence for front/occlusion/boundary;
- complete ray-stack/view evidence for global visibility;
- ABSTAIN/validity when source authority is absent.

### 2.4 TESSA structuralization

TESSA V2 already implements a separately typed scene-memory + surface/mechanics-memory autoregressive topology structuralizer with adjacency-aware topology serialization. It is proposal-only; Compiler alone may mint canonical M.

Current TESSA V2 does **not** yet receive explicit appearance boundary, alpha boundary, front/occlusion or ray-stack visibility evidence as a typed structural input.

## 3. Geometry boundary after SCALE-XFER

`H-GEO-P3-SCALE-XFER` is sealed `MIXED`: target multi-surface structure, frozen view-token spatial separability and carrier transfer all contributed. The result did not authorize P4, SURF-A2 or TESSA training.

Geometry remains required, but it is treated as a typed provider boundary:

```text
QUALIFIED_SURFACE_PROVIDER
        ↓
XYZ / normals / local scene evidence / mechanics
```

The provider may later be a RealSaS P4 lineage or another preregistered geometry specialist. No automatic P4 follows from `MIXED`.

The endpoint diagnostic remains evidence about the current metric carrier lineage. It must not silently become a universal product requirement without proving that it is necessary for qualified final-surface / animation behavior.

## 4. Active source-only workstream — STRUCTURAL_EVIDENCE_V3

No optimizer, notebook scientific run, diagnostic run or scientific court is authorized by this section.

### 4.1 Truth/readiness

Complete and audit source-defensible structural evidence:

- per-surface geometry: XYZ, normal, local H, mechanics;
- appearance/source support: RGBA/material support, alpha/appearance boundaries;
- pair relations: front/order, occlusion, structural-continuation/attachment only when authoritative;
- ray-stack evidence: complete ordered hits + global visibility;
- motion evidence: deformation coherence, attachment persistence, bend/stretch/contact validity where authoritative;
- explicit validity/ABSTAIN for unavailable truth.

No owner/drawable label may be fabricated from material IDs or semantic names.

### 4.2 Presentation-conditioned TESSA source design

Extend the structuralizer source contract without changing canonical authority:

```text
SCENE MEMORY
shared IRIS context

SURFACE / MECHANICS MEMORY
XYZ + normal + H + deformation evidence

PRESENTATION MEMORY
appearance/source boundaries
pair front/occlusion evidence
ray-stack visibility evidence
validity masks
        ↓
TESSA topology proposal decoder
```

The first implementation must preserve the existing tokenizer/grammar and as much decoder capacity as possible so the later experiment isolates the value of presentation evidence rather than confounding it with a new topology language.

### 4.3 Compiler contract

TESSA proposes persistent structural topology. IRIS provides evidence. Compiler remains responsible for qualification and product authority.

For pose `t` and camera `v`, Compiler/runtime must be able to derive or consume:

- persistent fragment/owner support;
- visible fragment set;
- partial/local draw-order graph;
- attachment anchors;
- source-art sample/support mapping;
- pose-dependent order flips without changing canonical structural identity.

## 5. Next scientific experiment — not yet authorized

After source/readiness gates are green, preregister one controlled A/B experiment:

```text
A — CURRENT STRUCTURALIZER
scene + surface + mechanics
        ↓
TESSA

B — REALSAS STRUCTURAL-EVIDENCE HYPOTHESIS
same scene + same surface + same mechanics
+ role-free appearance/boundary evidence
+ pair front/occlusion evidence
+ complete ray-stack visibility evidence
        ↓
TESSA
```

Controls:

- same sealed IRIS authority unless the preregistration explicitly says otherwise;
- same subjects / cameras / teacher corpus;
- same topology tokenizer/grammar;
- matched decoder capacity where practical;
- teacher topology never enters inference conditioning;
- no named semantic taxonomy in B;
- identical Compiler qualification.

Primary downstream measurements must include topology validity, structural/source fidelity, fragment stability, global visibility correctness, local/order correctness, pose-dependent order flips, motion owner stability, gap/tear behavior and source-art preservation.

A semantic-control arm may be added later; it is not required for the first A/B.

## 6. Scientific execution boundary

The assistant may continue source-only work autonomously under this plan:

1. authority/doc synchronization;
2. truth/readiness source audits;
3. typed evidence APIs;
4. TESSA presentation-memory source implementation;
5. regression/source tests and CI repairs;
6. preregistration drafting.

Stop and hand off to the user before any step that requires:

- a new notebook to be run;
- a scientific diagnostic/court execution;
- an A100/accelerator experiment;
- optimizer/backward/training steps;
- new scientific threshold evaluation on run outputs.

No such execution is authorized implicitly by `continue`.

## 7. Decision logic after the future A/B

If B materially and preregisteredly beats A, the next research question is whether geometry+graphics supervision can enrich the shared scene without geometry forgetting. Only then consider a protected joint-training experiment.

If B does not beat A, do not assume more joint training will fix it; first localize whether the presentation evidence, truth, structuralizer consumption or Compiler contract failed.

## 8. Product invariant

There is one canonical animation-ready asset. There is no independent deformable visual mesh used to hide mechanical/structural debt.

```text
observations
   ↓
shared structural scene + qualified metric surface
   ↓
role-free structuralization
   ↓
Compiler-qualified canonical M/G/W + graphics structure
   ↓
motion
   ↓
source-preserved editable 2D presentation
```
