# RealSaS-OPT — Current Plan

**Date:** 2026-10-16
**Status:** USER-AUTHORIZED NARROW ARCHITECTURE SIMPLIFICATION BEFORE FIT8
**Canonical code line:** `main` only
**Deferred:** broad Unified-IRIS / StdGEN++ architecture intervention and semantic teacher-corpus program

This plan intentionally narrows the intervention. RealSaS already produced a working Knight animated demo. The current objective is not to redesign the whole system. It is to remove one demonstrated information bottleneck between IRIS and TESSA while preserving the proven Compiler→M→AXIS→MIRA→CAA/runtime architecture.

---

## 0. Executive decision

The active product/research line returns to:

```text
8-view input
   ↓
IRIS signed scene representation
   ↓
TESSA topology proposal role
   ↓
Compiler → canonical M
   ↓
AXIS → G
   ↓
MIRA → W_M on exact M
   ↓
Compiler proof / CAA / presentation / runtime
```

The only architecture change authorized before Fit8 is:

> **TESSA must stop relearning the scene from a 17-D GSA summary. Its topology decoder should consume rich IRIS scene evidence directly.**

Everything else is held fixed unless this change exposes a concrete blocker.

---

## 1. Confirmed bottleneck

Current path:

```text
IRIS image tokens
  ↓
IRIS scene triplane
  ↓
SDF
  ↓
tetra zero surface
  ↓
GSA / RiggingSurface
  ↓
17-D node rows
  ↓
TESSA SurfaceEncoder (17 → 768, learned scene reconstruction)
  ↓
TESSA topology token decoder
```

The 17-D conditioning contains only:

```text
XYZ(3)
normal XYZ + valid(4)
8 view-support bits
graph degree(1)
persistence(1)
```

IRIS already contains a much richer spatial representation: in the current Knight/Fit8 configuration, `scene_planes` are `3 × 64 × 64 × 64` and are the same representation from which the continuous signed field is queried.

The information loss is therefore real and local: rich IRIS scene evidence is discarded and TESSA then trains a second surface encoder to infer a new latent from 17 numbers per GSA node.

---

## 2. Target simplification

### 2.1 Preserve IRIS geometry core

Do not rewrite `SceneFirstSignedGeometryV4` merely to implement this intervention.

Preserve:

- image/camera token fusion;
- current scene triplane;
- current continuous signed field;
- current Fit8 IRIS training objective and checkpoint semantics;
- deterministic tetra decoding/courts.

This is important because the stopped Fit8 IRIS checkpoint may remain resumable under the exact binding guards.

### 2.2 Turn TESSA into a topology head over IRIS scene memory

New learned path:

```text
IRIS scene_planes [B,3,64,64,64]
          ↓
IRIS→topology memory adapter
          ↓
fixed bounded topology memory [B,K,768]
          ↓
existing TESSA topology token decoder role
          ↓
TESSAMeshProposal / M*
```

Recommended first adapter:

1. flatten the three high-resolution IRIS planes into spatial tokens;
2. add explicit plane identity + normalized 2-D coordinate encoding;
3. project the source tokens to a compact key/value width;
4. use a fixed set of learned topology-memory queries (`K`, initially the existing 384-memory scale) to cross-attend once to the rich scene tokens;
5. allow at most a small bounded latent-refinement stack;
6. feed the resulting memory directly to the existing autoregressive topology decoder.

The adapter is an **information bridge/aggregator**, not a second scene model.

### 2.3 Retire the canonical 17-D topology input

For the new Fit8 topology path:

- `build_tessa_conditioning_v1()` is not the scene representation;
- `TESSASurfaceEncoderV1(17-D GSA)` is not the canonical topology encoder;
- GSA may still provide Compiler/mechanics evidence and diagnostics;
- the old TESSA path stays available for exact historical reproduction and A/B comparison.

### 2.4 Logical fusion, controlled training

IRIS and the TESSA topology role may be packaged under one wrapper/model boundary because they share the same scene representation.

However, the first Fit8 training intervention must remain causally controlled:

```text
phase A: IRIS geometry
  - resume the existing Fit8 checkpoint only if exact binding passes
  - otherwise restart the same frozen geometry recipe

phase B: topology head
  - load the sealed IRIS scene model
  - freeze IRIS initially
  - train one shared topology adapter + decoder across all 8 subjects

phase C: optional joint fine-tune
  - DEFERRED unless separately preregistered and justified by evidence
```

Thus the architecture can be one model at inference while the first scientific experiment still isolates the new topology intervention.

---

## 3. Compiler repositioning

The Compiler must be repositioned **logically**, not discarded.

The key rule:

> **Compiler courts no longer sit between IRIS scene understanding and TESSA scene understanding, because there is no second scene-understanding model. They still sit between learned proposal and canonical product authority.**

### 3.1 Pre-model authority retained

Keep:

- source/camera admission;
- Stage08 canonical normalization;
- exact source provenance.

### 3.2 Learned proposal produced early, held non-authoritative

One IRIS/topology inference may produce:

```text
signed field evidence
scene planes
learned topology proposal M*
```

`M*` is sealed as immutable learned evidence only. It has **no product authority** at this point.

### 3.3 Independent Compiler geometry court remains

From the signed field, independently execute the existing path:

```text
Stage12 deterministic tetra zero surface
→ Stage13 geometry substrate qualification
→ Stage14 GSA build
→ Stage15 qualified RiggingSurface
→ Stage17 qualified mechanical/component partition
```

These stages no longer compress the neural scene for topology prediction. Their jobs are:

- independent geometry witness;
- source/field lineage;
- qualified surface domain;
- structural/component evidence;
- later material/mechanical support binding;
- independent cross-check against learned `M*`.

### 3.4 Stage18 remains the meeting point

At Stage18 consume both:

```text
learned M*
+
qualified Stage12–17 evidence
```

Preserve/adapt the existing responsibilities:

- immutable proposal reference;
- bind proposal vertices to qualified support evidence;
- keep material/mechanical support separate from geometry authority;
- run generic static quality repair;
- record proposal→repaired reference lineage;
- require independent source/geometry checks;
- mint no product authority.

`TESSAMaterialSupportFieldIR` remains valid here as a **post-proposal support binding**. It must not be used as the neural topology input.

### 3.5 Stage19 remains canonical M authority

Stage19 remeasures the exact repaired candidate and seals canonical `M` only if the frozen policy passes.

Downstream remains unchanged:

```text
Stage19 M
  ↓
AXIS G
  ↓
MIRA W_M on exact M
  ↓
Compiler exact M/G/W proof
  ↓
CAA / presentation / motion / runtime
```

No AXIS, MIRA or skin-basis redesign is part of this intervention.

---

## 4. Why this is preferable to the broad Unified-IRIS redesign now

This change:

- fixes a demonstrated information bottleneck;
- reuses the IRIS representation that already trained on Knight/Fit8;
- reuses the TESSA topology grammar/decoder work;
- preserves deterministic tetra/GSA/Compiler courts;
- preserves canonical M and downstream AXIS/MIRA contracts;
- does not require a new semantic ontology;
- does not require an appearance teacher corpus;
- does not require one-image synthesis now;
- does not reopen CAA/presentation architecture;
- is directly testable against the old 17-D TESSA path.

The broad research remains deferred, not rejected.

---

## 5. Required implementation

### 5.1 New model pieces

Add a checkpoint-safe module outside the existing IRIS core source, tentatively:

```text
models/iris/topology/scene_memory_v1.py
models/iris/topology/head_v1.py
```

Responsibilities:

- consume existing IRIS `scene_planes`;
- deterministic plane/coordinate tokenization;
- bounded learned memory aggregation;
- reuse TESSA token decoder blocks/grammar;
- emit the existing `TESSAMeshProposalV1` contract where practical.

Prefer refactoring decoder-only TESSA code into a reusable class rather than copying decoder logic.

### 5.2 Backward compatibility

Do not delete:

- `models/tessa/v1/conditioning_v1.py`;
- `TESSASurfaceEncoderV1`;
- old historical checkpoint loaders;
- legacy TESSA proposal schemas.

Mark them compatibility/reproduction paths once the new route is qualified.

### 5.3 New tests

At minimum prove:

1. topology memory shape is bounded and independent of raw scene token count;
2. plane permutation/coordinate semantics are explicit and deterministic;
3. changing rich IRIS scene features changes topology memory even if legacy 17-D GSA rows are held constant;
4. the new topology head does not require teacher geometry during inference;
5. the emitted learned proposal still carries non-authoritative provenance;
6. Stage18/19 still fail closed when proposal/support/lineage mismatches occur;
7. legacy Knight downstream M→AXIS→MIRA contracts remain untouched.

### 5.4 A/B scientific test

Use the same Fit8 cohort and teacher geometry.

Compare:

```text
A: legacy GSA 17-D → TESSA surface encoder → decoder
B: IRIS scene planes → topology memory adapter → same decoder role
```

Primary topology/mechanical measurements should include:

- teacher/source topology reconstruction where defined;
- static mesh qualification failures;
- repair magnitude/count;
- source silhouette fidelity;
- non-manifold/sliver/flip/self-intersection metrics;
- final canonical-M dynamic consequences under the same AXIS/MIRA protocol.

Do not declare success merely from lower topology token loss.

---

## 6. Fit8 restart policy

Fit8 subject identities remain unchanged:

```text
Snake
Bat
T-Rex
Horse
Spider
GreenBlob
Shark
George
```

Rules remain:

- one shared parameter set per learned model/head across all subjects;
- no subject-specific production branches;
- same Compiler contracts;
- Knight remains separate regression;
- Fit8 is architecture closure, not unseen-generalization proof.

The stopped IRIS checkpoint around global step 4400 may be reused only if existing atomic resume/binding checks pass exactly. The implementation should deliberately avoid changing the existing IRIS source hash before this compatibility decision is made.

If the checkpoint cannot resume under exact binding, restart the same IRIS recipe rather than weakening the guard.

---

## 7. Explicitly deferred research

Preserve for later, but do not activate now:

- PR #113 broad Unified-IRIS core contracts;
- PR #114 semantic partition census;
- generic semantic/owner teacher corpus;
- one-image canonical-view synthesis;
- appearance/RGBA head;
- semantic/ownership head;
- FlexiCubes-like extraction head;
- StdGEN-class full-capacity replacement;
- broad topology/appearance/semantics multi-task training;
- joint IRIS/AXIS/MIRA mechanics model;
- presentation-system redesign.

Revisit only after Fit8 or when a measured failure specifically demands one of them.

---

## 8. Immediate order of operations

```text
1. freeze broad research state (done)
2. implement scene-plane topology memory adapter without changing existing IRIS core checkpoint semantics
3. refactor/reuse TESSA decoder-only role
4. wire non-authoritative proposal artifact into existing Stage18 consumer
5. run focused model + Compiler regression tests
6. inspect self-hosted/CI result
7. verify stopped Fit8 IRIS checkpoint exact-resume compatibility
8. resume or exact-recipe restart IRIS Fit8
9. train shared topology head
10. qualify M for all eight
11. proceed to AXIS then MIRA
```

When there is ambiguity, choose the option that changes fewer proven downstream contracts.
