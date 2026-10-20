# IRIS Shared Structural Evidence Plan — 2026-10-20

Status: **SOURCE BRIDGE + MATERIALIZATION AUTHORITY COMPLETE / NOTEBOOK HANDOFF NOW / NO SCIENTIFIC RUN AUTHORIZED**

Long-term research branch: `research/iris-vnext-hybrid-tessa-v2-20261019`

Materialization source gate: PR #133 — **MERGED**  
Frozen materialization implementation source: `e9f8e42445f0edca28a869a9164e54892095d9d4`

This is the active continuation after the sealed `H-GEO-P3-SCALE-XFER` result. It defines the role-free geometry+graphics+mechanics hypothesis and the controlled path by which that hypothesis may eventually be tested.

## 1. Research thesis

RealSaS does not require a named semantic taxonomy (`arm`, `cape`, `hair`, ... ) to be the structural authority for 2D presentation.

```text
8-view observations
        ↓
IRIS shared structural scene
        ↓
geometry + graphics + mechanics evidence
        + qualified metric surface provider
        ↓
role-free structuralization / topology reconstruction
        ↓
Compiler
        ↓
ONE persistent editable canonical asset
        ↓
3D mechanics → source-preserved 2D presentation
```

The scientifically relevant questions are structural: what remains connected/separate, what deforms together, what continues behind what, what is attached, what is visible under a camera and pose, what local draw-order relation is required, and which source appearance support belongs to the persistent structural element. Semantics may be auxiliary metadata; they are not canonical structural authority.

## 2. Implemented source boundary

### 2.1 Shared-scene evidence

IRIS V8 provides a sealed shared scene carrier. Post-M1 appearance, relation and topology heads can learn from that frozen carrier without rewriting geometry. This supports useful cross-modal evidence in H; it is not proof of full joint geometry+graphics shared-scene training.

### 2.2 Relation / visibility authority

Historical V8 relation truth was partly proxy/misaligned and is not scientific authority for order/visibility quality. Current V3 source separates:

- pairwise front/order: depth ordering between two same-view surface samples;
- positive occlusion support: same-ray overlap where source-defensible;
- global visibility: rank-zero membership in a **complete ordered ray stack**;
- incomplete global visibility: ABSTAIN / validity false.

Pairwise front is never promoted to global visibility. Material IDs and named semantic labels never become drawable-owner truth.

### 2.3 Typed role-free presentation evidence

`RoleFreePresentationEvidenceV1` carries source/boundary/pair/ray-stack/deformation/attachment/support values with explicit validity. View-dependent fields remain camera-conditioned; unsupported evidence is invalid rather than guessed.

### 2.4 TESSA V3 structuralization source

PR #132 added matched-capacity tri-stream conditioning:

```text
SCENE MEMORY ───────────────┐
SURFACE / MECHANICS MEMORY ┼─→ TESSA topology proposal decoder
PRESENTATION MEMORY ───────┘
```

Reference memory budget is `384 + 384 + 256 = 1024` tokens. The null-presentation control uses the same V3 decoder architecture and total conditioning budget. TESSA is proposal-only; Compiler alone may mint canonical M. No TESSA V3 training has occurred.

### 2.5 Compiler source coverage

Compiler/runtime regressions exercise presentation-consumption boundaries. Green source tests are source evidence, not scientific PASS.

## 3. Geometry boundary after SCALE-XFER

`H-GEO-P3-SCALE-XFER` is sealed `MIXED`: target multi-surface structure, frozen view-token spatial separability and carrier transfer all contributed. It authorized no P4, SURF-A2 or TESSA training.

Geometry remains behind:

```text
QUALIFIED_SURFACE_PROVIDER
        ↓
XYZ / normals / local scene evidence / mechanics
```

The provider may later be RealSaS P4 or another preregistered specialist. No automatic P4 follows from `MIXED`. The current endpoint diagnostic must not silently become a universal product requirement absent proof that it is necessary for final-surface / animation behavior.

## 4. Evidence authority — source complete, artifact absent

PR #133 merged the source-defensible materialization authority. The exact implementation source is:

`e9f8e42445f0edca28a869a9164e54892095d9d4`

Authority files:

- `canonical/STRUCTURAL_EVIDENCE_INPUT_BINDING_V1.json`
- `tools/iris_vnext/build_fit8_relation_all_hit_evidence_v1.py`
- `tools/iris_vnext/validate_structural_evidence_input_binding_v1.py`

The binding fixes:

- exact Stage01 bytes/bootstrap;
- exact legacy Fit8 shard hashes and normalization authority;
- exact authored-teacher batch and per-subject teacher/receipt hashes;
- exact sealed prepared Fit8 receipt/manifest;
- `8192` candidate rays/view and grid `32` producer policy;
- pairwise-front/global-visibility semantic separation;
- positive-only same-ray occlusion support;
- separate current-view RGBA/alpha boundary authority;
- no semantic/material owner shortcut;
- no teacher topology/rig/skin in inference.

The source gate was green before merge across the structuralization, mainline, architecture-freeze and retained historical source contracts.

Target output remains:

- one `ALL_HIT_RELATION_EVIDENCE_V1_MANIFEST.json`;
- eight per-subject `.all_hit_relation_evidence_v1.npz` files;
- exact hashes/source bindings;
- an evidence-only admission receipt.

**No such manifest has been admitted from Drive. Materialization has not occurred.**

## 5. Immediate next execution — zero-optimizer materialization

The source-only autonomous phase is complete. **Stop and hand off to the user before the next notebook.**

Execution ID: `STRUCTURAL-EVIDENCE-INPUT-MATERIALIZATION-V1`.

The run-all notebook may only:

1. hydrate exact sealed Stage01, legacy Fit8 and authored-teacher inputs;
2. verify every bound SHA256 before use;
3. derive complete ordered all-hit stacks and pair rows for Fit8 using the frozen producer;
4. write the eight evidence files and manifest;
5. run fail-closed materialization admission;
6. persist exact hashes and the typed receipt.

It must execute:

- `0` optimizer steps;
- `0` backward steps;
- no Relation training;
- no TESSA training;
- no scientific diagnostic/court;
- no research-hypothesis PASS/FAIL classification.

Materialization only establishes that a source-defensible B-arm input corpus exists.

## 6. Future scientific A/B — still not authorized

After and only after the evidence corpus is admitted, complete the preregistration before training:

```text
A — MATCHED NULL-PRESENTATION CONTROL
same scene + same surface + same mechanics
+ null presentation memory
        ↓
TESSA V3

B — REALSAS STRUCTURAL-EVIDENCE HYPOTHESIS
same architecture / capacity / subjects / cameras / teachers
+ role-free presentation evidence
        ↓
TESSA V3
```

Before any training output is observed, lock:

- seed set;
- optimizer and schedule;
- update budget;
- checkpoint-selection rule;
- exact topology/surface/presentation/motion court metrics;
- numeric thresholds and paired decision rule.

No threshold may be relaxed after results are observed. Teacher topology never enters inference conditioning. Named semantics are not a B-arm input.

Primary downstream measurements must include topology validity, structural/source fidelity, fragment stability, global visibility correctness, local/order correctness, pose-dependent order flips, motion owner stability, gap/tear behavior and source-art preservation.

## 7. Execution boundary

The immediate handoff is the zero-optimizer evidence-materialization notebook. A later optimizer/training A/B requires a second separate handoff.

Stop before:

- any new notebook execution without explicit handoff;
- any scientific diagnostic/court execution;
- any accelerator scientific experiment;
- any optimizer/backward/training step;
- any fresh-output scientific threshold evaluation.

## 8. Decision logic after the future A/B

If B materially and preregisteredly beats A, the next research question is whether geometry+graphics supervision can enrich the shared scene without geometry forgetting. Only then consider protected joint-training.

If B does not beat A, first localize whether presentation evidence, truth, structuralizer consumption or Compiler contract failed. Do not assume more joint training will fix it.

## 9. Product invariant

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
