# IRIS Shared Structural Evidence Plan — 2026-10-20

Status: **SOURCE BRIDGE COMPLETE / EVIDENCE MATERIALIZATION HANDOFF NEXT / NO SCIENTIFIC RUN AUTHORIZED**

Long-term research branch: `research/iris-vnext-hybrid-tessa-v2-20261019`

This plan is the active continuation after the sealed `H-GEO-P3-SCALE-XFER` result. It makes explicit the role-free geometry+graphics+mechanics hypothesis and the controlled path by which that hypothesis may eventually be tested.

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

## 2. Implemented source boundary

### 2.1 Shared-scene evidence

IRIS V8 provides a sealed shared scene carrier. Post-M1 appearance, relation and topology heads can learn from that frozen carrier without rewriting geometry. This supports the existence of useful cross-modal evidence in H, but it is not proof of jointly trained geometry+graphics scene learning.

### 2.2 Corrected relation / visibility abstraction

Historical V8 relation truth was partly proxy/misaligned and is not scientific authority for order/visibility quality.

The current V3 source explicitly separates:

- pairwise front/order: depth ordering between two same-view surface samples;
- positive occlusion support: same-ray overlap where source-defensible;
- global visibility: rank-zero membership in a **complete ordered ray stack**;
- incomplete global-visibility authority: ABSTAIN / validity false.

Pairwise front is never promoted into global visibility. Material IDs and named semantic labels never become drawable-owner truth.

### 2.3 Typed role-free presentation evidence

`RoleFreePresentationEvidenceV1` carries source/boundary/pair/ray-stack/deformation/attachment/support values with explicit validity. View-dependent fields remain camera-conditioned. Unsupported evidence is zeroed and invalid rather than guessed.

### 2.4 TESSA V3 structuralization source

PR #132 added a matched-capacity tri-stream source architecture:

```text
SCENE MEMORY
shared IRIS context

SURFACE / MECHANICS MEMORY
XYZ + normal + H + deformation evidence

PRESENTATION MEMORY
appearance/source boundaries
pair front/occlusion evidence
complete ray-stack visibility evidence
validity masks
        ↓
TESSA topology proposal decoder
```

Reference memory budget is `384 scene + 384 surface/mechanics + 256 presentation = 1024` tokens. The null-presentation control uses the same V3 decoder architecture and total memory budget. Tokenizer/grammar, coordinate vocabulary and reference decoder width/depth remain compatible with the V2 control family.

TESSA is proposal-only. Compiler alone may mint canonical M. No TESSA V3 optimizer/training run has occurred.

### 2.5 Compiler source coverage

Compiler/runtime source regressions exercise presentation consumption boundaries. Green source tests are not a scientific PASS.

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

## 4. Current gate — evidence authority before training

The V3 source bridge exists, but arm B of the future A/B still requires an actual sealed evidence corpus. Source code that *can* derive truth is not the same thing as a hash-bound artifact that a scientific run may consume.

Authority:

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

Target output:

- one `ALL_HIT_RELATION_EVIDENCE_V1_MANIFEST.json`;
- eight per-subject `.all_hit_relation_evidence_v1.npz` files;
- exact hashes and source bindings;
- admission receipt declaring evidence-only status.

At the time of this plan update, no such manifest has been admitted from Drive. Materialization has not occurred.

## 5. Next execution — zero-optimizer materialization, not a scientific experiment

Once source CI is green and the implementation source is frozen, stop and hand off to the user before execution.

The run-all notebook may only:

1. hydrate exact sealed Stage01, legacy Fit8 and authored-teacher inputs;
2. verify every bound SHA before use;
3. derive complete ordered all-hit stacks and pair rows for Fit8;
4. write the eight evidence files and manifest;
5. run fail-closed materialization admission;
6. persist exact hashes / typed receipt.

It must execute:

- `0` optimizer steps;
- `0` backward steps;
- no Relation training;
- no TESSA training;
- no scientific A/B court;
- no research-hypothesis PASS/FAIL classification.

Materialization only establishes that a source-defensible B-arm input corpus exists.

## 6. Future scientific experiment — still not authorized

After the evidence corpus is admitted, complete the preregistration before any training:

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

Before execution lock:

- seed set;
- optimizer and schedule;
- update budget;
- checkpoint-selection rule;
- exact topology/surface/presentation/motion court metrics;
- numeric thresholds and paired decision rule.

No threshold may be relaxed after results are observed.

Primary downstream measurements must include topology validity, structural/source fidelity, fragment stability, global visibility correctness, local/order correctness, pose-dependent order flips, motion owner stability, gap/tear behavior and source-art preservation.

Teacher topology never enters inference conditioning. Named semantics are not a B-arm input.

## 7. Scientific execution boundary

The assistant may continue source-only work autonomously under this plan, but must stop and hand off before:

- any new notebook execution;
- any scientific diagnostic/court execution;
- any accelerator scientific experiment;
- any optimizer/backward/training step;
- any new scientific threshold evaluation on fresh outputs.

The immediate handoff is the zero-optimizer evidence-materialization notebook. A later optimizer/training A/B requires a second, separate handoff.

## 8. Decision logic after the future A/B

If B materially and preregisteredly beats A, the next research question is whether geometry+graphics supervision can enrich the shared scene without geometry forgetting. Only then consider a protected joint-training experiment.

If B does not beat A, do not assume more joint training will fix it; first localize whether the presentation evidence, truth, structuralizer consumption or Compiler contract failed.

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
