# RealSaS — TESSA Mechanical Mesh Model Execution Plan V1

**Date:** 2026-10-04  
**Status:** `ACTIVE_P0_UPSTREAM_MECHANICS_PROGRAM`  
**Name:** `TESSA = Topology Estimation for Stable Surface Animation`

## 0. Executive decision

The learned/mechanical stack is:

```text
source views
   ↓
IRIS
   ↓
Stage08 NormalizationDomainIR             canonical object coordinate authority
   ↓
GSA / RiggingSurfaceIR S                  observation-grounded surface truth
   ↓
mechanical partition / support domains
   ↓
TESSA                                    learned mesh proposal only
   ↓
Compiler support binding + mesh qualification
   ↓
sealed canonical carrier M
   ↓
ATLAS                                    rig proposal
   ↓
Compiler rig qualification
   ↓
MIRA                                     skin-field proposal
   ↓
Compiler skin + dynamic motion proof
```

TESSA owns no canonical truth. It proposes a production mesh. The Compiler owns
support binding, legality, source fidelity, mechanical qualification, repair,
attempt lineage and final carrier seal.

The first-class map is:

`TESSA : (N_stage08, S_GSA) -> M_proposal`

with the explicit rule:

`S_GSA != M_proposal` is legal.

Stage08 owns the object coordinate frame. GSA owns where the admitted surface is
inside that frame. TESSA learns how that surface should be discretized to become
a useful deformable production carrier.

## 1. Why TESSA exists

Current Stage18 is a strong deterministic geometry/discretization baseline. It
preserves compact dense-face provenance and can locally improve triangles with
CDT, but it does not learn or globally design animation topology. A bad parent
connectivity/edge-flow pattern can therefore remain a bad mechanical basis even
when XYZ/source fidelity is excellent.

The decisive scientific observation is that static geometry quality and dynamic
mechanical quality are different objectives. A product carrier must satisfy
both.

TESSA is not an IRIS replacement. IRIS/GSA already solves the observation-side
surface problem. TESSA is the learned bridge from admitted surface truth to a
riggable, weightable, deformation-stable carrier.

## 2. Clean-room reference and deliberate departures

Primary public architecture reference: MeshAnything V2.

Useful solution-class ideas retained in clean-room form:
- point/surface + normal conditioning;
- fixed-size shape/surface latent memory;
- autoregressive mesh generation;
- adjacent-face continuation to avoid repeating an entire triangle;
- teacher artist meshes as topology priors.

RealSaS deliberately differs in six places:

1. **Input distribution.** Training input must be actual IRIS/GSA evidence made
   from teacher renders, not a pristine teacher point cloud unavailable at
   product inference.
2. **Coordinate authority.** TESSA may not refit a frame from the finite GSA
   cloud. Stage08 `NormalizationDomainIR` is the sole canonical object-frame
   authority used by both GSA conditioning and teacher targets.
3. **Authority.** Learned mesh output is `PROPOSAL`; Compiler qualification is
   mandatory before ATLAS/MIRA.
4. **Source support.** Every admitted generated vertex must resolve to a typed
   `SurfaceSupportBinding` inside its admitted GSA/component domain.
5. **Mechanics.** Training includes rig/skin/motion consequence objectives using
   coherent teacher assets; static token likelihood is insufficient.
6. **Scale.** A ~1.6K-face research ceiling is not a RealSaS product contract.
   TESSA V1 is configured for 32,768 faces and must admit the Knight source mesh
   (3,665 vertices / 6,952 faces) as ordinary workload.

## 3. V1 architecture

### 3.1 Coordinate authority and conditioning

The deterministic adapter is:

`NormalizationDomainIR + RiggingSurfaceIR -> tensor`

Stage08 normalization has world relation:

`P_world = center_xyz + P_stage08 * half_extent`

with admitted source coordinates inside `[-1,+1]`. TESSA coordinate tokens use
`[-0.5,+0.5]`, therefore the exact conversion is:

`P_tessa = (P_world - center_xyz) / (2 * half_extent)`

No GSA-bbox refit, teacher-derived recenter/rescale, padding chosen from teacher
truth, or clipping is permitted. This rule is hash-bound by the Stage08
`normalization_hash` and is part of the T1 checkpoint resume contract.

Current GSA conditioning has 17 features/node:
- Stage08-bound normalized XYZ: 3;
- derived normal: 3;
- normal-valid bit: 1;
- 8-view support mask: 8;
- normalized local-relation degree: 1;
- persistence-group-presence bit: 1.

No teacher topology, rig or skin enters product conditioning.

Generation remains component-aware. A learned proposal may not escape the
admitted support domain and then recover by unconstrained nearest-neighbor
binding across a nearby disconnected sheet.

### 3.2 Surface encoder

Point features -> MLP -> learned latent queries -> cross-attention -> latent
Transformer.

Default latent memory: 384 x 768.

The expensive latent stack is fixed-size. Raw GSA size affects the one
point-to-latent cross-attention rather than every autoregressive layer.

### 3.3 Mesh token grammar

V1 clean-room adjacent-face grammar:
- seed face: `FACE_BREAK + 9 coordinate tokens`;
- adjacent face sharing the active trailing edge: only 3 coordinate tokens for
  the new vertex;
- explicit source-connected component delimiters;
- chart delimiters retained by the grammar for compatibility/structured
  continuation, but canonical T1 does not split a connected source component
  into independently generated training charts;
- EOS/PAD.

Coordinates are Stage08-bound RealSaS object coordinates quantized to 1,024 bins
in V1.

The whole asset is serialized as one global topology sequence. Each true
face-connected source component receives one explicit component block. Arbitrary
face-count chart cuts are forbidden in canonical T1 because independent chart
generation would sever identity/connectivity at artificial boundaries.

V1 also fails closed if two distinct vertices inside the same source-connected
component quantize to the same coordinate identity. Cross-component equal XYZ is
not globally welded.

The strict decoder reconstructs indexed proposal geometry with component-local
coordinate identity. Face orientation remains unqualified until Compiler
support/normal evidence resolves it.

This sequence is a learned representation, not a canonical mesh identity.
Compiler decode/support resolution may reject ambiguous welding, cross-sheet
binding, non-manifoldness, boundary drift or source-fidelity failure.

### 3.4 Scalable decoder and training windows

A full L-token self-attention matrix is forbidden as the product scaling
contract.

TESSA V1 decoder uses bounded causal attention:

`cost ~ O(L * W)`

where default trailing window `W = 2048`, with query chunks of 256 tokens.
Every decoder block also cross-attends to the fixed surface latent memory.

The global topology sequence is not cut into independent meshes for training.
Instead, constant-memory teacher-forcing uses overlapping truncated LM windows:
- every target token position is scored exactly once;
- up to `W` preceding tokens from the same global sequence are provided as
  causal context;
- context-prefix labels are ignored/PAD;
- absolute sequence offset is retained for wrapped positional phase.

Default asset budget:
- 32,768 faces;
- 32,768 vertices;
- 128 source-connected components;
- local causal attention window 2,048 tokens;
- query chunk 256 tokens.

`max_faces_per_chart` is not a canonical T1 scaling mechanism. Legacy bounded
face-chart partitioning remains diagnostic only.

Future challengers may borrow the solution class of hierarchical/hourglass and
sliding-context mesh transformers, but V1 does not make any external
implementation a runtime dependency.

## 4. Proposal -> canonical carrier handoff

TESSA learned output is conceptually:

`TESSAMeshProposalV1 { V_proposal, F_proposal, Stage08/GSA lineage, model provenance }`

For each proposed vertex the model may expose a primary GSA anchor for routing,
but that pointer is not a qualified `SurfaceSupportBinding`.

Compiler must:
1. decode topology;
2. preserve explicit component separation during identity/weld resolution;
3. restrict support search to the admitted component domain;
4. project/bind generated vertices onto admitted GSA/source support;
5. construct `MeshDiscretizationCandidateIR` with typed support bindings;
6. run static carrier qualification;
7. run source/silhouette/coverage replay;
8. seal only a passing candidate;
9. invalidate downstream rig/skin if carrier lineage changes.

No direct `TESSAMeshProposalV1 -> QualifiedEditableMeshIR` conversion exists.

Current deterministic Stage18/CDT stays as:
- baseline;
- fallback/repair primitive;
- causal comparison arm.
It is no longer assumed to be the final learned production-topology solution.

## 5. Training data contract

For each coherent teacher asset:

```text
teacher source asset M*, R*, W*
        ↓ render using frozen observation contract
source views
        ↓ frozen/current IRIS + deterministic Stage08 + GSA
(N_stage08, S_train)
```

Training pair:

`(N_stage08, S_train) -> M*`

Teacher vertices are transformed into the same Stage08-bound TESSA coordinate
frame. The target may never be clipped or independently normalized to make the
fit easier.

This is essential. Training on pristine source geometry while product inference
uses GSA geometry would recreate the substrate-distribution mismatch TESSA is
meant to remove.

Teacher source topology is a high-value prior, not a unique truth. Multiple
meshes may be mechanically equivalent or better while representing the same
surface.

## 6. Training ladder

### T0 — deterministic/oracle dataset audit
- prove teacher source mesh/rig/skin lineage;
- generate Stage08 + GSA input through the real observation path;
- prove Stage08 normalization hash and coordinate-frame integrity;
- prove GSA and teacher source geometry both lie inside the canonical domain
  without refit or clipping;
- reject V1 examples with within-component quantized vertex identity collision;
- record component and support-domain correspondence;
- reject examples whose source mechanics cannot pass the oracle court.

### T1 — supervised topology fit
Primary loss:
- autoregressive mesh-token cross entropy over one global asset sequence using
  overlap/context-preserving truncated windows.

Auxiliary losses/metrics:
- source-surface residual;
- boundary/component preservation;
- manifold/degenerate statistics;
- density/triangle quality diagnostics.

T1 does not claim mechanical consequence learning.

### T2 — mechanical consequence fit
Use coherent teacher `R*, W*` and a broad source + synthetic pose bank.
Generated vertices query a source/material-coordinate skin field; direct
same-index weight copying is forbidden unless mesh identity is exact.

Differentiable optimization terms include:
- edge stretch;
- area ratio;
- condition number;
- deformation-Jacobian residual;
- orientation residual relative to oracle deformation;
- source/surface fidelity.

These are training surrogates only.

### T3 — hard Compiler court
A frozen checkpoint must pass deterministic Compiler qualification on held-out
poses/assets. Training reward cannot self-authorize shipping.

## 7. Knight causal court — mandatory before promotion

Use four carrier arms where available:
- `M_source`: original artist Knight mesh;
- `M_stage18`: current RealSaS deterministic carrier;
- `M_retopo`: deterministic/global retopology baseline;
- `M_tessa`: TESSA proposal after Compiler support binding.

Hold reference mechanics as constant as scientifically possible:
- raw 41-bone source rig;
- source continuous skin field queried at candidate material/surface points;
- source motions plus synthetic stress poses.

Measure separately:
1. vertex/material-point trajectory residual;
2. deformation Jacobian residual;
3. area/condition/stretch;
4. orientation/fold failures;
5. boundary/component/source fidelity.

This separation is mandatory because under ordinary per-vertex LBS, changing
triangle connectivity alone does not change identical vertex trajectories. A
topology claim must therefore be supported by interior/surface/Jacobian evidence
or by an explicit topology-dependent deformation operator.

Promotion invariant:

> A carrier that cannot reproduce source/oracle rig + skin + motion consequences
> within the frozen tolerance cannot qualify as the product mechanical carrier.

## 8. Interaction with ATLAS and MIRA

TESSA is upstream of final ATLAS/MIRA fitting.

Final learned stack:

`IRIS -> TESSA -> ATLAS -> MIRA`

- IRIS supplies surface/perception evidence.
- TESSA supplies a Compiler-qualified production carrier.
- ATLAS infers/preserves the admitted rig/control contract on that carrier.
- MIRA predicts/queries a geometry-conditioned skin field on that carrier.

Source skeleton complexity and skin sparsity remain separate. In particular,
zero-skin source/control bones are not deleted merely because their skin mass is
zero. Historical control-count pruning is not TESSA authority.

ATLAS/MIRA final retraining should wait until the TESSA carrier evidence contract
is frozen, otherwise a carrier-contract change can invalidate the fit target.

## 9. Promotion gates

TESSA cannot become shipping authority until all are true:

- G0: code/unit invariants PASS;
- G1: Stage08/GSA-only conditioning / teacher-leak audit PASS;
- G2: Knight 6,952-face scale admission + coordinate/tokenization T0 PASS;
- G3: supervised topology fit beats deterministic baseline on source fidelity
  without mechanical regression;
- G4: teacher rig/skin mechanical consequence court PASS;
- G5: held-out subject mechanical court PASS;
- G6: Compiler support binding + static carrier qualification PASS;
- G7: end-to-end `IRIS -> TESSA -> ATLAS -> MIRA -> motion proof` PASS;
- G8: learned artifact registry/checkpoint provenance sealed.

Until G8, TESSA outputs are research/proposal artifacts only.

## 10. Current implementation files

- `models/tessa/v1/contracts_v1.py`
- `models/tessa/v1/conditioning_v1.py`
- `models/tessa/v1/model_v1.py`
- `models/tessa/v1/tokenization_v1.py`
- `models/tessa/v1/sequence_codec_v1.py`
- `models/tessa/v1/training_data_v1.py`
- `models/tessa/v1/proposal_v1.py`
- `models/tessa/v1/mechanical_objective_v1.py`
- `tools/training/run_tessa_supervised_fit_v1.py`
- `notebooks/RealSaS_TESSA_KNIGHT_T1_RUN_ALL.ipynb`
- `tests/models/test_tessa_v1.py`
- `tests/models/test_tessa_training_data_v1.py`

This document is the canonical TESSA V1 program. It extends, rather than
silently rewrites, the existing carrier-first mechanical-coherence lineage.
