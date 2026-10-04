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
GSA / RiggingSurfaceIR S                 observation-grounded surface truth
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

TESSA owns no canonical truth.  It proposes a production mesh.  The Compiler
owns support binding, legality, source fidelity, mechanical qualification,
repair, attempt lineage and final carrier seal.

The new first-class map is:

`TESSA : S_GSA -> M_proposal`

with the explicit rule:

`S_GSA != M_proposal` is legal.

GSA owns where the admitted surface is; TESSA learns how that surface should be
discretized to become a useful deformable production carrier.

## 1. Why TESSA exists

Current Stage18 is a strong deterministic geometry/discretization baseline.  It
preserves compact dense-face provenance and can locally improve triangles with
CDT, but it does not learn or globally design animation topology.  A bad parent
connectivity/edge-flow pattern can therefore remain a bad mechanical basis even
when XYZ/source fidelity is excellent.

The decisive scientific observation is that static geometry quality and dynamic
mechanical quality are different objectives.  A product carrier must satisfy
both.

TESSA is not an IRIS replacement.  IRIS/GSA already solves the observation-side
surface problem.  TESSA is the learned bridge from admitted surface truth to a
riggable, weightable, deformation-stable carrier.

## 2. Clean-room reference and deliberate departures

Primary public architecture reference: MeshAnything V2.

Useful solution-class ideas retained in clean-room form:
- point/surface + normal conditioning;
- fixed-size shape/surface latent memory;
- autoregressive mesh generation;
- adjacent-face continuation to avoid repeating an entire triangle;
- teacher artist meshes as topology priors.

RealSaS deliberately differs in five places:

1. **Input distribution.** Training input must be actual IRIS/GSA evidence made
   from teacher renders, not a pristine teacher point cloud unavailable at
   product inference.
2. **Authority.** Learned mesh output is `PROPOSAL`; Compiler qualification is
   mandatory before ATLAS/MIRA.
3. **Source support.** Every admitted generated vertex must resolve to a typed
   `SurfaceSupportBinding` inside its admitted GSA/component domain.
4. **Mechanics.** Training includes rig/skin/motion consequence objectives using
   coherent teacher assets; static token likelihood is insufficient.
5. **Scale.** A ~1.6K-face research ceiling is not a RealSaS product contract.
   TESSA V1 is configured for 32,768 faces and must admit the Knight source mesh
   (3,665 vertices / 6,952 faces) as ordinary workload.

## 3. V1 architecture

### 3.1 Conditioning

Deterministic `RiggingSurfaceIR -> tensor` adapter, currently 17 features/node:
- normalized XYZ: 3;
- derived normal: 3;
- normal-valid bit: 1;
- 8-view support mask: 8;
- normalized local-relation degree: 1;
- persistence-group-presence bit: 1.

No teacher topology, rig or skin enters product conditioning.

Generation is component/chart scoped.  A learned proposal may not escape the
admitted support domain and then recover by unconstrained nearest-neighbor
binding across a nearby disconnected sheet.

### 3.2 Surface encoder

Point features -> MLP -> learned latent queries -> cross-attention -> latent
Transformer.

Default latent memory: 384 x 768.

The expensive latent stack is fixed-size.  Raw GSA size affects the one
point-to-latent cross-attention rather than every autoregressive layer.

### 3.3 Mesh token grammar

V1 clean-room adjacent-face grammar:
- seed face: `FACE_BREAK + 9 coordinate tokens`;
- adjacent face sharing the active trailing edge: only 3 coordinate tokens for
  the new vertex;
- explicit component/chart delimiters;
- EOS/PAD.

Coordinates are normalized RealSaS object coordinates and quantized to 1,024
bins in V1.

This sequence is a learned representation, not a canonical mesh identity.
Compiler decode/support resolution may reject ambiguous welding, cross-sheet
binding, non-manifoldness, boundary drift or source-fidelity failure.

### 3.4 Scalable decoder

A full L-token self-attention matrix is forbidden as the product scaling
contract.

TESSA V1 decoder uses bounded causal attention:

`cost ~ O(L * W)`

where default trailing window `W = 2048`, with query chunks of 256 tokens.
Every decoder block also cross-attends to the fixed surface latent memory.

Default mesh budget:
- 32,768 faces;
- 32,768 vertices;
- 4,096 faces per generation chart;
- 128 components;
- 512 charts/component upper contract.

Large assets are represented as component/chart sequences with explicit seam
and support-domain bookkeeping.  Chart decomposition is a scalability device;
it may not create a second geometry authority.

Future challengers may borrow the solution class of hierarchical/hourglass and
sliding-context mesh transformers (e.g. Meshtron-like scaling), but V1 does not
make any external implementation a runtime dependency.

## 4. Proposal -> canonical carrier handoff

TESSA learned output is conceptually:

`TESSAMeshProposalV1 { V_proposal, F_proposal, GSA lineage, model provenance }`

For each proposed vertex the model may expose a primary GSA anchor for routing,
but that pointer is not a qualified `SurfaceSupportBinding`.

Compiler must:
1. decode topology;
2. restrict support search to the admitted component/chart domain;
3. project/bind generated vertices onto admitted GSA/source support;
4. construct `MeshDiscretizationCandidateIR` with typed support bindings;
5. run static carrier qualification;
6. run source/silhouette/coverage replay;
7. seal only a passing candidate;
8. invalidate downstream rig/skin if carrier lineage changes.

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
        ↓ frozen/current IRIS + deterministic GSA
S_train
```

Training pair:

`S_train -> M*`

This is essential.  Training on pristine source geometry while product inference
uses GSA geometry would recreate the substrate-distribution mismatch TESSA is
meant to remove.

Teacher source topology is a high-value prior, not a unique truth.  Multiple
meshes may be mechanically equivalent or better while representing the same
surface.

## 6. Training ladder

### T0 — deterministic/oracle dataset audit
- prove teacher source mesh/rig/skin lineage;
- generate GSA input through the real observation path;
- record component and support-domain correspondence;
- reject examples whose source mechanics cannot pass the oracle court.

### T1 — supervised topology fit
Primary loss:
- autoregressive mesh-token cross entropy.

Auxiliary losses/metrics:
- source-surface residual;
- boundary/component preservation;
- manifold/degenerate statistics;
- density/triangle quality diagnostics.

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
poses/assets.  Training reward cannot self-authorize shipping.

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
triangle connectivity alone does not change identical vertex trajectories.  A
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

Source skeleton complexity and skin sparsity remain separate.  In particular,
zero-skin source/control bones are not deleted merely because their skin mass is
zero.  Historical control-count pruning is not TESSA authority.

ATLAS/MIRA final retraining should wait until the TESSA carrier evidence contract
is frozen, otherwise a carrier-contract change can invalidate the fit target.

## 9. Promotion gates

TESSA cannot become shipping authority until all are true:

- G0: code/unit invariants PASS;
- G1: GSA-only conditioning / teacher-leak audit PASS;
- G2: Knight 6,952-face scale admission PASS;
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
- `models/tessa/v1/mechanical_objective_v1.py`
- `tests/models/test_tessa_v1.py`

This document is the canonical TESSA V1 program.  It extends, rather than
silently rewrites, the existing carrier-first mechanical-coherence lineage.
