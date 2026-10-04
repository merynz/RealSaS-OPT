# TESSA / MeshAnything V2 Clean-room Reference Audit V1

**Date:** 2026-10-04  
**Purpose:** record public solution-class evidence used to design TESSA without creating a source dependency or copying an implementation.

## Public reference snapshot

Repository: `buaacyw/MeshAnythingV2`  
Observed public tree SHA: `461d3b6ed750ab3443281b2e4a0e30e8ee98097e`

Relevant public files inspected:
- `meshanything_train/models/single_gpt.py`, blob `6565e62d8edbd14f2372155e1ab1658dee38a3dc`;
- `adjacent_mesh_tokenization.py`, blob `ef98152cb8b118e8b633b516a66d794520595d89`.

Observed solution-class facts:
1. point-cloud + normal features condition a mesh autoregressive model;
2. a shape/point encoder produces a fixed conditioning memory;
3. a decoder-only causal model predicts mesh/coordinate tokens;
4. adjacent-face serialization can avoid repeating two shared vertices;
5. the public research implementation has a much smaller practical mesh regime than RealSaS requires.

## What TESSA reuses conceptually

Only the following high-level ideas are treated as references:
- surface-conditioned mesh generation;
- autoregressive topology/coordinate representation;
- adjacent-face sequence compression;
- artist/source meshes as topology examples.

No MeshAnything source file is imported, vendored, subclassed or made a runtime/training dependency.

## RealSaS-specific departures

TESSA is not a reimplementation of MeshAnything V2.

### Authority

Mesh model output is never canonical.  TESSA emits a proposal; Compiler support resolution, mesh qualification and attempt sealing own product truth.

### Input distribution

TESSA trains on the actual product-side substrate:

`teacher asset -> source views -> IRIS -> GSA -> TESSA`

not on a pristine teacher-only point cloud that product inference cannot observe.

### Source support

Every admitted generated vertex must acquire a typed `SurfaceSupportBinding` inside an admitted GSA/component domain before it can become a `MeshDiscretizationCandidateIR`.

### Mechanical objective

Teacher rig/skin/motion consequence is available during training and hard Compiler motion proof is available after inference.  Static artist-topology imitation alone is not the target.

### Scale

TESSA V1 default contract admits 32,768 faces and explicitly treats the Knight source mesh (6,952 faces) as ordinary workload.  Decoder self-attention is bounded-window `O(LW)` rather than global `O(L^2)`.

### Representation

TESSA generation is component/chart scoped to preserve RealSaS structural/support-domain semantics and to permit large meshes without a monolithic token sequence becoming the product architecture.

## Clean-room rule

Future TESSA work may compare behavior, published equations and public interface/architecture facts against external systems.  Any code promoted into `models/tessa/` must be independently written to the RealSaS contracts and reviewed against this authority/provenance boundary.
