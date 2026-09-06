# RealSaS Rigging Line Pre-C Closure V1

**Date:** 2026-09-06  
**Branch:** `e2e/mage-scene-first-v1-20260905`  
**Implementation parent:** `53067213c4b05ab0b6b9c8c38aa2424daf4b7321`  
**Status:** `PRE_C_RIGGING_INFRASTRUCTURE_CLOSED__MECHANISM_PROMOTION_PENDING_C_EXPERIMENTS`  
**Scope stop:** `QualifiedSkeletonIRV2`  
**Explicitly out of scope:** Arachne, skin weights, SkinTokens, deformation/skin promotion.

## 1. Closed product route

The authoritative scene-first rigging route is now:

```text
IRIS SceneFirstSigned geometry
  -> GSA SceneFirstSigned
  -> RiggingSurfaceIR
  -> GeppettoProductConditioningAdapterV2
       - fail-closed RiggingSurfaceIR validation BEFORE feature construction
       - exact surface-lineage binding
       - exact topology fingerprint
       - product conditioning certificate
  -> Geppetto / future C challenger
  -> SkeletonProposalIR
  -> compile_scene_first_rigging_v1
       - independently revalidate scene-first surface
       - verify product conditioning certificate
       - explicit SkeletonAdmissionPolicyV1
       - canonical graph optimizer
       - canonical joint-ID minting
       - single connected deform-tree product assertion
  -> QualifiedSkeletonIRV2
```

The older `qualify_skeleton()` and `qualify_skeleton_v2()` entrypoints remain available for historical diagnostics and compatibility. Their qualification reports now explicitly carry `promotion_authority=False`. They are not the product-promotion route.

## 2. R0 — GSA / RiggingSurfaceIR boundary closed

`GeppettoProductConditioningAdapterV2` is the mandatory product adapter. Before computing the 24D conditioning rows it calls:

`validate_rigging_surface_ir_v1(..., require_scene_first_signed_contract=True)`

The strict scene-first profile now requires:

- exact SceneFirstSigned GSA builder identity;
- non-empty scene-first surface;
- finite 3D positions;
- unique surface IDs;
- unit normals where present;
- support-view/raster-view equality;
- at least one observed surface node;
- non-zero raster evidence;
- non-empty local topology;
- valid local-relation endpoints;
- source run/checkpoint/zero-surface provenance;
- exact metadata count consistency;
- no teacher-truth contamination flags.

The product adapter returns a hash-pinned `GeppettoProductConditioningCertificate.v1`. The Compiler independently recomputes the boundary audit hash and topology fingerprint and refuses product compilation if the certificate does not match the exact `RiggingSurfaceIR`.

Historical/diagnostic `GeppettoConditioningAdapterV2` remains usable but mints only:

`DIAGNOSTIC_UNVALIDATED_V1`

and cannot satisfy the product compiler certificate gate.

## 3. Raster evidence closure

The production adapter retains the existing 24D feature contract, including:

- `raster_mean_x`
- `raster_mean_y`
- `raster_std_x`
- `raster_std_y`

The strict product boundary refuses a SceneFirstSigned surface with stripped observed raster bindings. Therefore the old compact Mage fixture remains a historical diagnostic witness but cannot silently masquerade as the new product-substrate path.

No historical conditioning-hash definition was rewritten. Boundary/topology authority is carried in separate fields, preserving old witness comparability.

## 4. Topology contract closed without prematurely choosing A2/A3

`RiggingSurfaceIR.local_relations` are no longer semantically invisible at the Geppetto boundary.

The conditioning batch now preserves:

- deterministic indexed local-relation pairs;
- exact `RiggingSurfaceTopologyFingerprint.v1`;
- topology contract `RIGGING_SURFACE_LOCAL_RELATIONS_INDEXED_V1`;
- explicit current C0 locality policy `EUCLIDEAN_KNN_V2`.

`topology_neighbor_index_v1()` converts the exact GSA graph into deterministic fixed-width neighbor rows for future topology-aware C/A arms.

Important: this patch does **not** promote GSA graph locality over Euclidean KNN. The earlier Mage measurement showed current KNN already recalls nearly all direct GSA edges. The product contract now preserves both evidence and policy explicitly so later experiments can compare them without reconstructing or losing topology.

## 5. Proposal admission seam closed

A new typed pre-optimizer object exists:

`AdmittedSkeletonProposalIR.v1`

through:

`admit_skeleton_proposal_v1(...)`

The admission stage validates:

- surface lineage;
- non-empty proposal;
- unique proposal IDs;
- finite 3D loci;
- finite bounded root/confidence values;
- known support-surface IDs;
- unique edge IDs;
- legal edge endpoints;
- finite bounded edge evidence;
- no self edges;
- no hard-required/hard-forbidden contradiction.

The product policy additionally requires each admitted deform proposal joint to have surface support.

### Native count is intentionally preserved

The current policy is:

`FAIL_CLOSED_ALL_CONTRACT_VALID_PROPOSALS`

It does not use an uncalibrated confidence threshold to prune joints. Therefore Geppetto STOP/cardinality remains visible and testable.

### Geometry-only duplicate fusion is forbidden

The exact-FIT witness contains 41 authored controls on 31 exact loci. Therefore two controls sharing the same or near-identical 3D locus cannot safely be assumed mechanically equivalent.

Product policy:

`PRESERVE_DISTINCT_PROPOSAL_IDS`

The admission report records coincident-locus multiplicities as telemetry, but:

`geometry_only_fused_joint_count = 0`

### Deform-node completion is forbidden in V1

There is currently no separately proven teacher-free owner capable of inventing a missing deform joint without hiding model failure.

Product policy:

`max_synthesized_deform_nodes = 0`

and the report records:

`synthesized_deform_node_count = 0`

A future bounded completion owner may be added only under its own proof contract. It is not silently implemented inside graph optimization.

## 6. Canonical graph authority clarified

`optimize_canonical_graph_v18_98` remains the exact global root/parent authority.

It owns:

- one globally selected root under the current product policy;
- one parent for each other admitted node;
- cycle-free connected arborescence;
- hard edge/root constraints;
- exact/global optimality proof;
- optional MILP shadow/escalation behavior already present.

It does **not** own:

- learned/native joint count;
- proposal pruning;
- geometry duplicate fusion;
- deform-node completion.

Those responsibilities are now explicit rather than implied by comments.

## 7. Product deform-graph policy frozen for this rigging line

`QualifiedSkeletonIRV2` remains a forest-capable semantic type for future architecture work.

The current scene-first product route, however, is explicitly:

`SINGLE_CONNECTED_DEFORM_TREE_V1`

`compile_scene_first_rigging_v1()` verifies exactly one deform root, legal parents, no cycles and root connectivity before returning product-authoritative `QualifiedSkeletonIRV2`.

Any optional technical assembly root remains separate and must be non-deforming.

This resolves the previous ambiguity between the forest-capable type and the single-arborescence optimizer without deleting future representational capacity.

## 8. Compatibility routes vs product authority

| Route | Intended use | Promotion authority |
|---|---|---:|
| `GeppettoConditioningAdapterV2` | historical/diagnostic conditioning | no |
| `qualify_skeleton` | compatibility/diagnostic | no |
| `qualify_skeleton_v2` | compatibility/diagnostic | no |
| `GeppettoProductConditioningAdapterV2` + `compile_scene_first_rigging_v1` | scene-first product rigging | yes |

This prevents a compact/stripped historical fixture from being confused with a fully validated production boundary.

## 9. Behavioral gates added

`tests/compiler/test_rigging_line_closure_v1.py` verifies:

1. strict product conditioning passes on a valid scene-first surface;
2. raster evidence reaches the 24D conditioning;
3. stripped scene-first raster fails closed;
4. exact GSA topology pairs/fingerprint are preserved;
5. deterministic topology neighbor rows are available;
6. diagnostic conditioning never mints product authority;
7. coincident controls are preserved rather than fused;
8. unsupported product joints fail admission;
9. diagnostic certificates cannot enter the product compiler;
10. the product compiler returns one connected canonical deform tree;
11. proposal IDs do not leak into canonical IDs;
12. legacy V2 qualification is explicitly non-promoting.

A dedicated CPU GitHub Actions workflow runs source compilation and this behavioral panel on the active experimental branch.

## 10. What is deliberately NOT closed by this patch

The C mechanism winner is not selected here.

Still pending experimental evidence:

- C0: current Geppetto baseline;
- C1: per-step full-surface token access;
- C2: C1 + conditional joint diffusion;
- C3: C2 + safe generated joint/parent geometry feedback;
- C4: equivalent sibling-order augmentation if still required;
- native STOP/count closure under the winning architecture;
- optional topology locality comparison if C results make it causal/relevant.

Those experiments must share the newly closed product substrate/certificate/admission/compiler route.

## 11. Next authorized step

Before writing the C notebooks:

1. inspect the previously successful IRIS/Mage notebook design conventions;
2. preserve the exact FIT regression contract (41 controls / 31 loci, structural serialization, proven phase-dependent optimizer policy);
3. build contemporaneous C0/C1/C2 first, with C3/C4 only as evidence requires;
4. measure exact-FIT quality, convergence speed, late stability, STOP/count and compiler-product status separately;
5. promote/freeze only the mechanism supported by evidence.

Only after the rigging architecture is promoted/frozen does work move to Arachne/SkinTokens.

## Verdict

`PRE_C_RIGGING_INFRASTRUCTURE_CLOSED`

The remaining uncertainty is now intentionally experimental mechanism choice, not an untyped or missing rigging seam.
