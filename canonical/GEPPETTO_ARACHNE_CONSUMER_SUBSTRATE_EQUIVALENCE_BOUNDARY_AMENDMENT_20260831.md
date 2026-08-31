# RealSaS — Geppetto/Arachne Consumer-Substrate Equivalence Boundary Amendment — 2026-08-31

**Status:** `CANONICAL_AMENDMENT__SUPERSEDES_RAW_IRIS_INTERPRETATION_OF_REFERENCE_R5`

## Why this amendment exists

The sealed external-reference clean-room prereg correctly names `RiggingSurfaceIR` / `QualifiedSkeletonIR` as the intended RealSaS comparison substrate, but its R5 discussion could still be misread as requiring raw partial IRIS output itself to be information-equivalent to the complete 3D surface input consumed by an external auto-rigger.

That interpretation conflicts with the already-canonical `OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`, whose scientific target is explicitly the **functional 8-view observational equivalent of the geometric shape input consumed by 3D auto-riggers after deterministic SurfaceBuilder processing**.

This is therefore a contract-clarification amendment before any Geppetto/Arachne learned output is opened. It does not authorize training and does not weaken the clean-room or no-completion rules.

## Binding equivalence boundary

### Geppetto

Reference-system information must be compared against the complete consumer-facing RealSaS observation stack:

```text
8 views + known cameras
  -> IRIS learned observation evidence
  -> ObservationEvidenceIR
  -> deterministic analytic P = O + dF
  -> deterministic SurfaceBuilder / admitted teacher-free persistence
  -> RiggingSurfaceIR
  -> optional declared deterministic GeppettoConditioningAdapter
  -> Geppetto
```

The scientific equivalence object is therefore:

`B_G = GeppettoConditioningAdapter(RiggingSurfaceIR)`

not raw DINO tokens, raw forward depth, or raw `ObservationEvidenceIR` alone.

The adapter may only derive information from the exact consumed `RiggingSurfaceIR` and public product constants. It may resample, normalize, derive qualified normals/local differential features, construct neighborhood descriptors, or package tensors when those operations are deterministic and observation-grounded. It may not use source mesh truth, source-rig IDs, hidden teacher fields, learned completion, or another geometry authority.

### Arachne

Reference skinning-system information must be compared against:

```text
RiggingSurfaceIR
  + Compiler-qualified QualifiedSkeletonIR
  -> optional declared deterministic ArachneConditioningAdapter
  -> Arachne
```

The scientific equivalence object is:

`B_A = ArachneConditioningAdapter(RiggingSurfaceIR, QualifiedSkeletonIR)`.

The adapter may deterministically derive surface samples, normals/local frames, bone-relative coordinates/distances, neighborhood structure, masks and other conditioning quantities that are functions only of the admitted surface and qualified skeleton. It may not recover or import authored helper identity, source-mesh-only topology, hidden weights, or unobserved completed geometry.

## Current executable RealSaS substrate

`IR_TYPE_SYSTEM_V1.md` and `compiler/realsas_compiler_core/types.py` define the current typed boundary:

- `ObservationEvidenceIR`: view/raster binding, known ray origin/forward, learned forward depth, support/provenance, validity;
- `SurfaceBuilder`: deterministic analytic geometry and admitted persistence;
- `RiggingSurfaceIR`: `SurfaceNode(P, support_views, provenance_refs, source_observation_ids, raster_bindings, persistence_group_id, optional derived_normal, validity_flags)` plus local `SurfaceRelation`s and a geometry lineage hash;
- `QualifiedSkeletonIR`: compiler-owned joint positions, parents/root, support and skeleton lineage.

Thus reference-equivalence is allowed to credit information **deterministically recoverable at these boundaries**, even when it is not a direct learned IRIS head.

## Two distinct equivalence questions

Every reference field/mechanism must now be evaluated at two levels:

### E1 — information availability / derivability

Can the reference-required quantity be obtained from `B_G` or `B_A` without hidden source truth?

Statuses remain:

- `EXACT_EQUIVALENT`
- `STRICTLY_STRONGER_REALSAS`
- `APPROXIMATE_EQUIVALENT`
- `MISSING`
- `NOT_REQUIRED_AT_INFERENCE`
- `UNKNOWN_FROM_PUBLIC_RELEASE`

A quantity counts as available if it is either explicitly present or deterministically derivable under a frozen, tested adapter.

### E2 — coverage / accessibility

Even if the field type exists, does RealSaS expose enough of it to solve the same functional problem?

Examples:

- a normal field can be type-equivalent while surface coverage is weaker;
- a sampled point cloud can be shape-compatible while unseen/back-side surface support is missing;
- local adjacency can be derivable while disconnected-sheet ambiguity makes it unreliable;
- a qualified skeleton can be structurally sufficient while its joint-position error changes skinning accessibility.

Therefore `same tensor fields` is not sufficient for a positive equivalence conclusion. Coverage, support, resolution, uncertainty and downstream oracle ceilings must also pass.

## Oracle comparison protocol

The first Geppetto/Arachne solvability experiments may use authoritative corpus geometry to create **oracle observation evidence**, but must then pass that evidence through the same deterministic RealSaS consumer stack.

Allowed oracle route:

```text
authoritative geometry
  -> exact observation/raster-aligned evidence permitted by shipping observations
  -> same analytic reconstruction
  -> same SurfaceBuilder
  -> same declared conditioning adapter
  -> learned consumer
```

Forbidden oracle shortcut for equivalence claims:

```text
source master mesh / hidden rig truth
  -> directly feed extra fields unavailable to product RiggingSurfaceIR
  -> call resulting success a RealSaS-input ceiling
```

This keeps the oracle ceiling focused on consumer apparatus rather than granting the consumer a richer substrate than product inference can ever provide.

A separate **reference-upper-bound diagnostic** may intentionally feed richer source geometry, but it must be labeled as such and may not establish RealSaS information equivalence.

## Reference-specific consequence

### RigAnything / Geppetto

The public RigAnything inference path samples normalized 3D surface points and normals. The Geppetto audit must therefore ask whether an equivalent point/normal conditioning set can be deterministically constructed from `RiggingSurfaceIR`, and separately whether partial observation-supported coverage is sufficient compared with RigAnything's mesh-surface sampling.

The relevant scientific question is not `raw IRIS depth == RigAnything point cloud`; it is:

> `Can the admitted RealSaS pre-Geppetto stack expose a consumer conditioning substrate with the information and coverage needed by a RigAnything-class skeleton solution?`

### SkinTokens / Arachne

SkinTokens conditions its skin representation/decoding on sampled geometry and, in the unified model, on skeleton tokens. Arachne equivalence must therefore be judged after deterministic surface conditioning and **after Compiler skeleton qualification**, not against IRIS alone.

The relevant scientific question is:

> `Can RiggingSurfaceIR + QualifiedSkeletonIR expose the geometry/skeleton conditioning needed by a SkinTokens-class skin-field solution without hidden completion or authored-rig leakage?`

## Authority firewall remains unchanged

This amendment changes the **measurement boundary**, not system ownership:

- IRIS remains learned observation evidence;
- SurfaceBuilder remains deterministic and non-owning;
- Geppetto/Arachne remain proposal models;
- Compiler remains canonical rig authority;
- deterministic adapters are packaging/derivation only and must be versioned, tested and hashed;
- unobserved geometry remains UNKNOWN unless a separately revised contract explicitly authorizes another source of evidence.

## Change-control effect

All R5 information-equivalence matrices and R6 oracle-substrate ceiling designs under `GEPPETTO_ARACHNE_EXTERNAL_REFERENCE_CLEANROOM_AUDIT_PREREG_20260831.md` must use `B_G` / `B_A` above.

Any future comparison that uses raw IRIS output as the sole RealSaS side of the reference-equivalence test is invalid unless explicitly labeled as a narrower diagnostic.

Scientific training authority remains unchanged.