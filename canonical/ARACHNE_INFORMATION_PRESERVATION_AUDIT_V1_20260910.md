# RealSaS — Arachne Information-Preservation Audit V1

**Date:** 2026-09-10  
**Status:** `OPEN_RESEARCH_AUDIT__EVIDENCE_ONLY_BRANCH__NO_ARCHITECTURE_CHANGE_AUTHORIZED`  
**Audit branch:** `audit/arachne-information-preservation-v1-20260910`  
**Base main commit:** `67a1ff1f67d5d2e7f8e6ba2630f388aab9432440`  
**Continuation authority:** `main/CURRENT_STATE.md`  
**Current A0 experiment:** external/live 4/8/16/32 internal-token-capacity work continues unchanged; this audit does not alter its preregistration, optimizer, gates, result interpretation, A1 authorization, or product status.

> This document is an in-progress architecture/evidence audit, not scientific promotion authority. Findings below distinguish observed source facts from hypotheses and open questions. No A1 implementation or training is authorized by this file.

## 0. Why this audit exists

Arachne is the first current downstream stage that must combine two already-qualified information systems:

`RiggingSurfaceIR S + QualifiedSkeletonIR G -> learned skin/deformation proposal`.

The immediate concern is not whether RealSaS has enough upstream information in the abstract. It is whether product-legal information that IRIS/GSA/Geppetto/Compiler already produced is preserved and made accessible across subsystem boundaries without either:

1. erasing useful evidence during model/Compiler transitions; or
2. leaking teacher/source-only information into product inference.

The working rule is:

> **Upstream evidence preservation and downstream feature admission are separate problems.** Product-legal, task-relevant evidence should remain recoverable in typed/certified state. A learned consumer may use only the subset justified by controlled ablation. Qualification chooses canonical state; it is not synonymous with evidence erasure.

This audit follows the binding ownership rule in `canonical/SUBSYSTEM_OWNERSHIP_ENVELOPES_V1.md`: one semantic owner per question; deterministic qualification may constrain but may not secretly re-infer learned semantics.

## 1. Audit taxonomy

### 1.1 Information classes

- `CANONICAL_AUTHORITY` — final accepted state owned by a deterministic authority, e.g. Compiler-selected parent/tree/canonical IDs.
- `PRODUCT_LEGAL_EVIDENCE` — learned or observed evidence available from legal shipping inputs.
- `DERIVED_PRODUCT_LEGAL_EVIDENCE` — deterministic function of product-legal state/cameras/constants.
- `PROVENANCE_ONLY` — hashes/IDs/source references required for lineage but not appropriate neural shortcut features.
- `MODEL_PRIVATE_LATENT` — hidden activations with no current public semantic contract; not automatically required downstream.
- `TEACHER_ONLY` — training/evaluation truth forbidden as A1 shipping input.
- `UNKNOWN_CALIBRATION` — a typed channel exists, but usefulness/calibration or intended downstream role is not established.

### 1.2 Transition verdicts

- `PRESERVED_EXACT`
- `PRESERVED_DERIVED`
- `PRESERVED_PROVENANCE_ONLY`
- `CONSUMED_BY_OWNER`
- `COMPRESSED`
- `DROPPED_CANDIDATE`
- `PRIVATE_NOT_REQUIRED`
- `UNKNOWN`

Severity is separate from verdict:

- `P0` — likely architecture-blocking before V7-native A1 preregistration.
- `P1` — important evidence-access issue; must resolve or explicitly reject before interface freeze.
- `P2` — useful hardening/ablation item; not presently a blocker.
- `INFO` — expected/intentional behavior.

## 2. Authority and source anchors

This audit is bound to the following inspected source identities.

| Area | Path / ref | Blob SHA |
|---|---|---|
| Current IR types | `compiler/realsas_compiler_core/types.py@main` | `27b6bfa3a9959b0026683a250e3583ee25b5c19c` |
| Scene-first IRIS V3 | `models/iris/v3/scene_first_signed_v3.py@main` | `65819277910e91bb9106d593c7d5eb851dd58cec` |
| Promoted Mage IRIS witness | `models/iris/v3/PROMOTED_MAGE_FIT_WITNESS_V1.json@main` | `28e1aed73a943bfc60c8bb0f331fca5348dbcf9b` |
| Scene-first GSA bridge | `compiler/realsas_compiler_core/substrate/scene_first_signed.py@main` | `a55e431faf0680e0d7f2920a85fda6341d45117d` |
| Promoted Geppetto surface tensorizer | `models/geppetto/reference_strength_v1/rigging_surface_tensorization_v1.py@main` | `647fdcb98c305c3f7d3afd05dd61449999721034` |
| Promoted Geppetto candidate | `models/geppetto/reference_strength_v1/geppetto_reference_strength_candidate_v1.py@main` | `e9b626815c63a96ac1d390580e1aa19f8bf1cfb2` |
| No-learned-slot Geppetto wrapper | `models/geppetto/reference_strength_v1/geppetto_reference_strength_no_learned_slot_v1.py@main` | `6f2505aa9e4169cec29c2c1508cde7264971073c` |
| Compiler skeleton qualifier | `compiler/realsas_compiler_core/rig.py@main` | `4569997706bc13f124406d8696df1bf6d266869d` |
| Old A1 conditioning V1 | `models/arachne/v2/conditioning_v1.py@main` | `b3539ed6c4580ac69338bdb9bd4b93c6e78a262c` |
| Old A1 conditioning V2 | `models/arachne/v2/conditioning_v2.py@main` | `3383cb7801b22aeb634f05551a847a04e674c7c8` |
| Old A1 pair geometry | `models/arachne/v2/arachne_geometry_v2.py@main` | `0a39450b69de48da3b7748cb08972d0b69ff9700` |
| Old A1 predictor | `models/arachne/v2/arachne_candidate_v2.py@main` | `ba5bf0a2757e8d702176332fcfa91f804e0d545e` |
| Compiler skin qualifier | `compiler/realsas_compiler_core/skin.py@main` | `7d3379df9d13338d8f560588c0f04aed47ec36d7` |
| V7 A0 research codec | `models/skin_field_codec/v7/skin_field_codec_v7.py@exp/arachne-skintokens-cleanroom-fit1-20260908` | `be85ce3b16c7114fca5dfa39e4ed2f4ae2dda921` |

Historical comparison/contract anchors include:

- `canonical/GEPPETTO_ARACHNE_CONSUMER_SUBSTRATE_EQUIVALENCE_BOUNDARY_AMENDMENT_20260831.md`
- `canonical/REALSAS_RIGANYTHING_SKINTOKENS_END_TO_END_CLEANROOM_MATRIX_20260903.md`
- `experiments/m4_identity_audit/VERY_IMPORTANT_AUDIT_M4_IDENTITY_AMBIGUITY_EQUIVALENT_SUBSTRATE_20260823.md`

Those records already establish the important precedent that the consumer boundary is the complete legal substrate, not raw IRIS output, and that historical fixed-width adapters could be scientific information-isolation proxies rather than product-optimal consumers.

## 3. End-to-end transition map — preliminary

| Boundary | Current status | Preliminary verdict |
|---|---|---|
| RGB/cameras -> IRIS signed field | promoted Mage FIT1 witness | geometry path strong; uncertainty role open |
| IRIS signed zero-surface -> GSA | current deterministic bridge | compaction/topology/support/raster preserved in structured form; uncertainty not observed in bridge |
| GSA -> promoted Geppetto | FIT1-frozen | **strong / fieldwise evidence-preserving** |
| Geppetto proposal -> Compiler -> QualifiedSkeletonIR | current qualifier | canonical graph authority correct; selected proposal evidence accessibility is narrower than proposal evidence |
| Qualified S+G -> old A1 conditioning | prior scaffold | **materially compressed** |
| A1 latent -> V7 A0 decoder | not current A1 | interface pending A0 winner/freeze; separate from S+G conditioning richness |
| Arachne SkinProposalIR -> Compiler skin qualification | implemented | scalar weight legality path clean; future structured uncertainty/evidence channel not yet typed |

## 4. IRIS -> GSA

### 4.1 What current IRIS V3 actually emits

`ContinuousSignedSurfaceFieldV3` returns two public named outputs at arbitrary queried 3D points:

- `sdf`
- `log_uncertainty`

`scene_planes` is also returned by the outer module but is treated here as a model-private latent unless separately promoted into a semantic evidence contract.

The current promoted Mage witness proves the signed-geometry route on the Mage FIT1 witness from `8x1024_RGBA + 8 exact orthographic cameras`, with no teacher mesh at inference. It does not report an uncertainty calibration metric.

### 4.2 What the scene-first GSA bridge consumes

`rigging_surface_from_scene_first_zero_mesh_v1` consumes:

- predicted zero-surface vertices;
- triangle faces from the decoded zero surface;
- implicit normal orientation hints;
- exact cameras;
- normalization/provenance identities.

It then deterministically:

- estimates robust local-PCA normals on the dense predicted zero surface;
- adaptive-voxel compacts the surface to the target carrier budget;
- maps dense triangle connectivity through the compaction inverse and retains unique non-self compact edges;
- re-derives exact 8-view raster projection and self-zbuffer visibility/support;
- emits `SurfaceNode` and `SurfaceRelation` typed state;
- records `source_dense_vertex_count`, compact count, observed/completed count, per-view support counts, source run/checkpoint/zero-surface hashes and operator identities in metadata.

### 4.3 Preliminary finding F-01 — uncertainty seam

**Status:** `OPEN / UNKNOWN_CALIBRATION`  
**Severity:** `P1 pending calibration`  
**Observed fact:** current mainline `log_uncertainty` search resolves to the IRIS V3 head and its shape test; the scene-first GSA bridge has no explicit per-surface uncertainty input/output field.  
**Do not conclude:** uncertainty is required downstream, or that its absence currently harms Mage.  
**Required question:** after scene-first signed-field/zero-set extraction, is `log_uncertainty` calibrated, non-redundant with deterministic support/local-geometry diagnostics, and predictive of downstream error?

Until that is measured, its correct status is `???`, not “must consume” and not “delete.”

### 4.4 Preliminary finding F-02 — scene-first compaction is not evidence erasure by default

**Status:** `PRESERVED_DERIVED / INFO`, with quantitative sufficiency still open.  
Mage current witness is `950` compact nodes, `2813` unique undirected topology relations, `897` observed nodes and `53` model-completed signed-zero-surface nodes. The average undirected degree is approximately `2*2813/950 = 5.92`.

The compactor does not simply subsample XYZ. It reconstructs compact connectivity from source faces and preserves/re-derives per-view support/raster evidence and robust normals. The exact Mage `source_dense_vertex_count` is still **TODO**; do not substitute the earlier unverified ~120k/131072 recollection.

### 4.5 Scene-first provenance nuance

For current scene-first nodes, `source_observation_ids=()` is intentionally empty at this bridge; per-view support/raster binding is re-derived from predicted surface + exact cameras, while higher-level provenance is bound through authority label/source run/checkpoint/zero-surface hashes. This is currently classified `PROVENANCE_ONLY / P2 review`, not a neural-information blocker.

## 5. GSA -> promoted Geppetto

### 5.1 Preliminary finding F-03 — this boundary is the positive control

**Status:** `PRESERVED_EXACT/PRESERVED_DERIVED`  
**Severity:** `INFO / desired pattern`

The promoted `rigging_surface_tensorization_v1.py` explicitly rejects the legacy summary-only design and carries fieldwise:

- world and normalized positions;
- normals + validity;
- 8-view support;
- per-view normalized raster XY + validity;
- observed/completed status;
- complete validity bits;
- exact GSA relation graph;
- edge score/distance/type and unknown-crossing/bridge state;
- degree and normalization state;
- certificate/provenance/operator hashes.

Teacher contamination fails closed. Provenance/operator metadata is certified without being made a neural shortcut feature.

This is the main architectural precedent for Arachne: **preserve legal evidence at the consumer boundary, then decide feature admission separately.**

## 6. Geppetto -> Compiler -> QualifiedSkeletonIR

### 6.1 What proposal evidence exists

Current reference-strength Geppetto proposal joints carry canonical-input candidates including:

- position;
- root score;
- confidence;
- support-surface IDs;
- metadata including `mechanical_salience_probability` and `position_sigma_normalized`.

Proposal edges carry directed parent evidence, confidence and legality flags.

### 6.2 What Compiler correctly consumes

`qualify_skeleton()` consumes root/confidence and directed-edge score/confidence into the exact graph optimizer, checks support references, fails closed on invalid proposals, then mints canonical IDs only after exact solve. Raw parent alternatives must not be given downstream authority to override the selected canonical tree.

### 6.3 Preliminary finding F-04 — selected evidence is not first-class in QualifiedJoint

**Status:** `COMPRESSED / EVIDENCE_ACCESS_SEAM`  
**Severity:** `P1`

Current `QualifiedJoint` carries:

- canonical joint ID;
- qualified position;
- canonical parent;
- support-surface IDs;
- source proposal ID.

It does **not** first-class type selected-joint confidence, `position_sigma_normalized`, or `mechanical_salience_probability`. The qualification lineage commits to the full proposal and `source_proposal_id` preserves referential joinability in principle, so this is not “data destroyed forever.” But the declared Arachne boundary `RiggingSurfaceIR + QualifiedSkeletonIR` does not itself expose these selected evidence values.

Open design question: should a future qualified evidence sidecar preserve product-legal selected-joint evidence while leaving canonical geometry/tree immutable? Any such sidecar must be evidence only, never a second skeleton authority.

## 7. Qualified S+G -> old A1 conditioning

### 7.1 Preliminary finding F-05 — confirmed lossy consumer adapter

**Status:** `COMPRESSED`  
**Severity:** `P0 before V7-native A1 preregistration`

The prior `ArachneConditioningAdapter` reduces rich `RiggingSurfaceIR` to 20 scalar features per node. Specifically:

- exact `local_relations` graph -> normalized degree + mean relation score;
- per-view `raster_bindings` -> only raster count;
- support views remain as 8 bits, but raster coordinates are not retained;
- joint `support_surface_ids` -> only support count;
- parent indices survive separately;
- P, N, normal validity and coarse root/depth state survive.

V2 then adds a useful explicit 10D point-joint/parent-segment geometry contract:

`dXYZ, point-control distance, point-parent-segment distance, segment t, segment length, normal-axis |cos|, normal-axis validity, parent-exists`.

This 10D relation should not be discarded merely because the surrounding adapter is old; it is a good, product-legal deterministic mechanical relation.

### 7.2 Old A1 is a scaffold, not a falsified idea

`ArachneCandidateV2` is a small relational latent predictor: 128-wide, 2-layer surface transformer; joint query with immediate-parent context; pair-biased/value surface attention; per-joint latent mean/log-sigma; frozen old codec decode. The important lineage idea survives:

`surface evidence + qualified skeleton + point/bone relation -> learned field representation`.

The audit does **not** claim its parameter count or architecture failed. It establishes that its consumer adapter does not expose the full modern S/G substrate.

## 8. A1 conditioning vs A0/V7 decoder interface

### 8.1 Preliminary finding F-06 — keep these two boundaries separate

V7 A0 has a narrow codec geometry contract `xyz + normal + normal_valid` and field observation `geometry7 + scalar W` during A0 training. Its condition encoder sees geometry only; its field encoder sees teacher W because A0 is a representation test. The decoder consumes continuous field tokens + condition tokens + query geometry.

This does **not** imply V7-native A1 should only see geometry7. A1 may consume the richer legal `S+G` evidence and predict the frozen field-token language. The codec interface and the predictor conditioning interface are separate design objects.

Current strict V7 source binds four field tokens. The live 4/8/16/32 experiment is intentionally testing internal token cardinality beyond that strict source contract. Therefore exact `K`, winner hash and latent semantics remain unresolved and must not be hardcoded into A1 during this audit.

## 9. Arachne proposal -> Compiler skin qualification

### 9.1 Current scalar-weight boundary

`SkinProposalIR` types per-pair `(surface_id, canonical_joint_id, weight)` plus S/G binding hashes, provenance and metadata. `qualify_skin()` validates lineage/references, rejects duplicates/nonfinite/material negatives, applies only bounded legal/simplex/top-k projection and records correction/discarded mass. It does not invent semantic skin weights.

**Current verdict:** `PRESERVED_EXACT + CONSUMED_BY_OWNER / GREEN` for the scalar-weight proposal.

### 9.2 Preliminary finding F-07 — future evidence channel risk

**Status:** `OPEN DESIGN RISK`  
**Severity:** `P2 now; P1 before A1 interface freeze`

The old A1 already computes per-joint latent uncertainty but collapses it to one aggregate metadata scalar when proposing skin. If V7-native A1 later produces meaningful per-joint/per-row/per-pair uncertainty or boundary ambiguity, current `SkinProposalIR` has no first-class typed channel for it.

Do not add fields speculatively now. Before freezing A1 proposal semantics, decide whether such evidence is calibrated/useful; if yes, type it rather than burying it in an opaque metadata blob.

## 10. Evidence that must **not** be promoted into neural inputs by default

The goal is preservation, not indiscriminate feature dumping.

- raw hashes, lineage IDs, asset/family IDs -> `PROVENANCE_ONLY`, not neural shortcut features;
- teacher W -> training/evaluation target only, never A1 product input;
- source/hidden teacher mesh geometry -> forbidden product input;
- private IRIS `scene_planes` -> `MODEL_PRIVATE_LATENT` unless a separate semantic contract and causal need are established;
- rejected Geppetto parent alternatives -> may remain audit/proposal evidence, but must not let Arachne re-own the Compiler-selected tree;
- source proposal serialization order/generation index -> not canonical identity;
- arbitrary product witness identity -> forbidden shortcut.

## 11. Required zero-optimizer follow-up audits

These are audit measurements, not A0/A1 training authorization.

### A. IRIS uncertainty calibration/redundancy audit

Measure `log_uncertainty` against signed-surface error and downstream GSA error; compare against deterministic support count/view disagreement/local-geometry diagnostics. Classify as useful, redundant, uncalibrated, or private/deprecated. No downstream feature admission before this result.

### B. Dense zero-surface -> compact GSA information-funnel census

For Mage record exact:

`dense zero-surface vertices/faces -> compact carriers -> compact edges -> observed/completed nodes -> per-view support/raster counts`.

Add geometry-preservation diagnostics that do not use hidden source truth as shipping input: compact-to-dense distance, normal deviation, connected-component/topology sanity, support/raster consistency. The source mesh may be used only as FIT evaluation truth where already allowed.

### C. Geppetto selected-evidence preservation audit

Inventory every proposal joint/edge evidence field, classify whether Compiler consumes it for canonical decision, whether it remains recoverable after qualification, and whether Arachne could lawfully benefit from it. Explicitly separate canonical authority from evidence annotations.

### D. Arachne conditioning information-preservation audit

Construct a machine matrix comparing:

`RiggingSurfaceIR + QualifiedSkeletonIR` vs old A1 adapter vs promoted Geppetto tensorization capabilities.

Do not train. Quantify exactly which node, edge, view/raster, joint-support, uncertainty and provenance channels are exact, summarized, omitted, or non-neural by policy.

### E. V7 latent-interface audit — only after A0 winner is known

Before A1 loss/final interface freeze, measure token permutation sensitivity, per-token ablation, prefix/rank sensitivity and decoder-equivalent token-set behavior. This determines whether A1 latent supervision should be ordered or set-matched. It does not belong inside the current 4/8/16/32 result contract.

## 12. Candidate controlled A1 evidence-admission ladder — design only

After A0 terminal/interface closure and a fresh A1 preregistration, evidence should be admitted by controlled arms rather than all at once. A possible ladder is:

1. modern baseline: P/N + qualified G + existing 10D point-segment geometry;
2. + exact GSA local relation graph;
3. + per-view raster XY/support payload;
4. + exact joint<->surface support edges;
5. + selected qualified Geppetto evidence if F-04 justifies a typed sidecar;
6. + IRIS uncertainty only if audit A shows calibrated, non-redundant value;
7. optional derived codec-condition alignment representation as an inductive-bias ablation, explicitly not new evidence.

Exact architecture, model size, loss weights and token cardinality are deliberately **not** frozen here.

## 13. Preliminary audit verdict

The current evidence does **not** support “RealSaS repeatedly throws away all the information it learns.” The more precise picture is:

1. the scene-first GSA compaction retains a structured physical carrier representation and rebuilds topology/support/raster evidence;
2. the promoted GSA->Geppetto boundary is deliberately fieldwise and evidence-preserving and should be treated as the positive design control;
3. some learned Geppetto evidence used around qualification is not first-class accessible in the qualified skeleton boundary;
4. the prior Arachne A1 adapter is definitely more lossy than the current S/G substrate and is the clearest present P0 seam;
5. IRIS uncertainty is genuinely unresolved after the signed-SDF transition: existence is proven, downstream necessity is not;
6. the V7 codec interface must not be confused with the richness of future A1 conditioning;
7. Compiler skin qualification is semantically clean today, but future A1 evidence types should not be silently collapsed if they prove useful.

**Current priority:** finish the field-level matrix and zero-optimizer evidence census while the external A0 token-capacity experiment continues unchanged. No repair or A1 coding is authorized by this open audit.
