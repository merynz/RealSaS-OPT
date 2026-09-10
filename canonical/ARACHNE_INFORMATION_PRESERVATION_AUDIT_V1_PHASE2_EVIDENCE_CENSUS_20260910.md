# RealSaS — Arachne Information-Preservation Audit V1 — Phase 2 Evidence Census

**Date:** 2026-09-10  
**Status:** `OPEN_RESEARCH_AUDIT__EVIDENCE_ONLY__NO_ARCHITECTURE_OR_TRAINING_CHANGE_AUTHORIZED`  
**Audit branch:** `audit/arachne-information-preservation-v1-20260910`  
**Parent audit:** `canonical/ARACHNE_INFORMATION_PRESERVATION_AUDIT_V1_20260910.md`  
**Continuation authority remains:** `main/CURRENT_STATE.md`  
**A0 4/8/16/32 experiment:** unchanged; this census does not alter its preregistration, optimizer, gates, winner rule, A1 authorization, or product status.

This file records exact artifact-level follow-up evidence discovered after Phase 1. It is an evidence appendix, not a new architecture authority.

## 1. Exact Mage dense-zero-surface cardinality recovered

The promoted/consumed Mage zero-surface artifact `ZERO_SURFACE_PRODUCT_CLIPPED.npz` was fetched from the Drive ID bound by the promoted Geppetto evidence manifest.

Exact arrays:

- `vertices`: **118,388 × 3**, float32;
- `faces`: **234,962 × 3**, int64;
- `normals`: **118,388 × 3**, float32.

Therefore the previously remembered “~120k” quantity was real and corresponds to the **dense predicted zero-surface vertex count**, not to an assumed 8×128×128 raster lattice.

Current compact Mage substrate remains:

- dense predicted zero-surface vertices: **118,388**;
- dense predicted zero-surface faces: **234,962**;
- compact GSA carriers: **950**;
- compact GSA undirected local relations: **2,813**;
- observed compact nodes: **897**;
- model-completed compact nodes: **53**.

Simple cardinality ratios, descriptive only:

- compact carriers / dense vertices = `950 / 118388 ≈ 0.008024` = **0.8024%**;
- dense vertices per compact carrier = `118388 / 950 ≈ 124.62`.

This large cardinality reduction is not by itself evidence destruction: the scene-first GSA bridge maps dense triangle connectivity through compaction and re-derives view support/raster bindings and robust normals. The audit question is whether the resulting typed compact state is task-sufficient, not whether raw cardinality is preserved one-for-one.

## 2. IRIS V3 uncertainty seam is now localized more precisely

Phase 1 recorded `log_uncertainty` as a named learned IRIS V3 output whose downstream use was unclear. Source inspection now localizes the exact handoff:

1. `ContinuousSignedSurfaceFieldV3` returns `sdf` and `log_uncertainty`.
2. `dense_signed_grid_v3(...)`, when its query function returns a dict, explicitly selects only `y["sdf"]`.
3. `ZeroSurfaceMeshV3` contains only normalized vertices, faces, normals, and scalar field min/max.
4. The scene-first GSA bridge consumes zero-surface geometry/cameras; it has no per-surface uncertainty carrier.

So the current source-level classification is stronger than “no GSA consumer found”:

`IRIS log_uncertainty -> DROPPED_AT_SIGNED_GRID/ZERO_SURFACE_INTERFACE`.

However **scientific severity remains conditional** because the current promoted witness does not establish calibration or unique downstream value for this uncertainty head after the signed-SDF/zero-set transition.

Current verdict:

- existence: **proven**;
- loss at current public zero-surface boundary: **proven**;
- calibrated usefulness: **unknown**;
- redundancy with deterministic GSA support/local-geometry diagnostics: **unknown**;
- immediate architecture repair authorized: **no**.

The right next step remains a zero-optimizer calibration/redundancy audit, not speculative wiring.

## 3. Exact current Arachne conditioning-cache state

The current Mage A0 cache `ARACHNE_MAGE_FS1_CONDITIONING_CACHE_V2.npz` was fetched from the active Mage A0 Drive folder and inspected.

Exact arrays include:

- `surface_ids`: `[950]`;
- `rest_points_world`: `[950,3]`;
- `surface_features`: `[950,20]`;
- `surface_positions_normalized`: `[950,3]`;
- `joint_ids`: `[22]`;
- `joint_features`: `[22,8]`;
- `parent_indices`: `[22]`;
- `pair_geometry`: `[950,22,10]`;
- `pair_mask`: `[950,22]`;
- `teacher_weights`: `[950,22]`;
- `teacher_supervision_mask`: `[950]`;
- `confidence_code`: `[950]`.

This confirms the Phase-1 distinction:

- rich upstream `RiggingSurfaceIR + QualifiedSkeletonIR` exists;
- the historical/current cached Arachne conditioning view is a **consumer-specific compressed representation**, including the old 20D surface summary and 8D joint summary;
- the useful 10D point↔joint/parent-segment geometry is preserved in the cache.

This cache is valid A0/A1 lineage evidence; it must not be mistaken for the complete legal information boundary available to a future V7-native A1.

## 4. Geppetto `support_surface_ids` semantics: sparse support anchors, not surface assignment

This is a critical semantic clarification for Arachne design and for interpretation of structural gates.

Promoted Geppetto proposal code does the following for each generated joint:

- predict a support-presence probability;
- if admitted, sort that joint's support logits over all surface nodes;
- retain only `support_topk` nodes;
- current support top-k is **8**.

Therefore `QualifiedJoint.support_surface_ids` is **not** a partition/coverage assignment of all surface nodes to joints. It is a sparse set of high-support anchors per joint.

The exact promoted Mage qualified skeleton contains:

- 22 joints;
- 8 support IDs per joint;
- **176 total joint→surface references**;
- **155 unique surface nodes** referenced by at least one joint;
- **795 / 950 surface nodes are not present in any qualified joint's support list**;
- unique support-anchor coverage of compact surface nodes = `155/950 ≈ 16.32%`.

This is not automatically a defect. The support lists were designed as sparse mechanical-support evidence, not as skin weights or exhaustive semantic ownership.

But it changes two interpretations:

1. `unsupported_joint_count == 0` means **every qualified joint has at least one support anchor**. It does **not** mean every surface node is assigned/supported by some joint.
2. Future A1 must not interpret `support_surface_ids` as an exhaustive hard mask. If consumed, they are sparse positive mechanical evidence / attention bias, while surface↔joint skin ownership remains Arachne's learned problem.

This finding strengthens the earlier rule: preserve the exact mapping, but do not overstate its semantics.

## 5. Mage head/hood question — exact artifact-level check

A visual concern was raised because the qualified skeleton overlay appears visually sparse/empty inside the large head/hood silhouette.

The exact qualified artifact shows that the head region is **not control-empty**:

- highest central qualified control: `J:6c57ffd28437d5a273fd`;
- source proposal: `P:GRS:0012`;
- world position approximately `(0.00597, 0.00088, 1.24477)`;
- parent: `J:33d2c674ed9153028054` / source `P:GRS:0006` at z≈`0.97430`;
- head control has 8 sparse support anchors.

Those 8 support-anchor z values lie approximately between `1.0673` and `1.2644`; they do not extend over the full upper hood volume. Per Section 4 this is expected because support IDs are top-k anchors, not exhaustive coverage.

A teacher-only diagnostic on the existing Mage A0 cache gives a much more useful answer about deformation ownership. For all **387 supervised compact surface rows with world z ≥ 1.3**, the teacher skin field is exactly:

`weight(head_joint) = 1.0`, all other joint weights = `0.0`.

The same exact one-hot head ownership holds in the higher z slices inspected (`z ≥ 1.6` and `z ≥ 1.9`).

Interpretation:

- the large upper hood/hat silhouette is present in the compact A0 substrate;
- current teacher semantics treat that upper spatial region as a **rigid head-bound region**;
- a deform joint does not have to geometrically extend through every pixel/vertex it controls; one rigid head transform can legitimately move a much larger bound hood region;
- therefore “there is no long bone drawn through the hat” is **not evidence that the upper hood is uncontrolled**.

What this does **not** prove:

- that one head control is product-optimal for every character;
- that a flexible hood tip/hair/cloth should never get secondary controls/deformers;
- that face/eye expressive controls are unnecessary;
- that current Mage attachment/render semantics are complete.

For current FIT1 target fidelity, one rigid head control is consistent with the teacher skin field. Product capability for flexible headwear/secondary motion remains a separate later contract.

## 6. Correction to the proposed “hat should trigger unsupported_joint_count” test

The statement “if hat surface nodes are assigned to no joint, `unsupported_joint_count == 0` should catch it” is **incorrect for the current schema**.

The actual measurement is equivalent to:

`sum(not joint.support_surface_ids for joint in skeleton.joints)`.

It checks joint→some-support existence. It does not compute the inverse surface→some-joint coverage relation.

A separate surface-coverage diagnostic would be needed if the product contract ever requires one. It must not be conflated with the current structural joint-support gate.

For Arachne, exhaustive surface coverage is naturally expressed by qualified skin rows after skin inference/qualification: `qualify_skin()` explicitly fails if any surface has no skin influences. That is the appropriate later boundary for full per-surface skin coverage.

## 7. Rigid attachments: relevant but distinct from the current Arachne A0 audit

The supplied front render visibly contains a staff and shield while the displayed GSA body substrate does not visibly trace those rigid pieces. Historical R6 render extraction records already distinguish rigid bone attachments from unrelated scene geometry: a helmet was classified `RIGID_BONE_ATTACHMENT`, while an `Icosphere` was excluded by the subject gate.

The exact current `QualifiedSkeletonIRV2` type includes an `assembly_root_binding`, but the current `qualify_skeleton_v2()` implementation constructs it as an empty dict. The promoted Mage qualified skeleton artifact likewise has `assembly_root_binding: {}`.

Therefore the critique is directionally correct that **rigid attachment/assembly semantics are not yet demonstrated by the current Mage qualified-skeleton artifact**. What is not yet proven is the exact current classification of this Mage staff/shield; do not claim deliberate exclusion vs lost support until their source/render classification witness is recovered.

This is not an A0 codec blocker. It is a product-coverage item and should be tracked separately:

- recover the Mage staff/shield extraction classification;
- verify intended rigid attachment parent/socket semantics;
- ensure a future FIT-k/FIT8 selection includes at least one asset with a genuine rigid attachment so attachment capability cannot be accidentally avoided by family selection.

## 8. Regional A0 error audit queued, but must not mutate the running experiment

The suggestion to inspect head/hood, cape, and other spatially distinctive regions is scientifically useful. Current geometry-causality evidence is global; it does not prove equal field fidelity by region.

Do **not** add a new acceptance gate to the running 4/8/16/32 preregistration.

After the current arms finish, run a zero-optimizer diagnostic on the frozen winner/control artifacts with row-level stratification such as:

- upper rigid-head spatial slice;
- central torso;
- lateral limb bands;
- cape/back-protruding geometry where a product-legal spatial/component rule can be defined;
- pure one-joint rows vs blend-boundary rows;
- GSA950 vs disjoint-surface holdout for matched region definitions.

Semantic names like “hat” or “cape” must be treated as diagnostic labels/manual masks unless a product-legal typed component exists. They must not become hidden teacher-derived predictor inputs.

## 9. Phase-2 implications for V7-native A1 design

No architecture is authorized here, but three design constraints are now stronger:

1. **Do not use `QualifiedJoint.support_surface_ids` as a hard/exhaustive assignment mask.** It is sparse top-k mechanical evidence.
2. **Do not equate structural joint support with skin surface coverage.** Full surface coverage belongs to the Arachne/Compiler skin boundary.
3. **Do not decide head-control sufficiency from line-art occupancy.** Evaluate deformation behavior/weight ownership and intended secondary-motion capability separately.

The primary Phase-1 P0 remains unchanged: the old A1 consumer view compresses legal S/G evidence materially. The running A0 token-capacity experiment remains causally untouched.

## 10. Updated next actions

Zero-optimizer audit order after this census:

1. quantify whether IRIS V3 `log_uncertainty` is calibrated/non-redundant; if not, explicitly deprecate or leave private rather than wiring it by habit;
2. finish dense→GSA geometry/support census using the now exact `118,388 → 950` cardinality;
3. complete proposal→qualified selected-evidence field census;
4. complete S+G→old-A1 field-by-field information matrix, including the newly clarified sparse-support semantics;
5. after A0 winner selection, run latent-interface audit and regional row-error diagnostic;
6. separately recover Mage rigid-attachment classification and add rigid-attachment presence as a future FIT-k/FIT8 corpus-selection coverage dimension.

**No repair, A1 implementation, optimizer run, or product PASS is authorized by this Phase-2 evidence appendix.**