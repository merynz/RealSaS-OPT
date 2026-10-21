# IRIS / TESSA V2 Decision Ledger — 2026-10-19

Status: **ACTIVE RESEARCH AUTHORITY FOR PR #122**

This ledger supplements the 2026-10-18 post-V8 decision ledger. It does not rewrite V8/V9 history.

## D1 — TESSA role

**Decision:** TESSA is an optional learned structuralization/retopology layer after IRIS surface evidence, not a mandatory small topology head embedded in IRIS.

**Why:** StdGEN++-class continuous/high-resolution surface reconstruction and MeshAnything-class structural mesh generation solve different problems. Their intersection is useful only if the learned structuralizer adds production topology value beyond the direct surface.

**Falsification:** exact same IRIS evidence must support both direct-surface→Compiler and TESSA→Compiler A/B arms.

## D2 — IRIS owns continuous evidence; Compiler owns canonical M

**Decision:** IRIS owns learned scene/field/evidence. TESSA may propose explicit structured meshes. Compiler remains the only authority that may mint canonical M.

**Rejected:** treating a TESSA free-running decode as product mesh authority by itself.

## D3 — reference-class TESSA capacity from the first serious fit

**Decision:** first serious TESSA V2 uses 1024 width / 16 heads / 24 decoder layers / 4096 FFN, 1024 coordinate bins, 512 scene + 512 surface/mechanics memory tokens.

**Rejected:** beginning with an arbitrarily reduced decoder and using failure to infer architectural inadequacy.

## D4 — preserve useful AMT grammar, add long-range history

**Decision:** retain the 1024-bin AMT-compatible coordinate grammar and component boundaries, while adding causal global structural-history summaries beyond the exact 4096-token local window.

**Why:** local adjacency is valuable but a 32k policy ceiling cannot be justified from a local trailing window alone.

## D5 — topology loss is not topology qualification

**Decision:** token CE remains necessary but is supplemented by metric coordinate loss and structural-token loss. Scientific topology quality is decided by free-running decoded-mesh courts, not teacher-forced loss.

**Required court:** grammar/EOS, components, boundaries/open chains, non-manifold edges, degenerates, source-surface fidelity, support binding and downstream mechanics/Compiler behavior.

## D6 — deterministic geometry must be explicit conditioning

**Decision:** Relation V3 receives full/raw H plus XYZ_A/B, deltas, camera-space depth, projected UV, same-ray residual and camera features.

**Rejected:** forcing a relation MLP to rediscover deterministic geometry from a compressed latent while the geometry is already available.

## D7 — visibility is not a pairwise relation

**Decision:** front/order remains pairwise. Visibility belongs to a complete-ray stack/view abstraction or deterministic raster authority.

**Rejected:** V9-style pairwise `visibility_a = front`, `visibility_b = 1-front` as global visibility truth.

## D8 — no orphan authority outputs

**Decision:** inference outputs that influence authority must be supervised, calibrated, deterministic, or explicitly diagnostic-only.

**Consequences:** new relation API removes free neural confidence; appearance/mechanical V3 remove unsupervised confidence/descriptor/uncertainty outputs; historical scene uncertainty is frozen/hidden from V3 evidence.

## D9 — contact persistence abstains until source authority exists

**Decision:** current Fit8 preparation does not supply `contact_face_mask`, therefore stored contact-persistence zero is not a negative label. Validity remains false and gradient zero.

## D10 — direct surface baseline uses exact historical V8

**Decision:** the A arm is produced by exact completed V8 `IRISVNextV1`, not by re-instantiating/migrating into V3.

**Authority:** source `5083c991...`, step 12000, checkpoint `0be15df...`, binding `735b37f...`, sealed scene_core.

## D11 — historical sparse T512→T1024 decoder was an experimental A-arm candidate

**Original decision:** test `SparseRegularTetraPolicyV5()` exactly: T512 coarse grid, sparse T1024 refinement, no deformation.

**Rejected:** treating the V8 differentiable 24^3 training court as final spatial capability.

**Rejected:** using a reduced 8/16, 64/128 or similar extraction as scientific direct-surface evidence.

**2026-10-19 runtime result:** this raw-field sparse adaptation is now **FALSIFIED as the direct A-arm baseline**. Exact Snake execution at apparatus source `d9701923b2c19bd1c9b9771fe071bb92264a041d` failed closed at `INDEXED_MT_BUILD_AND_NORMALS` with `SPARSE_SHELL_TRUNCATES_ZERO_SURFACE:62735467 boundary crossing tetrahedra`. The same run measured 6,525,575 refined coarse cells, 313,227,600 candidate fine tetrahedra, 95,514,148 crossing tetrahedra and 125,076,400 exact output faces before the fail-closed truncation court.

**Scope:** this falsifies compatibility of the exact V8 **raw scalar field alone** with the historical distance-like sparse selector under the selected policy. It does not falsify V8's native learned FlexiCubes extraction path, sealed H, or TESSA.

## D12 — raw-field identity bridge is preserved as a failed experiment, not promoted authority

**Original decision:** the vNext/V8 scalar-field adapter preserved normalized coordinates and field sign exactly and consumed neither `extraction_delta` nor uncalibrated uncertainty.

**Runtime interpretation:** identity made the experiment clean, but the experiment tested the wrong surface semantics for a V8-trained-witness baseline. V8 G2 surface supervision flowed through `field + extraction_delta + learned 21-D FlexiCubes cube parameters`, not through the raw zero-field alone.

**Replacement:** keep the raw-field result as falsification evidence; do not widen the refinement band, disable boundary truncation, change field sign or add hidden repair to force a mesh.

## D13 — direct surface numeric policy is FP32/no-autocast

**Decision:** scene encode and field query/control extraction remain FP32 without autocast unless a separately preregistered experiment says otherwise.

**Why:** relation-V9 BF16 inference is not authority for surface location, and a precision change would add an unnecessary confound.

## D14 — reuse sealed V9 prepared observations read-only

**Decision:** reuse the exact V9 prepared V4 cohort at source `7029ee3f...`; do not rebuild teachers/cohort for the surface control.

**Binding:** `canonical/VNEXT_DIRECT_SURFACE_INPUT_BINDING_V1.json` remains the exact input-byte authority. Its historical T512/T1024 purpose records the falsified experiment; it must not be silently reinterpreted as a successful A-arm authority.

## D15 — A100 preflight is engineering-only

**Decision:** resource preflight may benchmark exact V8 scene encode and a real field-query chunk, but may not generate a reduced-resolution mesh or make a quality claim.

**Earned result:** the preflight passed before the Snake extraction run; therefore the later failure is scientific/representation-extraction evidence, not a CUDA/resource admission failure.

## D16 — TESSA promotion requires an A/B win

**Decision:** TESSA becomes mandatory only if the preregistered B arm preserves static/multiview/dynamic/support/source-fidelity qualification while adding meaningful structural economy/topology quality.

If a corrected direct A qualifies and B adds no meaningful structural value, direct A is preferred.

## D17 — relation V10 and surface/TESSA are parallel post-V8 workstreams

**Decision:** PR #122 does not erase the relation Truth V3/V10 evidence program. Relation V10 remains blocked on its own truth/conditioning/support gates. Surface/TESSA work proceeds without reopening H or authorizing relation optimization.

## D18 — source green is not scientific green

**Decision:** CI success proves source/contract integrity only. Surface scientific status changes only after real artifacts and independent courts; TESSA scientific status changes only after real free-running A/B evidence.

## D19 — native V8 FlexiCubes R24 is the mandatory control before any carrier or high-resolution redesign

**Decision:** after the raw-field sparse failure, materialize the exact surface path that actually received V8 G2 extraction supervision at its native training resolution `R24`:

```text
exact V8 scene planes
        ↓
query_evidence on native R24 grid
        ↓
field + learned extraction_delta + H
        ↓
learned 21-D cube parameters from cube_extraction(H corners)
        ↓
pinned FlexiCubes @ exact upstream commit
        ↓
native V8 surface witness
```

**Purpose:** separate `V8 native surface quality` from `post-V8 high-resolution transition compatibility`.

**Not a claim:** R24 success would not prove final spatial capability or product suitability.

## D20 — do not naively scale V8 extraction delta to T1024

**Decision:** no direct R24→T1024 reuse of the learned absolute `extraction_delta` is authorized.

**Why:** V8 bounds extraction delta by `0.03` in normalized coordinates. At R24 the nominal cell step is about `0.0833`; at T1024 it is about `0.00195`. The same absolute deformation therefore changes from sub-cell-scale behavior to many fine cells. A high-resolution transition must define resolution semantics explicitly rather than silently reuse the native head.

## D21 — raw-field compatibility forensic is measurement-only

**Decision:** `audit_v8_raw_field_sparse_compatibility_v1.py` may measure sign support, refined-shell boundary structure and finite-difference roughness from the preserved Snake T512 memmap.

These are distance-like/Lipschitz compatibility indicators only. They are not an Eikonal proof and cannot override the failed materialization.

## D22 — TESSA V2 training remains blocked on the replacement A-arm boundary

**Decision:** no TESSA optimizer step until the direct surface program reaches a qualified replacement A-arm. The Snake raw-field sparse failure is evidence to redesign the A-arm transition, not permission to skip A and train TESSA.

## D23 — native V8 R24 control completed and closes the pending-control state

**Runtime authority:** apparatus source `ccb71011f41b041bc5c137544bc1b97ca394ea9c`; Drive evidence root `RealSaS_IRIS_TESSA_V2_FIT8_20261019/ccb71011f41b/V5_NATIVE_V8_CONTROL`; batch receipt SHA256 `a83c0c466d5251c8cc1bc13475866c766c4796e7b0b08c7b96d031c0662667de`; independent control court SHA256 `656f674e12cf3a4ffe79fd16dd25f8ffe4949f736a8e39c9dcd4987df19f3617`.

All eight exact V8 native R24 FlexiCubes witnesses materialized with zero optimizer steps, no TESSA and no high-resolution capability claim.

Measured global values: mean subject mask IoU `0.5605675`, minimum subject mask IoU `0.4053549`, minimum view IoU `0.3425416`, maximum mask gap `392.4207 px`, mean symmetric Chamfer `0.0612046` IRIS units.

Because no PASS/FAIL threshold was preregistered for this diagnostic, the control remains **measurement-only**.

## D24 — native extraction debt and high-resolution transition debt are both open

**Decision:** the runtime evidence no longer supports the simpler hypothesis “native V8 surface is good and only high-resolution adaptation is wrong.”

Compact-teacher fragmentation includes Snake `3→85`, T-Rex `1→59`, Spider `15→88`, GreenBlob `4→92`, Shark `1→67` components; Bat is close at `63→65`. Combined with mask and distance measurements, this establishes actionable native extraction-quality debt without converting the diagnostic into a retroactive formal FAIL.

**Rejected:** immediately blaming or reopening sealed H. Native extraction quality and representation capacity are separate questions.

## D25 — SURF-A1-FROZEN-H is the next information-maximizing experiment

**Decision:** freeze exact V8 scene core/H and train only a new extraction decoder.

**Hypothesis:** sealed V8 H contains enough geometric information for a resolution-aware, raster-supervised surface decoder to recover materially better geometry/source fidelity.

**Why:** this isolates carrier information sufficiency from the known weaknesses of V8's native R24 extraction training. It is cleaner than simultaneously retraining H, surface extraction and topology.

**Interpretation:** A1 PASS permits design of `SURF-A2-HIRES`; A1 FAIL after all apparatus gates pass opens a dedicated H representation investigation. Neither result directly authorizes TESSA.

## D26 — SURF-A1 deformation is cell-relative and zero-step native identity is mandatory

**Decision:** inherited V8 extraction delta is converted from absolute normalized coordinates into native R24 cell units. Learned correction is bounded in cell space; current-grid absolute displacement is obtained by multiplying by `2/resolution`.

At zero step R24, field and V8 21-D cube parameters remain exact baseline values; faces must match the native V8 witness exactly and vertices must differ by at most `2e-6` IRIS units.

**Rejected:** naive reuse of absolute `0.03` at finer resolution; hidden migration that changes the baseline before optimizer step 1.

## D27 — source mask truth and raster-apparatus truth are separate

**Decision:** exact Stage01 source alpha masks remain silhouette optimization truth. They are not used to prove the differentiable renderer is correct.

Independent authored-rest teacher geometry is rasterized with first-party pixel-center semantics to create a teacher-geometry coverage reference. PyTorch3D hard coverage must reach precision and recall ≥ `0.98` on all 64 Fit8 views before optimizer admission.

Teacher depth/normal supervision is valid only where teacher geometry coverage intersects source foreground. Teacher topology/rig/skin remains training-side truth and never inference conditioning.

## D28 — Fit8 is a generic-system forcing function, not merely eight subject scores

**Decision:** every post-V8 experiment must preserve one shared architecture/configuration, one typed contract family and generic Compiler/runtime consumption across all Fit8 subjects. No subject-specific trainable parameters, subject-name branching or hidden semantic shortcuts are allowed.

Fit8 does not claim unseen generalization. Its role is to eliminate one-subject assumptions in models, Compiler boundaries, failure semantics, receipts and execution machinery before future large unseen campaigns.

## D29 — single-consent notebook worker and lane discipline

**Decision:** research notebooks are compute workers. Required UX is `open → Run all → approve Drive once → autonomous execution` with exact-SHA source, precompile/source tests, input hash validation, meaningful progress telemetry and resumable typed checkpoints/receipts.

A second Google OAuth path, `auth.authenticate_user()`, interactive rclone/gcloud auth, manual cell sequencing or hidden reduced-resolution fallback is forbidden.

Operationally only `SURFACE_TESSA` is active now. `RELATION_V10` is parked, not cancelled. TESSA V2 training is blocked specifically on a qualified `SURF-A2-HIRES` direct A-arm; historical raw T512/T1024, native R24 control and SURF-A1 are not eligible substitutes.

## D30 — downstream module boundaries follow the qualified carrier, not vice versa

**Decision:** the long-term target is one generic visual/geometric/mechanical animation-ready canonical carrier M. AXIS, MIRA, GSA and later consumers must be evaluated against that carrier contract rather than freezing historical module boundaries as requirements on the learned representation.

GSA may later shrink, change role or disappear if IRIS/TESSA/Compiler can prove the same responsibilities with stronger typed authority and no regression. This ledger does **not** authorize a GSA removal today; that decision waits for a qualified direct A-arm / TESSA comparison and downstream contract audit.
