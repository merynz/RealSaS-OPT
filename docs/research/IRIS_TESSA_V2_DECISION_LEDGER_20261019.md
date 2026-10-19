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

**Decision:** no TESSA optimizer step until:

1. native V8 R24 FlexiCubes witness is materialized for Fit8;
2. independent geometry + exact Stage01 mask measurements are recorded;
3. the program decides from those measurements whether the debt is primarily native V8 surface quality or high-resolution transition design;
4. a corrected high-resolution direct A-arm is defined and reaches its preregistered readiness boundary.

The Snake raw-field sparse failure is evidence to redesign the A-arm transition, not permission to skip A and train TESSA.