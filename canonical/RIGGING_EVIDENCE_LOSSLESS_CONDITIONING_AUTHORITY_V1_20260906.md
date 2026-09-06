# RealSaS Rigging Evidence Lossless Conditioning Authority V1

Date: 2026-09-06
Status: EXPERIMENTAL BRANCH ONLY — NOT PROMOTED
Branch: `exp/lossless-rigging-evidence-v1-20260906`
Base: `c6b5108f672bb1af5354a2c6abd86cda36c2c845`

## Decision

The historical Geppetto V2 **24D summary-only product-conditioning authority is revoked**.

The V2 source remains byte-preserved for historical replay and diagnostics. Its 24D tensor may still be derived and used explicitly in controlled experiments, but it is no longer allowed to stand as the sole product-authoritative representation of `RiggingSurfaceIR` evidence.

New product conditioning authority:

`PRODUCT_VALIDATED_LOSSLESS_RIGGING_EVIDENCE_V3`

Evidence contract:

`RIGGING_SURFACE_IR_FULL_PAYLOAD_PLUS_VIEW_TENSORS_V1`

Revocation ID:

`REALSAS_20260906_CAUSAL_REPAIR_233ec3bcd77e0f02`

## Why authority was revoked

The repaired causal run removed the earlier seed/backend/runtime confounds. Historical parity passed under the frozen runtime. Under that controlled protocol, the corrected production-raster R2 baseline did not close FIT-OPT within the budget, while the C1 full-surface-attention challenger repeatedly entered the exact solution region.

That evidence does **not** prove that every future learner must consume every raw per-view field, nor does it prove a theoretical necessity of one specific attention mechanism. It does prove that the prior engineering assumption — that a 24D summary-only seam was sufficient to serve as the product-authoritative evidence boundary — is no longer justified.

The source substrate already contains richer typed evidence. Destroying that information before the learned consumer has chosen how to use it is therefore forbidden.

## Exact information-loss defect

`RiggingSurfaceIR.SurfaceNode` carries view-indexed raster bindings:

`((view_id, (x, y)), ...)`

and view-indexed support, plus typed local topology.

Historical `GeppettoConditioningAdapterV2` collapses the raster bindings into only:

- `raster_mean_x`
- `raster_mean_y`
- `raster_std_x`
- `raster_std_y`

This mapping is many-to-one. Different view-to-raster assignments can have identical mean/std values. The old 24D representation therefore cannot reconstruct the original evidence.

## New invariant

**Authoritative evidence may be transformed, indexed, padded or normalized, but it may not be irreversibly summarized before the consuming learner boundary.**

Derived summaries are allowed only if all of the following remain available in the same product-conditioning envelope:

1. complete typed source payload;
2. source payload hash;
3. view-indexed raster coordinates;
4. raster-valid mask;
5. view-indexed support mask;
6. typed local-relation topology and fingerprint;
7. boundary audit hash;
8. explicit declaration that any compact feature tensor is derived and non-authoritative.

## Implemented V3 seam

`models/geppetto/v3/conditioning_lossless_v3.py`

The V3 batch carries:

- `surface_ids`
- `positions_normalized`
- `valid_mask`
- `normalizations`
- `raster_xy_by_view [B,N,8,2]`
- `raster_valid_by_view [B,N,8]`
- `support_by_view [B,N,8]`
- exact indexed topology pairs
- full `RiggingSurfaceIR.to_dict()` payload per sample
- raw payload hashes
- boundary-audit hashes
- topology fingerprints
- evidence hashes
- historical 24D tensor only as `derived_features_24d`
- `derived_feature_authority=False`

There is intentionally **no `.features` compatibility alias** on the V3 product batch. Existing Geppetto V2 consumers therefore cannot silently accept the product batch and ignore the lossless evidence. A consumer must be adapted explicitly and must prove its own evidence-consumption behavior before product promotion.

Canonical product entrypoint:

`models/geppetto/product_conditioning.py`

The package selector now records:

- current learned model package: `models.geppetto.v2`
- current product-conditioning package: `models.geppetto.v3`
- lossy V2 product-conditioning authority: revoked

## Regression lock

`tests/models/test_geppetto_lossless_conditioning_v3.py`

The key regression constructs two scene-first surfaces with:

- identical geometry;
- identical support sets;
- identical unordered raster coordinate sets;
- different **view -> raster-coordinate assignments**.

Expected behavior:

- historical V2 24D feature tensors are identical;
- V3 view-indexed raster tensors differ;
- V3 evidence hashes differ.

This test encodes the exact class of lossy aliasing that the new authority rule forbids.

## What remains frozen

This authority change does **not** reopen the IRIS signed-geometry core.

It does **not** alter GSA projection, self-zbuffer visibility, compaction, robust normals or topology generation.

The evidence loss was downstream of GSA: GSA already emitted the view-indexed raster bindings that were later summarized by the learner adapter.

## What is deliberately not claimed

- C1 is not yet promoted as a stable architecture winner.
- Full per-view raster consumption is not yet proven necessary for every learner.
- Geppetto V2 is not yet a valid V3 product-evidence consumer.
- No generalization claim is made.
- No native count/STOP claim is made.
- No Arachne/skinning claim is made.
- No main-branch promotion is authorized by this document.

## Next gate

Before Geppetto can regain product authority, a candidate consumer must take the V3 lossless batch and demonstrate, under a preregistered controlled FIT-OPT protocol, that:

1. the lossless evidence is actually consumed rather than discarded;
2. historical diagnostic replay remains available separately;
3. product evidence cannot silently collapse to the V2 summary-only seam;
4. stability is evaluated with a late-stability criterion that cannot remain latched after later collapse.
