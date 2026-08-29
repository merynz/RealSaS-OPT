# RealSaS — Structured Predicted-Depth Bridge Apparatus Closure

**Date:** 2026-08-29  
**Status:** `M0_M3_APPARATUS_PASS__SURFACE_GRID_AUTHORIZED__NO_PRODUCT_TOLERANCE_CLAIM`  
**Canonical parent before promotion:** `6689393f66f3fd6cefff087a1902e2a7ac3a8061`

## Decision

The minimum executable apparatus for the structured ray-aligned predicted-depth bridge is closed.
The next canonical step is to freeze and execute the surface/substrate scientific grid over structured
`(epsilon, ell, A)` corruptions.

This closure does **not** claim a product-safe IRIS depth envelope.
A later product-level interpretation is consumer-profile dependent and is blocked by the sealed
`CONSUMER_VALIDITY_INTERLOCK_V1.md` until minimal real Geppetto/Arachne consumers are validated on
the RealSaS rigging substrate.

## Scientific firewalls preserved

- Proxy27 was not opened or used for apparatus tuning.
- DEV32 remains closed.
- E0/N-B3 source and result authorities were not modified.
- no learner was trained;
- no per-cell consumer retraining occurred;
- M3 cells are apparatus smoke only and cannot define a tolerance threshold.

## M1 — corruption operator

CPU unit tests: **5/5 PASS**.

Closed properties:
- epsilon=0 is bit-exact P identity;
- deterministic PCG64 replay;
- corruption is aligned to the known camera forward ray;
- per-affected-view RMS equals requested epsilon;
- asymmetry cardinalities are deterministic.

The corruption intervention is:

`P_hat = P_exact + delta_d * F_v`.

## M2 — zero-corruption end-to-end identity

Deterministic already-open E0 calibration family:

`asset_551ea351b43a1787d0f55536`

Replay:

`raw authoritative geometry/raster -> exact observable P -> epsilon=0 -> sealed MUTUAL_P003 -> D2-style carrier -> typed Compiler RiggingSurfaceIR`.

PASS:
- source view IDs exact;
- source row IDs exact;
- P bit-exact;
- support bit-exact;
- matched rows exact;
- source-grid bit-exact;
- support pairs `1909 == 1909`;
- matched-row difference count `0`;
- typed Compiler surface accepted;
- typed surface nodes `512`.

Typed surface lineage hash:

`5392bbeddf80c02f7099862b913607b8a64b96bfa0621a50385302e980f99d13`

This proves the corruption layer has an identity point and does not silently redefine the sealed E0
persistence path at epsilon zero.

## M3 — non-binding structured smoke

The following cells were sealed before their completion result was interpreted:

1. `epsilon=0.001, ell_px=0, ALL8`
2. `epsilon=0.001, ell_px=16, ONE_BAD_HASHED`
3. `epsilon=0.003, ell_px=8, TWO_OPPOSITE_HASHED`

All three traversed:

`corrupted view-local P -> re-derived normals -> MUTUAL_P003 -> D2-style carrier -> typed Compiler surface`.

Observed apparatus facts only:
- ALL8 support pairs: `1921`;
- ONE_BAD_HASHED selected V5; support pairs: `1910`;
- TWO_OPPOSITE_HASHED selected V3/V7; support pairs: `1896`;
- all cells emitted 512 typed surface nodes;
- maximum measured ray-normal leakage was approximately `2.11e-8`;
- requested RMS epsilon was met on every affected view.

These values are **not** a tolerance conclusion and must not be used to choose the scientific grid
based on apparent success/failure.

## Provenance closure

`APPARATUS_STAGE_B_M2_RESULT_V1.json` and `APPARATUS_STAGE_C_M3_RESULT_V1.json` are retained as
machine-readable evidence.

`APPARATUS_MANIFEST_V2.json` binds the apparatus files by byte count and SHA-256.
The M3 result and V2 manifest were normalized after write so repository bytes match their sealed
manifest values exactly.

## Consumer-validity interpretation interlock

The current bridge can establish a versioned **surface/substrate robustness chart**.
Before such a chart becomes the final model-selection acceptance region, validate a minimal real
consumer chain:

`ClosedRiggingVolume V0 -> InteriorRiggingSubstrate V0 -> Geppetto G0 -> Compiler -> Arachne A0 -> Compiler/proof`.

This is an interpretation interlock, not authorization for three parallel research programs.

## Authorized next step

1. keep `calibration_anchor8` as the apparatus/grid-development population;
2. freeze the full scientific corruption grid and decision/reporting operator before opening outcomes;
3. do not use Proxy27 or DEV32 to tune the grid;
4. execute the frozen surface/substrate bridge;
5. only after the resulting surface chart and consumer-validity interlock are both available may a
   product-level depth acceptance region be frozen for foundation/IRIS model selection.
