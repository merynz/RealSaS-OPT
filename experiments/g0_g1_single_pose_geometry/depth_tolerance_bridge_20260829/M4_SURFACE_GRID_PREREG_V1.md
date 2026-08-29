# RealSaS — M4 Structured Surface/Substrate Depth Grid Prereg V1

**Date:** 2026-08-29  
**Status:** `FROZEN_BEFORE_M4_SCIENTIFIC_OUTCOMES__PROXY27_DEV32_CLOSED`  
**Machine-readable authority:** `M4_SURFACE_GRID_PREREG_V1.json`

## Question

Which structured forward-depth residual regimes remain safe for observable persistence and the current
frozen surface/substrate consumer path?

This gate is **not** a product-safe IRIS tolerance claim. The later real-consumer validity interlock
remains mandatory.

## Population

Surface + typed Compiler route:
- exact frozen `calibration_anchor8`, all eight FIT families;
- no family may be added/removed using bridge outcomes.

Frozen D2 proxy evaluation:
- only the four preregistered calibration families with legal Geppetto+Arachne truth;
- same frozen D2 information-isolation proxy checkpoints;
- no retraining per cell.

`qualification_proxy32` is not used to choose this grid. `DEV32` remains closed.

## Full structured grid

There is one canonical zero cell plus the full Cartesian product for every nonzero magnitude:

- `epsilon`: `0`, `0.0015`, `0.003`, `0.006`, `0.012`;
- `ell_px`: `0`, `4`, `16`, `64`;
- `A`: `ALL8`, `ONE_BAD_HASHED`, `TWO_OPPOSITE_HASHED`, `FOUR_ALTERNATING_HASHED`.

Because `ell` and `A` are irrelevant at epsilon zero, duplicate zero cells are forbidden. Total:
**65 cells**.

The grid is chosen from frozen mechanism scales, not observed M3 smoke behavior:
- `0.003` = historical reciprocal-cycle threshold;
- `0.006` = frozen BASE common-frame P gate;
- nonzero epsilon ladder = `0.5x, 1x, 2x, 4x` the reciprocal threshold;
- `ell=4` = frozen raster candidate search-radius scale;
- `ell=16` and `64` stress increasingly coherent regional drift.

M3 smoke values are explicitly non-binding and were not used to choose a pass/fail boundary.

## Canonical route

```text
P_exact
 -> ray-aligned corruption P_hat = P_exact + delta_d F_v
 -> re-derived view-local normals
 -> BASE_DERIVED_MATCHER
 -> reciprocal reverse match
 -> cycle_P <= 0.003
 -> D2_MUTUAL_P003
 -> typed RiggingSurfaceIR
 -> frozen D2 proxy consumers where legally applicable
```

BASE and D2 are reported separately. BASE support is diagnostic; scientific qualification uses the
post-cycle D2 route.

## Decision operator

Canonical baseline: `E0000_L000_ALL8`.

### Surface-route veto

A cell is `SURFACE_ROUTE_FAIL` if any preregistered calibration family has:
- non-finite corruption/output values;
- corruption invariant failure;
- typed `RiggingSurfaceIR` construction failure;
- schema mismatch; or
- typed surface node count other than 512.

BASE/D2 support retention is always recorded but no new support-count threshold is invented after
seeing outcomes.

### Frozen proxy margins

On the four fixed truth-capable calibration families, corrupted D2 is compared against zero-corruption
D2 with the already-frozen E0 practical numerical margins:

- Arachne CE ratio <= `1.05`;
- Arachne influence-displacement ratio <= `1.05`;
- Geppetto joint-mean ratio <= `1.05`;
- Geppetto family-P95 ratio <= `1.10`;
- Geppetto PCK@0.05 delta >= `-0.10`;
- Geppetto PCK@0.08 delta >= `-0.10`.

All six checks must pass for `SURFACE_ROUTE_PASS__PROXY_PASS`.
Every eligible family is also reported separately so aggregate behavior cannot hide hard tails.

These margins are frozen here **before** corrupted-grid outcomes. Their original numerical authority is
`E0_DOWNSTREAM_PROXY_MARGIN_DECISION_V1.json`; this prereg explicitly fixes the new comparison as
`corrupted D2 / zero-corruption D2`.

## Reporting

Every asset/cell record must preserve:
- epsilon, ell, asymmetry and affected views;
- BASE and D2 support counts/fractions;
- corruption diagnostics;
- typed surface status/node count/lineage;
- Geppetto/Arachne metrics when legally available.

Every cell must preserve:
- all-family surface-route result;
- BASE and D2 aggregate diagnostics;
- frozen proxy aggregate + six checks;
- per-family tails;
- one of the preregistered cell labels.

No single scalar epsilon may replace the `(epsilon, ell, A)` chart.

## Pre-open execution seal still required

This prereg freezes the scientific design and decision operator only. Before the first M4 corrupted-cell
outcome is admissible, a separate execution seal must bind:
- raw authoritative geometry/raster SHA-256 for all eight calibration families;
- exact grid-runner bytes;
- exact additive grid-runtime bytes;
- typed Compiler surface-adapter bytes;
- D2 proxy checkpoint bytes;
- this prereg JSON SHA-256.

No outcome may be opened between this prereg and that execution seal.

## Firewalls

- scientific M4 outcomes opened at prereg: `false`;
- Proxy27: `CLOSED`;
- DEV32: `CLOSED`;
- training steps: `0`;
- grid edits after first outcome: `FORBIDDEN`;
- decision-operator edits after first outcome: `FORBIDDEN`;
- source/checkpoint substitutions after first outcome require a new versioned prereg;
- product-safe interpretation remains blocked by `CONSUMER_VALIDITY_INTERLOCK_V1.md`.
