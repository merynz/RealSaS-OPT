# RealSaS — Structured Predicted-Depth Bridge — Pre-Open Prereg V1

**Date:** 2026-08-29  
**Status:** `APPARATUS_PREOPEN__NO_SCIENTIFIC_OUTCOMES_OPENED`  
**Canonical parent:** `CURRENT_STATE.md` at main `6689393f66f3fd6cefff087a1902e2a7ac3a8061`

## Question

Which structured forward-depth residual regimes can IRIS tolerate before teacher-free persistence,
frozen proxy consumers, Compiler qualification, or functional proof becomes unsafe?

## Scope now

This document freezes the *apparatus semantics* before any bridge outcome is opened.
It intentionally does **not** yet freeze the final scientific corruption grid.
The final grid will be sealed only after zero-corruption identity and one-family end-to-end apparatus
preflight pass, while Proxy27 and DEV32 remain closed.

## Legal corruption

For every visible row of view `v`:

`P_hat = P_exact + delta_d * F_v`

where `F_v` is the known unit orthographic forward direction.

Forbidden:
- lateral/image-plane displacement as a primary bridge intervention;
- teacher triangle/barycentric identity in forward persistence;
- camera perturbation in this V1 bridge;
- per-cell consumer retraining;
- outcome-dependent corruption-field normalization.

## Frozen corruption semantics

`epsilon`
: RMS value of `delta_d` on each affected view after sampling the field on that view's visible rows.

`ell_px`
: Gaussian spatial smoothing sigma in native-raster pixel units. `ell_px=0` is independent visible-row Gaussian noise. For `ell_px>0`, a dense PCG64 Gaussian raster is Fourier-filtered with
`exp(-0.5*(2*pi*ell)^2*|f|^2)`, sampled on visible rows, then standardized to zero mean / unit RMS.

`A`
: deterministic affected-view topology. Apparatus-supported patterns are:
- `ALL8`
- `ONE_BAD_HASHED`
- `TWO_OPPOSITE_HASHED`
- `FOUR_ALTERNATING_HASHED`

The affected view orientation is SHA256-derived from `asset_id` so no hand-picked camera direction is privileged.

## Persistence intervention point

1. Load exact observable view-local `P` and raster provenance.
2. Freeze the 512 source rows with the existing clean P-only deterministic anchor sampler.
3. Apply ray-aligned corruption to every view-local visible P row.
4. Re-derive view-local normals from corrupted P with the sealed E0 deterministic normal operator.
5. Move each source anchor to its corrupted source-view P.
6. Re-run the sealed teacher-free `derived_match_row` / MUTUAL_P003 persistence on corrupted views.
7. Construct the same D2-style 36D carrier semantics:
   - 0:3 P
   - 3:11 support
   - 11:27 raster XY
   - 27 support fraction
   - 28:36 camera-forward depths
8. Route the resulting admitted surface through the current typed Compiler boundary.
9. Evaluate frozen consumer/proof bytes without per-cell retraining.

Freezing source *rows* but not source P isolates predicted-depth/persistence robustness from a second
anchor-sampler intervention.

## Firewalls

- Proxy27: CLOSED for apparatus development and tuning.
- DEV32: CLOSED.
- N-B3 qualification results may be read only as historical context; their isotropic epsilon ladder is not a product depth envelope.
- E0/N-B3 source files and results are immutable.
- no downstream checkpoint is retrained per corruption cell.
- no single scalar P95 may replace structured family-tail reporting.
- any result file is forbidden until apparatus identity preflight is sealed.

## Required pre-open gates

A. CPU unit tests:
- epsilon=0 bit-exact P identity;
- deterministic PCG64 replay;
- exact ray alignment;
- epsilon RMS calibration;
- asymmetry cardinality.

B. One real open/FIT asset:
- epsilon=0 carrier parity against the sealed clean D2 semantics;
- same source row IDs;
- same support/matched rows;
- same X36 within frozen numeric tolerance;
- typed Compiler adapter accepts the clean replay.

C. One tiny open/FIT apparatus smoke:
- a few non-binding structured cells;
- only runtime/failure-shape inspection;
- no scientific threshold or model conclusion.

Only after A/B/C apparatus gates close may the full `(epsilon, ell, A)` scientific grid be sealed.

## Current unresolved execution dependency

The GitHub-contained E0 SurfaceBuilder contains the teacher-free matching implementation needed for
the replay. Historical `e0_downstream_proxy_v1` helper source used by old compact-pack runners is not
currently GitHub-contained. The bridge must not reconstruct that helper by guesswork.

The bridge can build P/support/raster/X36 directly from the sealed SurfaceBuilder functions.
For frozen proxy evaluation, exact model/checkpoint adapter parity must be established before outcome open.
