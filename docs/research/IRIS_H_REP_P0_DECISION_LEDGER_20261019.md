# IRIS H-REP-P0 Decision Ledger — 2026-10-19

Status: **ACTIVE REPRESENTATION-INVESTIGATION AUTHORITY**

This ledger starts at the exact scientific boundary created by completed `SURF-A1-FROZEN-H`. It supplements, rather than rewrites, `IRIS_TESSA_V2_DECISION_LEDGER_20261019.md`.

## D1 — SURF-A1 is a real scientific FAIL, not an apparatus failure

All preregistered runtime admission gates passed before optimizer step 1: exact A100 lane, exact V8 checkpoint/source binding, native R24 face identity on all eight subjects, frozen scene-core hash, and 64/64 hard raster parity with precision/recall `1.0`.

The shared decoder then completed all `2400` optimizer steps and the independent R48 court returned `SCIENTIFIC_FAIL_SURF_A1_FROZEN_H_REPRESENTATION_RESCUE`.

Therefore do not attribute the result to PyTorch3D import/build, Drive transport, wrong checkpoint, partial training or a reduced-resolution fallback.

## D2 — preserve the magnitude of the A1 rescue; do not convert FAIL into “H is empty”

Relative to the native V8 R24 control, A1 improved:

- mean subject mask IoU `0.56057 → 0.77953`;
- minimum subject mask IoU `0.40535 → 0.67901`;
- minimum view mask IoU `0.34254 → 0.65213`;
- mean symmetric Chamfer `0.06120 → 0.04305`;
- maximum mask gap `392.42 → 258.66 px`.

Nevertheless A1 missed the locked global IoU, Chamfer, mask-gap, normal-alignment and per-subject p95 gates. The correct interpretation is:

> the tested sealed-H→shared-extraction path is insufficient across Fit8, while the amount of rescue leaves local representation sufficiency unresolved.

Rejected: lowering A1 thresholds after results, promoting it to SURF-A2, or treating improvement as a PASS.

## D3 — do not retrain the carrier before separating representation from extraction

A1 still passed meaningful local/global subcourts and some subjects decode much better than others. Reopening V8 H immediately would confound representation capacity with extraction-transition capacity.

Decision: run `H-REP-P0` first.

Carrier retraining remains blocked until a separate future preregistration earns it.

## D4 — H-REP-P0 is extractor-free

H-REP-P0 queries exact sealed V8 scene planes at point locations and evaluates local evidence directly. It does not invoke:

- SURF-A1 decoder;
- V8 cube-extraction head as an output target;
- FlexiCubes;
- marching tetrahedra;
- TESSA;
- Compiler repair;
- any mesh materialization.

This makes the experiment answer a narrower question: whether local geometry can be decoded before the mesh-extraction bottleneck.

## D5 — use four exact equal-capacity probe arms

All arms use the same `387-D` input envelope and exactly the same predictor parameter count.

```text
H_XYZ
    384-D sealed local H + XYZ

RAW_FIELD_XYZ
    raw field in evidence channel 0
    remaining evidence channels = 0
    + XYZ

XYZ_ONLY
    all 384 evidence channels = 0
    + XYZ

SHUFFLED_H_XYZ
    fixed within-subject permutation of H
    + original XYZ
```

Why both negative controls:

- `XYZ_ONLY` tests coordinate/template memorization;
- `SHUFFLED_H_XYZ` preserves subject-level H statistics while destroying local H↔XYZ alignment.

A local-H claim requires beating both controls, not merely fitting held-out points.

Rejected: different-width probe networks, subject-ID conditioning, per-subject adapters or semantic-class routing.

## D6 — probe truth is an explicit point-cloud proxy, not exact SDF truth

H-REP-P0 reuses sealed V4 geometry already accepted by independent surface courts:

- `teacher_surface_points_iris`;
- `mechanical_points`;
- `mechanical_normals`.

Nearest-point distance/offset to the sealed point sets is called a **point-cloud geometry proxy**. It is not silently renamed SDF, occupancy or exact surface truth.

Normals are validity-gated nearest mechanical normals and evaluated up to sign.

Teacher truth is target-only and never enters the probe input.

## D7 — train/holdout queries are independently deterministic

For each Fit8 subject, the probe dataset contains separately seeded train and held-out point sets. Near-surface queries use deterministic normal offsets plus independently sampled uniform queries in the normalized IRIS domain.

The dataset receipt is hash-bound to the current H-REP input binding, exact V4 manifest and exact prepared shards.

## D8 — H-REP-P0 scientific gates are preregistered before runtime

`H_XYZ` must pass both absolute held-out geometry quality and local-information controls versus `XYZ_ONLY` and `SHUFFLED_H_XYZ`.

`RAW_FIELD_XYZ` has its own absolute quality and XYZ-control gates.

The carrier-local-geometry result is positive if either H or raw-field arm earns its entire preregistered gate set.

Thresholds may not change after results.

## D9 — PASS and FAIL have deliberately narrow consequences

### If H or raw field passes

Interpretation: useful local geometry is decodable without mesh extraction.

Next action:

`DESIGN_EXTRACTION_TRANSITION_ALTERNATIVE__DO_NOT_RETRAIN_H_YET__DO_NOT_TRAIN_TESSA`

This does not itself authorize SURF-A2.

### If neither passes

Interpretation: local geometry decodability was not demonstrated under the locked point-probe court.

Next action:

`PREREGISTER_CARRIER_REPRESENTATION_RETRAINING_EXPERIMENT__DO_NOT_TRAIN_TESSA`

The H-REP FAIL is evidence for opening that study; it is not permission to start carrier optimization immediately.

## D10 — SURF-A2 and TESSA remain blocked

No H-REP outcome can bypass the direct-surface qualification chain.

```text
H-REP-P0
   ↓
representation/extraction decision
   ↓
new direct-surface experiment + prereg
   ↓
qualified SURF-A2-HIRES
   ↓
only then TESSA-V2-B1 may start
```

Historical raw field, native R24, SURF-A1 and H-REP point probes are not eligible TESSA training-surface authority.

## D11 — H-REP notebook must inherit the successful V8 operator contract

Notebook UX is part of the research apparatus:

- `Run all`;
- one native `drive.mount` consent;
- physically separated phase cells;
- exact source SHA;
- direct exact Drive paths for known inputs;
- no `os.walk`, `rglob` or full-MyDrive discovery;
- no second OAuth;
- resumable hash-bound checkpoints;
- independent final court.

Rejected: another giant worker cell or a notebook that rediscovers known Drive artifacts at runtime.
