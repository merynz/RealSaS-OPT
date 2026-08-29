# RealSaS — Structured Depth Bridge — Minimum First Result Path V1

**Status:** `EXECUTION_ORDER_FREEZE__KEEP_APPARATUS_SMALL`

The current goal is not to build the entire future IRIS program. It is to obtain the first valid,
end-to-end structured-depth bridge result without repeating the historical “designed but never run”
failure mode.

## Gate M0 — source / byte audit

- main HEAD and branch parent recorded;
- E0 geometry and SurfaceBuilder source hashes verified;
- frozen D2 checkpoint hashes and compiler entry surface verified;
- no Proxy27/DEV32 access.

## Gate M1 — CPU corruption unit tests

Run `tests/test_depth_corruption_v1.py`.

No corpus, CUDA, model checkpoint, or scientific outcome is required.

PASS closes:
- deterministic corruption;
- ray alignment;
- epsilon calibration;
- asymmetry pattern semantics.

## Gate M2 — one real FIT/open asset, epsilon=0

Use one already-open E0 FIT asset.

Replay:
`exact observable P -> zero corruption -> sealed MUTUAL_P003 -> D2-style carrier -> typed compiler surface`.

Require identity/parity with the clean sealed path before any nonzero cell is evaluated.

No model training.

## Gate M3 — tiny non-binding structured smoke

One small FIT/open family subset only.

Run a deliberately tiny fixed smoke set sufficient to prove the apparatus traverses:
- independent vs coherent residual;
- all-view vs asymmetric residual;
- persistence;
- typed Compiler routing.

This is apparatus validation, not the product tolerance result.

## Gate M3.5 — consumer-validity interpretation interlock

M2/M3 apparatus PASS is sufficient to open a **surface/substrate robustness** bridge.
It is not sufficient to call the resulting envelope the final product-safe IRIS tolerance.
Before using a bridge envelope to select the final foundation/IRIS model, close
`CONSUMER_VALIDITY_INTERLOCK_V1.md` with minimal real Geppetto/Arachne consumers.

This gate does not block sealing/running the current surface-level corruption grid; it blocks only
the stronger product-level interpretation.

## Gate M4 — freeze full scientific grid

Only after M0-M3 PASS:
- seal the actual `(epsilon, ell, A)` cell list;
- bind consumer/checkpoint/code hashes;
- bind output schema and hard-family reporting;
- then open the scientific FIT qualification population.

## Deferred until after first end-to-end result

Do not open:
- C1 robust consumer training;
- DINOv2 capacity ladder;
- MapAnything/DA3;
- OOD appearance probes;
- external-camera Mode E;
- robust uncertain-camera hull;
- full prospective dual-replay program.

Those are sealed prospective work, not prerequisites for the current bridge.

## User handoff rule

Do not ask the user to run anything that is available through repository/connected-source tooling.
Request user execution only when the next gate genuinely requires unavailable local hardware,
CUDA/GPU runtime, or a non-materializable heavy authority.
