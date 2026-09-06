# RealSaS — AR-01 Skeleton-Causal Autoregression Notebook Delivery Audit

**Date:** 2026-09-07  
**Status:** `DELIVERY_PREFLIGHT_PASS__SCIENTIFIC_RUN_NOT_EXECUTED_HERE`

## Artifacts

- Prereg: `AR01_SKELETON_CAUSAL_AUTOREGRESSION_PREREG_V1.md`
  - SHA-256: `78981b0215c88462d7a2be607274f638494dc1517f88ed882084a4e83d11fae1`
- Research source: `models/geppetto/challengers/ar01_skeleton_causal_v1.py`
  - SHA-256: `6f52e168883c7385c62fb7705bf526c740ad980a5fb0eed79356cdfde65d540d`
- Unit test: `tests/models/test_geppetto_ar01_skeleton_causal_v1.py`
  - SHA-256: `854240d0c2fbedfad7897615075317251fac396ec7e966a5e93d65ffcf220205`
- Run-All notebook: `RealSaS_MAGE_GEPPETTO_AR01_SKELETON_CAUSAL_RUN_ALL_NO_TOKEN_FINAL.ipynb`
  - SHA-256: `aafc50eecfb6c942f03f775cbae3dca57021a59a6278b2d8e17f91b9bfe72e92`
  - size: `213516` bytes
  - cells: `13 total / 12 code`

## Authority / quarantine

- main audit-hold commit: `b7025a00bcc4b8b209b74704555cbc56dcdd5a0d`
- experiment branch base: `d840d96ece57705e0f519d1650119db180d9f5c4`
- canonical Geppetto source is not mutated by the notebook.
- no promotion authority is encoded in any output.
- `MECHANICAL_SALIENCE_FUNCTIONAL_SIMPLIFICATION` is recorded as separate/out-of-scope.
- existing FIT1 mechanism authority is left in its own prereg/result ledgers and is not re-adjudicated here.

## Causal isolation

AR0 and AR1 are one class / one parameterization.

Only runtime gate:

`mechanical_feedback_enabled`

differs.

Shared in both arms:

- V3 lossless fixed-eight-view `(x,y,valid,support)` evidence path;
- per-step full-surface attention;
- direct three-mode locus generation;
- causal parent scores restricted to prior controls;
- exact 41-control structural serialization;
- geometry + parent + root loss;
- AdamW/epsilon-switch schedule;
- seed/backend;
- final 16,384-step horizon.

AR1 alone allows joint+parent mechanical geometry to alter the next recurrent/history state.

The feedback residual's final projection is zero-initialized, so the treatment does not introduce a random-init output difference.

## Stability discipline

The old one-time 3-check latch is absent.

Both arms always run to step 16,384.

Promotion-level structural stability requires:

- final geometry exact;
- parent accuracy = 1.0;
- root exact;
- final contiguous structural-PASS streak >= 48 checks;
- check interval = 64;
- terminal directly observed horizon = 3,072 optimizer steps.

This follows the repaired V3 notebook standard derived from the independent historical B1s 53-check terminal streak.

## Static validation performed

- `nbformat.validate`: PASS
- every code cell Python compilation: PASS
- source Python compilation: PASS
- unit-test Python compilation: PASS
- lowercase JSON literals `true|false|null` in Python cells: 0
- notebook outputs cleared: PASS
- notebook execution counts reset: PASS

## CPU mechanism smoke

The full scientific run remains CUDA/A100-parity-only.

A local CPU smoke used:

- the exact embedded historical Geppetto runtime source from the notebook;
- the exact AR-01 notebook mechanism class;
- reduced synthetic tensor cardinality/model dimension only for control-flow verification.

Measured smoke:

- AR0 vs AR1 at zero-init: positions bit-identical = PASS
- parent-logit tensor shape/causal path: PASS
- gate-OFF teacher-mechanics perturbation max output delta = `0.0`
- gate-ON perturbation:
  - steps before causal availability delta = `0.0`
  - later-step position delta = `0.0013568401336669922` (>0)
- zero-init feedback projection gradient:
  - AR0 = absent/zero
  - AR1 = non-zero (`288` non-zero elements in reduced smoke)
- full-surface attention width matched the synthetic surface-token count.

These are implementation/preflight checks only; they are not AR-01 scientific evidence.

## Frozen outcome operator

- AR1 PASS / AR0 FAIL -> `AR01_PASS_SKELETON_CAUSAL_RESCUE`
- AR1 PASS / AR0 PASS -> `AR01_COMPATIBLE_NOT_NECESSARY_ON_FIT1`
- AR1 FAIL / AR0 PASS -> `AR01_FAIL_FEEDBACK_HARMS_FIT1`
- AR1 FAIL / AR0 FAIL -> `AR01_NO_TERMINAL_CLOSURE`

No secondary metric can override this operator.

## Next action

Run the notebook exactly as delivered. Do not alter the gate, loss, optimizer, causal preflight,
teacher-forcing semantics, free-running evaluation, or terminal-stability threshold after observing outcomes.
