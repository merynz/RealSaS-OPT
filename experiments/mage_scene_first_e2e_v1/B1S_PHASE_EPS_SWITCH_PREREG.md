# B1s gate-relative AdamW EPS phase-switch preregistration

Date: 2026-09-06
Branch: `e2e/mage-scene-first-v1-20260905`

## Motivation

Two already-observed results must be reconciled without changing Geppetto architecture or GSA conditioning:

1. Original B1s, constant AdamW `eps=1e-8`: stable structural serialization materially improves learning and approaches the gate, but late training catastrophically collapses.
2. Rescue from the parity-certified B1s step-10752 state with `eps=1e-4`: PASS (first stable PASS 12544, final outside 0, final occupancy error 0, final p95 0.007887238636612892).
3. Constant-from-step-0 `eps=1e-4`: FAIL by 16384 (best outside 15, best p95 0.0424167774617672; final outside 16, final p95 0.040598511695861816).

Therefore the next optimizer hypothesis is phase-dependent: historical epsilon for coarse learning, larger denominator floor only after the frozen structural-slot geometry is near the gate.

## Frozen policy

- initial AdamW epsilon: `1e-8`
- stabilization epsilon: `1e-4`
- LR: `3e-4` unchanged
- weight decay: `1e-4` unchanged
- check cadence: every 64 optimizer steps
- switch statistic: structural-slot p95
- switch threshold: `structural_slot_p95 <= 1.5 * capture_radius`
- switch requires 3 consecutive qualifying checks
- once switched, epsilon remains `1e-4` permanently
- no LR change, clipping, recurrent-state feedback, diffusion, loss change, serializer change, GSA/raster/topology change, or count/STOP change is allowed in this run
- primary budget 8192; continuation to 16384 is frozen
- original B1s geometry/multiplicity gate remains unchanged; stable PASS still requires 3 consecutive PASS checks

The `1.5 * capture_radius / 3 checks` rule is selected now, after the constant-epsilon full-step-0 result and before this experiment. It must not be tuned after seeing the result.

## Mage replay parity assertion

The policy itself contains no Mage step number. However, because the pre-switch trajectory is exactly the original deterministic B1s `eps=1e-8` trajectory, the Mage witness is expected to trigger the frozen policy at step 10240. The notebook aborts if the switch occurs at a different step. This is a replay-integrity assertion only, not policy authority.

## Claims allowed

PASS: the frozen B1s structural serialization plus the preregistered phase-dependent AdamW epsilon policy is sufficient for the Mage 41-on-31 geometry/multiplicity gate within the frozen horizon.

FAIL: the policy is insufficient within the frozen horizon. No decoder/capacity impossibility claim follows.

No result from this experiment by itself authorizes generalization, native count/STOP, mechanical-role, GSA-boundary, product, or FIT8 promotion.

## Next gate after this experiment

Regardless of outcome, B2 remains closed until the separate GSA/RiggingSurface boundary information audit/ablation and topology-preserving compactor challenger are resolved. Recurrent architecture changes remain downstream of those deterministic-boundary checks.
