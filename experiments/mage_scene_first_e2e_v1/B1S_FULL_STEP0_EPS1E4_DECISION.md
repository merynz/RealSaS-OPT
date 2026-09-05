# B1s full step-0 AdamW eps=1e-4 decision

Date: 2026-09-05/06
Branch: `e2e/mage-scene-first-v1-20260905`

## Question

Does the frozen B1s structural-serialization training path close the original 41-authored / 31-unique-locus geometry+multiplicity gate from initialization when the only optimizer change is AdamW `eps=1e-4`?

## Authority

A100 run package: `RealSaS_MAGE_GEPPETTO_B1S_FULL_STEP0_EPS1E4_A100_V1_392520b1b89d5072.zip`

SHA-256: `8135d7e0b9ba6ff50160f40f2b2f1eb6dbda7e66d85b623fd6d422e42cfbc0a1`

Dynamic A100 preflight PASSed with historical J4 source-control parity under `eps=1e-8`, exact same initialization for the intervention arm, and `eps=1e-4` active before the first experimental optimizer step.

## Result

FAIL within the preregistered 16384-step horizon.

- first single-check PASS: none
- first stable PASS: none
- best outcome step: 16256
- best nearest p95: `0.0424167774617672`
- best nearest max: `0.04284873604774475`
- best outside capture count: `15`
- final step: 16384
- final nearest p95: `0.040598511695861816`
- final nearest max: `0.044211164116859436`
- final outside capture count: `16`
- final occupancy L1 error: `16`
- assignment changes: `0`
- training Hungarian calls after start: `0`

The corresponding late rescue from the parity-certified step-10752 B1s state with the same `eps=1e-4` intervention had previously PASSed (first stable PASS 12544; final outside 0; final occupancy error 0; final p95 0.007887238636612892).

## Interpretation

`eps=1e-4` is not a valid constant-from-step-0 optimizer policy for this frozen B1s regime. It suppresses or distorts the early/mid optimization dynamics enough that the model never reaches the near-converged state from which the same epsilon intervention successfully rescues late training.

This does **not** falsify structural serialization, decoder capacity, or the late optimizer-stability diagnosis. The evidence supports a phase-dependent optimization requirement: the original `eps=1e-8` regime is materially better for coarse/early learning, while a larger denominator floor is beneficial after the system enters the near-gate regime.

No generalization, count/STOP, mechanical-role, product, or capacity-impossibility claim follows.

## Next gates

1. Keep B2 closed.
2. Do not change recurrent architecture yet.
3. Run the already-planned GSA/RiggingSurface boundary information audit/ablation separately; current substrate is held constant across B1/B1s/B1a and cannot explain the optimizer collapse, but boundary sufficiency and silent evidence loss must be closed before architecture freeze.
4. For optimizer policy, test exactly one preregistered phase-switch rule rather than a constant high epsilon. The switch must be generic and gate-relative, not a Mage step number. Candidate rule to preregister separately: remain at `eps=1e-8` until structural-slot p95 is at or below `1.5 * capture_radius` for three consecutive 64-step checks, then set `eps=1e-4` and keep all other optimizer/model/loss settings unchanged through the frozen horizon. This candidate is not yet promoted or scientifically authorized by this decision record.
