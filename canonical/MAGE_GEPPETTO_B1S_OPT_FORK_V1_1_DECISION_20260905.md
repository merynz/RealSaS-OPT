# Mage Geppetto B1s Optimizer Fork V1.1 — Corrected Decision

Date: 2026-09-05
Status: DIAGNOSTIC_ONLY
Scientific source SHA: `8b3eb659abbec3a128a314abc59ad07fdc628329`
Repository design head at experiment construction: `1fdb45da713fa189273c8686d63991203db7b3d0`

## Authority

Drive root: `RealSaS_MAGE_GEPPETTO_B1S_OPT_FORK_A100_V1_1_BRANCH_ISOLATION_FIX_NO_TOKEN`
Contract: `contract_2549648d07b748c6`
Result ZIP: `RealSaS_MAGE_GEPPETTO_B1S_OPT_FORK_A100_V1_1_2549648d07b748c6.zip`

V1 intervention branches are scientifically invalidated because optimizer moment tensors were shared through an in-memory state dictionary after the baseline branch mutated them. V1 baseline parity remains valid. V1.1 reloads the parity-certified step-10752 checkpoint from disk independently for every branch and requires identical model, optimizer-moment and RNG authority hashes before intervention.

V1.1 also requires baseline and grad-clip branches to be identical at step 10753 while clipping is inactive. This no-op parity passed.

## Corrected fork outcome

At step 10816:

| branch | outside | occupancy L1 | nearest p95 | global grad L2 | global parameter delta L2 | output delta L2 | Adam ratio L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 41 | 41 | 0.2717448 | 23.9264 | 0.891113 | 3.88491 | 2970.38 |
| grad clip 1.0 | 41 | 41 | 0.2572599 | 9.01589 pre / 1.0 post | 0.405550 | 1.59642 | 1351.83 |
| LR / 10 | 3 | 3 | 0.0206668 | 0.002042 | 0.00015195 | 0.00017743 | 5.06465 |
| AdamW eps=1e-4 | 3 | 3 | 0.0205462 | 0.002014 | 0.00029600 | 0.00075158 | 0.987659 |

Baseline reproduces the historical catastrophe. Gradient clipping first activates at step 10811 but does not prevent the catastrophe. Both global LR reduction and a larger Adam epsilon prevent the collapse over the 512-step fork window.

No branch reaches the original B1s multiplicity gate by step 11264. The best retained outside count remains 3. The eps=1e-4 branch has the best nearest-p95 among stabilizing branches (`0.01855236` at step 11264) and is still improving at the end of the window. LR/10 ends at `0.01998197`.

## Interpretation

The corrected evidence falsifies the simple claim that gradient clipping alone is sufficient. The collapse contains a true gradient explosion, but clipping the gradient norm to 1.0 still leaves an excessive effective step and does not prevent global AR-trajectory destruction.

The fact that `eps=1e-4` and LR/10 both prevent collapse supports an optimizer effective-step / small-denominator mechanism as an important contributor. The current evidence does not prove that this is the only mechanism, and it does not yet establish a production optimizer setting.

B1s' remaining three errors are still slots 34, 35 and 36; only slot 36 overlaps one unresolved automorphic class. Therefore no serializer change is authorized by this fork.

## Frozen next gate

Select exactly one confirmatory stabilization before any recurrent-architecture intervention. Choose `AdamW eps=1e-4` because:

1. it directly targets the small-denominator mechanism;
2. unlike clipping, it prevents the collapse;
3. unlike LR/10, it does not globally reduce the intended learning-rate scale;
4. among the two stabilizing branches it has the lower end-of-window p95 and a continuing downward trajectory.

Confirmatory experiment:

- restart from the same parity-certified step-10752 checkpoint;
- change **only** AdamW `eps: 1e-8 -> 1e-4`;
- preserve model, loss, structural serialization, learning rate `3e-4`, weight decay `1e-4`, forced 41 decode steps, fixture, runtime, seed and gate;
- continue to step 16384;
- original B1s PASS authority remains: all 41 captured within radius `0.016685275360941887`, exact 31-bin occupancy/multiplicity, three consecutive check points;
- no early capacity/impossibility claim;
- if PASS, this is a rescue sufficiency result only. A full step-0 B1s rerun with eps=1e-4 is still required before optimizer promotion.

GSA/RiggingSurface boundary ablations remain a separate causal branch and must not be mixed into this confirmatory optimizer run.

No B2, no previous-locus feedback, no serializer change, and no optimizer promotion before this confirmatory gate closes.
