# R6.2 Execution Isolation Note — 2026-08-20

## Scope

R6.2 changes **only the process boundary** used to execute R6.1 observable Phase-1. The scientific source bundle, frozen model/checkpoint, candidate construction, geometry, mechanics, population, witness definition, thresholds, evaluator policy, and decision tree are unchanged.

## Observed execution pathology

The frozen R6.1 `observable_phase.run_family()` performs:

1. `runner.run(...)`, which internally executes the frozen V8 baseline in a subprocess and then evaluates the hybrid route in the parent process;
2. a second model/context construction in that same parent process;
3. full candidate-pool reconstruction.

On the current execution container, the sequence `runner.run -> second model context -> _candidate_pools` repeatedly stalls for minutes even though its components are individually fast. Diagnostics without evaluator truth showed:

- direct frozen V8 baseline for family 11032/e01: prediction SHA `87acfd1b15e87b6c04cde9ada77142d62136fbf1b7352ead9fc2332a732360cd`, finishing normally;
- fresh-process full support candidate reconstruction for 11032/e01: 101 support carriers in ~19.2 s, output candidate total 13,728;
- same-process sequence reproduced the stall on both 11032 and 09908.

This is treated as an execution-process-state pathology, not a scientific failure and not evidence about the RealSaS hypothesis.

## R6.2 harness

`R6_2_PROCESS_ISOLATED_OBSERVABLE_HARNESS.py`

- SHA-256: `d1d607c4d8b95da59950dbae6210f8a019bac27de49256848c91be562c7aa3f6`
- GitHub commit: `9acb6ae245f8d2bb8be553db3e5764a38d1cbd4f`
- GitHub blob: `ad03279ea266bcda7b9aa2b43bb351646ec18454`

The harness has two explicit modes that MUST be invoked as separate processes:

1. `baseline`: calls the unchanged frozen R6.1 `runner.run`, verifies `truth_access=NONE`, checkpoint SHA, prediction SHA, and all 16 raster hashes, then freezes the baseline cache.
2. `state`: starts in a fresh process, verifies the cached baseline bytes and raster hashes, substitutes only the already-verified baseline-return operation, then calls the unchanged frozen R6.1 `observable_phase.run_family()` for all remaining computation.

No evaluator sidecar is opened or parsed by the harness.

## Bit-exact parity gate

On family 09908/e01, which already had an R6.1 observable state created before any evaluator truth-open, R6.2 reproduced every state array bit-exact:

`P_A, P_B, N_A, N_B, V_A, V_B, XY_A, XY_B, H_xyz, H_reproj_px, H_desc_score, H_offsets`.

Both compressed state files have the same SHA-256:

`482181d79ab45c4f12d246a1eaa0a7960d59e2b5c73aeb7e83c86d8b067b7ccd`

Parity record SHA-256:

`b65bacd0a0ae05053894614812504ba943168b6062c433aa855c76b892d684da`

## Authority rule

R6.2 is permitted only if the harness SHA and all inherited R6.1 source/input hashes match the frozen preflight. Any mismatch invalidates execution. R6.1 scientific contract remains authoritative; R6.2 is an execution-isolation revision only.
