# P-V5 R256 One-Cell Optimizer Localization V1 — Result Interpretation
Date: 2026-08-25
Status: CLOSED / PASS

## Parent reproduction
The exact selected one-cell checkpoint was reproduced before any restart optimization:
- checkpoint SHA-256: `cad4421ae728848bbf9181c87e0d4912c38ec6f94d61c86cd114ad0a251c79ee`;
- parent selected P p95: `0.005682396539486942`;
- reproduced P p95: `0.005682396539486942`;
- absolute difference: `0.0`;
- scientific optimizer steps at reproduction: `0`.

The parent tail was not uniform. Highest zero-step per-view P p95:
- V6: `0.008282015100121498`;
- V2: `0.007690753322094679`;
- V3: `0.006184491445310414`;
- V1: `0.005966349854134023`.

This confirms the prior gate miss was a localized residual tail rather than a global P failure.

## Frozen restart arms
All arms started from the exact same parent model state with fresh AdamW moments.

| arm | lr | first PASS step | minimum/final P p95 | result |
|---|---:|---:|---:|---|
| A control | `3e-4` | — | `0.005420877947472036` | FAIL |
| B | `1e-4` | `256` | `0.00420133795123547` | PASS |
| C | `3e-5` | `64` | `0.003946938854642211` | PASS |

Threshold: `0.005`.

Best arm: `C_LR3E5`.

Canonical status:
`P_V5_R256_ONE_CELL_RECOVERY_PASS`

Canonical localization label:
`LATE_STAGE_LR_FLOOR_SUPPORTED`

## Interpretation
The original frozen 2048-step one-cell gate remains an immutable FAIL under its preregistered fixed-lr protocol. The localization experiment does not rewrite that result.

However, the controlled restart evidence establishes that:
1. the exact same R256 model state can cross the canonical P threshold without changing representation, P ontology, V5 analytic reconstruction, data membership or objective;
2. the original `3e-4` late-stage rate is too high to close the residual tail within the tested restart budget;
3. lower late-stage rates recover the one-cell gate, with `3e-5` giving the earliest and best preregistered recovery.

Therefore one-cell **learner/optimizer sufficiency is closed**, and the prior miss is localized to late-stage optimizer/LR behavior rather than a need to reopen P representation or model capacity.

## Repaired optimizer evidence frozen for promotion
For the next rung, no new LR search is authorized.

The repaired schedule is:
- fresh model initialization;
- MAIN AdamW `3e-4` for 2048 steps;
- fresh AdamW restart on current weights;
- TAIL AdamW `3e-5` for 512 steps.

The next rung must test one asset × two styles jointly and must require both style cells to pass individually.

## Promotion
Authorized next gate only:
`1 asset × 2 styles R256 joint overfit`.

Not yet authorized:
- 8 assets × 2 styles;
- unseen-family generalization.
