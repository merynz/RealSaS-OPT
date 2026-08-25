# P-V5 R256 One Asset × Two Styles — Canonical Result

**Date:** 2026-08-25  
**Status:** `P_V5_R256_TWO_STYLE_OVERFIT_PASS`

## Frozen question

Can one shared fresh R256 model jointly fit both frozen render styles of the same FIT asset to the canonical per-cell P threshold, without using a one-style checkpoint as initialization?

Frozen cells:
- `asset_76313e4bd82b82fcd1659c70 / cel_clean / FIT`
- `asset_76313e4bd82b82fcd1659c70 / ink_cel / FIT`

Threshold: each style cell separately requires `P_p95 <= 0.005` at the same preregistered checkpoint.

## Result

Selected checkpoint: `TAIL_0512`  
Selected total optimizer steps: `2560`  
Selected aggregate P p95: `0.003779542224947363`  
Selected worst-cell P p95: `0.0038324856432154623`

| style | P p50 | P p90 | P p95 | P mean | PASS |
|---|---:|---:|---:|---:|---|
| `cel_clean` | `0.00086122757056728` | `0.0024969751713797444` | `0.003733412444125855` | `0.0015121092099903422` | YES |
| `ink_cel` | `0.0008622108143754303` | `0.0025506053352728486` | `0.0038324856432154623` | `0.0015198510400674659` | YES |

Both cells pass independently; the result is not an aggregate-only pass.

## Trajectory

| candidate | LR | worst-cell P p95 | both cells pass? |
|---|---:|---:|---|
| `MAIN_0512` | `3e-4` | `0.029882849100977146` | NO |
| `MAIN_1024` | `3e-4` | `0.01868599979206918` | NO |
| `MAIN_2048` | `3e-4` | `0.00982439313083885` | NO |
| `TAIL_0064` | `3e-5` | `0.005000768043100823` | NO — ink_cel misses by ~7.68e-7 |
| `TAIL_0128` | `3e-5` | `0.0046900292858481395` | YES |
| `TAIL_0256` | `3e-5` | `0.004198873043060298` | YES |
| `TAIL_0512` | `3e-5` | `0.0038324856432154623` | YES |

Interpretation: the low-LR tail-polish behavior identified by the prior one-cell optimizer-localization experiment transfers to joint two-style fitting on the same asset. The two styles do not show a joint-capacity/style-interference blocker at this rung.

## Provenance / safety

- output field: `256 × 256`
- parent status: `P_V5_R256_ONE_CELL_RECOVERY_PASS`
- best checkpoint SHA-256: `cab207e1455f755b6931b9912fd6404d308216099ee1bab2042cb5dad0952b69`
- decision SHA-256: `2250f7c21076c0ae0b04093b5e711d2a2c0d26ababb2f2960ea0759a007cb5eb`
- preopt authority SHA-256: `16ac4eaa3444b2213340e4a6d0fb62260e67e75224b1142b1dcf87c2e12585b9`
- `camera.json` consumed: false
- TUNE consumed: false
- sealed splits opened: false

## Promotion decision

The frozen ladder advances exactly one rung:

```text
R256 representation                       PASS
  ↓
1 asset × 1 style learner/optimizer        PASS
  ↓
1 asset × 2 styles joint fit               PASS
  ↓
8 assets × 2 styles                        NEXT
  ↓ PASS
unseen-family generalization
```

Next policy: **preregister 8 assets × 2 styles R256 only**.

No unseen-family/generalization claim is made by this result.
