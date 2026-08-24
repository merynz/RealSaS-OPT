# RealSaS-OPT — Current State

**Date:** 2026-08-25  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_CLOSED__P_V5_CLOSED__R256_FIELD_CLOSED__ONE_CELL_SUFFICIENT__TWO_STYLE_JOINT_FIT_PASS__EIGHT_BY_TWO_PREREG_NEXT`

## Single continuation authority
Active implementation: `experiments/iris_single_pose_v2/`.

Before continuing, read this file and root `README.md`.

## Canonical architecture — evidence controlled

```text
8 ordered neutral-pose views
        ↓
      IRIS
        ↓
rigging-sufficient observable 2.5D substrate
        ↓
    Geppetto
        ↓
editable skeleton / hierarchy proposal
        ↓
     Arachne
        ↓
editable skinning / weight proposal
        ↓
     Compiler
        ↓
verified editable puppet
```

IRIS stops at the observable 2.5D substrate. Geppetto/Arachne training organization remains intentionally unfrozen while IRIS is being closed.

**Architecture change control:** architecture/responsibility boundaries may change only under recorded controlled evidence. Convenience, analogy, intuition, implementation ease or conversational drift are not authority.

### IMPORTANT external precedent — do not lose

`audit/IMPORTANT_EXTERNAL_PRECEDENT_PATCHMATCH_RL_20260825.md`

PatchMatch-RL (ICCV 2021) is the closest open-code working precedent identified so far for the current IRIS formulation: calibrated multi-view images + known cameras -> pixelwise depth/normal/visibility -> reprojection-consistent oriented surface/point cloud.

This is an **important feasibility precedent and future intervention library**, especially for geometry-in-the-loop cross-view hypothesis verification if a family-disjoint P/N hard tail later survives the direct R256 ladder. It is **not current architecture authority** and does not authorize changing the frozen ladder by analogy alone.

## Closed P authority
- `P_GEOMETRY_SUFFICIENT`: CLOSED/PASS.
- P-V5 native-scale-once analytic reconstruction: CLOSED/PASS.
- legal P factorization: native1024 RGBA -> estimate `h_native` once -> canonical yaw -> learn camera-forward scalar depth -> analytic canonical P.
- `camera.json` / teacher camera half extent remain forbidden learner inputs.

## R256 field representation — CLOSED
| field | interpretation | worst-cell P p95 |
|---|---|---:|
| 64×64 | NOT_CERTIFIED | `0.03889907157958461` |
| 128×128 | NOT_CERTIFIED | `0.01284720621837844` |
| 256×256 | CERTIFIED 16/16 | `0.0008174655519194024` |

Smallest tested certified field = **R256**.

## Frozen learner promotion ladder
```text
R256 field representation                     PASS
        ↓
1 asset × 1 style learner/optimizer sufficiency PASS
        ↓
1 asset × 2 styles joint fit                    PASS
        ↓
8 assets × 2 styles                             ← NEXT
        ↓ PASS
unseen-family generalization
```

No rung may be skipped without a preregistered evidence-backed revision.

## One asset × one style — learner/optimizer sufficiency CLOSED
The original frozen fixed-lr gate remains an immutable FAIL:
- `cel_clean`, 2048 steps, AdamW `3e-4`;
- P p95 `0.005682396539486942` vs threshold `0.005`.

Optimizer localization reproduced the exact checkpoint at step 0 with absolute P-p95 difference `0.0` and then ran three fresh-AdamW restarts:

| arm | lr | first PASS | min P p95 |
|---|---:|---:|---:|
| A control | `3e-4` | — | `0.005420877947472036` |
| B | `1e-4` | 256 | `0.00420133795123547` |
| C | `3e-5` | 64 | `0.003946938854642211` |

Canonical recovery status: `P_V5_R256_ONE_CELL_RECOVERY_PASS`.  
Localization label: `LATE_STAGE_LR_FLOOR_SUPPORTED`.

Interpretation authority:
`experiments/iris_single_pose_v2/P_V5_R256_OPTIMIZER_LOCALIZATION_RESULT_20260825.md`

## One asset × two styles — CLOSED/PASS
Canonical result:
`experiments/iris_single_pose_v2/P_V5_R256_TWO_STYLE_RESULT_20260825.md`

Frozen cells:
- `asset_76313e4bd82b82fcd1659c70 / cel_clean / FIT`;
- `asset_76313e4bd82b82fcd1659c70 / ink_cel / FIT`.

One shared fresh R256 model jointly fit both styles. Selected checkpoint `TAIL_0512`, total optimizer steps `2560`:

| style | selected P p95 | threshold | status |
|---|---:|---:|---|
| `cel_clean` | `0.003733412444125855` | `0.005` | PASS |
| `ink_cel` | `0.0038324856432154623` | `0.005` | PASS |

Selected aggregate P p95: `0.003779542224947363`.  
Selected worst-cell P p95: `0.0038324856432154623`.

MAIN `3e-4` at 2048 steps remained insufficient (`worst-cell P p95 = 0.00982439313083885`). Fresh-moment TAIL `3e-5` crossed the two-cell gate by tail step 128 (`0.0046900292858481395`) and improved through step 512. This independently supports the prior late-stage LR localization under joint two-style fit.

Safety/provenance:
- best checkpoint SHA-256 `cab207e1455f755b6931b9912fd6404d308216099ee1bab2042cb5dad0952b69`;
- decision SHA-256 `2250f7c21076c0ae0b04093b5e711d2a2c0d26ababb2f2960ea0759a007cb5eb`;
- no `camera.json`;
- no TUNE;
- sealed splits unopened.

## CURRENT GATE — preregister 8 assets × 2 styles R256
Frozen next policy from the two-style result: **preregister `8 assets × 2 styles R256` only**.

No 8×2 scientific training is authorized until its membership, checkpoint-selection rule, optimizer schedule, per-cell PASS rule and executable preflight are frozen prospectively.

The one-asset/two-style result is an overfit/joint-capacity result only. It does not authorize unseen-family or product-domain claims.

## Research rule
`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
