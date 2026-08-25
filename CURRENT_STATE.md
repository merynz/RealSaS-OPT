# RealSaS-OPT — Current State

**Date:** 2026-08-25  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_CLOSED__P_V5_CLOSED__R256_FIELD_CLOSED__ONE_CELL_PASS__TWO_STYLE_PASS__EIGHT_BY_TWO_FAIL__RESIDUAL_LOCALIZED_TO_SHARED_DEPTH_FIT__CONTINUATION_LOCALIZATION_RUN_NEXT`

## Single continuation authority
Active implementation: `experiments/iris_single_pose_v2/`. Read this file and root `README.md` before continuing.

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
IRIS stops at the observable substrate. Architecture/responsibility boundaries may change only under recorded controlled evidence.

## Important external precedent
`audit/IMPORTANT_EXTERNAL_PRECEDENT_PATCHMATCH_RL_20260825.md` remains preserved. PatchMatch-RL geometry-in-the-loop verification is a future intervention candidate only if controlled residual evidence localizes a geometric extraction/coherence blocker. It is not admitted by analogy or convenience.

## Closed P authority
- `P_GEOMETRY_SUFFICIENT`: CLOSED/PASS.
- P-V5 observable native-scale analytic reconstruction: CLOSED/PASS.
- R256 field representation: CLOSED/PASS; smallest tested certified field.
- one asset × one style learner/optimizer: PASS.
- one asset × two styles shared joint fit: PASS.

## Promotion ladder
```text
R256 field representation                       PASS
        ↓
1 asset × 1 style learner/optimizer              PASS
        ↓
1 asset × 2 styles joint fit                     PASS
        ↓
8 assets × 2 styles joint fit                    FAIL / IMMUTABLE
        ↓
8×2 failure localization                         CURRENT
        ↓
new 8×2 certification PASS required
        ↓
unseen-family generalization
```
Unseen-family remains closed.

## 8×2 V1 — IMMUTABLE FAIL
Canonical result: `experiments/iris_single_pose_v2/P_V5_R256_8X2_RESULT_20260825.md`.

Selected `TAIL_0512`, 2560 optimizer steps:
- worst-cell P95 `0.008320469176396726`;
- aggregate P95 `0.0051227101590484376`;
- 8/16 cells PASS;
- required 16/16 at one preregistered checkpoint.

Style interference is strongly disfavored: all four passing assets pass both styles and all four failing assets fail both styles. The prior solo-PASS control `asset_76313...` degrades from ~`0.00373/0.00383` solo to ~`0.00679/0.00675` in the 8-asset shared model.

## Residual localization
Canonical note: `experiments/iris_single_pose_v2/P_V5_R256_8X2_RESIDUAL_LOCALIZATION_20260825.md`.

Optimizer-zero microscopy reproduces the frozen failures and finds:
- `36fb...` error broad across all 8 views, not one broken view;
- no strong top-error enrichment at silhouette boundary;
- `76313...` shows the same broad degradation despite prior solo PASS;
- analytic screen-plane reconstruction remains stable; degradation is concentrated in learned camera-forward depth.

Current evidence ranking:
- style interference: strongly disfavored;
- silhouette/occlusion hard-tail as primary blocker: disfavored;
- intrinsic non-extractability of `76313...`: falsified by solo PASS;
- shared multi-asset optimization/capacity burden: strongly supported;
- extra optimization versus true shared capacity/interference: unresolved.

PatchMatch remains important but **not currently justified as the next intervention**.

## CURRENT GATE — additional low-LR optimization localization
Prereg: `experiments/iris_single_pose_v2/P_V5_R256_8X2_CONTINUATION_LOCALIZATION_PREREG_20260825.md`.

Starts from exact failed `TAIL_0512` model weights (SHA-256 `0228c8c939490c9ac33cee6ca360c9228a5b4ee462b30b9774888c12129f3776`). Same architecture, same 16 cells, same truth, same objective; no PatchMatch or new data.

Because the parent checkpoint does not contain optimizer state, the frozen intervention is:
- fresh AdamW moments;
- LR `3e-5`;
- +2048 continuation optimizer steps;
- eval at 128/256/512/1024/1536/2048;
- every update represents all 16 cells through 8-cell ×2 accumulation.

PASS means only `ADDITIONAL_LOW_LR_OPTIMIZATION_SUFFICIENT`; it does not rewrite the old FAIL and does not authorize unseen-family. PASS -> preregister fresh 8×2 V2 certification with adequate schedule. FAIL -> shared capacity/interference localization.

## Downstream sequencing after IRIS closure
Before starting new Geppetto/Arachne/downstream implementation work, audit the most canonical Compiler + Runtime code preserved on Drive and bring the valuable surviving code into GitHub as canonical authority. Do not abandon the existing compiler/runtime codebase.

## Research rule
`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
