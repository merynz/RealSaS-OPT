# RealSaS-OPT — Current State

**Date:** 2026-08-25  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_CLOSED__P_V5_CLOSED__R256_FIELD_CLOSED__ONE_CELL_PASS__TWO_STYLE_PASS__EIGHT_BY_TWO_FAIL__CONTINUATION_FAIL__SCHEDULE_UNDERBUDGET_SUPPORTED__36FB_TWO_STYLE_PREREG_FROZEN__A100_RUN_NEXT`

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

`SurfaceBuilder` remains a **deterministic** candidate canonicalization/topology layer in the current contract. The linked alternative architecture note may allow optional learned surface-relation evidence upstream, but SurfaceBuilder itself is not promoted to a neural model.

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
8 assets × 2 styles joint fit V1                 FAIL / IMMUTABLE
        ↓
residual microscope                              COMPLETE
        ↓
+2048 low-LR continuation localization           FAIL / IMMUTABLE
        ↓
36fb × 2-style solo learner sufficiency          ← FROZEN / RUN NEXT
        ↓
new fresh 8×2 V2 certification PASS required
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

PatchMatch remains important but **not currently justified as the next intervention**.

## +2048 low-LR continuation — IMMUTABLE FAIL
Canonical result: `experiments/iris_single_pose_v2/P_V5_R256_8X2_CONTINUATION_RESULT_20260825.md`.

Starting from exact failed `TAIL_0512` weights, the frozen continuation used fresh AdamW moments at `3e-5` for +2048 shared optimizer steps.

Final `CONT_2048` / total step 4608:
- aggregate P95 `0.003706276847515254`;
- worst-cell P95 `0.005740759451873588`;
- 13/16 cells PASS;
- scientific status `P_V5_R256_8X2_ADDITIONAL_LOW_LR_OPTIMIZATION_NOT_SUFFICIENT`.

Remaining failures:
- `36fb / cel_clean = 0.005740759451873588`;
- `36fb / ink_cel = 0.005718213529326022`;
- `76313 / ink_cel = 0.005042715766467144`.

Critical localization: **all 16/16 cells improve from parent `TAIL_0512` to `CONT_2048`**. The worst-cell curve remains descending at the final checkpoint; `36fb` two-style mean improves by ~`4.27e-4` in each of the final two 512-step intervals. Therefore the continuation proves that the frozen +2048 budget was insufficient, but does **not** establish a plateau or true shared-capacity wall.

Updated evidence ranking:
- style interference: strongly disfavored;
- localized geometry/occlusion hard-tail as primary blocker: disfavored;
- shared multi-asset fit burden: supported;
- true shared-capacity wall: not established / weakened;
- schedule or optimization under-budget: strongly supported;
- intrinsic two-style extractability of `36fb...`: unresolved.

Do not add PatchMatch and do not extend this failed checkpoint again by inspection-driven training.

## CURRENT GATE — 36fb one-asset × two-style fresh sufficiency
Prereg: `experiments/iris_single_pose_v2/P_V5_R256_36FB_TWO_STYLE_PREREG_20260825.md`.
Membership: `experiments/iris_single_pose_v2/P_V5_R256_36FB_TWO_STYLE_MEMBERSHIP_V1.json`.
Release authority: `experiments/iris_single_pose_v2/P_V5_R256_36FB_TWO_STYLE_RELEASE_V1.json`.

```text
asset_36fb02305846592b1ecdf3d4
  × cel_clean
  × ink_cel
        ↓
fresh R256 model
MAIN 2048 @ 3e-4
TAIL 2048 @ 3e-5, fresh moments
        ↓
2/2 cells P_p95 <= 0.005 ?
```

The longer tail is frozen prospectively because the 8×2 continuation remained descending through +2048 low-LR steps. This gate does not reuse any failed 8×2 weights.

Prepared release checks:
- Python syntax/compile: PASS;
- CPU B=2 full-R P-only forward/backward: PASS;
- fake 36fb staging: PASS;
- fake 4096/view cache with exact shared truth loci across styles: PASS;
- parent continuation + prior two-style authority verifier: PASS;
- scientific optimizer steps during preparation: 0;
- prior identical B=2 R256 GPU preflight: PASS on Tesla T4 at `8,115,611,136` peak allocated bytes.

Notebook SHA-256: `eff053232806e762c2a1e88c76109a34e05706a691b8d1e5f70586c4f77939d0`.
Bundle SHA-256: `c002cb7510722565fb48d4a311c25616530256e4edd057fe80d4b828beb5ca2a`.

Interpretation:
- PASS -> directly supports shared optimization/interference burden; preregister a fresh 8×2 V2 certification with evidence-backed adequate schedule;
- FAIL -> localize `36fb` itself before any capacity or geometry-intervention claim.

No architecture change is authorized. Unseen-family remains closed.

## Downstream sequencing after IRIS closure
Before starting new Geppetto/Arachne/downstream implementation work, audit the most canonical Compiler + Runtime code preserved on Drive and bring the valuable surviving code into GitHub as canonical authority. Do not abandon the existing compiler/runtime codebase.

## Research rule
`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
