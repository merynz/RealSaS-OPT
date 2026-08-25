# RealSaS-OPT — Current State

**Date:** 2026-08-25  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_CLOSED__P_V5_CLOSED__R256_FIELD_CLOSED__ONE_CELL_PASS__TWO_STYLE_PASS__EIGHT_BY_TWO_V1_FAIL__CONTINUATION_FAIL__36FB_TWO_STYLE_PASS__FRESH_8X2_V2_PREREG_FROZEN__A100_RUN_NEXT`

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

`SurfaceBuilder` remains a **deterministic** candidate canonicalization/topology layer in the current contract. The linked alternative architecture note may allow optional learned surface-relation evidence upstream, but SurfaceBuilder itself is not a neural model.

## Important external precedent
`audit/IMPORTANT_EXTERNAL_PRECEDENT_PATCHMATCH_RL_20260825.md` remains preserved. PatchMatch-RL is a future geometry-in-the-loop intervention candidate only if residual evidence localizes an extraction/coherence blocker. Current evidence does not justify it.

## Promotion ladder
```text
R256 field representation                         PASS
        ↓
1 asset × 1 style learner/optimizer                PASS
        ↓
1 asset × 2 styles joint fit                       PASS
        ↓
8 assets × 2 styles joint fit V1                   FAIL / IMMUTABLE
        ↓
residual microscope                                COMPLETE
        ↓
+2048 low-LR continuation localization             FAIL / IMMUTABLE
        ↓
36fb × 2-style fresh sufficiency                   PASS
        ↓
fresh 8×2 V2 adequate-budget certification         ← FROZEN / RUN NEXT
        ↓
unseen-family generalization                       CLOSED
```

## Immutable 8×2 V1 FAIL
`experiments/iris_single_pose_v2/P_V5_R256_8X2_RESULT_20260825.md`

Selected `TAIL_0512`, total step 2560:
- 8/16 cells PASS;
- worst P95 `0.008320469176396726`;
- aggregate P95 `0.0051227101590484376`.

Do not rewrite this result.

## Immutable +2048 continuation FAIL
`experiments/iris_single_pose_v2/P_V5_R256_8X2_CONTINUATION_RESULT_20260825.md`

Selected `CONT_2048`, total step 4608:
- 13/16 cells PASS;
- worst P95 `0.005740759451873588`;
- aggregate P95 `0.003706276847515254`.

All 16/16 cells improved relative to the parent checkpoint and the worst-cell curve remained descending. This established that the frozen +2048 budget was insufficient but did not establish a hard shared-capacity wall.

## 36fb × two-style sufficiency — PASS
Canonical result: `experiments/iris_single_pose_v2/P_V5_R256_36FB_TWO_STYLE_RESULT_20260825.md`.

Fresh model, MAIN `2048 @ 3e-4`, fresh-moment TAIL `2048 @ 3e-5`:
- `36fb / cel_clean` P95 `0.002594304538797585`;
- `36fb / ink_cel` P95 `0.0025984761072322723`;
- 2/2 PASS;
- selected worst-cell P95 `0.0025984761072322723`.

This falsifies intrinsic non-extractability of the dominant hard asset under current R256 P representation. Combined with the shared continuation trajectory, schedule/shared-optimization burden is now the strongest supported explanation. A demonstrated hard capacity wall is not established.

## CURRENT GATE — fresh 8×2 V2 certification
Prereg: `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_PREREG_20260825.md`.  
Release: `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_RELEASE_V1.json`.

Frozen design:
```text
same 8 FIT assets × 2 styles = 16 cells
same R256 architecture / P-only objective
fresh model; NO checkpoint reuse

MAIN 2048 @ 3e-4
        ↓
fresh AdamW moments once
TAIL 4096 @ 3e-5, continuous
        ↓
PASS iff one prereg checkpoint has
16/16 cells P_p95 <= 0.005
```

Tail eval checkpoints: `256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096`. Full schedule runs even if an earlier checkpoint passes.

V2 checkpoint authority is strengthened: every prereg checkpoint contains model + optimizer + GradScaler + Python/NumPy/Torch CPU/Torch CUDA RNG state. Exact phase-local continuation will therefore remain possible if ever scientifically authorized.

Prepared local validation:
- Python compile: PASS;
- CPU full-R / accumulation equivalence: PASS (`max abs ~2.086e-7`, `max rel ~4.889e-6`);
- resumable checkpoint schema: PASS;
- optimizer-state checkpoint roundtrip after a synthetic optimizer step: PASS;
- membership SHA exactly equals V1 membership;
- scientific optimizer steps during preparation: 0.

Firewalls:
- no architecture change;
- no PatchMatch;
- no augmentation;
- no camera JSON / hidden teacher camera;
- N/U/Z frozen;
- unseen-family CLOSED;
- original FAILs remain immutable.

### Next policy
- V2 PASS → freeze/preregister unseen-family generalization **before opening unseen data**.
- V2 FAIL → remain at 8×2 and localize true shared capacity/interference before architecture expansion or PatchMatch.

## Current evidence ranking
- P representation: CLOSED/PASS;
- style interference: strongly disfavored;
- intrinsic hard-asset extractability blocker: falsified by 36fb PASS;
- localized silhouette/occlusion geometry hard-tail: disfavored as primary blocker;
- shared multi-asset fitting burden: confirmed;
- schedule/optimization under-budget: strongest current hypothesis;
- true shared-capacity wall: not established;
- PatchMatch: preserved but not currently justified.

## Downstream sequencing after IRIS closure
Before starting new Geppetto/Arachne/downstream implementation work, audit the most canonical Compiler + Runtime code preserved on Drive and bring valuable surviving code into GitHub as canonical authority. Do not abandon the existing compiler/runtime codebase.

## Research rule
`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
