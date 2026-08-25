# RealSaS-OPT — Current State

**Date:** 2026-08-25  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_CLOSED__P_V5_CLOSED__R256_FIELD_CLOSED__ONE_CELL_PASS__TWO_STYLE_PASS__36FB_TWO_STYLE_PASS__8X2_V1_FAIL__8X2_CONTINUATION_FAIL__8X2_V2_FAIL__EXACT_V2_CONTINUATION_PREREG_FROZEN__A100_RUN_NEXT`

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

Compiler placement is still under an explicitly non-canonical alternative hypothesis: it may ultimately act as a cross-cutting authority/control plane between neural stages rather than only as the final stage. See `audit/ALTERNATIVE_DOWNSTREAM_ARCHITECTURE_HYPOTHESES_20260825.md`. Do not promote this without controlled evidence.

## Important external precedent
`audit/IMPORTANT_EXTERNAL_PRECEDENT_PATCHMATCH_RL_20260825.md` remains preserved. PatchMatch-RL is a future geometry-in-the-loop intervention candidate only if residual evidence localizes an extraction/coherence blocker. Current evidence does not justify it.

## Promotion ladder
```text
R256 field representation                         PASS
        ↓
1 asset × 1 style                                  PASS
        ↓
1 asset × 2 styles                                 PASS
        ↓
8 assets × 2 styles V1                             FAIL / IMMUTABLE
        ↓
residual microscope                                COMPLETE
        ↓
+2048 fresh-moments low-LR continuation            FAIL / IMMUTABLE
        ↓
36fb × 2-style fresh sufficiency                   PASS
        ↓
fresh 8×2 V2, 6144 steps                          FAIL / IMMUTABLE
        ↓
exact full-state V2 continuation +1024             ← FROZEN / RUN NEXT
        ↓
unseen-family generalization                       CLOSED
```

## Immutable earlier FAILs
### 8×2 V1
`experiments/iris_single_pose_v2/P_V5_R256_8X2_RESULT_20260825.md`

Selected `TAIL_0512`, total step 2560:
- 8/16 PASS;
- worst P95 `0.008320469176396726`;
- aggregate P95 `0.0051227101590484376`.

### +2048 fresh-moments continuation
`experiments/iris_single_pose_v2/P_V5_R256_8X2_CONTINUATION_RESULT_20260825.md`

Selected `CONT_2048`, total step 4608:
- 13/16 PASS;
- worst P95 `0.005740759451873588`;
- aggregate P95 `0.003706276847515254`.

All 16 cells improved versus the V1 parent and the curve remained descending. This supported schedule/optimization insufficiency but did not preserve exact optimizer-state continuation.

## 36fb × two-style sufficiency — PASS
`experiments/iris_single_pose_v2/P_V5_R256_36FB_TWO_STYLE_RESULT_20260825.md`

Fresh model, MAIN `2048 @ 3e-4`, fresh-moment TAIL `2048 @ 3e-5`:
- cel_clean P95 `0.002594304538797585`;
- ink_cel P95 `0.0025984761072322723`;
- 2/2 PASS.

This falsifies intrinsic non-extractability of the dominant hard asset under the current R256 P representation.

## Fresh 8×2 V2 — IMMUTABLE FAIL
Canonical result: `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_RESULT_20260825.md`.

Frozen schedule:
- fresh model;
- MAIN `2048 @ 3e-4`;
- continuous TAIL `4096 @ 3e-5`;
- total 6144 optimizer steps;
- exact full-state checkpoint authority enabled.

Selected/final `TAIL_4096`:
- status `P_V5_R256_8X2_V2_CERTIFICATION_INSUFFICIENT`;
- 14/16 PASS;
- aggregate P95 `0.003398841607850042`;
- worst P95 `0.005129679851233959`.

Only `asset_36fb...` remains above threshold:
- cel_clean `0.005108391284011302`;
- ink_cel `0.005129679851233959`.

Late-tail worst-cell trajectory:
```text
TAIL_2048  0.0063670447  12/16
TAIL_2560  0.0061471324  12/16
TAIL_3072  0.0056674025  12/16
TAIL_3584  0.0056157707  14/16
TAIL_4096  0.0051296799  14/16
```
The final 512-step interval improves worst-cell P95 by ~`4.86e-4`; the final checkpoint is the best authority checkpoint. V2 is therefore a real FAIL but does not demonstrate a late-tail plateau. The remaining miss is ~`1.30e-4` and is confined to a hard asset already known to fit independently near `0.00260`.

## CURRENT GATE — exact V2 full-state continuation V1
Prereg: `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_EXACT_CONTINUATION_PREREG_20260825.md`.  
Parent authority: `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_EXACT_CONT_PARENT_AUTHORITY.json`.

Frozen design:
```text
exact parent = V2 TAIL_4096
checkpoint SHA = 8c3872412124644885b66aea1b00413a8da5ab91a7471e83fcae001d43544049

restore:
  model
  AdamW state
  GradScaler state
  Python / NumPy / Torch CPU / Torch CUDA RNG

same 16 cells
same objective
same LR = 3e-5
NO fresh moments

+1024 exact continuation steps
eval = 128 / 256 / 512 / 768 / 1024
PASS iff one prereg checkpoint has 16/16 P_p95 <= .005
```

Before scientific optimizer step 1:
- parent run-complete, decision and checkpoint SHA must match;
- regenerated frozen truth SHA mapping must match the canonical 8-asset cache truth mapping;
- a zero-step parent re-evaluation must reproduce 14/16 and parent worst P95 within `5e-5`;
- CPU exact-state and GPU accumulation preflights must PASS;
- scientific optimizer steps remain 0 until all checks pass.

The full +1024 schedule runs even if PASS appears early.

Interpretation:
- PASS -> original V2 remains immutable FAIL, but exact same optimizer trajectory proves the final miss was still budget-limited; freeze/preregister unseen-family generalization before opening unseen data;
- FAIL -> stop blind budget extension; shared capacity/interference becomes the primary localization branch before any architecture expansion. PatchMatch still requires geometry-localized evidence.

No architecture change is authorized. Unseen-family remains CLOSED.

## Current evidence ranking
- P representation: CLOSED/PASS;
- style interference: strongly disfavored;
- intrinsic hard-asset extractability blocker: falsified by 36fb PASS;
- localized silhouette/occlusion geometry hard-tail: disfavored as primary blocker;
- shared multi-asset fitting burden: confirmed;
- schedule/optimization under-budget: still plausible and directly tested by exact continuation;
- true shared-capacity wall: not yet established;
- PatchMatch: preserved but not currently justified.

## Downstream sequencing after IRIS closure
Before starting new Geppetto/Arachne/downstream implementation work, audit the most canonical Compiler + Runtime code preserved on Drive and bring valuable surviving code into GitHub as canonical authority. Do not abandon the existing compiler/runtime codebase.

## Research rule
`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
