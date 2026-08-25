# RealSaS-OPT — Current State

**Date:** 2026-08-25  
**Active branch:** `audit/iris-architecture-discipline-20260824`  
**Draft PR:** `#4` — audit only, not merged  
**Status:** `P_GEOMETRY_CLOSED__R256_FIELD_CLOSED__8X2_EXACT_CONTINUATION_PASS__DEV32_ZERO_SHOT_PREREG_FROZEN__RUN_NEXT`

## Single continuation authority
Active implementation: `experiments/iris_single_pose_v2/`.

## Canonical architecture — evidence controlled
```text
8 ordered neutral-pose views
        ↓
      IRIS
        ↓
rigging-sufficient observable 2.5D substrate
        ↓
deterministic SurfaceBuilder candidate
        ↓
    Geppetto
        ↓
     Arachne
        ↓
Compiler authority / verified editable puppet
```
`SurfaceBuilder` remains deterministic. Learned surface-relation evidence and cross-cutting Compiler placement remain non-canonical linked hypotheses only.

## P-V5 R256 ladder
```text
R256 field representation                         PASS
1 asset × 1 style                                  PASS
1 asset × 2 styles                                 PASS
8 assets × 2 styles V1                             FAIL / IMMUTABLE
residual microscope                                COMPLETE
+2048 fresh-moments continuation                   FAIL / IMMUTABLE
36fb × 2-style fresh sufficiency                   PASS
fresh 8×2 V2, 6144 steps                          FAIL / IMMUTABLE
exact full-state continuation +1024                PASS
DEV32 zero-shot family generalization               ← FROZEN / RUN NEXT
EXTERNAL_HOLDOUT 169                               SEALED / CLOSED
```

## Exact 8×2 continuation — PASS
Canonical result: `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_EXACT_CONTINUATION_RESULT_20260825.md`.

Parent V2 remains immutable FAIL at 14/16, worst `0.005129679851233959`.

Exact-state continuation restored the same model + AdamW + GradScaler + RNG state, no fresh moments:
- zero-step parent reproduction: exact, max per-cell P95 delta `0.0`;
- first strict crossing: `EXACT_CONT_0512`, 16/16, worst `0.00494269179180264`;
- selected/final: `EXACT_CONT_1024`, total optimizer step `7168`;
- selected aggregate P95 `0.003200733161065726`;
- selected worst P95 `0.004825880285352466`;
- selected checkpoint SHA-256 `672a92030ce1a62acd8228791eb92c7af93c34fa36ac1d5866153f38ef8708de`.

Scientific interpretation: the fresh V2 miss was still optimization-budget limited along the same optimizer trajectory. A hard shared-capacity wall is not established.

## Corpus split authority
Post-corpus canonical production selection = 3993 non-QA assets:
- FIT 2981
- TUNE 318
- CAL 251
- DEV 274
- EXTERNAL_HOLDOUT 169

The builder's deterministic split recomputation had zero mismatches. `EXTERNAL_HOLDOUT` remains sealed.

The completed consumer export under `exports/IRIS` is only a partial 11-record export; master evidence is intact under `master/assets`.

## CURRENT GATE — DEV32 zero-shot probe
Prereg: `experiments/iris_single_pose_v2/P_V5_R256_DEV32_ZERO_SHOT_PREREG_20260825.md`  
Membership: `experiments/iris_single_pose_v2/P_V5_R256_DEV32_ZERO_SHOT_MEMBERSHIP_V1.json`

Fixed checkpoint before DEV evidence opens:
`672a92030ce1a62acd8228791eb92c7af93c34fa36ac1d5866153f38ef8708de`

Membership:
- 32 DEV assets × 2 styles = 64 cells;
- 26 Objaverse + 3 Quaternius + 3 KayKit;
- deterministic SHA selection within source;
- canonical JSON membership SHA-256 `afc20747a6154ac514f3c791ef03496b9e71f99dd7727b7804518e77767b157d`;
- selection uses metadata only, not raster/geometry/difficulty.

Evaluation:
- optimizer steps = 0;
- same P-V5 R256 observable geometry contract;
- 4096 exact raster-authority samples/view;
- no camera JSON, no PatchMatch, no architecture change;
- strict PASS iff 64/64 cells have P95 <= 0.005;
- source-stratified metrics mandatory.

If strict PASS: freeze larger/full DEV evaluation; still do not open EXTERNAL_HOLDOUT.
If NOT_CERTIFIED: keep EXTERNAL_HOLDOUT closed; localize by source/family and decide FIT-scale training before architecture changes.

## Firewalls
- TUNE/CAL/EXTERNAL_HOLDOUT unopened by current gate.
- DEV is evaluation-only; no gradients/checkpoint selection.
- Previous scientific FAILs remain immutable.
- PatchMatch remains preserved precedent, not current intervention.

## Downstream sequencing after IRIS closure
Before new Geppetto/Arachne implementation, audit canonical Compiler + Runtime code on Drive and recover surviving implementation into GitHub.

## Research rule
`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
