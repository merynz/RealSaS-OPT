# RealSaS-OPT

Private canonical research workspace for RealSaS.

> **START HERE — active branch:** `audit/iris-architecture-discipline-20260824`  
> **Continuation authority:** `CURRENT_STATE.md`  
> **Canonical architecture:** `canonical/PRODUCT_CONTRACT_V1.md` + `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`  
> **IMPORTANT external precedent:** `audit/IMPORTANT_EXTERNAL_PRECEDENT_PATCHMATCH_RL_20260825.md` — preserved precedent, not current intervention authority.

## Canonical architecture plan — evidence controlled
```text
ONE neutral pose × 8 ordered views
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

`SurfaceBuilder` is currently a **deterministic** candidate geometric canonicalization/topology layer. The linked alternative note may later admit learned surface-relation evidence upstream, but SurfaceBuilder itself is not a neural stage.

The cross-cutting Compiler/control-plane idea remains a linked architecture hypothesis, not canonical authority.

## Current IRIS frontier
Closed/PASS:
- P geometry sufficiency;
- P-V5 observable native-scale analytic geometry;
- R256 field representation;
- one-asset/one-style;
- one-asset/two-style;
- dominant hard asset `36fb` × two styles independently (`~0.00260` worst P95).

Immutable FAILs:
- 8×2 V1: 8/16, worst `0.008320469176396726`;
- +2048 fresh-moments continuation: 13/16, worst `0.005740759451873588`;
- fresh 8×2 V2 (6144 steps): 14/16, worst `0.005129679851233959`.

Fresh V2 leaves only `36fb` above threshold (`0.005108 / 0.005130`). The last 512-step interval still improves the worst cell by ~`4.86e-4`, so the V2 FAIL does not establish a plateau.

**Current executable gate:** exact full-state continuation from V2 `TAIL_4096`.

```text
restore exact model + AdamW + GradScaler + RNG
same 16 cells / same objective / same LR 3e-5
+1024 steps
checks at 128 / 256 / 512 / 768 / 1024
PASS = 16/16 P95 <= .005 at one prereg checkpoint
```

No fresh optimizer moments, no PatchMatch, no architecture change, no unseen-family access. Parent re-evaluation and canonical truth SHA mapping must reproduce before optimizer step 1.

Authority files:
- `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_RESULT_20260825.md`;
- `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_EXACT_CONTINUATION_PREREG_20260825.md`;
- `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_EXACT_CONT_PARENT_AUTHORITY.json`.

Unseen-family remains CLOSED until the exact continuation resolves and a separate unseen-family preregistration is frozen.

## Downstream sequencing
After IRIS is genuinely closed, and before new Geppetto/Arachne/downstream work, audit the most canonical Compiler + Runtime code on Drive and recover the valuable surviving implementation into GitHub as canonical authority.

## Authority order
1. `CURRENT_STATE.md`
2. canonical product/substrate contracts
3. active preregistration/result/localization files under `experiments/iris_single_pose_v2/`
4. external precedent notes under `audit/`
5. historical experiments

## Research rule
`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
