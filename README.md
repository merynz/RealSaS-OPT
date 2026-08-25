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

`SurfaceBuilder` is currently a **deterministic** candidate geometric canonicalization/topology layer. The linked alternative architecture note may later admit learned surface-relation evidence upstream, but SurfaceBuilder itself is not a neural stage.

## Current IRIS frontier
Closed/PASS:
- P geometry sufficiency;
- P-V5 observable native-scale analytic geometry;
- R256 field representation;
- one asset × one style;
- one asset × two styles;
- dominant hard asset `36fb` × two styles independently (`~0.00260` worst P95).

Immutable FAILs:
- fresh 8×2 V1: 8/16 PASS, worst `0.008320469176396726`;
- preregistered +2048 low-LR continuation: 13/16 PASS, worst `0.005740759451873588`.

The continuation improved all 16 cells simultaneously and remained descending. The subsequent fresh `36fb × 2-style` PASS falsifies intrinsic non-extractability of the dominant blocker. Current evidence therefore favors **shared optimization/schedule under-budget** over a demonstrated capacity wall or localized geometry-information failure.

**Current executable gate:** fresh 8×2 V2 adequate-budget certification.

```text
same frozen 16 cells
fresh unchanged R256 model
MAIN 2048 @ 3e-4
TAIL 4096 @ 3e-5 continuous
PASS = 16/16 P95 <= .005 at one prereg checkpoint
```

Every V2 prereg checkpoint stores model + optimizer + GradScaler + RNG state. No PatchMatch, no architecture change, no unseen-family access.

Authority files:
- `experiments/iris_single_pose_v2/P_V5_R256_36FB_TWO_STYLE_RESULT_20260825.md`;
- `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_PREREG_20260825.md`;
- `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_RELEASE_V1.json`.

Unseen-family remains CLOSED until V2 passes and an unseen-family preregistration is frozen.

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
