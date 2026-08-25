# RealSaS-OPT

Private canonical research workspace for RealSaS.

> **START HERE — active branch:** `audit/iris-architecture-discipline-20260824`  
> **Continuation authority:** `CURRENT_STATE.md`  
> **Canonical architecture:** `canonical/PRODUCT_CONTRACT_V1.md` + `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`

## Current IRIS frontier
P-V5 R256 multi-asset fit is now **closed on the frozen 8×2 training membership**.

Exact full-state continuation result:
- first 16/16 crossing at `EXACT_CONT_0512`;
- selected/final `EXACT_CONT_1024`;
- total optimizer steps `7168`;
- worst cell P95 `0.004825880285352466`;
- checkpoint SHA-256 `672a92030ce1a62acd8228791eb92c7af93c34fa36ac1d5866153f38ef8708de`.

Earlier V1/V2 FAILs remain immutable. The exact-state PASS proves the final V2 miss was still budget-limited, not a demonstrated hard capacity wall.

## Current executable gate
**DEV32 fixed-checkpoint zero-shot family generalization probe.**

```text
fixed EXACT_CONT_1024 checkpoint
        ↓
32 deterministic DEV families
26 Objaverse + 3 Quaternius + 3 KayKit
        ↓
cel_clean + ink_cel = 64 cells
        ↓
NO optimizer / NO checkpoint selection
        ↓
strict PASS = 64/64 cell P95 <= .005
```

Canonical JSON membership SHA-256: `afc20747a6154ac514f3c791ef03496b9e71f99dd7727b7804518e77767b157d`.

`EXTERNAL_HOLDOUT` has **169** canonical families and remains sealed. It is not consumed by DEV32.

Authority:
- `CURRENT_STATE.md`
- `experiments/iris_single_pose_v2/P_V5_R256_8X2_V2_EXACT_CONTINUATION_RESULT_20260825.md`
- `experiments/iris_single_pose_v2/P_V5_R256_DEV32_ZERO_SHOT_PREREG_20260825.md`
- `experiments/iris_single_pose_v2/P_V5_R256_DEV32_ZERO_SHOT_MEMBERSHIP_V1.json`

## Architecture note
`SurfaceBuilder` remains deterministic in the current contract. Learned surface-relations and cross-cutting Compiler placement remain linked non-canonical hypotheses.

## Downstream sequencing
After IRIS closes, audit and recover the canonical Compiler + Runtime implementation from Drive before starting new Geppetto/Arachne work.
