# RealSaS-OPT

Private canonical research workspace for RealSaS.

> **START HERE — active branch:** `audit/iris-architecture-discipline-20260824`  
> **Continuation authority:** `CURRENT_STATE.md`  
> **Canonical architecture:** `canonical/PRODUCT_CONTRACT_V1.md` + `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`  
> **IMPORTANT external precedent:** `audit/IMPORTANT_EXTERNAL_PRECEDENT_PATCHMATCH_RL_20260825.md` — reference/feasibility evidence, not current architecture authority.

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

IRIS ends at the observable rigging-sufficient 2.5D substrate. It does not own skeleton topology/parents, authored mechanical identity, skinning weights or source-rig exactness. Geppetto owns rigging structure; Arachne owns skinning; Compiler verifies and exports the editable puppet. The future training organization of Geppetto/Arachne remains intentionally unfrozen until IRIS is qualified.

### Architecture change rule
This plan may change only under recorded controlled evidence. Convenience, intuition, analogy, implementation ease or conversational drift are not sufficient authority.

## Current IRIS frontier
Closed/PASS:
- P geometry sufficiency;
- P-V5 observable native-scale analytic geometry;
- R256 field representation (smallest tested certified field);
- one-asset/one-style learner/optimizer sufficiency;
- one-asset/two-style shared-model joint fit.

**Current executable gate:** `8 assets × 2 styles R256` joint fit. Its preregistration and fixed 16-cell membership are frozen. The run uses a fresh model, the previously evidenced `3e-4 → 3e-5` schedule, and a full 16-cell objective every optimizer step via A100-targeted 8-cell ×2 gradient accumulation. PASS requires all 16 cells individually `P_p95 <= 0.005` at one checkpoint.

No unseen-family claim is authorized before this gate.

## Important external precedent
PatchMatch-RL (ICCV 2021), *Deep MVS with Pixelwise Depth, Normal, and Visibility*, remains the closest open-code precedent for known-camera multi-view image -> oriented geometric surface reconstruction. Its geometry-in-the-loop verification is retained as a future intervention candidate **only if controlled results show the current direct R256 extraction/coherence path is insufficient**.

## Authority order
1. `CURRENT_STATE.md`
2. `canonical/PRODUCT_CONTRACT_V1.md`
3. `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`
4. active preregistration/result files under `experiments/iris_single_pose_v2/`
5. external precedent notes under `audit/`
6. historical experiments

## Research rule
`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
