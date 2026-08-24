# RealSaS-OPT

Private canonical research workspace for RealSaS.

> **START HERE — active branch:** `audit/iris-architecture-discipline-20260824`
>
> **Continuation authority:** read `CURRENT_STATE.md` first.
>
> **Canonical architecture authority:** `canonical/PRODUCT_CONTRACT_V1.md` + `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md`.

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

**Responsibility boundary:** IRIS ends at the observable, rigging-sufficient 2.5D substrate. It does not own skeleton topology, parents, authored mechanical owner identity, skinning weights or source-rig exactness. Geppetto owns rigging/skeleton structure; Arachne owns skinning; Compiler verifies, normalizes, repairs/reselects where authorized, and exports the editable puppet.

The exact future training organization of Geppetto/Arachne — corpus layout, separate versus shared encoders/checkpoints, joint versus staged training — is **not frozen yet**. That is a later evidence question after IRIS qualification.

### Architecture change rule

This plan is the default authority. **It may change only in response to recorded controlled evidence.**

A proposed change must be tied to an explicit experiment, ablation, closure result or downstream non-inferiority/necessity test showing that the current boundary or representation is insufficient or that the proposed change materially improves the frozen objective. Convenience, intuition, analogy, implementation ease, or conversational drift are not sufficient authority to rewrite the plan.

Until such evidence exists, alternatives remain hypotheses and must not silently replace this architecture in `README`, `CURRENT_STATE`, preregistrations or implementation.

## Current IRIS frontier

Closed scientific authority:

- `P_GEOMETRY_SUFFICIENT` remains CLOSED/PASS.
- P-V5 native-scale-once analytic geometry remains CLOSED/PASS.
- The historical R/2 depth-overfit run completed insufficiently, but its interpretation was reopened at the output-field layer rather than assigned directly to learner capacity.
- P-V5 field-representation closure is now CLOSED: 256×256 is the smallest tested certified depth field; 128×128 is not certified under the frozen 16-cell gate.

Current executable research step:

**preregister and run R256 one-asset / one-style learner overfit.**

No broader P learner/generalization claim is authorized before that gate.

## Authority order

1. `CURRENT_STATE.md` — single continuation authority.
2. `canonical/PRODUCT_CONTRACT_V1.md` — visible product architecture and responsibility boundaries.
3. `canonical/OBSERVABLE_RIGGING_SUBSTRATE_CONTRACT_V1.md` — IRIS scientific problem definition.
4. active preregistration/result files under `experiments/iris_single_pose_v2/`.
5. historical experiment branches/files — evidence only, not execution authority unless explicitly re-authorized.

## Research rule

`apparatus/data -> representation/target -> learner/optimizer -> evidence consumer -> downstream sufficiency -> only then information limit`
