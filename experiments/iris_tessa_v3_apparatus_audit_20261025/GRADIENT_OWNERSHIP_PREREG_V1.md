# Gradient Ownership / Private Relation Adapter Court V1 — preregistration

**Status:** SEALED BEFORE OUTCOME  
**Authority:** bounded follow-up to `CONTINUOUS_JOINT_SYSTEM_PREREG_V1` after the V1 joint court isolated a geometry-interference failure while endpoint relation learning remained strong.

## Motivation / failure owner

The continuous joint V1 court used the V6-selected `DENSE3D` carrier, frozen `Q192_D3_FOURIER4` query class, S2 GM/GP private adapters, endpoint-bound relation head, equal unweighted normalized GM/GP losses plus relation BCE, and 6000 steps/arm. The joint arm learned the relation but failed the GM/GP geometry gates and negative-transfer gates.

This court changes **only relation gradient ownership and relation-private low-rank capacity**. It does not change the carrier, scene family, truth functions, endpoint binding, optimizer, step budget, batch construction, GM/GP adapters, query class, or scientific gates.

External systems (including StdGEN++-style private adaptation) are hypothesis generators only; no adapter enters V3.1 unless it wins this matched court.

## Frozen dependency

Consume exactly `ORACLE_Z_FINAL_CARRIER_SELECTION_V6.json` and require:

- schema `RealSaS.OracleZFinalCarrierSelection.v6`;
- status `FINAL_PRETRUTH_CARRIER_SELECTION_V6`;
- selected carrier `DENSE3D`;
- all V6 evidence gates true;
- continuous-joint dependency authority true.

## Fixed apparatus

Reuse without modification from `continuous_joint_system_probe_v1.py`:

- five scene seeds;
- `SURFACE_STRUCTURED` and `THIN_SURFACE`;
- factors `4,2,1 * h_ref`;
- exact GM and independently seeded GP analytic targets;
- directed endpoint relation target `GP_1(a) > GP_1(b)`;
- width 192, depth 3, Fourier4 query;
- S2 GM and GP residual adapters, bottleneck 64;
- LR `3e-3`;
- 6000 steps/arm;
- batch 32;
- train pool 4096/scene;
- eval points 2048/scene;
- no runtime-based truncation;
- GitHub 360-minute timeout is only a runaway guard.

## Factorial arms

All four scientific joint arms have the same S2 GM/GP architecture and endpoint-bound relation semantics.

1. `CURRENT_FULL_GRAD`
   - exact current S2 joint-bound behavior;
   - relation BCE backpropagates through endpoint GP state, GP adapter, shared Q and carrier;
   - no extra relation-private adapter.

2. `STOPGRAD_RELATION`
   - relation consumes `stopgrad(H_GP(a)), stopgrad(H_GP(b))`;
   - relation BCE trains the relation head but cannot update GP adapter, shared Q or carrier;
   - GM/GP continuous losses still train the full geometry path normally.

3. `PRIVATE_REL_ADAPTER_FULL_GRAD`
   - add one shared relation-private low-rank residual adapter to each endpoint GP state before relation compression;
   - rank `16`, linear low-rank residual, initialized near identity;
   - relation gradient remains free to flow through its input into GP adapter/shared Q/carrier;
   - tests private relation capacity without gradient isolation.

4. `PRIVATE_REL_ADAPTER_STOPGRAD`
   - same rank-16 private relation adapter;
   - adapter input is detached from GP/shared geometry path;
   - relation BCE updates only relation-private adapter + relation head;
   - tests combined private capacity + explicit gradient ownership.

Reference arms:

5. `GM_SINGLE_ORACLE` — unchanged single-family GM reference.
6. `GP_SINGLE_ORACLE` — unchanged single-family GP reference.
7. `RANDOM_RELATION_FIREWALL` — use the selected best ownership form only after selection logic is applied; retrain with independent balanced random relation labels to detect leakage. This arm may not rescue a scientific arm.

No arm may alter GM/GP loss weights to win.

## Parameter fairness

Report exact trainable parameter count for every arm and the low-rank adapter overhead relative to `CURRENT_FULL_GRAD`.

The rank-16 relation-private adapter is admissible only if its trainable-parameter overhead is <=5% of `CURRENT_FULL_GRAD`. If overhead exceeds 5%, adapter-containing arms are invalid rather than silently receiving extra capacity authority.

## Primary metrics / unchanged scientific gates

For every scientific joint arm report:

- GM overall mean nRMSE;
- GP overall mean nRMSE;
- endpoint relation accuracy;
- relation target positive fraction;
- GM negative transfer relative to `GM_SINGLE_ORACLE`;
- GP negative transfer relative to `GP_SINGLE_ORACLE`;
- GM and GP surface-local mean nRMSE;
- exact parameter count.

An arm is **joint-valid** only if all are true:

- GM overall mean nRMSE <=0.15;
- GP overall mean nRMSE <=0.15;
- relation mean accuracy >=0.95;
- relation target positive fraction in `[0.35,0.65]`;
- GM negative transfer <=15%;
- GP negative transfer <=15%.

The selected arm must additionally pass:

- random-relation firewall accuracy in `[0.48,0.52]`;
- rolled scene-carrier local-error increase >=25%;
- low-rank parameter-overhead gate if adapter is present.

## Selection rule

Evaluate all four scientific arms before looking at the firewall outcome.

1. If `CURRENT_FULL_GRAD` is joint-valid, retain current behavior unless another arm reduces both GM and GP error by >=10% without relation degradation >0.01.
2. Otherwise select among joint-valid arms by lowest `max(GM negative transfer, GP negative transfer)`.
3. Tie within 0.01 absolute worst-transfer: prefer fewer trainable parameters.
4. If still tied: prefer the less restrictive gradient policy (`FULL_GRAD` over `STOPGRAD`).

After provisional selection, execute the random-relation firewall using the selected ownership/adapter form. Final selection requires all selected-arm ownership/control gates.

## Interpretation classes

- `OPTIMIZATION_COUPLING`: a stop-gradient arm closes the court while its corresponding full-gradient arm does not, and private capacity is not required.
- `PRIVATE_RELATION_CAPACITY`: private adapter full-gradient closes the court while no-adapter full-gradient and stop-gradient do not.
- `GRADIENT_OWNERSHIP_AND_PRIVATE_CAPACITY`: only `PRIVATE_REL_ADAPTER_STOPGRAD` closes the court.
- `CURRENT_COUPLING_NOT_CAUSAL`: current full-gradient closes/reproduces or isolation/adapters provide no material rescue.
- `DEEPER_REPRESENTATION_INTERFERENCE`: no scientific arm closes the unchanged gates.

## Authority

This court can select the **relation-to-geometry gradient ownership/private-adapter treatment only**. It cannot by itself authorize V3.1 freeze.

If a treatment closes all gates, rerun the full continuous joint system as V2 with that treatment plus all original controls before architecture freeze.

If none closes, do not weaken gates; move to a narrower shared-carrier/subspace or structured-reunion court.

No real-model training, Truth Builder adequacy, FIT8 authority, product authority, or final relation-loss choice is claimed.
