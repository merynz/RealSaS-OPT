# Shared-Scene Premise Court V1 — preregistration

Status: SEALED BEFORE OUTCOME
Authority: research-only.

## Premise under test

The RealSaS hypothesis requires useful cross-modal coupling between geometry/mechanics/graphics evidence. It does **not** require that every evidence family be forced through one undifferentiated latent tensor.

This court therefore treats *degree and location of sharing* as an architectural variable.

## Matched-budget arms

All arms must use the same observations, teacher truth, sampling schedule, optimizer family, total learned parameter budget (within 2%), and downstream head capacity (within 2%). Token/carrier budgets are reported explicitly.

### S0 — INDEPENDENT
Two independent scene encoders/states:
- geometry+mechanics state `Z_gm`
- appearance+presentation state `Z_gp`
No cross-state communication before final task heads.

Purpose: negative-control for cross-modal sharing.

### S1 — LATE_FUSION
Independent `Z_gm` and `Z_gp`, with a budget-matched bidirectional fusion block only at local query / relation stage.

Purpose: test whether shared reasoning is useful without shared early representation.

### S2 — SHARED_TRUNK_PRIVATE_SUBSPACES
One shared scene trunk followed by two learned private subspaces for GM and GP, with optional cross-attention between subspaces.

Purpose: permit common factors while protecting family-specific information.

### S3 — FULL_SHARED_Z
Current strongest form of the hypothesis: one shared scene state `Z`; all local evidence comes from `Q(Z,x)` plus family-specific readouts.

Purpose: test whether maximal sharing is actually beneficial.

### S4 — SHARED_Z_PLUS_FAMILY_ADAPTERS
One shared `Z`, but each evidence family receives a small budget-matched adapter before its readout.

Purpose: distinguish a bad fully-shared readout geometry from a bad shared scene state.

## Truth decomposition required before real run

Truth families are grouped only for experimental analysis, not supplied as semantic labels to the student:
- geometry field / surface localization;
- mechanics / deformation response;
- appearance/source response;
- pair causal response;
- ray/compositing response;
- topology-useful presentation evidence.

The Teacher/Truth Builder may use privileged authored data to construct these targets. Student inputs never include privileged IDs or names.

## Measurements

For every arm:
1. held-out error per truth family;
2. calibration where probabilistic;
3. cross-family transfer: delta from single-family training to joint training;
4. worst-family degradation under joint training;
5. gradient cosine matrix between family losses at shared parameters;
6. representation ablation: zero/private-shuffle/cross-state-shuffle tests;
7. TESSA B-arm improvement using the produced presentation evidence;
8. Compiler court agreement after binding, when available.

## Negative-transfer gate

An arm is not preferred merely because its average loss is lower. It must not improve one family by sacrificing another family that is required for canonical downstream use.

For joint training define for family f:
`interference_f = metric_joint_f - metric_single_f` using an error-oriented metric.

A sharing arm is rejected as a default if any required family has >15% relative degradation versus its matched single-family oracle **without** a downstream TESSA/Compiler benefit that compensates under a preregistered product metric.

## Selection logic

- If S3 > S0/S1/S2/S4 across required truth and downstream metrics, maximal shared scene is supported.
- If S2 or S4 wins, the hypothesis survives but changes form: shared substrate is useful, undifferentiated sharing is not.
- If S1 wins, late cross-modal fusion is sufficient; early shared-Z premise is weakened.
- If S0 wins robustly and downstream structure gains disappear, the shared-scene premise is falsified for the tested evidence families.

## Two-stage execution

### Stage A — synthetic causal-factor court
Construct scenes with known shared factors plus deliberately private factors and conflicting nuisance variables. This validates that the court can detect both positive transfer and negative transfer.

### Stage B — authored causal truth court
Run only after Truth Builder produces qualified response truth. This is the scientific result that can select the RealSaS architecture.

## Non-claims

Synthetic success cannot establish the real shared-scene hypothesis. It only validates the apparatus and identifies obvious architectural pathologies before expensive Truth/FIT8 runs.
