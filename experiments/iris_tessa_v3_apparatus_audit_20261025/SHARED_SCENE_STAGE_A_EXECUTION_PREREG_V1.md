# Shared-Scene Premise Court — Stage A Execution Preregistration V1

**Parent:** `SHARED_SCENE_PREMISE_PREREG_V1.md`  
**Status:** `SEALED_BEFORE_EXECUTION`  
**Authority:** synthetic apparatus / pre-Truth architecture evidence only.

## Purpose

Execute parent Stage A with exact constants. This court may reject obvious pre-Truth sharing pathologies or retain a provisional baseline. It cannot establish the real shared-scene hypothesis; authored causal truth Stage B remains mandatory.

## Synthetic causal factors

Observation vector has 16 channels:

- 4 shared causal inputs `c`;
- 4 GM-private inputs `g`;
- 4 GP-private inputs `p`;
- 4 nuisance channels `n`.

Teacher GM and GP targets are four continuous channels each. Both depend nonlinearly on the shared factors plus their own private factors; neither target depends causally on nuisance.

Training nuisance is spuriously correlated with mixtures of causal factors. Evaluation includes both IID nuisance and a shifted nuisance where those correlations are broken/reversed. This tests shortcut robustness in addition to ordinary transfer.

Train/eval coordinates and nuisance draws are disjoint.

## Arms

Parent S0–S4 are implemented with a common downstream readout shape. Total trainable parameter count must be within `2%` of S0. Exact counts are reported before optimization.

- `S0_INDEPENDENT`: two independent encoders.
- `S1_LATE_FUSION`: independent compact encoders; bidirectional fusion only immediately before readout.
- `S2_SHARED_TRUNK_PRIVATE`: one shared trunk plus GM/GP private subspaces.
- `S3_FULL_SHARED_Z`: one shared scene encoder, two readouts.
- `S4_SHARED_Z_FAMILY_ADAPTERS`: one shared scene encoder plus family residual adapters.

The implementation widths are fixed from parameter-count algebra before outcomes and may not be changed after execution starts.

## Optimization

Seeds: `731,947,1217,1597,2017`.

- AdamW, LR `1e-3`, weight decay `0`;
- 1400 joint-training steps;
- 900 single-family-oracle steps per family;
- batch 512;
- evaluation 16384 samples for IID and 16384 for nuisance-shift per seed;
- error metric: per-family MSE normalized by teacher-target variance.

## Measurements

For each arm/seed:

- GM and GP normalized held-out error, IID and shifted nuisance;
- relative joint-training degradation versus the same arm's GM-only / GP-only oracle;
- worst-family degradation;
- gradient cosine between GM and GP losses on genuinely shared parameters when such parameters exist;
- representation ablation: zero shared state where defined, and cross-sample family/private-state shuffle where defined;
- exact parameter count.

TESSA/Compiler downstream metrics are explicitly `NOT_AVAILABLE_PRE_TRUTH` in Stage A and are not fabricated.

## Gates

Validity:

- S0 joint IID mean normalized error must be `<=0.08` for both families, proving the teacher/task is learnable;
- random-target regression firewall mean normalized error must be `>=0.90`;
- all arm parameter counts must be within `2%` of S0.

A sharing arm is `NEGATIVE_TRANSFER_REJECTED_AS_DEFAULT` if either family has >`15%` relative IID degradation versus that arm's single-family oracle **or** >`20%` relative nuisance-shift degradation versus S0 on a required family without a compensating real downstream metric (none exists in Stage A).

Among non-rejected sharing arms, Stage A provisional preference is the Pareto set over:

1. worst-family IID error;
2. worst-family nuisance-shift error;
3. worst single-family interference;
4. parameter count.

S3 is retained as the current low-compute default only if it is non-rejected and no other sharing arm improves both worst IID and worst shifted error by `>=10%` relative. If S3 is rejected and S2/S4 is non-rejected, prefer the lower-error robust S2/S4 arm. If all sharing arms are rejected, fall back to S1 then S0 for V3.1 pre-Truth.

## Non-claims

No Stage A result authorizes final real GM/GP sharing, Truth139/FIT8 training, product activation, ModelRelease promotion, or a claim that V3.1 is scientifically final. Stage B authored causal truth is mandatory after Truth Builder exists.
