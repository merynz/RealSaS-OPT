# RealSaS — Geppetto V2 Behavioral Closure

**Date:** 2026-09-03  
**Status:** `PASS_GEPPETTO_V2_BEHAVIORAL_CLOSURE__REFREEZE_NOT_YET_SEALED`  
**Working branch at closure:** `behavioral/geppetto-v2-integrity-v1-20260903`  
**Frozen base:** `7f39a846ad05f91560836026e5ef6dfbc74dc731`  
**PR:** `#25`

## 1. Scope and scientific firewall

This closure covers generic Geppetto V2 implementation behavior from optimizer through shipping decode and actual Compiler qualification. It does **not** constitute formal Family-1 evidence and does not authorize family selection.

`rigxl_03490` is a development/integration witness only. It may reveal generic failures but it may never dictate a repair. No `rigxl_03490`, `J=67`, generation-limit `75`, real-family identifier, family-specific threshold, family-specific branch, or family-derived tuning constant was used in the repairs recorded here.

The IRIS privileged-input issue is separately scoped in `canonical/IRIS_LEAK_SCOPE_20260903.md`. Affected learned IRIS results remain quarantined and were not used as evidence for this Geppetto closure. The Geppetto behavioral witnesses are synthetic `RiggingSurfaceIR` witnesses and do not consume a learned IRIS checkpoint.

## 2. Epistemic correction: historical 3-control result

The historical 3-control witness must not be retrospectively relabeled.

- Under the then-active **teacher-exact topology** criterion, the case was **FAIL**.
- A later criterion changed product acceptance to anonymous canonical mechanical semantics. Under that later criterion the same witness can PASS.
- These are different propositions. The later PASS does not erase or reinterpret the earlier FAIL.
- Teacher-exact topology remains a diagnostic field, not current product acceptance authority.

This distinction is part of the permanent record because acceptance criteria were changed after a failure had been observed.

## 3. Current behavioral acceptance authority

For the small witness and preregistered heterogeneous panel, behavioral PASS requires all of the following for **three consecutive evaluation checks**:

1. shipping `generate()` returns the exact target control count;
2. shipping geometry has `matched_p95 < 0.5 * nearest_teacher_separation`;
3. the actual `propose() -> qualify_skeleton_v2()` path returns a mechanically qualified canonical graph;
4. canonical `MECHANICAL_STRUCTURE` semantics require:
   - `joint_count > 0`;
   - `deform_root_count > 0`;
   - `illegal_parent_count == 0`;
   - `unsupported_joint_count == 0`.

Teacher topology equality is logged only as a diagnostic.

Frozen optimizer protocol:

- AdamW
- LR `3e-4`
- weight decay `1e-4`
- maximum `2048` steps
- evaluation every `32` steps
- required stable PASS checks `3`

The preregistered witnesses, seeds, thresholds and protocol were not relaxed to obtain closure.

## 4. Failure chain and generic repairs

### 4.1 Shipping-MAP objective split — CONFIRMED, REPAIRED

Historical training could reward any mixture component while shipping emitted the MAP component. A low-confidence or secondary hypothesis could therefore explain teacher geometry without correcting the emitted locus.

Repair:

- direct shipping-MAP locus regression;
- explicit WTA-prepared secondary geometry;
- mode-ranking supervision;
- mixture NLL retained as uncertainty calibration.

### 4.2 Cardinality / STOP dilution — CONFIRMED, REPAIRED

Generation cardinality was controlled by first STOP crossing while training used a mean BCE in which one terminal positive could be diluted by sequence length.

Repair:

- STOP is sole cardinality authority;
- first-hit boundary loss gives terminal positive and worst earlier false-stop equal authority;
- existence is confidence evidence, not count authority.

### 4.3 Root / parent / support objective-consumer mismatch — CONFIRMED, REPAIRED

Sparse all-pairs BCE objectives did not match rank-based downstream consumers.

Repair:

- root pairwise ranking;
- parent per-child categorical one-parent ranking;
- support pairwise ranking aligned with top-k proposal consumption.

### 4.4 Hard-MAP recurrent feedback — CONFIRMED, REPAIRED

The emitted MAP position had been fed back into the autoregressive recurrence. A discrete mode-confidence crossover could therefore rewrite all later anonymous control states.

Repair:

- recurrence is latent-state-only;
- emitted MAP geometry is never recurrence input;
- source regression proves forcing a MAP switch changes emitted loci but leaves later `control_states` bit-exact.

### 4.5 Teacher root identity inside anonymous Hungarian correspondence — CONTRACT VIOLATION, REMOVED

Training matching had used:

`L1 geometry + 0.25 * teacher-root discrepancy`.

The root term could numerically dominate close spatial alternatives and, independently of empirical behavior, violated the contract that anonymous query↔teacher correspondence may not use teacher root identity.

Repair:

- `match_root_cost` removed entirely;
- root mask cannot affect anonymous matching.

### 4.6 Hungarian assignment cliff — CONFIRMED

Direct instrumentation logged assignment permutations and best-vs-second assignment margins.

Findings:

- float32 and float64 could choose different assignments near machine-scale margins;
- exact zero-margin assignment ties persisted in float64;
- therefore `FLOAT64_ALONE_FIXES_IT` is **FALSIFIED**;
- `DISCRETE_ASSIGNMENT_CLIFF_EXISTS` is **CONFIRMED**.

Permanent repair keeps hard assignment:

- L1 geometry is quantized to a fixed `2^-16` numerical identity grid;
- exact-primary ties receive a deterministic bounded secondary rank;
- total secondary authority is less than `0.25` of one primary geometry bin and therefore cannot overturn a one-bin geometric advantage;
- no soft assignment was introduced.

### 4.7 Teacher-row-order leak in deterministic tie-break — CONFIRMED, REPAIRED

The first deterministic tie-break used raw teacher column indices. This made the loss non-invariant to a permutation of teacher rows even though teacher row identity is not authoritative.

Repair:

- teacher columns are canonicalized from geometry before deterministic secondary ranking;
- assignments are mapped back to original teacher indices afterward;
- source regressions require teacher-row permutation invariance.

### 4.8 Train L1 vs eval L2 assignment mismatch — REAL, NOT SUFFICIENT

A causal diagnostic compared deterministic L1 and L2 matching on frozen `fork_5`.

- L1 assignment switching: approximately `222` switches;
- L2 assignment switching: approximately `77` switches;
- neither variant reached behavioral PASS.

Conclusion:

`L1_L2_MISMATCH_IS_SUFFICIENT_ROOT_CAUSE` = **FALSIFIED**.

The mismatch is real but was not sufficient to explain the failure.

### 4.9 MAP↔WTA multimodal ownership churn — REAL, NOT SUFFICIENT

Diagnostics compared current dual authority, no-mode-rank, MAP-owned and winner-owned variants.

MAP-owned training dramatically reduced MAP/WTA disagreement and mode switching, but `fork_5` still failed its frozen geometry gate.

Conclusion:

`MAP_WTA_OWNERSHIP_CHURN_IS_SUFFICIENT_ROOT_CAUSE` = **FALSIFIED**.

### 4.10 Uncertainty NLL leaked into shared latent state — CONFIRMED CAUSAL ROOT SEAM, REPAIRED

Although position means and mode logits were detached inside mixture NLL, `log_sigma = log_sigma_head(h)` allowed NLL gradients to flow backward through the sigma head into shared decoder latent `h`. That latent also drives locus, STOP, root, parent and support evidence.

Direct NLL-only gradient probe before repair:

- sigma-head gradient: nonzero;
- shared-latent gradient norm sum: approximately `10.55`;
- approximately `41` shared parameters received NLL gradient.

Causal A/B on frozen `fork_5`:

- current path: FAIL through step `2048`, final p95 approximately `0.1966 > 0.11824`;
- `position_nll = 0`: PASS at step `544`;
- sigma-head input detached while NLL retained: PASS at step `544`, final p95 approximately `0.04621`, while sigma NLL continued learning.

Conclusion:

`NLL_SHARED_LATENT_GRADIENT_LEAK_CAUSES_FORK_INSTABILITY` = **CONFIRMED** on the generic witness.

Permanent repair:

`mode_ls = self.log_sigma(h.detach())`

This changes the training gradient graph only. Sigma remains trainable and is still emitted/consumed as uncertainty and proposal confidence evidence. A source regression now requires NLL-only backward to produce nonzero sigma-head gradients and zero non-sigma model gradients.

## 5. Clean closure run

Source closure commit before freeze-workflow wiring:

`bb0b2eaa414507a1c41820ea451b5149964086ca`

GitHub Actions workflow:

- run `33703334408`
- first job `100487117556`
- runner region `chilecentral`
- result: **27 passed, 14 warnings**

Behavioral results:

| witness | stable PASS step | final count | final p95 | unique radius | mechanical status |
|---|---:|---:|---:|---:|---|
| small 3-control | 224 | 3 | 0.1832485199 | 0.1855348349 | roots=1, illegal=0, unsupported=0 |
| wave_chain_4 | 544 | 4 | 0.0927352309 | 0.1120960265 | roots=1, illegal=0, unsupported=0 |
| offset_star_4 | 288 | 4 | 0.0331314728 | 0.1544334143 | roots=1, illegal=0, unsupported=0 |
| fork_5 | 544 | 5 | 0.0462099798 | 0.1182369515 | roots=1, illegal=0, unsupported=0 |

All four reached three consecutive product-authoritative PASS checks.

## 6. Cross-region deterministic replay

The exact successful workflow job was rerun without source or commit changes.

- second job `100487622297`
- runner region `westus3`
- result: **27 passed, 14 warnings**

The pass steps and reported final witness values matched the first run:

- 3-control: `224`
- wave: `544`
- star: `288`
- fork: `544`

Thus:

`GEPPETTO_V2_CROSS_REGION_DETERMINISM = PASS`

for the unchanged closure source across `chilecentral` and `westus3`.

## 7. Architecture-freeze integration

Behavioral closure gates were promoted into `.github/workflows/architecture_freeze_source_gate.yml` as refreeze prerequisites, with deterministic environment controls:

- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `PYTHONHASHSEED=0`

The freeze gate now includes:

- IRIS source/generic-strength gates;
- Compiler proof-engine source gate;
- Geppetto source/generic-strength gates;
- Geppetto behavioral integrity;
- Geppetto behavioral overfit;
- Geppetto heterogeneous behavioral panel;
- Arachne/Codec source and generic-strength gates.

Integration commit:

`92cc2b9ffe5747d8107be16205b1b4945c392df0`

Architecture-freeze run:

- run `33703637411`
- job `100488030297`
- runner region `centralus`

Results:

- compile: PASS;
- family-independent/freeze-eligible source audit: PASS;
- expanded generic + behavioral prerequisite suite: **61 passed, 27 warnings**;
- candidate source count: `38`;
- candidate source fingerprint: `95b0c7670911468c4aafad600c96dae2fb86f715bdf8c7a149d1057332284e6e`;
- candidate status: `PASS_SOURCE_ELIGIBLE_FOR_FREEZE`;
- `family_selection_authorized = false`;
- final authorization check: expected FAIL with `FAMILY_SELECTION_BLOCKED__SOURCE_CHANGED_AFTER_FREEZE`.

Therefore the new fingerprint is a **refreeze candidate only**. It is **not a seal** and does not yet authorize Family-1 selection.

## 8. Closure decision

### PASS

- shipping geometry objective aligned with shipping MAP;
- dynamic cardinality/STOP semantics behaviorally exercised;
- root/parent/support rank semantics regression-covered;
- latent recurrence insulated from hard-MAP crossover;
- anonymous matching contains no teacher-root authority;
- hard matching is numerical-cliff resistant and teacher-row-order invariant;
- NLL uncertainty gradients are confined to sigma-head parameters;
- strengthened anonymous mechanical gate rejects unsupported joints;
- small witness sustained PASS;
- preregistered heterogeneous panel 3/3 sustained PASS;
- same source passed cross-region replay;
- behavioral gates are now architecture-freeze prerequisites.

### Still blocked / not claimed

- no formal untouched Family-1 evidence has been generated;
- `rigxl_03490` remains dev-only and cannot serve as formal evidence;
- architecture has not yet been refrozen to fingerprint `95b0c767...84e6`;
- family selection remains blocked;
- IRIS privileged-input repair remains open and affected learned IRIS evidence remains quarantined;
- this closure does not claim system-wide generalization.

## 9. Final status

`GEPPETTO_V2_BEHAVIORAL_CLOSURE = PASS`

`GEPPETTO_V2_CROSS_REGION_DETERMINISM = PASS`

`ARCHITECTURE_SOURCE_ELIGIBLE_FOR_REFREEZE = PASS`

`ARCHITECTURE_REFREEZE_SEALED = FALSE`

`FAMILY_SELECTION_AUTHORIZED = FALSE`

The next architectural action, after this closure record and PR metadata are synchronized, is an explicit refreeze decision. No family selection is permitted before that seal.