# RealSaS — Arachne / SkinFieldCodec behavioral failure record

**Date:** 2026-09-03  
**Status:** `FAIL__FIRST_FAILING_LAYER_CODEC_A0__OBJECTIVE_BUG_REPAIRED__REPRESENTATION_CAUSE_UNRESOLVED`  
**Preregistration:** `canonical/ARACHNE_CODEC_BEHAVIORAL_PANEL_PREREG_20260903.md`  
**First workflow:** `33752571671`  
**First job:** `100639284377`  
**First runner region:** `eastus`

## 1. Epistemic rule

This is the first execution of the preregistered Arachne/Codec behavioral panel. Its witness definitions, seeds, thresholds, optimizer horizons and stable-PASS protocol were committed before this run and are now immutable for the repair cycle.

No threshold or witness may be changed to turn this failure into PASS.

Historical FAIL evidence in this record is never overwritten by later repairs. A later PASS answers a new proposition about repaired source.

## 2. Suite result

Source compilation: `PASS`.

Combined source + behavioral pytest result: `2 failed, 20 passed`.

Panel result:

`ARACHNE_CODEC_BEHAVIORAL_PANEL = FAIL`

The first failing layer is `SkinFieldCodec A0` for two witnesses. Arachne A1 is therefore not yet the repair target.

## 3. `chain_blend_3` — full-chain PASS

Seed: `20260921`.

Codec A0:
- sustained 3-check PASS at step `640`;
- row-L1 p95 `0.047307368367910385` <= frozen `0.05`;
- deformation ratio `0.012499332427978516` <= frozen `0.05`;
- max simplex residual `1.1920928955078125e-07`;
- negative weight count `0`.

Arachne A1 shipping chain:
- sustained 3-check PASS at step `320`;
- shipping path is `Arachne.propose -> Compiler.qualify_skin -> QualifiedSkinIR -> verified LBS`;
- qualified row-L1 p95 `0.05972529575228691` <= frozen `0.10`;
- qualified deformation ratio `0.019142622128129005` <= frozen `0.10`;
- Compiler total correction L1 `2.5319604889772324e-07` <= frozen `1e-5`;
- qualified row count `9`;
- max simplex residual `0`;
- negative weights `0`.

Interpretation: the complete Codec/Arachne/Compiler/LBS authority chain is executable and can satisfy the behavioral contract on at least one generic witness. A broad proposal/Compiler plumbing failure is therefore falsified.

## 4. `branch_blend_4` — FAIL_A0

Seed: `20260922`.

A1 was not run because A0 never produced three consecutive PASS evaluations.

First-run final step `1536`:
- row-L1 p95 `0.09382732957601547` > frozen `0.05`;
- deformation ratio `0.036894265562295914` <= frozen `0.05`;
- max simplex residual `5.960464477539063e-08`;
- negative weights `0`.

The trace improves strongly from initial row-L1 p95 above `1.4`, but never reaches the frozen row-tail ceiling. Late deformation consequence is already acceptable while the dense row-tail remains wrong.

Interpretation:

> `GOOD_DEFORMATION != SUFFICIENT_W_RECONSTRUCTION`

The behavioral panel correctly detects that the chosen probe bank can be compatible with materially different dense W rows. The row-tail criterion must remain; it may not be removed merely because deformation passes.

## 5. `sharp_fork_5` — FAIL_A0

Seed: `20260923`.

A1 was not run.

First-run final step `1536`:
- row-L1 p95 `0.7438475489616394` > frozen `0.05`;
- deformation ratio `0.24094338715076447` > frozen `0.05`;
- max simplex residual `1.1920928955078125e-07`;
- negative weights `0`.

The trace starts near row-L1 p95 `1.60`, deformation ratio `0.71`, improves early, then plateaus/oscillates for a long late interval around row-L1 p95 approximately `0.72–0.76` and deformation ratio approximately `0.23–0.25`.

Interpretation: this does not look like a simple near-threshold horizon miss. A generic Codec A0 representation, objective or optimization limitation is exposed on a sharper heterogeneous weight field.

## 6. Initial causal hypotheses

The following hypotheses were opened after the first FAIL:

1. **Encoder information bottleneck.** `encode_teacher_weights()` reduces each joint's entire surface weight field to pooled summaries before the latent MLP; sharp spatial field structure may be compressed.
2. **Decoder capacity / latent-field bottleneck.** Even oracle/free per-joint latents may be unable to drive the shared decoder to the required W field.
3. **Shared temperature floor/dynamics.** A global softmax temperature may limit sharp rows or interact with optimization.
4. **A0 objective geometry.** The weighted cross-entropy construction and deformation MSE may optimize a surrogate that stalls before the frozen dense-row tail ceiling.
5. **Optimizer dynamics.** Possible but not assumed; `sharp_fork_5` late plateau argues against claiming that more steps alone is the explanation.

## 7. Causal decomposition run

Workflow run `33753462948`, job `100642169460`, runner region `northcentralus` added diagnostics while leaving the frozen behavioral panel unchanged.

### D1 — exact pair-logit feasibility oracle

The exact valid pair logits `log(W_teacher)` followed by the same row softmax and verified LBS pass the frozen A0 behavior by construction and numerically:

- `branch_blend_4`: row-L1 p95 `9.295e-08`, deformation ratio `3.077e-05`;
- `sharp_fork_5`: row-L1 p95 `1.201e-07`, deformation ratio `2.622e-05`.

Therefore:

`FROZEN_A0_METRICS_AND_THRESHOLDS_FEASIBLE = TRUE`

The panel is not failing because the metric/probe implementation asks for an impossible weight field.

### D2 — free per-joint latent, same decoder

Teacher encoder was bypassed while retaining the same tiny Codec decoder/softmax and the same `latent_dim=8`.

Under the current Codec reconstruction + deformation objective:

- `branch_blend_4` achieved a best frozen-behavior PASS checkpoint: row-L1 p95 `0.0260529`, deformation ratio `0.00715175` at step `1440`;
- `sharp_fork_5` improved dramatically relative to normal A0 but remained FAIL: best/final row-L1 p95 `~0.10496`, deformation ratio `~0.02495` at step `1536`.

Interpretation:

- the current decoder/softmax has enough capacity for the branch witness when the pooled teacher encoder is removed;
- teacher-encoder compression is therefore **materially causal** for at least part of the failure;
- it is **not sufficient root cause** for the sharp witness because encoder bypass alone does not close the frozen row-tail criterion.

### D3 — teacher distribution telemetry

`branch_blend_4` contains no mixed active/inactive row at the `1e-3` active threshold.

`sharp_fork_5` contains `15` mixed active/inactive rows, but the maximum inactive teacher mass per row is only about `8.17e-4`.

This matters for interpreting the objective bug below: that bug is real, but it cannot explain the branch failure and its theoretical target shift is far smaller than the observed sharp plateau.

## 8. Confirmed independent objective contract bug

A family-independent exact-truth probe used teacher row:

`[0.6000, 0.3991, 0.0009]`

under historical `active_threshold=1e-3`, `active_weight=2.0`.

The old cross-entropy multiplied active classes by `3` but the inactive class by `1`. At exact teacher probabilities the measured pre-softmax gradient was:

`[-0.00036, -0.00023946, +0.00059946]`

with max absolute gradient `5.9946e-4`.

Therefore:

`EXACT_TEACHER_W_STATIONARY_UNDER_OLD_CODEC_CE = FALSE`

The old CE optimum is proportional to `class_multiplier * teacher`, not teacher itself. This is a generic loss-contract violation independent of any behavioral witness.

The theoretical teacher-to-old-CE-optimum L1 shift for that adversarial row is about `0.00119964`. That proves the contract bug, but it is too small and too selectively present to be declared the sole explanation of the observed panel failures.

### Objective repair

Commit `f0fe52ab625695d46bed7007acba39fe4cdfb248` changes active emphasis from per-class multipliers to one teacher-only scalar per simplex row. This preserves approximately the historical emphasis scale while preventing within-row target distortion.

Commit `9692ac12a44769212906616b6bca13861022d42c` converts the positive falsification probe into a permanent negative regression:

`exact teacher W -> cross-entropy pre-softmax gradient ~= 0`.

This repair closes only the truth-stationarity bug. It does **not** authorize declaring the A0 behavioral panel repaired.

## 9. Representation diagnostic now opened

The next causal question is whether the Codec decoder is being forced to rediscover point-to-control relations inefficiently from separate absolute surface/joint embeddings.

A diagnostic lane has been added without changing the frozen panel:

- same teacher encoder;
- same hidden dim `32`;
- same latent dim `8`;
- same A0 optimizer/horizon/objective;
- decoder receives only four additional generic analytic relation channels already represented in the canonical Arachne pair-geometry contract: `dx`, `dy`, `dz`, point-control distance.

This is a diagnostic, not yet a source repair. Its result will determine whether explicit relational geometry is causally justified.

## 10. Repair boundary

Until Codec A0 closes:
- do not modify Arachne A1 architecture/loss;
- do not relax behavioral panel thresholds/seeds/horizons;
- do not use a real family for repair selection;
- do not refreeze architecture;
- do not authorize Family-1 selection.

Any source repair must be accompanied by a cause-level generic regression, then the original preregistered panel must rerun unchanged.
