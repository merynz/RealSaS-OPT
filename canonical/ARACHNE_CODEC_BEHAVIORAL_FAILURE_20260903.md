# RealSaS — Arachne / SkinFieldCodec behavioral failure record

**Date:** 2026-09-03  
**Status:** `FAIL__FIRST_FAILING_LAYER_CODEC_A0__ROOT_CAUSE_UNRESOLVED`  
**Preregistration:** `canonical/ARACHNE_CODEC_BEHAVIORAL_PANEL_PREREG_20260903.md`  
**Workflow:** `33752571671`  
**Job:** `100639284377`  
**Runner region:** `eastus`

## 1. Epistemic rule

This is the first execution of the preregistered Arachne/Codec behavioral panel. Its witness definitions, seeds, thresholds, optimizer horizons and stable-PASS protocol were committed before this run and are now immutable for the repair cycle.

No threshold or witness may be changed to turn this failure into PASS.

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

Final step `1536`:
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

Final step `1536`:
- row-L1 p95 `0.7438475489616394` > frozen `0.05`;
- deformation ratio `0.24094338715076447` > frozen `0.05`;
- max simplex residual `1.1920928955078125e-07`;
- negative weights `0`.

The trace starts near row-L1 p95 `1.60`, deformation ratio `0.71`, improves early, then plateaus/oscillates for a long late interval around row-L1 p95 approximately `0.72–0.76` and deformation ratio approximately `0.23–0.25`.

Interpretation: this does not look like a simple near-threshold horizon miss. A generic Codec A0 representation, objective or optimization limitation is exposed on a sharper heterogeneous weight field.

## 6. Current causal hypotheses — NOT YET CONFIRMED

The following are diagnostics only until intervention distinguishes them:

1. **Encoder information bottleneck.** `encode_teacher_weights()` reduces each joint's entire surface weight field to pooled summaries before the latent MLP; sharp spatial field structure may be irrecoverably compressed.
2. **Decoder capacity / latent-field bottleneck.** Even oracle/free per-joint latents may be unable to drive the shared decoder to the required W field.
3. **Shared temperature floor/dynamics.** A global softmax temperature may limit sharp rows or interact with optimization.
4. **A0 objective geometry.** The weighted cross-entropy construction and deformation MSE may optimize a surrogate that stalls before the frozen dense-row tail ceiling.
5. **Optimizer dynamics.** Possible but not assumed; `sharp_fork_5` late plateau argues against claiming that more steps alone is the explanation.

No one of these is currently declared root cause.

## 7. Next causal diagnostics

Keep the frozen panel unchanged and run generic A/B probes that decompose A0:

### D1 — decoder-only oracle-latent capacity

Using the same synthetic conditioning and teacher W, replace the teacher encoder output by free trainable per-joint latent parameters while retaining the exact same decoder/softmax structure. Optimize decoder + free latents under the same reconstruction/deformation objectives.

- If this also fails sharply, encoder compression is not sufficient root cause; decoder/softmax/objective capacity remains implicated.
- If this passes while normal A0 fails, encoder-summary bottleneck becomes causally implicated.

### D2 — free pair-logit upper bound

Optimize unconstrained valid pair logits followed by the same row softmax. This establishes whether the frozen W/deformation thresholds are themselves feasible and verifies the metric/probe implementation. Expected mathematical capacity is exact up to optimization/numerics; failure would indicate a diagnostic/test bug.

### D3 — objective decomposition

On the same frozen witness, log reconstruction CE, L1, row-tail and deformation trajectories and compare gradients/consequences without changing the preregistered production objective.

### D4 — temperature telemetry/intervention

Record learned effective temperature and decoder logit span. Only if D1/D3 show a sharpness restriction should a generic temperature intervention be tested.

## 8. Repair boundary

Until Codec A0 closes:
- do not modify Arachne A1 architecture/loss;
- do not relax behavioral panel thresholds/seeds/horizons;
- do not use a real family for repair selection;
- do not refreeze architecture;
- do not authorize Family-1 selection.

Any source repair must be accompanied by a cause-level generic regression, then the original preregistered panel must rerun unchanged.
