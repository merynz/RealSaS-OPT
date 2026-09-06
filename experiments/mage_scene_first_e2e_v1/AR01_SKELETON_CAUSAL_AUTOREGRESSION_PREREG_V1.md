# RealSaS — AR-01 Skeleton-Causal Autoregression Closure — Preregistration V1

**Date:** 2026-09-07  
**Status:** `PREREGISTERED_BEFORE_ANY_AR01_OPTIMIZER_STEP`  
**Gate:** `AR-01_SKELETON_CAUSAL_AUTOREGRESSION_CLOSURE`  
**Experiment branch:** `exp/geppetto-ar01-skeleton-causal-v1-20260907`  
**Branch base:** `d840d96ece57705e0f519d1650119db180d9f5c4` (`exp/lossless-rigging-evidence-v1-20260906`)  
**Main audit-hold authority:** `b7025a00bcc4b8b209b74704555cbc56dcdd5a0d`  
**AR-01 source SHA-256:** `6f52e168883c7385c62fb7705bf526c740ad980a5fb0eed79356cdfde65d540d`  
**AR-01 unit-test SHA-256:** `854240d0c2fbedfad7897615075317251fac396ec7e966a5e93d65ffcf220205`

## Scientific question

With the evidence seam, target, generation head, full-surface access, structural serialization,
optimizer policy, seed/backend, count and evaluation horizon fixed, does feeding the developing
skeleton's **joint geometry + parent geometry** into the next recurrent generation step causally
improve FIT1 reachability/retention and structural parent prediction?

This gate tests the missing RigAnything-class mechanism:

`previous generated/teacher mechanical structure -> next joint state`.

It does **not** test whether old hard-MAP feedback should be restored.

## Prior facts frozen before this experiment

1. Current canonical Geppetto V2 is latent-state recurrent. The emitted hard-MAP locus is not fed
   back into recurrence and parent logits are computed after the latent/locus sequence exists.
2. Historical hard-MAP feedback was deliberately removed because discrete multimodal crossover
   could rewrite all later anonymous states. This proves that feedback path brittle, not that safe
   structural feedback is unnecessary.
3. The repository already contains a broader C3 challenger with joint+parent geometry feedback,
   but C3 has not received a controlled scientific closure and bundles multiple mechanism changes.
4. Lossless V3 evidence preservation is required at the learned-consumer boundary.
5. The retired promotion rule “three consecutive PASS checks observed once” is invalid for late
   stability. AR-01 uses the repaired terminal-retention gate.
6. `MECHANICAL_SALIENCE_FUNCTIONAL_SIMPLIFICATION` is a separate neural responsibility and is
   explicitly out of scope here.
7. FIT1-specific optimizer/mechanism authority remains in its own existing prereg/result ledgers;
   AR-01 does not reopen or re-adjudicate those experiments.

## Causal-isolation design

There are exactly two scientific arms:

### AR0 — no mechanical feedback

- exact AR-01 shared class and parameterization;
- V3 lossless fixed-eight-view `(x,y,valid,support)` evidence side path;
- per-step full-surface cross-attention;
- direct three-mode continuous locus head;
- causal parent logits over prior generated controls;
- teacher tensors are supplied identically but are **forbidden from changing recurrent state**;
- `mechanical_feedback_enabled=False`.

### AR1 — skeleton-causal mechanical feedback

Everything in AR0 is identical, plus:

- training: current teacher joint geometry + teacher parent geometry are encoded into a residual
  mechanical-state update;
- inference/evaluation: current generated joint geometry + the model's predicted prior parent
  geometry are encoded into the same residual update;
- that mechanical state becomes the recurrent/history state from which later joints are generated;
- `mechanical_feedback_enabled=True`.

## Initialization control

AR0 and AR1 are the **same model instance/class**, not separately implemented architectures.

The final projection of the mechanical-feedback residual is zero-initialized. Therefore, before
training:

- AR0 and AR1 must have identical `state_dict` bytes;
- teacher-forced AR0 and AR1 position outputs must be bit-identical;
- teacher-forced AR0 and AR1 parent logits must be bit-identical.

This prevents random initialization of a new feedback branch from becoming a hidden causal variable.

## Mandatory causal preflights

No AR-01 optimizer step may occur unless all pass:

1. historical R0 parity preflight from the repaired Mage notebook;
2. embedded Mage fixture SHA authority;
3. exact structural serializer/source SHA authority;
4. exact serialized target SHA authority;
5. lossless V3 tensor witness authority;
6. AR0/AR1 same-class/same-state initialization equality;
7. per-step attention width = 950 surface tokens;
8. lossy corrected 24D raster-summary coordinates remain zero in the compatibility trunk;
9. gate-OFF invariance: after deterministically opening the feedback projection in a disposable probe copy, changing teacher joint/parent geometry must not alter later outputs when the gate is OFF;
10. gate-ON causality: the same perturbation must alter a later joint when the gate is ON;
11. feedback-gradient check: the feedback projection must receive zero/no gradient in AR0 and non-zero gradient in AR1 from a later-step locus loss;
12. strict causal parent mask: step `t` may score only parents `< t`.

Any failure is an apparatus/source-contract failure. Do not train around it.

## Frozen training target and order

- witness: frozen Mage FIT1 witness;
- generated controls: forced 41 for this architecture-responsibility closure;
- target: exact existing `canonical_structural_serialization_v1` result;
- parent-before-child target order is frozen;
- no sibling-order augmentation in AR-01;
- no geometry deduplication;
- no hidden cardinality repair;
- no compiler deform-node completion;
- no teacher/source mesh at inference.

Forced 41 is deliberate: native STOP/count is not the variable under test.

## Frozen loss

Both arms use exactly the same loss:

`L = L_geometry + 2.0 * L_parent + 1.0 * L_root`

where:

- `L_geometry` is the existing structural-slot direct three-mode FIT loss used by the repaired Mage/V3 notebooks;
- `L_parent` is cross-entropy over valid prior parent candidates in the frozen parent-before-child serialization, skipping roots;
- `L_root` is BCE on root logits against the serialized root mask.

Parent weight `2.0` matches the canonical Geppetto V2 parent-loss weight.

The feedback gate is the only arm-dependent term. No arm-specific loss weight exists.

## Teacher forcing and evaluation

Training uses teacher mechanical state only for the AR1 feedback path.

**All scientific evaluation is free-running**:

- generated current joint geometry is fed back;
- predicted parent among previous generated controls is fed back;
- no teacher geometry or teacher parent is available to recurrence at evaluation.

Teacher-forced metrics are diagnostic only and cannot satisfy a PASS gate.

## Frozen optimizer / runtime

Carry forward the repaired deterministic Mage policy identically to both arms:

- seed: `20260905`;
- AdamW;
- lr `3e-4`;
- weight decay `1e-4`;
- initial epsilon `1e-8`;
- stabilize epsilon `1e-4`;
- epsilon switch after 3 consecutive checks with slot-p95 <= `1.5 * capture_radius`;
- checks every 64 steps;
- checkpoints every 256 steps;
- mandatory final step 16,384;
- A100-SXM4-40GB parity runtime;
- deterministic algorithms ON;
- cuDNN benchmark OFF;
- TF32 OFF.

No new optimizer redesign is permitted inside AR-01.

## Persistent-stability gate

The historical one-time three-check latch is retired.

Each arm must run to 16,384 even if it enters the exact region earlier.

At every 64-step check record free-running:

- exact 31-bin multiplicity;
- outside-capture count;
- occupancy L1;
- slot MAE/P95/max;
- parent accuracy over non-root controls;
- root accuracy;
- optimizer phase/epsilon;
- parameter and gradient group norms;
- full-surface attention diagnostics;
- teacher-forced vs free-running exposure gap.

`terminal_structural_pass_now` requires all of:

- all 41 generated controls strictly inside the frozen capture radius;
- exact 31-bin multiplicity;
- parent accuracy = 1.0;
- root accuracy = 1.0.

`terminal_stability_pass` requires:

- final step = 16,384;
- final check is `terminal_structural_pass_now`;
- final contiguous structural-PASS streak >= 48 checks;
- therefore at least 3,072 terminal optimizer steps are directly observed.

The 48-check threshold is frozen from the independent historical B1s terminal streak of 53 checks.

## Decision operator

Both AR0 and AR1 always run. There is no staircase/early skip.

### `AR01_PASS_SKELETON_CAUSAL_RESCUE`

AR1 terminal-stability PASS and AR0 does not.

Interpretation: under this frozen FIT1 protocol, skeleton-causal mechanical feedback is a necessary
causal repair relative to the matched no-feedback arm. This authorizes preparation of a promotion/
refreeze package; it does not by itself establish family-disjoint generalization.

### `AR01_COMPATIBLE_NOT_NECESSARY_ON_FIT1`

Both AR0 and AR1 terminal-stability PASS.

Interpretation: AR feedback is compatible but FIT1 does not establish necessity. No necessity claim.
Promotion/refreeze requires a separate architecture judgment/transfer gate rather than outcome
preference.

### `AR01_FAIL_FEEDBACK_HARMS_FIT1`

AR0 terminal-stability PASS and AR1 does not.

Interpretation: the tested safe structural-feedback implementation is rejected for promotion.

### `AR01_NO_TERMINAL_CLOSURE`

Neither arm terminal-stability PASS.

Interpretation: no causal winner; diagnose the remaining instability without modifying this result.

No secondary metric may override these four primary outcomes.

## Non-claims

AR-01 does not claim:

- unseen-family generalization;
- native STOP/count closure;
- mechanical salience / rig simplification closure;
- Arachne/skinning closure;
- compiler product PASS;
- theoretical necessity of one AR implementation;
- superiority of teacher-forcing versus scheduled sampling;
- that RigAnything should be copied literally.

## Promotion rule

No automatic merge or refreeze is performed by the notebook.

Only `AR01_PASS_SKELETON_CAUSAL_RESCUE` may directly open the proposed
`promotion/refreeze` path from this gate. Promotion must atomically reconcile source, tests,
lossless consumer, result/prereg hashes, compiler compatibility, `CURRENT_STATE.md`, and the
supersession record for prior latent-only Geppetto architecture authority.
