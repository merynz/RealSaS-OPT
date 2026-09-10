# RealSaS — Arachne Information-Preservation Audit V1 — Phase 8 Post-A0 Zero-Optimizer Diagnostic Plan

**Date:** 2026-09-11  
**Status:** `PROSPECTIVE_DIAGNOSTIC_PLAN__PARTIAL_TOKEN_TRACE_ALREADY_OBSERVED__NO_OPTIMIZER__NO_A1_AUTHORIZATION`  
**Audit branch:** `audit/arachne-information-preservation-v1-20260910`  
**Execution barrier:** do not execute until the live 4/8/16/32 fixed-budget token experiment has completed and its immutable result/selected model artifacts are available.

## 0. Honesty boundary

This plan is written after partial current-experiment traces are known, including the completed 4-token and 8-token arms and partial 16-token trace. Therefore it is **not a blind preregistration of the original token experiment** and must never be described as one.

Its purpose is narrower: freeze the forensic questions and methods **before inspecting the final 16/32 outcomes or conducting post-run exploratory analyses**, so Phase 8 does not become unconstrained metric fishing.

No Phase-8 metric changes the already-preregistered token-capacity winner rule or the current FIT1 gate retroactively.

## 1. Inputs to bind at execution time

Execution must hash-bind at minimum:

- current token-capacity prereg SHA-256 `16e9aef4de9077c12637a8737ba6642d7ed30a9a29e7d5016dc9bc580ecf58ad`;
- exact final result JSON;
- exact trace JSON;
- exact rolling/final selected model artifact(s) available under the experiment storage policy;
- exact T1 parent SHA-256 `e28d9f3e2762724ef983e7e7b427340df1ef3c0dced33571cbe46c95f0ed0bf1`;
- exact FS1 cache SHA-256 `db87c42d65e777072b3a607178a2c7f19ab221a4969c380eac46070db2216edd`;
- exact QualifiedSkeletonIR SHA-256 `48754ad703c596ec9d332c6f733f1dd31e74d016ef15f3ce451263a724493992`;
- exact zero-surface SHA-256 `987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b`.

If the storage policy retains only one large winner model, diagnostics requiring a non-winner final model must be limited to metrics already persisted in the trace/result or rerun only under a separately declared reconstruction rule. Never silently recreate a different model state.

## 2. D1 — latent token-interface semantics

Question: is the selected A0 field-token interface ordered, set-like, prefix-ranked, redundant, or fragile?

Zero-optimizer probes per joint:

1. **Permutation sweep:** fixed latent tokens, deterministically permute token order; measure raw logits, normalized W row-L1 and deformation co-metric delta.
2. **Single-token ablation:** replace one token at a time by the frozen zero/reference vector; measure impact.
3. **Leave-k-out / prefix sweep:** where architecture permits, compare 1..K prefixes and complementary subsets without retraining.
4. **Duplicate-token test:** duplicate one token into another slot and measure effect.
5. **Token replacement/cross-joint donor test:** replace a joint token with another joint's token while condition/query state stays fixed; verify field identity remains causal.
6. **Encode repeatability:** repeated field encoding under eval mode must be deterministic within declared numerical tolerance.

Outputs:

- token permutation sensitivity matrix;
- per-token causal effect magnitude;
- effective token-count estimate;
- `ORDERED`, `SET_EQUIVARIANT/INVARIANT`, `PREFIX_RANKED`, or `MIXED/UNKNOWN` verdict.

This verdict is a **blocking input to A1 latent-loss design**. Do not choose positional latent L2/NLL vs set matching before D1.

## 3. D2 — raw scalar → normalized W information funnel

Question: where does the disjoint-holdout p95 improvement fail to appear in the current deformation co-metric?

For GSA950 and disjoint holdout separately, record at the same frozen state:

- per-joint pre-normalization sigmoid scalar predictions;
- raw scalar absolute error by joint;
- raw 22-field mass sum before normalization;
- normalized W;
- row-L1 mean/p50/p90/p95/p99/CVaR10;
- dominant accuracy/top3 inclusion;
- inactive mass / false-positive mass;
- teacher-support mass and missed support mass;
- normalization-induced row delta `||W_norm - W_raw||_1`;
- correlation between raw-mass error and normalized row error.

Stratify by:

- pure one-hot teacher rows;
- 2-way blend rows;
- 3+-way blend rows;
- active-support cardinality;
- high vs low normalization-induced delta.

Primary forensic distinction:

- `SCALAR_FIELD_ERROR_DOMINANT`;
- `CROSS_JOINT_NORMALIZATION/COMPETITION_DOMINANT`;
- `MIXED`.

No training change follows automatically.

## 4. D3 — deformation consequence attribution

Current metric uses fixed synthetic joint-index-dependent translations. D3 must leave that metric intact for historical comparison, then add independent diagnostic probe banks.

### Bank A — exact current translation probe

Reproduce current `PROBE_TRANSFORMS` bit-for-bit and verify persisted metric reproduction.

### Bank B — joint-permutation equivariance control

Permute joint serialization, W columns and probe-transform assignment consistently. Physical deformation ratio must remain invariant within tolerance. Then intentionally permute W columns without transforms as a negative control.

### Bank C — orthogonalized deterministic translation bank

Construct multiple fixed-seed translation banks designed to reduce near-collinearity/cancellation among joint displacement vectors. No optimizer.

### Bank D — parent-relative articulated rotation bank

Using the exact QualifiedSkeletonIR tree and joint rest positions, construct small deterministic rotations around selected parent/joint pivots. The transform construction must be independently unit-tested before metrics are interpreted.

For each bank record:

- teacher motion RMS;
- prediction-vs-teacher RMS;
- deformation error ratio;
- per-row squared-error contribution;
- per-joint / joint-pair contribution where decomposable;
- rank correlation with row-L1 error.

Do not promote Bank D as product motion proof; it is a sensitivity diagnostic unless separately bound to the Compiler motion-proof contract.

## 5. D4 — regional/component error census

Diagnostic-only source component labels may be used **only after prediction**, never as model inputs.

Frozen region groups:

- `Mage_Hat` nearest-source carriers: current diagnostic count 11;
- Spellbook + Spellbook_open nearest-source carriers: 5;
- 2H_Staff nearest-source carriers: 3;
- body/head/left-right arm/left-right leg nearest-source groups;
- nearest-component ambiguity margin strata;
- high-z head-owned slice as a separate geometry-only diagnostic;
- pure vs blended teacher rows.

Report GSA row errors and deformation contributions per region. For holdout, construct the same nearest-component diagnostic from source geometry only if the exact holdout points and source authority hashes reproduce; otherwise mark region unavailable.

`Mage_Hat`, `Staff`, etc. are source/teacher diagnostic labels, not shipping semantics.

## 6. D5 — rigid-attachment fidelity

For source components whose teacher semantics are exact one-hot attachment controls:

- handslot.l book rows;
- handslot.r staff/wand rows;
- head-bound hat rows;
- chest-bound cape where object identity can be resolved diagnostically;

measure:

- predicted parent-control mass;
- top1 attachment-control accuracy;
- false mass on unrelated joints;
- post-normalization drift;
- current and alternate probe deformation consequence.

This directly tests whether the learned field representation can reproduce rigid attachment behavior without requiring an explicit assembly schema.

## 7. D6 — A0 frozen-interface readiness report

The final Phase-8 report must separate:

- `FIT1_GSA_GATE_STATUS` from the original prereg;
- token-capacity causal verdict from the original prereg;
- latent-interface semantics from D1;
- disjoint-surface weight-transfer forensic from D2;
- deformation-probe sensitivity from D3;
- regional/attachment diagnostics from D4/D5;
- whether the exact selected A0 encoder/decoder/token cardinality is safe to freeze for A1.

A1 readiness does **not** require disjoint holdout <=0.05 unless a new preregistration explicitly says so.

The selected A0 interface may be declared `READY_TO_FREEZE_FOR_A1` only if:

1. the chosen interface has the required Mage/GSA stable FIT evidence or a separately authorized terminal stability closure;
2. D1 establishes how latent tokens must be matched/ordered;
3. exact encoder/decoder/config hashes are known;
4. no latent-interface bug or non-determinism is found;
5. the audit's P0 `S+Qualified G -> A1` information-preservation requirement is incorporated into the new A1 boundary contract.

## 8. No-retroactivity rule

D1-D6 are diagnostics. They may motivate a new experiment, but they may not:

- rewrite the token-capacity prereg;
- change the historical 4/8/16/32 winner rule;
- add a new current-run gate after seeing results;
- turn holdout into a shipping gate retroactively;
- authorize A1 optimizer steps by themselves.

**Execution remains blocked until the current token experiment completes.**
