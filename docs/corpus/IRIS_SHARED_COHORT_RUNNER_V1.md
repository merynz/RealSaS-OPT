# Shared IRIS cohort runner

The historical Knight post-SDF-head IRIS line is an apparatus and chronology reference,
not the generic authority for Fit8. Generic Fit8 evaluation is cohort-owned: all eight
subjects are equal witnesses under one shared formulation and no subject name may select
a production algorithm or qualification rule. Knight remains a separate regression witness.

The shared runner is a research execution boundary: it uses one image-only IRIS model,
one active optimizer state, and one deterministic cohort contract. It does not create
Product PASS, Fit8 PASS, unseen claims, or a second workflow/qualification authority.

## Inputs and forward contract

`tools/training/prepare_iris_cohort_v1.py` admits the eight controlled bind
observations, validates source/camera/raster/mask proofs, loads exact frozen
DINOv2-S, and materializes subject shards. Only RGB-derived DINO tokens, token
positions, and qualified cameras enter IRIS forward. Masks, source geometry, rigs,
weights, depth, R512 audit coordinates, and Stage13 outcomes are never forward
inputs.

Each subject owns immutable F* query pools and source-ray pools. Training and
heldout points are disjoint. The source mask labels silhouette rays; F* certifies
that the finite-depth renderer/sampler gives those frozen ray labels the correct
sign before training.

## Revision 3 — post-SDF Knight chronology

Revision 3 supersedes the earlier 48,000 sequential-update design. A subject
exposure is no longer treated as an Adam update. One **global optimizer step** means:

1. zero the shared gradients;
2. compute one subject loss for every cohort member;
3. backpropagate each loss with equal `1 / subject_count` weight;
4. clip the resulting shared gradient once;
5. call `optimizer.step()` once and `scheduler.step()` once.

With eight subjects, 19,200 global updates therefore contain 153,600 subject
exposures. Every subject is exposed exactly once per global update.

The stage schedule reproduces the preserved post-SDF Knight C lineage rather than
using the unrelated later 6,000-step demo intervention:

| Stage | Global steps | Objective | LR → eta-min | Heldout eval | Replay |
|---|---:|---|---|---:|---|
| `DIRECT_FSTAR_COLD` | 4,800 | direct F* Huber | `8e-5 → 4e-6` | 200 | rank at init, refresh every eval |
| `C_OBJECTIVE` | 4,800 | direct F* + source silhouette | `8e-5 → 4e-6` | 200 | rank once from selected direct start, then freeze |
| `C_LONG_HORIZON` | 9,600 | direct F* + source silhouette | `2e-5 → 1e-6` | 400 | keep the same fixed C replay bank |

Each stage starts a fresh AdamW/cosine optimizer, matching the historical Knight
experiments. Weight decay remains zero. The 32,768-point query mixture remains
16,384 global + 8,192 near-zero + 4,096 sign-straddle + 4,096 replay. The C ray
apparatus remains 256 rays/update, balanced across 8 views × 4 strata, with 128
depth samples and `tau = 1/512`. The historical C schedule seeds are restored:
query `260924101`, ray schedule `260924202`, ray-pool construction `260924203`.

The shared-cohort intervention is only the cohort dimension itself: subject pools
are independent and deterministic, all eight contribute equally to each global
update, and checkpoint selection uses a **worst-subject envelope** over the Knight
heldout F* axes. Cohort means are telemetry and cannot hide a failed subject.

## Revision 3.1 — rejection-pool engineering guard

Revision 3.1 changes no proposal distribution, F* acceptance semantics, pool size,
training objective, optimizer schedule, heldout selection rule, or subject weighting.
It only replaces the historical Knight-local 32,000,000-point rejection-sampler draw
ceiling with a Fit8 engineering ceiling of 536,870,912 points and emits progress every
8,388,608 draws. The wider ceiling is activated by the Fit8 policy; the pool helper's
historical default remains 32,000,000.

This guard exists because rejection rate is geometry-dependent. A subject exhausting a
Knight-sized compute ceiling before optimizer step 1 is an apparatus-capacity signal, not
a scientific reason to exclude that geometry. The same seed and acceptance rule must
produce byte-identical accepted straddle points whenever both ceilings are large enough;
that sequence parity is covered by contract test. A 512M exhaustion remains fail-closed
and must trigger apparatus redesign rather than another casual ceiling increase.

## Logging, evaluation, and checkpointing

`[TRAIN100]` is emitted every 100 global updates with shared mean/worst training
loss/MAE, silhouette telemetry, training wrong-sign/ray fractions, LR, elapsed/ETA,
and GPU memory. The persisted trace also contains per-subject window means.

At stage-specific evaluation intervals, every subject is evaluated on its complete
frozen heldout set. `[EVAL_SUBJECT]` reports sign disagreement, MAE, near-zero MAE,
p95 and RMSE. `[EVAL_SHARED]` reports cohort means plus the worst subject/metric
owners. Only these heldout F* metrics select checkpoints. R512 and Stage13 remain
strictly selection-blind.

Checkpoints contain the exact model, optimizer, scheduler, RNGs (including the
stateful direct-stage per-subject query RNGs), replay banks, stage state, partial
100-step log window, progress, and code/input/policy binding. Atomic payload write
precedes the latest pointer. Interrupted and uninterrupted training are tested to
produce byte-equal model state under the same inputs.

## Final surface court

After all optimizer stages close, the selected shared checkpoint is decoded for
every subject independently:

1. reproduce the prefit source-only R512 exterior census and require zero learned
   nonpositive certified exterior points;
2. materialize the learned zero level set with the existing exact-contained
   T512/sparse-T1024 decoder;
3. emit the downstream `SignedZeroSurface` (`zero_surface.npz`);
4. run unchanged P999 source-mask courts on all eight views of every subject.

The notebook prints `[R512]`, `[SIGNED_ZERO_SURFACE]`, `[P999]`, and
`[SUBJECT_COURT]` records so the pass-relevant quantities are visible in the run
log. Training loss or cohort-average metrics can never mint `PASS_RAW_COURT`.

Historical note: the preserved Knight C checkpoint `222350d1...` is the frozen
C/demo lineage used by later Knight work, but it was not itself a strict learned
IRIS R512+P999 scientific PASS. Fit8's all-subject raw court intentionally keeps
the stricter R512 + exact containment + P999 requirement rather than weakening it
to reproduce the historical fallback status.

## Notebook transport

`tools/notebooks/build_iris_cohort_v1.py` embeds the hash-bound RealSaS source
closure and input identities. It no longer embeds the multi-megabyte DINO Git tree
as base64 in a Colab code cell. The notebook shallow-checks out exact DINO revision
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`, verifies the public weight SHA, and
keeps all active work/cache under `/content`. This removes the prior Colab editor
lag without relaxing source authority.

Contract tests are loaded by verified file path instead of importing the generic
`tests` top-level package, avoiding Colab package-shadowing. A physical A100
preflight exercises all three training stages, all eight subjects, atomic
checkpoint restore and continuation before the production fit starts.

## Platform consumption

`shared_iris_execution_v1.py` validates the same joint-update semantics. For a
completed run it requires:

- one shared model and one shared optimizer authority;
- `actual_optimizer_steps == policy.steps`;
- every subject exposure count equals that global-step count;
- total subject exposures equal `subject_count * global_steps`;
- the three stage receipts are present in order;
- the final heldout evaluation contains every subject;
- no Fit8/product/unseen claim was minted by the trainer.

The platform owns the ResearchAttempt and canonical compiler run. At an external learned
stage boundary the same run may enter `WAITING_EXTERNAL_EVIDENCE`; A100/Colab is only a
compute worker. Exact checkpoint/result/receipt bytes are CAS-bound back to that stage and
the same run resumes. Import itself never mints qualification.

The imported checkpoint then continues through the existing Go/Compiler qualification
lane. The notebook is research execution bytes, not qualification state.
