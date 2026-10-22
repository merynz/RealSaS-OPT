# Structural-Evidence A/B Directional Screen V1 — 2026-10-22

## Why this experiment exists

The parent Structural-Evidence A/B experiment was designed close to a qualification run: both sealed final checkpoints, all eight FIT subjects, strict greedy free-running to EOS, and downstream courts. That is appropriate for qualification, but it proved unnecessarily expensive for the immediate research decision: **should the next architecture iteration follow A (null presentation) or B (role-free presentation evidence)?**

The V9.6 execution exposed two distinct facts that make a smaller directional screen justified:

1. `A_NULL_PRESENTATION / Snake` reached the evaluator's operational `131072` token ceiling at `26008` faces without EOS. The model was still producing grammar-admissible topology. Therefore the stop is an operational truncation, not a scientifically valid EOS failure.
2. `A_NULL_PRESENTATION / Bat` later produced EOS, but codec decoding raised `TESSA_DECODE_DUPLICATE_FACE`. That is a real structural-invalid candidate. The evaluator incorrectly converted that scientific failure state into a process crash instead of recording it and continuing.

The parent training itself is complete. Re-training is neither needed nor authorized by this screen. The expensive part is autoregressive evaluation, so the correct intervention is to reduce the evaluation question rather than alter the trained models.

## Question

**Given the already sealed final A and B checkpoints, which arm is the better research direction to continue?**

This is deliberately narrower than the parent question. It does **not** establish FIT8 qualification, unseen generalization, product readiness, or canonical-M authority.

## Subjects

The primary screen uses exactly two subjects:

- **George** — deliberately chosen as the difficult/complex case.
- **Bat** — morphologically distinct and diagnostically informative because the prior A run exposed a duplicate-face EOS failure.

Bat selection is explicitly post-diagnostic. That makes this an engineering/research direction-selection screen, not an unbiased confirmatory test or a generalization claim.

## Measurements

Each sealed arm is evaluated on the same two subjects with two complementary probes.

### 1. Teacher-forced deterministic probe

Up to eight evenly spaced existing topology windows per subject are evaluated with no parameter updates. Metrics are the existing training objective components:

- total objective,
- token cross-entropy,
- coordinate metric,
- expected absolute coordinate error in bins,
- structural-special-token cross-entropy.

This is cheap and sensitive, but it cannot decide the experiment by itself because teacher forcing can hide autoregressive instability.

### 2. Exact bounded free-running probe

The same exact greedy, grammar-masked decoder is used with the reference-parity gate.

- Primary horizon: **16,384 new tokens**.
- Extension horizon: **32,768 new tokens**, only when the 16K result is inconclusive.
- Reaching the bounded horizon is **not** an EOS failure.
- Grammar dead-end, EOS codec-invalid, and EOS topology-court-invalid are typed hard-invalid states and are recorded rather than crashing the run.

The bounded probe records stop reason, EOS/codec validity, face/component/restart counts, tokens per face, and selected-token log probability.

## Directional decision rule

A hard-invalid state has precedence within a subject: if exactly one arm becomes hard-invalid and the other does not, the non-invalid arm wins that subject.

When validity class is tied, an arm wins a subject only when:

- its teacher-forced total objective is at least **1% lower**, and
- coordinate absolute-bin error does not regress by more than **3%**, and
- structural-special-token CE does not regress by more than **3%**.

At 16K:

- both subjects vote for the same arm → select that arm;
- one subject votes for an arm and the other ties → select only when two-subject macro total objective is also at least 1% better and there is no additional hard-invalid state;
- otherwise run the 32K extension.

At 32K the same rule is applied. If the two subjects are still split/tied, the result is **INCONCLUSIVE**. No threshold may be relaxed after seeing results.

## Interpretation boundary

`SELECT_A` means the current role-free presentation-conditioning formulation has not earned further architectural investment relative to the matched null baseline.

`SELECT_B` means the role-free/shared-presentation-evidence direction has earned continuation into the next research iteration. It does **not** mean B passed the original FIT8 qualification courts.

`INCONCLUSIVE` means exactly that; a separately preregistered tie-break or redesigned measurement is required.

Canonical machine-readable authority: `canonical/STRUCTURAL_EVIDENCE_AB_DIRECTIONAL_SCREEN_V1_20261022.json`.
