# Structural Evidence A/B — FIT8 Teacher-Forced Directional Census V1

## Purpose

This experiment is a post-training **direction-selection screen** over the already sealed A/B checkpoints. It is not a replacement for the parent FIT8 qualification court and cannot authorize canonical M or product promotion.

The immediate research question is narrower than qualification: **which already-trained arm is the better direction to continue developing?**

## Why a new experiment is needed

The parent A/B experiment was designed close to qualification strength: full-capacity TESSA, FIT8, strict free-running generation, EOS requirements and downstream courts. That is appropriate for final qualification but unnecessarily expensive for the earlier directional decision.

The first optimized full free-running attempt established that autoregressive evaluation can dominate wall-clock cost. A/Snake reached the 131,072-token operational ceiling while still generating grammar-admissible structure. A/Bat reached EOS but decoded to a duplicate-face failure. Those observations are useful, but they also show that full-EOS FIT8 generation is a poor first-line tool for the simple A-vs-B direction question.

A cheaper George+Bat bounded screen was therefore preregistered and executed at 16K, with automatic 32K extension. It completed correctly but remained inconclusive: both subjects tied under the locked rule. The result was informative rather than failed infrastructure: Bat favored B on teacher-forced total objective but both arms were structurally invalid at EOS; George showed only a tiny A total-objective advantage while B improved coordinate error, and neither arm reached EOS by 32K.

The next information-efficient step is therefore a **full FIT8 teacher-forced census** using every available topology window for every subject. Training is not repeated. The exact sealed A/B final checkpoints are reused.

## Measurement design

For each of the eight FIT8 subjects and each arm:

1. Build the exact inference conditioning memory under the original A/B contract.
2. Evaluate **every topology teacher-forced window**. No window subsampling is allowed.
3. Aggregate the original training objective by valid-token weighting.
4. Record token cross-entropy, coordinate metric, coordinate expected absolute error, and structural-special-token cross-entropy as diagnostics.

The per-subject directional vote uses only **total objective**, because it is the scalar objective actually optimized by the matched A/B training. Diagnostic submetrics are reported but cannot override the vote after seeing results.

A subject votes for an arm only when that arm improves total objective by at least 1% relative to the other arm. Smaller differences are ties.

An arm is selected directly by the census only if both are true:

- it wins at least 5 of 8 subjects; and
- its macro mean total objective improves by at least 1% relative to the other arm.

This prevents a narrow subject-count majority with a poor macro result from becoming a directional selection.

## Horse tie-break

If the FIT8 census does not directly select A or B, the evaluator runs one short exact greedy grammar-masked free-run on **Horse** for each arm, capped at 16,384 new tokens.

Horse is used only as a structural-validity tie-break modality. A bounded horizon is not an EOS failure. A valid EOS or non-hard-invalid run may beat a hard-invalid candidate. Log probability, face count, component count and other continuous free-run diagnostics cannot by themselves select a winner.

If Horse does not provide a structural-validity separation, the correct result is `INCONCLUSIVE` rather than an invented winner.

## Scientific scope

This experiment executes zero optimizer steps, modifies no weights, changes no conditioning semantics, and changes no tokenizer or grammar. It answers only which sealed arm has the stronger directional evidence for continued research.

It does **not** claim:

- FIT8 qualification,
- unseen generalization,
- canonical-M authority,
- product promotion,
- or downstream presentation/motion success.

Those require separate qualification after a research direction is selected.
