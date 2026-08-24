# IRIS Single-Pose V2 — Evaluation Contract

Status: `MINI_V1_FROZEN_BY_PREREG`
Date: 2026-08-24

The active mini authority is `MINI_EXTRACTABILITY_GENERALIZATION_PREREG_V1.md`. Metric definitions, membership, checkpoint selection and final TUNE criteria are frozen before scientific optimizer use.

## Geometry panel

P: object-space Euclidean mean/median/p90/p95. N: angular error degrees mean/median/p90/p95. U_geo: Spearman with P error and risk-coverage error AUC; full calibration belongs to its dedicated gate.

CI104 established exact-P representation sufficiency. It did not establish `track_n_view` as exact persistent-locus correspondence authority. Therefore N is an observation-local geometry target/diagnostic only in Mini V1 and is excluded from correspondence ranking and checkpoint selection.

Relative improvement from random initialization is diagnostic only; absolute quality is mandatory.

## Coarse persistence

Candidate universe is alpha-supported cells on the actual Z_coarse lattice after the known-camera coarse-row corridor. Report exact coarse-cell Recall@1/4/8/16/32 plus family tails. No normalized `TRUTH_TOL` surrogate.

## Predicted-P addressability

On the same observation-derived coarse universe, report P-nearest coarse-cell Recall@1/4/8. This is a predicted-P retrieval diagnostic, not the exact representation ceiling.

## Fine local panel

Two measurements are mandatory: oracle-basin local discriminability and end-to-end retained-basin refinement. Report native-authority pixel error and PCK. This separates coarse miss from local discrimination failure.

## Reciprocal/cycle

Reciprocal/cycle are deterministic support evidence on retained candidate sets, not singleton truth. Report support on retained truth candidates. No learned ambiguity head is mandatory in Mini V1.

## Set-valued ambiguity

Primary output remains top-k hypotheses. No singleton precision threshold authorizes scale-up in Mini V1.

## Style aggregation

cel_clean and ink_cel are the same physical assets under two styles. Report each separately and equal-macro. Asset count remains physical asset count, not assets x styles.

## Checkpoint selection

Do not select by mixed training loss. Mini V1 freezes an observable normalized key over P tail quality, Z_coarse high-recall containment, oracle-basin Z_fine local tail quality, family non-collapse and style non-collapse. N is explicitly absent after the CI104 normal-authority finding.

FIT_SELECT alone chooses the checkpoint. TUNE_FINAL is opened once only after checkpoint freeze.

## Non-claims

This evaluator does not prove final 1024 localization quality, autonomous source-track extraction, complete SurfaceBuilder, Geppetto/Arachne sufficiency, deformation-proof compilation or artist-domain robustness.
