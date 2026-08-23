# IRIS Single-Pose V2 — Evaluation Contract

Status: `AUDIT_CANDIDATE__THRESHOLDS_NOT_YET_PROMOTION_AUTHORITY`
Date: 2026-08-24

The next mini must measure the architecture actually implemented. Metric definitions are frozen before any optimizer run; promotion thresholds are frozen only after apparatus/data/memory preflight.

## Geometry panel

P: object-space Euclidean mean/median/p90/p95 and cross-view P spread mean/p95. N: angular error degrees median/p90/p95 and cross-view normal spread median/p95. U_geo: Spearman with P error and risk-coverage AUC; full calibration belongs to its dedicated gate.

Relative improvement from random initialization is diagnostic only; absolute quality is mandatory.

## Coarse persistence

Candidate universe is alpha-supported cells on the actual Z_coarse lattice after the known-camera coarse-row corridor. Report exact coarse-cell Recall@1/4/8/16/32 plus family tails/hard-tail slices. No normalized `TRUTH_TOL` surrogate.

## Predicted-P addressability

On the same observation-derived coarse universe, report P-nearest coarse-cell Recall@1/4/8/16/32. This is a predicted-P retrieval diagnostic, not the sparse/oracle representation ceiling.

## Fine local panel

Two measurements are mandatory:

1. **oracle-basin local discriminability:** condition on a coarse basin containing truth and test Z_fine only inside its local patch;
2. **end-to-end retained-basin refinement:** use actually admitted basins.

Both report native-authority pixel mean/median/p90/p95, PCK@2/4/8/16 and top-k containment. This separates coarse miss from local discrimination failure.

## Reciprocal/cycle

Report reciprocal top1/topK support, cycle support, localization error conditioned on support, and gain/regret versus identical candidate sets without consistency evidence. Every qualifier must report truth retention before and after it.

## Set-valued ambiguity

Before calibration, primary output is top-k hypotheses. Report Recall@K, risk-coverage, and any adaptive set-size distribution. No singleton precision threshold may authorize scale-up before a calibrated singleton policy is explicitly frozen.

## Style aggregation

cel_clean and ink_cel are the same physical assets under two styles. Report each style separately and macro-average across styles. Asset count remains physical asset count, not assets x styles. Checkpoint selection must use the same frozen style policy as final TUNE evaluation.

## Checkpoint selection

Do not select best checkpoint by mixed training-loss scalar alone. The mini prereg must freeze an observable lexicographic/normalized key including P/N tail quality, coarse high-recall containment, oracle-basin fine local tail quality, and family-tail non-collapse.

## Non-claims

This evaluator does not prove autonomous source-track extraction, complete SurfaceBuilder, Geppetto/Arachne sufficiency or artist-domain robustness.
