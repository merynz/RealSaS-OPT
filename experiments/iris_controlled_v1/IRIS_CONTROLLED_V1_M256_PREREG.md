# IRIS Controlled V1 — M256 Full-Matcher Scale Gate Prereg

**Date:** 2026-08-23  
**Status:** FROZEN BEFORE M256 OPTIMIZER  
**Scope:** Open-split intermediate scale; no CAL/DEV/EXTERNAL access.

## Scientific question

After the 64-asset representation ceiling PASS and 51-FIT/13-TUNE learner pilot PASS, does the current **D3-free full IRIS evidence architecture** continue to learn and produce safe useful correspondence sets on a fresh 256-asset scale gate?

This is not product closure and not sealed generalization.

## Fresh M256 membership

- 208 FIT train
- 48 TUNE evaluation/model selection
- the deterministic 64 ceiling/pilot assets are excluded entirely
- selection is deterministic SHA order within the already frozen FIT/TUNE memberships
- no resplit and no sealed access

## Neural model

Unchanged `IRISControlledV1`:

- P
- geometric N
- U/log-risk
- Z_coarse
- Z_fine

Training uses corrected `train_iris_controlled_v1_v1_1.py` with invariant full-objective TUNE checkpoint selection.

M256 epochs: **12**. All other optimizer settings remain Controlled V1 values.

## Deterministic matcher V1 — NO D3

For each source observation and target view:

1. exact-camera row corridor defines the legal search domain (`|dy| <= 0.030` normalized);
2. retain coarse descriptor top-16;
3. union predicted-P nearest top-4;
4. within this retained pool, independently rank by coarse Z, fine Z, and predicted-P distance;
5. combine ranks with scale-free reciprocal-rank fusion (RRF, k=60);
6. keep final top-8 ordered candidates;
7. evaluate reciprocal top-1/top-4 consistency;
8. evaluate deterministic two-third-view cycle support;
9. use U only to **expand** the qualified output set under high relative risk; U cannot create a singleton;
10. output singleton only under strong consensus + reciprocal + cycle support, else top-4 or top-8 ambiguity set.

### Explicit exclusions

- no D3 learned local matcher;
- no hidden owner/joint/skin truth;
- no N coefficient in correspondence ranking in this gate;
- no TUNE-fitted matcher weights;
- no arbitrary weighted sum of P/Z scales.

N remains rich observable evidence for geometry/downstream use. Its correspondence contribution is deferred until calibrated conditional evidence is justified.

## Qualification policy

Singleton requires:

- reciprocal top-1;
- >=1 cycle support;
- >=2/3 agreement among coarse-top1, fine-top1, P-top1 with the fused top1;
- both endpoint U risks below the per-view 75th percentile.

Otherwise:

- very high risk (>=90th percentile) or reciprocal top-4 failure -> top-8 set;
- remaining ambiguous cases -> top-4 set.

No ambiguity case is silently forced to singleton.

## Evaluation

Evaluate the same fresh 48 TUNE assets at:

1. random initialization witness;
2. best 12-epoch checkpoint selected by invariant full TUNE objective.

Report:

- P Euclidean error and N error;
- raw coarse top1/top4/top8;
- raw fine top1/top4/top8;
- predicted-P top1/top4/top8;
- fused matcher top1/top4/top8;
- reciprocal rates;
- cycle support rate;
- singleton coverage and precision;
- top4/top8 set rates;
- mean output set size;
- output-set truth coverage;
- per-asset p10 tails;
- hard-tail examples.

## Frozen decision gates

### Learner gate

- TUNE P Euclidean error reduction vs init >= 50%;
- coarse top8 absolute gain vs init >= 0.40;
- fine top8 absolute gain vs init >= 0.40.

If this fails: `LEARNER_FAIL`.

### Matcher/safety gate

- truth row-domain recall >= 0.995;
- fused top8 >= 0.970;
- output-set truth coverage >= 0.970;
- p10 per-asset fused top8 >= 0.900;
- singleton coverage >= 0.100;
- singleton precision >= 0.950.

If learner passes but matcher/safety fails: `MATCHER_CONSUMER_FAIL` / `PARTIAL`, not an information-limit claim.

If both pass: `M256_PASS` and the next step is a larger open scale gate, not automatic sealed evaluation.

## Interpretation discipline

A high raw descriptor top-k is not the final matcher metric. A low raw descriptor top1 is not by itself a product failure. The relevant contract is whether the full deterministic evidence system produces high-coverage, high-precision singleton/set-valued surface correspondence on unseen TUNE families.
