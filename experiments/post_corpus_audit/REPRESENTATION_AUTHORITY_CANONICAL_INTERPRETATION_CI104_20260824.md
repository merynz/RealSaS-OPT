# Representation Authority — Canonical Interpretation — CI104 — 2026-08-24

Status: `P_GEOMETRY_SUFFICIENT__NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED__R4_NOT_OPENED__SOI2_NOT_OPENED`

This document interprets the frozen CI104 confirmatory measurement **without changing any preregistered measurement definition, score, threshold, panel membership, or result bytes**.

## Authority

- CI104 source head: `7d99eef46d2c071eca1d883e0c916bf4adecda35`
- GitHub Actions `IRIS V2 Preflight #104`, run ID `32679394221`: PASS
- frozen panel: 256 OPEN assets, 230 FIT / 26 TUNE
- panel asset-ID digest: `366b5fffb1ff93c1c7bbad0ac4746c4f2675a633ec01745c026cecb2b7820961`
- confirmatory queries: 12,288 per arm
- optimizer steps: 0
- sealed splits opened: false
- persisted result folder: `RealSaS_MASTER_CORPUS_1024_V3/runs/IRIS_SINGLE_POSE_V2_REPRESENTATION_AUTHORITY_V2_CI104_RESULT`

## Canonical scientific result

### R0 exact P

R0 is decisive for the immediate representation question.

- top1 = `1.000000`
- top4 = `1.000000`
- top8 = `1.000000`
- reciprocal = `1.000000`
- cycle = `1.000000`
- pooled physical-error median/p90/p95/max = `0 / 0 / 0 / 0`
- across 256 assets, family top1/top4/top8 p5/p10/median/p90/p95/max are all `1.0`

The legal SAME-locus ambiguity fraction is `0.2294108073`; this does **not** produce an R0 failure under the preregistered set-valued physical success definition.

Canonical label:

`P_GEOMETRY_SUFFICIENT`

Meaning: on the frozen controlled representation panel, exact common-frame surface position P is sufficient to identify the legal persistent physical-locus set under the current observable-substrate contract. No richer representation is required to explain an exact-representation failure, because no such R0 failure exists.

Therefore:

- R4 is **not opened** by this result;
- SOI-2 is **not opened** by this result;
- no information-limit claim is authorized;
- representation closure may proceed to learner extractability / mini-learning preparation.

## R2 — P perturbation robustness

R2 remains a valid controlled robustness diagnostic because it uses the same valid exact `track_p` authority and perturbs only P.

Pooled results:

| P sigma | top1 | top4 | top8 | reciprocal | cycle |
|---:|---:|---:|---:|---:|---:|
| 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.0005 | 0.9989 | 1.0000 | 1.0000 | 0.9982 | 0.9972 |
| 0.0010 | 0.9614 | 0.9998 | 1.0000 | 0.9564 | 0.9269 |
| 0.0025 | 0.6991 | 0.9740 | 0.9976 | 0.7528 | 0.6390 |
| 0.0050 | 0.4070 | 0.8096 | 0.9390 | 0.6052 | 0.4474 |
| 0.0100 | 0.1904 | 0.4843 | 0.6848 | 0.5340 | 0.3580 |

This curve is descriptive, not a post-hoc promotion threshold. It establishes a target regime for learner P accuracy and confirms graceful degradation rather than a binary information collapse.

## R1/R3 normal-arm authority finding

The measured R1/R3 numbers must **not** be interpreted as an exact-N sufficiency/failure result.

Frozen implementation inspection after opening the result shows an authority mismatch:

1. `track_p` is the exact persistent common-frame point used by R0/R2.
2. `track_n_view`, used as `n_exact` by R1/R3, is reconstructed separately in each view from the selected nearby raster visibility/surface witness `row[ok]`.
3. The witness is only required to lie within `max_surface_error=0.003` of the exact P locus; it is not the exact differential normal evaluated at the exact persistent locus.
4. R1/R3 rank with the frozen score `dP + 0.05*(1-cos N)`.

Consequently a view-specific witness-normal difference can outrank a candidate even when exact P alone has zero distance to the correct persistent locus. This is directly consistent with the observed R1 degradation:

- R0 exact P top1/top4/top8 = `1 / 1 / 1`
- R1 nominal `P+N exact` top1/top4/top8 = `0.96696 / 0.97884 / 0.98543`
- R1 reciprocal = `0.95980`
- R1 cycle = `0.94784`

The correct interpretation is **not** “P+N representation fails while P succeeds.” It is:

`NORMAL_CORRESPONDENCE_AUTHORITY_NOT_ESTABLISHED`

The R3 grid remains a diagnostic of this raster-witness normal field plus controlled perturbation, but it is not canonical evidence about exact surface-normal sufficiency. No R3 value may override the valid R0 P sufficiency result.

This does not block the active V2 architecture because the production matcher already assigns roles as:

- global basin admission: `Z_coarse + P`;
- local refinement: `Z_fine`;
- N: dense geometry/orientation evidence for the downstream substrate, not global correspondence admission authority.

If a future experiment needs N as correspondence authority, it must first bind N to the exact same persistent-locus authority as P (or explicitly preregister a different normal semantics). That is not required for the next learner gate.

## Ambiguity / reciprocal-cycle interpretation

The exact-P arm has legal ambiguity fraction `0.2294108073` yet achieves top1/top4/top8, reciprocal, and cycle all equal to 1.0 under physical set-valued truth.

This supports retaining the existing policy:

- preserve candidate sets;
- use reciprocal/cycle as deterministic support evidence;
- do not introduce a mandatory learned ambiguity/singleton head before evidence shows the current support/calibration machinery is insufficient.

It does **not** prove a future predicted-P matcher will be perfectly calibrated; that remains a learner/consumer question.

## What is now closed vs open

Closed:

- exact legal P representation contains sufficient persistent-locus information on the frozen controlled panel;
- no R4 feature search is justified by representation insufficiency;
- no SOI-2 / information-limit branch is justified;
- old hidden-owner/joint identity reconstruction is not required for this frontend target.

Still open:

- image -> P/N/Z extractability and generalization;
- predicted-P error distribution relative to the R2 robustness curve;
- learned `Z_coarse` high-recall containment;
- learned `Z_fine` local refinement;
- calibration of top-k + reciprocal/cycle on predicted evidence;
- SurfaceBuilder and downstream Geppetto sufficiency;
- product-domain appearance gap.

## Next authorized work

Training is still not automatically authorized by this document alone. Before the first optimizer run:

1. production-width GPU memory/throughput preflight with optimizer=0;
2. freeze mini FIT/TUNE membership;
3. freeze learner checkpoint-selection key and evaluator;
4. freeze style/appearance policy for the mini;
5. write the mini prereg;
6. only then open the learner optimizer.

No CAL, DEV, or EXTERNAL_HOLDOUT split is opened by this interpretation.
