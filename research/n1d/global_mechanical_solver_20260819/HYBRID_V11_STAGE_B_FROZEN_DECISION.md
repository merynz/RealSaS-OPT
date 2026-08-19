# RealSaS N1D Hybrid V11 — Frozen Stage-B Decision

Date: 2026-08-19

Verdict: **`STAGE_B_FROZEN_QUALIFICATION_FAIL__NO_RETUNE`**

Protocol integrity:
- Stage-A PASS was frozen before Stage-B truth access.
- 12/12 Stage-B raster-only predictions were frozen and rehashed before truth opening.
- prediction freeze ledger SHA-256: `f05ac9d48029a4bd1d62a9ec10967978e524213cb20f2b25a757fad351b48c19`
- truth-open ledger SHA-256: `80b2b6b371a199b8023718a8ae83fa96084592399c7296255394e5093fa392c2`
- frozen Stage-B result SHA-256: `33d5f55e9f2f8e8619d0e76638283e8bd2388dd3ae4db9974d85ce8952c1d472`
- canonical decision SHA-256: `564194e02bd8dc206a53915f639e7c8b001d5db8fc7f303081afb59f40340d4d`
- frozen result bundle SHA-256: `3ede770da3789bbb066dfae4967907572250392b22c8693c1c836947a3a88dc1`

Frozen Stage-B metrics:
- F activity Spearman `0.433160` — FAIL vs `>=0.50`
- F kernel Spearman `0.634387` — PASS
- D tangent `0.061331` — PASS
- R differential `0.042740` — PASS
- G direction `0.899471` — PASS
- G line `0.088411` — PASS
- valid F episodes `7` — PASS
- G supported carriers `215` — PASS
- family G robustness `3/4` — PASS
- episode-index G robustness `2/3` — PASS

Separate qualification-design defect: all three 12907 Stage-B episodes (`e02/e04/e06`) are truth-stratum `near_zero`, while the evaluator excludes near-zero episodes from the primary aggregate. Therefore the preregistered primary-only `minimum_supported_G_families=4` requirement is structurally unattainable on this Stage-B panel: maximum possible primary supported family count is 3, independent of prediction quality. This does not erase the genuine F-activity failure.

No post-truth change may rewrite Hybrid V11 Stage-B as PASS. Any subsequent treatment is a new research line and needs a new qualification plan.
