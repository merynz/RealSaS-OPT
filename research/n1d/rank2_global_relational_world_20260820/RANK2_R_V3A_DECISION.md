# V3-A Decision — Proposal-Level Global R

Status: **DIRECTIONALLY SUPPORTED DEVELOPMENT PASS — PROCEED TO V3-B GEOMETRY**

V3-A moved R to the identity/correspondence layer before H triangulation, using the actual frozen per-view top4 B proposal graph.

## Result
- 64 family-views; 63/64 have full real top4 matching.
- 2356 visible carrier-view rows; only 4 ABSTAIN (0.17%).
- Overall G-top1 oracle-best proposal hit: **21.42%**.
- Overall V3-A G+R injective hit: **29.68%** (+8.26 percentage points).
- Overall median proposal regret: **1.727 -> 1.406 px**.
- Overall mean regret: **3.640 -> 2.569 px**.
- Paired applicable rows: **522 improve / 319 worsen / 1443 unchanged**.

Rank-3 hard-tail subset (15 carriers, 79 applicable carrier-view rows):
- oracle-best hit **22.78% -> 26.58%**;
- median regret **1.554 -> 1.520 px**;
- mean regret **4.665 -> 3.224 px**;
- carrier-mean direction: **6 improve / 6 worsen / 3 unchanged**.

Large corrections include `14404/carrier32` mean regret **19.45 -> 1.05 px** and `11032/carrier42` **6.51 -> 1.63 px**.

## Interpretation
This is the first rank-2-lineage revision where the global R idea improves the directly targeted identity/correspondence observable while also improving the overall population. Therefore the architectural move is supported: R belongs before proposal-to-H compression, not as a late XYZ regularizer.

The result is not a complete identity solution: hard-tail gains are heterogeneous, and six of fifteen rank-3 hard-tail carriers worsen at carrier-mean level. Do not tune V3-A in place.

## Next gate
V3-B freezes the exact V3-A assignments and asks whether those correspondences produce better geometry/mechanics when passed through the existing frozen triangulation machinery. No V3-A assignment, anchor, score, or threshold may change in V3-B.

V3-B should use the frozen `candidate_pool` geometry construction with each view reduced to the frozen V3-A assigned proposal (ABSTAIN -> no proposal), while retaining the frozen baseline P_B as the candidate_pool seed/fallback. The full resulting world must be frozen before teacher mechanics evaluation.

This e01 result is development-open evidence, not fresh qualification. sealed21/external10 remain CLOSED.
