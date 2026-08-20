# Rank-2 / Identity R V3-A — Proposal-Level Global Address Assignment

Status: PRE-OUTCOME DEVELOPMENT DESIGN FREEZE

This is a development diagnostic on already-open e01. It is not fresh qualification.

## Question
Before triangulation/H, can the user's global relational-address R improve cross-pose identity/correspondence over descriptor-G alone on the *actual frozen per-view B proposal evidence*?

## Observable input
Replay the exact frozen R6.3 model/front door. For every output carrier i and every A-visible view v, persist exactly the coordinates and descriptor scores returned by the existing frozen `top4_with_scores(...)` call before `candidate_pool(...)` triangulation. No teacher data enters extraction.

Within a view, identical integer-pixel proposal coordinates from different carriers denote the same B proposal site.

## Per-view assignment problem
Rows: A-visible output carriers.
Columns: unique B proposal sites appearing in at least one row's frozen top4 list.
Allowed edge (i,s): site s occurs in i's frozen top4 proposals.
Constraint: each carrier chooses at most one real site and each real site is assigned to at most one carrier.

A private ABSTAIN dummy is available to each non-anchor carrier. Real proposal edges always have lower nominal cost than ABSTAIN; ABSTAIN exists only to keep a view feasible when the top4 graph has no full matching (the pre-outcome combinatorics audit found 63/64 views fully matchable; worst view 50/54).

## G unary
For each carrier row, rank its valid top4 proposal sites by frozen descriptor score descending. `G_rank` is normalized stable rank in [0,1]. No learned/tuned weight.

## Six high-confidence G anchors
For each visible carrier, compute descriptor margin `top1_score - top2_score`. Sort by descending margin with stable carrier-index tie break. Greedily take the first six carriers whose top1 B site is not already reserved by an earlier anchor. Anchor selection uses no teacher label.

The six selected anchor carrier->site pairs are fixed in the view assignment. If fewer than six unique top1 sites exist, the view is invalid for V3-A rather than changing the rule.

## R address
For every non-anchor carrier candidate site, compare its relation to all six fixed anchors against the A-pose relation of that carrier to the same six anchor carriers in the same view.

For each carrier-anchor pair use the structured 2D address components:
- signed dx, dy;
- radial distance;
- unit direction vector.

A and B relation vectors are each normalized by their own observation-derived per-view scene scale (median pairwise distance of the corresponding observable carrier/site configuration). Use Huber residual componentwise and mean across available anchor relations.

Within each carrier row, convert raw R residual to stable normalized rank in [0,1].

## Combined cost
`cost(i,s) = G_rank(i,s) + R_rank(i,s)`.

There is no lambda sweep. Both terms are row-rank normalized before addition.

Anchors are fixed. Solve the remaining sparse injective assignment by deterministic linear sum assignment. Missing edges are forbidden. Each non-anchor row has one private ABSTAIN column with cost 2.5 (> maximum real edge cost 2.0), so ABSTAIN is used only when real injective matching cannot cover the row under lower cost.

## Frozen outputs before truth
For every family/view persist:
- proposal coordinates/scores/valid mask and their content hash;
- chosen six anchors and anchor sites;
- G-only top1 site for every row;
- V3-A assigned site or ABSTAIN;
- assignment objective and abstain count.

Teacher truth remains closed until all 64 family-views are frozen.

## Evaluator-only truth metrics
After selection freeze, map the existing evaluator surface truth and project each mapped teacher B carrier into the view.

For each carrier/view where the evaluator target is applicable:
1. `oracle_best_top4`: which of that carrier's frozen top4 sites is geometrically closest to the projected teacher target;
2. G-top1 oracle-best hit rate;
3. V3-A oracle-best hit rate;
4. G-top1 pixel regret = selected distance - oracle-best distance;
5. V3-A pixel regret;
6. assignment ABSTAIN rate.

Report these metrics overall and post-hoc on the R6.3 hard-tail rank-3 subset. The hard-tail label is evaluator-only and never enters anchor/assignment construction.

## Decision rule
V3-A is directionally supported only if it improves oracle-best proposal selection and/or lowers median regret versus G-top1 on the rank-3 hard-tail subset **without materially worsening the overall applicable proposal population**. Exact numerical outcome is reported; no same-revision threshold/weight retuning is allowed.

If supported, V3-B will triangulate the frozen V3-A correspondences and evaluate geometry/mechanics. If not, V3-A is frozen as a failed formulation before any revision.

sealed21/external10 remain CLOSED.
