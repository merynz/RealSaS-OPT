# V5 — Full-Pairwise Proposal-Level Relational Address

Status: **PRE-OUTCOME DEVELOPMENT DESIGN FREEZE**

Parent evidence:
- Historical rank-3 R succeeded as a whole-structure relational address / quadratic assignment rather than a local signature.
- V3-A demonstrated that proposal-level R contains real identity signal, but used only six high-confidence anchors as an approximation.
- V3-B/V3-C showed that proposal R must not be collapsed directly to one XYZ realization.
- V4 established that canonical G/R must remain a full feasible set plus distinct evidence/global coupling; no G/R-only pruning is authorized before F/D.

## Question
Does removing the six-anchor approximation and using the **entire same-view carrier relation structure** improve proposal identity beyond descriptor-only G and beyond V3-A anchor-R?

This experiment evaluates R only at proposal/correspondence level. It does not produce geometry and does not authorize any XYZ collapse.

## Inputs
Use the exact frozen V3-A proposal evidence on the same eight e01 development-open families:
- per carrier/view up to four real B proposal sites and frozen descriptor scores;
- observable A carrier projections `XY_A` and visibility `V_A`;
- no teacher information in forward selection.

## G unary
For each visible carrier i in view v, deduplicate its real proposal sites and define

`G_i(q) = stable_rank_desc(score_i(q)) in [0,1]`.

No learned/tuned coefficient is used.

## Full pairwise R factor
For every pair of visible carriers `(i,j)` and every distinct-site proposal pair `(q_i,q_j)`:

1. normalize A displacement `d_A = (XY_A[i]-XY_A[j]) / scene_scale_A`;
2. normalize B proposal displacement `d_B = (q_i-q_j) / scene_scale_B`;
3. construct the same signed/directional/radial relation residual family used by V3-A:
   `[(d_B-d_A), (|d_B|-|d_A|), (unit(d_B)-unit(d_A))]`;
4. reduce with the frozen Huber delta 0.25 to a raw relational discrepancy;
5. convert the raw discrepancies for that carrier-pair's complete real top-k x top-k support into a stable ascending rank in `[0,1]`.

This pair-rank is `R_ij(q_i,q_j)`. No carrier is privileged as an anchor in the R objective.

## Injectivity / abstention legality
The top-k proposal graph may not admit a full real injective assignment. V5 first computes the **maximum real matching cardinality K** for each view from the proposal graph only.

Every optimization state must contain exactly K real assignments; all remaining visible carriers are private ABSTAIN. Therefore R cannot improve its objective by arbitrarily abstaining and no abstain penalty is scientifically tuned.

## Global objective
For a legal state Z with exactly K real assignments:

`E_G(Z) = mean G_i(z_i)` over real-assigned carriers.

`E_R(Z) = mean R_ij(z_i,z_j)` over all pairs of real-assigned carriers.

`E(Z) = E_G(Z) + E_R(Z)`.

G and R are therefore equally weighted **mean rank evidences**, with no fitted lambda.

## Solver
The underlying problem is a constrained quadratic assignment and V5 does **not** claim exact global optimization.

Use a deterministic multi-start injective local solver:
- restart 0: lexicographic maximum-cardinality minimum-G Hungarian initialization;
- remaining restarts: deterministic seeded random maximum-cardinality Hungarian initializations;
- local descent accepts only strict decreases in the exact full-pair objective;
- allowed moves preserve maximum cardinality: move an assigned carrier to an unused candidate site, feasible pair-site swaps, and real/ABSTAIN exchanges when applicable;
- stable lexicographic tie breaking;
- freeze 16 restarts per family/view.

Report best objective, G/R components, restart agreement, and objective spread. Solver failure means this optimization/operationalization is insufficient, not that full-pair R is false.

## Pre-truth gates
Before evaluator truth opens:
1. exact V3-A proposal evidence content digest parity;
2. maximum matching cardinality independently reproduced;
3. injectivity and fixed-cardinality legality for every final state;
4. exact objective recomputation parity;
5. deterministic repeat on at least one family;
6. all 64 family-view assignments frozen with source SHA and proposal evidence digests.

## Evaluation
Reuse the V3-A proposal truth evaluator semantics on the frozen assignments:
- overall oracle-best proposal hit and regret;
- exact rank-3 hard-tail subset hit/regret;
- carrier-mean improve / same / worsen on the 15 rank-3 hard-tail carriers;
- compare descriptor G, V3-A anchor-R, and V5 full-pair R.

V5 is directionally supported only if the full-pair factor improves proposal identity/regret without relying on teacher forward inputs. It does not authorize geometry collapse.

If full-pair R regresses V3-A, the next hypothesis is not to abandon global R but to **gate pair factors by observable articulation/mechanical compatibility (F/D or equivalent)** so relations across genuine moving boundaries are not treated as rigidly invariant.

sealed21/external10 remain CLOSED.
