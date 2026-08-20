# V4 Decision — Full Set Survives; G/R-Only Pruning Is Non-Canonical

Status: **V4 CLOSED AS REPRESENTATION DIAGNOSTIC**

V4 stopped forcing a single XYZ point. For every carrier it retained the complete frozen H support plus explicit baseline, attached independent G and global-R-derived evidence, and examined the non-dominated `(G_rank,R_rank)` frontier only as a diagnostic.

Results on the exact 23 R6.3 hard-tail witnesses:
- 77,926 pooled feasible candidates across 512 carriers;
- first G/R Pareto frontier: 1,525 candidates = 1.96% of support, mean 2.98/carrier;
- hard-tail frontier mean 2.91 candidates, max 7;
- a frontier candidate has lower truth-relative mechanics composite than baseline for 15/23 hard-tail carriers;
- median oracle-best frontier composite improves 0.5342 -> 0.5071;
- only 2/23 frontiers contain a material improvement and 0/23 contain a strict mechanics-equivalent alternative under the current single-carrier evaluator.

Therefore:
1. set-valued G/R is strongly preferable to premature point estimation;
2. Pareto compression is useful diagnostically/for acceleration, but **not authorized as an information-preserving reduction before F/D**;
3. the canonical representation must retain the full feasible geometry support with G and R as distinct evidence/factors;
4. R min-marginals are summaries of a global factor, not a replacement for that global coupling.

A candidate dominated in G and R alone cannot be discarded before F/D, because it may still dominate on mechanical/differential evidence. The eventual compiler must solve/quotient the joint GFDR world, not a G/R-only point or frontier.

Next R development: remove the six-anchor approximation at proposal level and test the user's original whole-structure relational-address idea as a full pairwise same-view factor. No geometry collapse is authorized by that diagnostic.
