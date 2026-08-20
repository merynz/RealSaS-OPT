# V3-B Decision — Hard Proposal Collapse Rejected

Status: **CLOSED / FAIL GEOMETRY LIFT — V3-A IDENTITY SIGNAL SURVIVES**

V3-B froze the exact V3-A proposal assignments, reduced each carrier/view to the single assigned proposal, and passed those hard assignments through the frozen `candidate_pool` geometry construction before the unchanged R6.3 GFDR evaluator.

The R6.3 hard-tail population reproduced exactly: 23/23 witnesses.

## Result

- V3-B changed 505/512 world points by >1e-4; only 7/512 remained exactly baseline.
- Hard-tail geometry moved closer to teacher for 14/23 carriers, including 9/15 rank-3 hard-tail carriers.
- Hard-tail mechanical material improvement: **0/23**.
- Hard-tail mechanical material degradation: **16/23**.
- Hard-tail strict mechanics-equivalence: **0/23**.
- Median mechanics composite: **0.5342 baseline -> 0.8489 V3-B**.
- Reliable+active material degradation: **161/202**; material improvement: **0/202**.

Therefore the V3-B lift is rejected. The failure is not evidence against V3-A's proposal-level global R signal: V3-A independently improved the proposal identity target. The failure is caused by collapsing uncertainty too early: a globally preferred proposal was treated as certain, all alternative top-4 sites were discarded, and triangulation was forced to rebuild almost the entire world.

## Architectural conclusion

`R` belongs before proposal-to-H compression, but it must **reweight ambiguity rather than destroy it**.

The next formulation must preserve the top-k proposal support and transport global relational information into multi-view G as a score/confidence over alternatives. A proposal should become uniquely selected only when the global constrained objective actually makes it unique.

## V3-C authorization — global min-marginal R

For every carrier/view proposal q, define a global constrained min-marginal

`Delta_iv(q) = min_{Z: z_iv=q} E_{G+R}(Z) - min_Z E_{G+R}(Z)`

under the same-view injectivity and relational-address constraints. `Delta=0` identifies proposals that participate in a global optimum; positive Delta measures how much global consistency is lost by forcing that proposal.

V3-C may use these teacher-free global min-marginals to re-score the **existing top-4 proposal support** and/or the **existing frozen H candidates**, but it may not hard-delete all alternatives merely because one proposal wins the MAP assignment. The frozen baseline/no-change realization must remain available.

No teacher-created proposal/XYZ, training, retune, sealed21, or external10 access is authorized.
