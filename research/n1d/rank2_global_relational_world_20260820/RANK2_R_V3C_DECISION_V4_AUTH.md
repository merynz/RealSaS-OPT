# V3-C Decision — Point Collapse Rejected; V4 Set-Valued G/R Authorized

Status: **V3-C CLOSED / FAIL POINT ESTIMATION — PROPOSAL-LEVEL R SIGNAL SURVIVES**

V3-C replaced V3-B hard MAP proposal collapse with exact global proposal min-marginals. The top-4 proposal support was preserved and global R was transported to frozen H candidates. However V3-C still made a final scalar/rank decision `argmin(rank_G + rank_R)` for each carrier.

The exact R6.3 hard-tail population reproduced 23/23.

## Result
- geometry moved closer to teacher on 17/23 hard-tail carriers;
- material mechanics improvement: 0/23;
- material mechanics degradation: 20/23;
- strict mechanics-equivalence: 0/23;
- median hard-tail composite: 0.5342 baseline -> 0.8752 V3-C;
- reliable+active degradation: 168/202; improvement: 0/202;
- 510/512 carrier points changed by >1e-4 despite explicit baseline candidates.

Thus the failure is no longer attributable to hard proposal collapse. The remaining invalid step is **forcing G and R into a point estimate before functional/mechanical evidence participates**.

## Representation conclusion
The evidence objects must remain distinct:

- `G_i`: complete observation-consistent geometry support H_i (plus explicit frozen baseline/no-change realization), with observable geometry evidence;
- `R`: global relational factor over proposal identities/correspondences. Its exact proposal min-marginals are transported to candidate geometry as relational support, but do not create geometry and do not force a point;
- downstream F/D/compiler may select a functional world only after considering mechanical evidence/global constraints.

## V4 authorization
V4 will freeze, for every candidate p in every carrier domain:
- exact xyz (existing frozen H or exact baseline P_B only),
- G evidence/rank,
- R min-marginal support/rank,
- deterministic Pareto layer in the 2-D evidence order `(G_rank, R_rank)`.

No `G+R` scalar lambda and no final XYZ top1 is permitted in V4 forward representation.

The first frontier is the set of candidates for which no other candidate is no-worse in both G and R and strictly better in at least one. Baseline remains explicit even if dominated and is separately marked as the safety realization.

After the full representation is frozen without truth, the evaluator may inspect hard-tail carriers and mechanically score candidates from the Pareto frontier (and baseline) using the unchanged observable decorator/GFDR evaluator. The diagnostic asks whether G/R has compressed the feasible family while retaining mechanically viable alternatives. It does NOT ask R to solve mechanics by itself.

A positive V4 result would support the formulation `G feasible set + R global relational posterior -> constrained functional compilation`. It would not yet qualify a deployable compiler selection rule.

No teacher-created candidate, learned score, retune, sealed21 or external10 access is authorized.
