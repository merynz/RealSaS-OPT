# V3-C — Soft Global Min-Marginal G/R Contract

Status: **PRE-OUTCOME DEVELOPMENT DESIGN FREEZE**

Parent evidence:
- V3-A proposal-level global R is directionally supported: proposal identity/regret improves overall and on the rank-3 hard tail.
- V3-B hard proposal collapse is rejected: reducing every view to one MAP proposal changes 505/512 world points and materially degrades mechanics.

## Scientific question
Can the global relational-address signal be transported into multi-view geometry **without destroying proposal ambiguity**, so that G retains its feasible set and R only disambiguates when the complete constrained system supports that choice?

## Frozen inputs
Use only:
- frozen R6.3 e01 observable states/H candidate sets;
- frozen V3-A top-4 proposal evidence;
- frozen observable A projections/visibility;
- V3-A G/R cost semantics and same-view injectivity;
- frozen baseline P_B as an explicit no-change option.

No teacher field is available before the final world selection is frozen.

## Layer 1 — global proposal min-marginals
For each family/view, reproduce the V3-A assignment problem exactly: six observation-derived anchors, descriptor G rank, relational-address R rank to those anchors, same-view injectivity, private ABSTAIN columns.

Let E* be the unconstrained optimum. For each non-anchor carrier i and each real top-4 proposal site q that is globally feasible under anchor reservation, compute the exact forced-assignment min-marginal:

`Delta_iv(q) = min_{Z: z_iv=q} E_V3A(Z) - E*`.

The forced solve must preserve all other assignment constraints. Delta is non-negative up to numerical tolerance. Proposal support is represented by stable ascending rank of Delta within that carrier/view; no learned/tuned lambda is introduced.

For the six high-confidence anchors, keep all observed top-4 proposal sites. Their support rank is the original descriptor G rank; anchors are reference observations, not hard geometry decisions in V3-C.

No real top-4 proposal is removed merely because it is not the MAP assignment, except a proposal that violates the frozen hard same-view injectivity/anchor reservation constraint.

## Layer 2 — transport proposal R support to frozen G geometry
Do not generate new XYZ.

For every carrier i, start from the complete frozen R6.3 H_i candidate set. Add the frozen baseline P_B[i] as an explicit no-change candidate if it is not already exactly present.

For each candidate p:
- G unary is the frozen observable geometry evidence: 0.75 * stable rank(reprojection error ascending) + 0.25 * stable rank(descriptor score descending), recomputed over the candidate domain including explicit baseline using the exact top-4 proposal evidence for baseline scoring when needed;
- project p into every A-visible view;
- in each view, map that projection to the nearest retained real top-4 proposal site for carrier i;
- R_raw(p) is the mean proposal support rank from Layer 1 across usable views;
- R_candidate_rank is stable ascending rank of R_raw over the candidate domain;
- G_candidate_rank is stable ascending rank of the G unary over the candidate domain.

Final per-carrier cost:

`E_i(p) = G_candidate_rank(p) + R_candidate_rank(p)`.

Choose the deterministic minimum with stable lowest-domain-index tie breaking. This equal-rank formulation mirrors V3-A's untuned G+R combination while preserving the full G candidate support.

## Critical semantics
- G owns proposal/geometry feasibility and remains set-valued when observations do not determine a point.
- R is a global relational compatibility over the complete same-view assignment, represented as min-marginal support; it does not create XYZ and does not collapse top-k ambiguity before G triangulation.
- Rank-3 carriers are allowed to change because V2 showed most hard-tail failures are rank-3 upstream correspondence failures; however the frozen baseline is always an explicit candidate.

## Solver / determinism gates
Before truth:
1. exact V3-A base assignment replay parity;
2. all finite proposal min-marginals >= -1e-10;
3. forced-assignment solver determinism;
4. no new XYZ beyond frozen H plus exact baseline P_B;
5. exact selected candidate indices/coordinates frozen for all 8 families;
6. repeat solve exact on at least one family.

## Evaluation
After all 8 V3-C worlds are frozen, reuse the unchanged R6.3 mapping, hard-tail identity, observable N/V decorator, GFDR-V2 mechanics, truth-block composite, material thresholds and rank-3/rank-deficient split.

Hard-tail identity must reproduce exact 23/23 R6.3 witnesses before interpretation.

Primary outcome: V3-C should show a positive hard-tail geometry/mechanics signal without V3-B-style broad reliable+active collateral degradation. Exact results decide; no threshold or weighting may change after truth opens.

A negative result rejects this soft min-marginal lift, not the broader G/R representation and not information-theoretic observability.

sealed21/external10 remain CLOSED.
