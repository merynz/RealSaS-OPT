# V3-B — Frozen Proposal Assignment -> Geometry -> Mechanics

Status: PRE-OUTCOME DEVELOPMENT DESIGN FREEZE

Parent: V3-A directional pass, decision commit `cf5f4b425953750c726c1cc513ba2929d1bbe4fb`.

This is an e01 development diagnostic. V3-A assignments are immutable in V3-B.

## Question
Do the already-frozen V3-A proposal-level G+R injective correspondences improve the actual observable 3D world and downstream GFDR mechanics when fed through the existing frozen triangulation machinery?

## Frozen authorities
- R6.3 observable panel manifest SHA256: `b5158b4133f1b9b24d84fee30b9c64bcf5ac9a785215279d7733aa86afbeffef`.
- V3-A solver SHA256: `911cf2be0e06b502f7591f25168d5114a563324a276a3cf41392f9576e7566c1`.
- V3-A frozen selection SHA256: `82f7e934179bedb50146e60f85978567a1efcd81197acc1ab302ed68468d9b41`.
- V3-A selection authority commit: `f80b9ea1080e3e8782c47066fd0a4667d6196fcc`.

No V3-A anchor, assignment, score, abstention or proposal site may change.

## Candidate-pool parity gate
Before producing a V3-B world, replay the frozen original proposal evidence through the exact route module's existing `candidate_pool(...)` using the original `_model_context` `seedB` value. The resulting per-carrier pool must reproduce the persisted R6.3 `H_xyz / H_reproj_px / H_desc_score` population within numerical serialization tolerance and with identical quantized candidate keys/order semantics. This gate validates that the same triangulation implementation is being used.

The parity replay uses original `seedB` ONLY to verify historical H construction. It is not the V3-B selection fallback described below.

## V3-B per-carrier geometry construction
For each carrier i and view v:
- if frozen V3-A assigned a real proposal site to i in v, `cands[v]` contains exactly that assigned pixel site;
- if V3-A ABSTAINed i in v, or i is not A-visible in v, `cands[v]` is empty;
- `cscores[v]` is the maximum frozen top4 descriptor score attached to that exact assigned integer-pixel site for carrier i/view v. No score is re-estimated and no teacher is used.

Call the exact frozen route module:

`candidate_pool(cands, cscores, P_A[i], P_B_baseline[i])`

where `P_B_baseline[i]` is the frozen R6.3 observable final baseline point.

Using final baseline `P_B` as `seed_i` is an intentional preregistered safety fallback: V3-A correspondence is allowed to propose a better triangulated realization but is not allowed to force deletion of the already-observable baseline world. `P_A[i]` remains the second built-in candidate exactly as in frozen `candidate_pool`.

Select `pool[0]`, i.e. the candidate chosen by the frozen `candidate_pool` deterministic sort `(mean reprojection error, -mean descriptor score, xyz)`.

No new XYZ operation beyond the frozen `candidate_pool` implementation is permitted.

## Pre-truth frozen outputs
For all 8 families, before sidecar truth opens, persist:
- full 64x3 `P_B_V3B`;
- for every carrier: selected xyz, reprojection error, descriptor score, candidate-pool size, selected-is-baseline exact flag, distance from baseline;
- V3-A selection SHA and proposal-evidence content SHA;
- world content SHA.

All eight worlds must be frozen before evaluator truth opens.

## Observable mechanics construction
After world freeze, reconstruct V3-B `N_B / V_B` only through the frozen real observable decorator:
`decorate_pb_observable(root, family, e01, route, P_A, P_B_V3B)`.
Then compute frozen GFDR-V2. Teacher normals/visibility are forbidden.

## Evaluation
Reuse exactly the R6.3 evaluator mapping, reliable/active definitions, local-scale k=4, hard-tail definition, truth-block composite and material improvement/degradation thresholds.

The evaluator must reproduce the exact 23 R6.3 e01 hard-tail carriers before any interpretation.

Report:
1. all 23 hard-tail baseline vs V3-B geometry error and mechanics composite;
2. the 15 rank-3 hard-tail carriers separately;
3. the 8 rank-deficient hard-tail carriers separately;
4. all reliable+active carriers;
5. fraction of carriers/views/world points that remain exactly baseline;
6. material improvement/degradation and strict mechanics-equivalence counts.

## Decision
V3-B is directionally supported if frozen V3-A correspondences produce a meaningful mechanics/geometry improvement signal on the hard-tail without broad collateral degradation of reliable+active carriers. Exact results are reported; no threshold, fallback, assignment or weighting may change within V3-B after truth opens.

A negative V3-B result freezes this geometry-lift formulation as failed; it does not invalidate the V3-A identity result or prove information-theoretic impossibility.

sealed21/external10 remain CLOSED.
