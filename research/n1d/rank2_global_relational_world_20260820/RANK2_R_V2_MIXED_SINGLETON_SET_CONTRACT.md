# Rank-2 Global Relational World V2 — Mixed Singleton/Set Contract

Status: PRE-OUTCOME DESIGN FREEZE

Parent evidence:
- R6.3 observable panel is canonical closed on untouched e01.
- V0 all-64 raw pairwise 3D relational invariance failed mechanically.
- V1 all-64 intrinsic graph-path address improved over V0 but still degraded too many mechanically healthy carriers.

## Scientific correction
The full 64-carrier optimization domain was over-broad. The historical structural G contract distinguishes projectively rank-3 carriers (singleton under the visible multi-view linear constraints) from rank-deficient/set-valued carriers.

V2 computes projective rank from observation-only visibility and the frozen canonical 8-yaw camera basis. For carrier i, stack the two orthographic projection rows (camera right and negative camera up) for every visible view. A carrier is FIXED iff this matrix has rank >=3 in both Pose A and Pose B. Otherwise it is VARIABLE.

No teacher field, sidecar geometry, joint, weight, parent, intervention metadata, or truth-relative score participates in this classification.

On the frozen R6.3 e01 observable panel this pre-outcome rule yields:
- 09908: 63 fixed / 1 variable
- 11032: 62 / 2
- 12772: 64 / 0
- 13203: 56 / 8
- 14404: 57 / 7
- 14702: 60 / 4
- 14758: 61 / 3
- 15290: 63 / 1
- pooled: 486 fixed / 26 variable

## World variables
For fixed rank-3 carriers, P_B is frozen exactly to the R6.3 observable baseline P_B.
For variable carriers only, choose c_i in the frozen observation-derived H_i candidate set after the same V1 observable pruning rule (top 10%, min 8, max 32 by 0.75 reprojection-rank + 0.25 descriptor-rank).

No new XYZ candidate may be created.

## R formulation
Use the V1 intrinsic observation graph idea, now in the correct mixed domain:
- graph topology: symmetric 8-NN graph constructed from observable P_A;
- anchors: ALL fixed rank-3 carriers, not an arbitrary top-k subset;
- address of every variable carrier: shortest-path distance to every fixed rank-3 anchor;
- reference address: Pose-A graph-path address;
- candidate-world address: same graph topology with fixed P_B nodes and selected H_i coordinates for variable nodes.

This preserves the user's global-address principle while allowing articulation: it compares global intrinsic position relative to the stable singleton substrate rather than raw 3D pose equality.

## Objective
E(W) = E_G(W)/s_G + E_R(W)/s_R.

E_G is the mean observable candidate cost over VARIABLE carriers only.
E_R is robust Huber loss over graph-path addresses from VARIABLE carriers to ALL FIXED anchors.
s_G and s_R are computed without truth from one-coordinate observable perturbations around the min-G mixed world, exactly as an energy-scale calibration rather than a tuned lambda.

## Solver
Deterministic coordinate descent with 12 frozen restarts and stable lowest-index tie breaking. Exact dynamic shortest-path updates are allowed only with numerical parity to brute-force Dijkstra.

Family with zero VARIABLE carriers returns the frozen R6.3 baseline world unchanged.

## Pre-truth gates
- camera/rank classifier deterministic and teacher-free;
- fixed/variable mask reproduced from frozen V_A/V_B only;
- source syntax/static firewall PASS;
- dynamic shortest-path parity <=1e-12;
- deterministic best-world repeat PASS on at least one variable family;
- exact selected H indices frozen before evaluator truth opens.

## Evaluation
Reuse the frozen R6.3 mapping, hard-tail witness definition, GFDR-V2 mechanics, truth-block composite, and material-improvement/degradation thresholds. Evaluate the FULL mixed V2 world, not single-node swaps.

The hard-tail population must reproduce exact R6.3 e01 witness identity (23 witnesses) before any scientific interpretation.

Primary question: does resolving only genuinely rank-deficient carriers under the global relational address preserve the already-observable rank-3 world while selecting a mechanically better functional realization of the set-valued tail?

No V2 outcome by itself is an information-theoretic impossibility proof.
