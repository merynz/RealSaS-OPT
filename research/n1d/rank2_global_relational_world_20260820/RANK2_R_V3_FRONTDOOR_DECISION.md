# Rank-2 / Identity R V3 — Front-Door Decision Checkpoint

Status: DESIGN FRONTIER — SOLVER NOT YET FROZEN

## Evidence accumulated
1. Historical R solved rank-3 identity as a global relational-address / quadratic-assignment problem, not a local distance signature.
2. V0 (all-64 raw 3D relation preservation) failed badly.
3. V1 (all-64 intrinsic graph-path address) improved but still damaged many healthy carriers.
4. V2 correctly froze observation-derived rank-3 carriers and varied only 26 projectively rank-deficient carriers. This reduced collateral degradation dramatically, but produced 0/23 material hard-tail improvements.
5. V2 diagnosis: only 8/23 hard-tail witnesses are projectively rank-deficient; 15/23 are rank-3.
6. Post-hoc evaluator-only diagnostic: for 12/15 rank-3 hard-tail carriers, the teacher target is geometrically nearer to another existing observable baseline B carrier site than to P_B[i]. This strongly implicates cross-pose identity/correspondence, not projective null-space, as the dominant rank-3 failure mode.
7. A teacher-optimal pure one-to-one permutation of the 64 baseline B points alone fixes only a minority, so geometry realization must remain coupled to identity.

## Source-level finding
The frozen observable front door still contains the evidence needed by R *before* H is built:
- for every A carrier i and every visible B view v, `top4_with_scores(...)` returns up to four B-image proposal coordinates and descriptor scores;
- `candidate_pool(...)` then combines pairs of these per-view proposals, triangulates them, projects the resulting 3D hypothesis back to every usable view, picks nearest proposals, and finally discards the proposal provenance;
- the persisted R6.3 state retains only `H_xyz`, mean reprojection error and mean descriptor score.

Therefore current H is downstream of the identity/correspondence ambiguity that historical R was designed to solve.

## V3 architectural correction
Do not keep inventing a more complicated R energy *after* H_xyz.

V3 will expose and freeze the teacher-free per-view B proposal evidence before `candidate_pool` compression. Global R will operate at the identity/correspondence layer first. The intended hierarchy is:

`A carriers + per-view B proposal evidence -> G unary support + global relational-address R / assignment -> cross-pose correspondence -> G triangulation / H only for genuinely residual geometric ambiguity -> mechanics`

The final V3 solver form is intentionally NOT frozen by this checkpoint. First required diagnostic is to characterize proposal-site multiplicity, overlap, visibility and whether a global one-to-one/partial assignment is well-posed from the actual frozen top4 proposal evidence.

## Guards
- No teacher field may enter proposal extraction, clustering, assignment, R, triangulation or candidate generation.
- Teacher may only evaluate a frozen correspondence/world.
- Existing e01 truth is development-open; any V3 design selected using e01 outcomes is a development revision, not fresh qualification.
- sealed21/external10 remain CLOSED.
