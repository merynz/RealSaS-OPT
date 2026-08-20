# RealSaS N1D Rank-2 Global Relational World — V0 Contract

Date: 2026-08-20
Parent authority: `research/n1d/observable_functional_audit_v2_20260820/R6_3_CLOSURE_AUTHORITY.json`
Parent status: `CANONICAL_CLOSED`

## Scientific question

Given the frozen observable R6.3 state for each family, with one observation-consistent candidate set `H_i` for each of the 64 persistent carriers, can a teacher-free global relational-address objective select one candidate per carrier such that the resulting 64-carrier world is globally coherent and closer to the mechanically correct functional world than the frozen local/baseline selection?

This is an open-development experiment. Historical Rank-3 R is a behavioral/design baseline, not a frozen formula that V0 must reproduce exactly.

## Historical design carried forward

The retained semantic invariant is: **R is a global address, not a local sorted-distance/kNN signature.** Historical Rank-3 R acted on the whole pairwise relational structure and was stabilized by observation-derived high-confidence G anchors. V0 lifts that idea from singleton points to set-valued candidate worlds.

## Frozen input population

- episode: `e01`
- families: `[09908,11032,12772,13203,14404,14702,14758,15290]`
- exactly the 8 R6.3 observable family states bound by `OBSERVABLE_PANEL_MANIFEST_R6_3.json`
- `sealed21=CLOSED`
- `external10=CLOSED`
- no training or retuning

## Forward/truth firewall

Solver forward may read only the frozen observable state arrays:

`P_A, P_B, N_A, N_B, V_A, V_B, XY_A, XY_B, H_xyz, H_reproj_px, H_desc_score, H_offsets`.

Teacher sidecar geometry, normals, visibility, skeleton, weights and intervention truth are **forbidden** inside candidate pruning, anchor selection, energy construction and optimization. Teacher may open only after a revision is source-frozen, to evaluate the selected world.

## V0 candidate reduction

For each carrier `i`, rank all candidates inside `H_i` using the already-frozen observable reduction

`g_i(h) = 0.75 * rank01(reprojection_error) + 0.25 * rank01(-descriptor_score)`.

Retain the best `K_i = min(32, max(8, ceil(0.10 * |H_i|)))` candidates. No truth-dependent candidate insertion is allowed.

## V0 observation-derived anchors

For each carrier, compute the covariance of its retained candidate coordinates around the coordinate-wise median. Define anchor sharpness from the maximum principal spread, with observable rank concentration as a secondary deterministic tie-break. Select the six sharpest carriers globally. Anchor identities and anchor candidate choices are fixed from observable evidence only.

## V0 relational kernel

For any carrier pair `(i,j)`, construct a normalized 3D pair relation from the selected coordinates in each pose:

- radial distance normalized by robust scene scale;
- signed x/y/z displacement normalized by the same scale;
- unit direction vector.

The relation discrepancy uses a robust Huber reduction. It is evaluated over the **entire 64-carrier pair graph**, not only local neighbours. Anchor-pair terms receive a fixed extra weight; they do not replace the full graph.

## V0 objective

For world assignment `c=(c_1,...,c_64)`, with `c_i` selecting one retained candidate in `H_i`:

`E_V0(c) = E_G(c) + lambda_R * E_R_global(c)`.

`E_G` is the normalized frozen observable candidate rank cost. `E_R_global` compares the complete pairwise relation kernel in Pose A against the candidate world in Pose B. V0 uses fixed `lambda_R=1.0` and anchor multiplier `2.0`; these values are frozen before truth evaluation and are not swept inside V0.

## Optimization

Deterministic coordinate descent over all 64 carriers plus 16 deterministic multi-start initializations generated from observable G ranks only. Ties choose the lower retained-candidate index. Stop at a fixed point or 50 full sweeps.

The output must report:

- selected original candidate index per carrier;
- objective decomposition `E_G/E_R/total`;
- restart agreement;
- best-vs-second-best restart margin;
- one-coordinate counterfactual minimum margin for the final world;
- whether the selected world is unique under the explored restarts.

## Evaluator-only metrics after source freeze

After V0 source/result-without-truth is frozen, evaluator may use the same R6.3 truth mapping machinery to report:

- selected geometry error over local truth scale, baseline vs V0;
- hard-tail witness geometry improvement/degradation;
- frozen GFDR-V2 truth-relative mechanical composite, baseline vs V0;
- material mechanical improvement/degradation using the R6.3 frozen thresholds;
- candidate-choice exact teacher-nearest rate as descriptive only.

Teacher-nearest geometry is never an optimization target.

## Interpretation

- V0 PASS signal: global R materially improves hard-tail mechanical truth and/or produces a stable unique world without teacher input.
- V0 failure does not falsify the R-address idea; it falsifies this first lifted kernel/objective and authorizes result-driven V1 design changes on this open-development panel.
- No result from e01 is product/sealed qualification.

## Persistence rule

Every material revision must persist source + machine-readable result + decision/continuity note to GitHub before the next revision is developed. Chat state is non-authoritative.