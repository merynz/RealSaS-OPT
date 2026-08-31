# RealSaS — MWB-1 Identity / Subset Deterministic Baseline Closure — 2026-09-01

**Status:** `MWB1_CLOSED_PASS__REAL_IDENTITY_SUBSET_MESH_AND_EXACT_WEIGHT_COPY_BOUND_TO_CANONICAL_PUPPET_V2__MWB2_ELIGIBLE_NOT_OPENED`

## Authority

Preregistration:

- `canonical/MWB1_IDENTITY_SUBSET_BASELINE_PREREG_20260901.md`;
- `canonical/MWB1_IDENTITY_SUBSET_BASELINE_PREREG_AMENDMENT_20260901.md`.

Machine result:

- `experiments/mwb1_identity_subset_20260901/MWB1_IDENTITY_SUBSET_BASELINE_RESULT_V1.json`.

## Real witness

The experiment reused the sealed real consumer-interlock fixture.

- asset: `asset_551ea351b43a1787d0f55536`;
- fixture transport SHA-256: `81634b7db6dbae3f7cc30d7f6942ace9be841416dd1db833185884d3e10584c5`;
- admitted surface nodes: `512`;
- Compiler-qualified joints: `5`;
- Compiler-qualified skin rows: `512`.

No new corpus, teacher identity, hidden full mesh or learned output was introduced.

## Deterministic identity subset

The preregistered local selection rule chose:

- `S:0000:724df3a7b364`;
- `S:0437:767a0fd965e4`;
- `S:0321:fd0447c91433`.

Smallest common support view: `3`.

Triangle cross-product magnitude: `0.0025998059716087446` with surface bbox diagonal `1.4628259823441812`.

Every mesh vertex was identity-bound to one admitted surface node with coefficient exactly `1.0`; no vertex position was created by interpolation or camera geometry.

Camera geometry was not consumed. The preregistered null-camera-dependency lineage sentinel was used only for MWB-1.

## Exact skin binding

`identity_weight_copy_exact = true`.

The mesh binder copied the exact corresponding `QualifiedSkinIR` influences onto each identity-bound mesh vertex without new joint support or semantic skin synthesis.

## Canonical V2 lineage

Observed hashes:

- candidate lineage: `60bcb450c58ef3e829c2c4ec48149cba5623bec99d25bcd1a3465c2361ca14c5`;
- qualified mesh lineage: `39c5b2d842e600970defbe6cc805e4c377ade0bcbe6bb80372be6a15884ec851`;
- qualified mesh-skin lineage: `4f548afea2c7145af3997785baf7cfaac20c57f28a5e2a762bdb913f4305b77e`;
- `CanonicalPuppetGraph.v2` product-state hash: `bd45f7751a4ce2a48f4c549e922b69ae54dd648aba15a6257457c87d53b486e0`.

## Mutation / stale-proof results

Topology mutation:

- mesh hash changed: **true**;
- product-state hash changed: **true**;
- original proof rejected as stale: **true**.

Weight mutation (`delta=1e-6` on `MV:0000` while preserving legal support/simplex):

- mesh-skin hash changed: **true**;
- product-state hash changed: **true**;
- original proof rejected as stale: **true**.

## CI regression evidence

Final pre-closure code head `aee49f31939c6d80e740f1f49fc6d96627cfd8ed` passed:

- `mwb1-identity-subset-v1` run `33440420335`: **PASS**;
- `mwb0-typed-seam-v1` run `33440420240`: **PASS**;
- `consumer-interlock-v0` run `33440420323`: **PASS**;
- `consumer-coupling-probe-v1` run `33440420444`: **PASS**.

Thus MWB-1 did not regress the earlier typed seam or the pre-existing real Compiler consumer route.

## Permitted claim

> The current typed V2 product route can bind a real admitted identity-subset mesh and exact qualified skin rows with coherent mesh/weight lineage and fail-closed stale-proof invalidation.

## Forbidden claims

This closure does not establish:

- product-quality full-character mesh generation;
- complete observed-domain face coverage;
- CDT necessity or correctness in the current product route;
- local interpolation safety;
- Arachne field-to-mesh transfer quality;
- deformation non-inferiority;
- hidden-surface completion authority.

Scientific optimizer steps remain `0`; training remains unauthorized.

## Next

MWB-2 is now **eligible**, but not automatically opened.

Before MWB-2 execution, freeze the exact constrained local interpolation / candidate-topology policy and, if CDT is proposed, verify the historical exact-predicate CDT source provenance and typed compatibility rather than silently importing a weaker substitute.
