# RealSaS S0-B Rigging-Substrate Pilot Report V1

**Date:** 2026-08-22  
**Status:** `S0_B_PILOT_COMPLETE__HEAD_DECISION_PARTIAL`  
**Scope:** open-development causal pilot; **not** product qualification; official 384-row M5 validation remained closed.

## Question

Given exact observation-equivalent surface geometry, which candidate substrate fields materially improve downstream rigging under fixed probes?

This tests the canonical S0 proposition:

> `HEADS = minimal downstream-sufficient observable quantities`.

It does **not** test whether IRIS can predict these fields from raster yet. It tests whether downstream rigging causally needs the information.

## Data / split

Source: `IRIS_M5_V18_76_PRODUCT_CORE_CORPUS_000.zip`  
SHA-256: `43a83db6afb206d0b240eba1acc1e3a474940e6d1b142e0f37cfbf31604977b5`

- 790 rows;
- 790 unique families;
- 512 surface points per row;
- deterministic family-disjoint split from SHA-256 ordering of `sample_id`;
- train 600;
- selection 95;
- qualification 95;
- official 384 M5 validation rows unopened.

Split hashes:

- train: `7e6ad2f93f7666d5a18c28c8c99946301d59932a066c98a99f13c75755dd47c9`;
- selection: `78d95eb6a19121557a7a3bb72e45d08d16a1e968372566788f5e8de67e589236`;
- qualification: `e4a36e683ba783c56293230089888a81ff7c54b0b66150d26186d788748be5d2`.

## Arms

| Arm | Substrate exposed |
|---|---|
| A0 | `P` only |
| A1 | `P + N_derived(P)` using deterministic kNN/PCA, `k=12` |
| A2 | `P + exact/direct N` |
| A3 | A2 + deterministic local graph descriptors |
| A4 | A3 + exact view visibility/support + provenance `XY` |

Every learned probe uses the same fixed feature dimensionality. Fields absent from an arm are zeroed, so model capacity and initialization remain matched.

## Deterministic normal audit

The A1 normal derivation is **not** a sufficiently faithful substitute for exact normals:

- oriented median angular error: **32.08°**;
- oriented p90: **154.18°**;
- oriented p95: **166.42°**;
- points with >90° orientation error: **26.44%**;
- sign-invariant median angular error: **23.13°**;
- sign-invariant p90: **70.51°**.

Therefore A1 tests one concrete deterministic reconstruction, not the theoretical best derivation from `P + provenance`.

## ArachneProbe — GT skeleton fixed

Probe contract: substrate + GT product skeleton/parents -> skinning weights.

Fixed model: padded 57D point/joint pair feature slots -> MLP `96 -> 96 -> 1`; softmax across legal controls.  
Training: seed `1862`, AdamW, LR `2e-3`, WD `1e-4`, 6 epochs, batch 24 rows, 96 points/row. BEST selected only by selection soft-label cross entropy. Qualification opened once after all arms were frozen.

### Qualification

| Arm | CE ↓ | weight MAE ↓ | top1 ↑ | GT mass in predicted top4 ↑ | deform mean ↓ | deform family-p95 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 1.318255 | 0.045887 | 0.614453 | 0.932155 | 0.002124 | 0.004661 |
| A1 | 1.307283 | 0.045600 | 0.615049 | 0.933326 | 0.002107 | 0.004562 |
| A2 | 1.269899 | 0.043967 | 0.636431 | 0.935987 | 0.002050 | 0.004493 |
| A3 | 1.266380 | 0.043743 | 0.640892 | 0.936106 | 0.002038 | 0.004510 |
| A4 | 1.281625 | 0.043790 | 0.642681 | 0.935084 | 0.002037 | 0.004554 |

### Matched causal deltas

- **A0 -> A1:** derived normals give a small but real CE gain: mean delta `-0.010973`, 95% bootstrap CI `[-0.0158, -0.0059]`.
- **A0 -> A2:** exact normals give a much larger gain: CE delta `-0.048357`, 95% CI about `[-0.060, -0.037]`; deformation mean delta `-0.0000744`, 95% CI about `[-0.000104, -0.000044]`.
- **A2 -> A3:** deterministic graph descriptors add a small further gain: CE delta `-0.003519`; deformation mean delta `-0.0000123`.
- **A3 -> A4:** support/provenance does **not** improve the fixed-GT-skeleton skinning objective. CE worsens on average and deformation is effectively unchanged.

### Arachne interpretation

1. `P` alone carries substantial skinning information.
2. Normal information is causally useful.
3. The tested `N_derived(P)` is not sufficient to close the gap to exact `N`.
4. Local graph structure should remain **deterministic-first**: it helps slightly, but nothing here requires a learned graph head.
5. View support/provenance is not needed as an extra Arachne skinning feature when `P/N` and the GT skeleton are already given.

## Geppetto probes

### Unconstrained set decoder

A 48-query learned set decoder was first trained to predict joint loci plus existence/count.

Result: joint-locus improved with training, but existence/count destabilized as queries spread to cover the skeleton; qualification count MAE remained very large. This probe is therefore marked:

`INVALID_FOR_HEAD_DECISION`

It must not be used to claim that a substrate field helps or does not help full skeleton generation.

### Oracle-count joint-locus isolation

To isolate *information content* from the failed count head, a second fixed probe receives GT joint count only as an experimental control and predicts joint locations from the substrate. This is **not a product Geppetto result**.

Fixed model: 36D point slots -> 48-query attention decoder, hidden 48.  
Training: seed `1862`, AdamW LR `1.5e-3`, WD `1e-4`, 8 epochs, batch 16 rows, 256 points/row.

| Arm | joint mean ↓ | family-p95 ↓ | joint p95 mean ↓ | PCK@0.05 ↑ | PCK@0.08 ↑ |
|---|---:|---:|---:|---:|---:|
| A0 | 0.128674 | 0.210421 | 0.253313 | 0.166913 | 0.358516 |
| A1 | 0.128534 | 0.214812 | 0.249572 | 0.167031 | 0.361194 |
| A2 | 0.126855 | 0.215006 | 0.247417 | 0.172651 | 0.362233 |
| A3 | 0.125293 | 0.206513 | 0.245912 | 0.177754 | 0.370615 |
| A4 | 0.118838 | 0.205213 | 0.231957 | 0.174391 | 0.405811 |

Key paired results:

- A0 -> A2 joint mean: `-0.001819`, 95% bootstrap CI about `[-0.00319, -0.00037]`.
- A3 -> A4 joint mean: `-0.006455`, 95% CI about `[-0.00928, -0.00372]`.
- A0 -> A4 joint mean: `-0.009836`, 95% CI about `[-0.01316, -0.00678]`.

Interpretation: exact normals add a small but reproducible joint-locus benefit; deterministic graph adds more; view support/provenance adds the largest additional benefit in this isolated joint-locus probe.

## Oracle-matched joint -> skin diagnostic

For a diagnostic only, oracle-count predicted joints were Hungarian-matched back to GT joint semantics, GT parents were retained, and the corresponding fixed ArachneProbe was run.

This is **not** a product end-to-end metric, but it tests whether better joint loci can translate into deformation.

- A0 deformation mean: `0.003522`;
- A4 deformation mean: `0.003319`;
- A0 -> A4 matched delta: `-0.000203`, 95% bootstrap CI about `[-0.000342, -0.000067]`, paired Wilcoxon `p≈0.0041`.

The direction is positive for the richer observation-grounded substrate.

## Decision

### SUPPORTED

- `P` is a core substrate primitive.
- **Normal information is downstream-useful.**
- Current naive deterministic `N_derived(P)` is materially weaker than exact `N`.
- Deterministic local surface graph features can add small value beyond exact normals.
- View support/provenance carries additional Geppetto-side joint-locus information.
- Exact mesh faces/source topology are not required by this pilot.

### NOT YET FROZEN

- **`N` as a separate learned IRIS head.** The information is required/useful, but S0 has only falsified one naive deterministic derivation. Before promoting a permanent `N` head, test the strongest deterministic reconstruction from `P + view provenance/correspondence`.
- Full Geppetto skeleton generation/count/topology.
- `U` and set-valued `H`; exact GT substrate does not instantiate prediction uncertainty or true observational multimodality.

## Architectural consequence now

```text
A×8
  ↓
IRIS
  ├─ P                     CORE
  ├─ N information         REQUIRED; direct-head vs derived still open
  ├─ persistence           REQUIRED capability
  └─ observation support / provenance preserved
  ↓
SurfaceBuilder
  ├─ deterministic graph / neighborhoods
  ├─ sheet/component logic
  ├─ derived local geometry
  └─ reprojection/cycle checks
  ↓
Geppetto
  ↓
Arachne
  ↓
Compiler
```

No evidence from S0-B justifies a learned adjacency/faces head.

## Next falsification

`S0-B2`: construct the strongest deterministic normal/surface derivation from `P + exact view provenance/correspondence` and compare it against direct/exact `N` under the same fixed Arachne and joint-locus probes.

If the gap to direct `N` survives, promote `N` as a required learned IRIS output. If the gap closes, keep normals in `SurfaceBuilder` and simplify IRIS.

Only after that should S0-C freeze the minimum `RiggingSurface` contract.
