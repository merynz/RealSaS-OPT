# RealSaS S0-B2 — Deterministic Normal Falsification Prereg V1

**Date:** 2026-08-22  
**Status:** `PREREG_FROZEN_BEFORE_QUALIFICATION_OPEN`  
**Authority:** research falsification only; not product qualification.

## Question

Can the downstream value of direct/exact surface normals be recovered deterministically from already-admitted surface geometry, or must normal information remain a direct learned IRIS output candidate?

This experiment tests **head assignment**, not whether normal information itself is useful; S0-B already supported normal information as downstream-useful.

## Source and fresh panel

Source bundle: `IRIS_M5_V18_76_PRODUCT_CORE_CORPUS_001.zip`  
Source bundle SHA-256 from frozen package report: `78be9007cd4e0df1deb64e2258dd090a11c54a55de649896c74ed3ed24253897`.

The bundle contains 792 `train` rows and no official validation rows. Families are unique at the row/sample level for this projected corpus slice.

Rows are sorted by `SHA256(sample_id)` and split:

- train: first 600;
- selection: next 96;
- qualification: final 96.

Frozen split hashes:

- train IDs: `ab09eccc375abac499e742ab26a22771969015dcaaf7f2eb1a8615ee188d4ac5`;
- selection IDs: `fbd2db34ed98b6e30ee10f89d2be20198e9c3fa4a80202ce64ff7da49e9aa0f2`;
- qualification IDs: `bd64de6914a1e00b6342617a6b689a7245a27185e121ee92fbf62f531dd334ce`.

The 96-row qualification panel must not be read for normal-method selection, architecture selection, checkpoint selection, hyperparameter selection or threshold tuning.

## Frozen deterministic normal derivation

`N_det(P)` is produced only from exact surface point positions; view provenance remains available to the downstream probe in every arm, so the N comparison is isolated.

Algorithm:

1. For each surface point, take 12 nearest neighbours in common/object-frame Euclidean geometry.
2. Fit a **distance-weighted local covariance** with weights `exp(-(d / median(d))^2)`.
3. Use the minimum-eigenvalue eigenvector as the unsigned local normal.
4. Orient the sign by local point-mass asymmetry: take 40 nearest neighbours, compute a Gaussian-weighted local neighbour centroid, and choose the sign that points **away** from that local mass centroid.
5. Normalize to unit length.

This method was selected on the 96-row selection surface only. Earlier naive 12-NN PCA remains historical S0-B evidence and is not reused as the strongest deterministic treatment.

No exact normals, rig truth, skin weights, authored IDs, Pose B, motion truth or deformation truth enter `N_det` construction.

## Fixed graph/provenance policy

Every comparison arm receives identical non-normal point slots:

- `P` common-frame XYZ;
- 6 deterministic P-derived local graph descriptors;
- 16 flattened ordered-view projected XY coordinates;
- 8 view-support bits.

Thus the only changed field among the three primary arms is the 3D normal slot.

Primary arms:

```text
B0  P + graph + provenance/support; normal slot = 0
B1  B0 + N_det(P)
B2  B0 + direct/exact N
```

## Fixed Arachne information probe

Contract: substrate + **GT product-core skeleton/parents/roles** -> dense skinning weights.

Point/joint pair feature size is frozen at 57D:

- 36D point slot: `P3 + N3 + graph6 + XY16 + V8`;
- 21D skeleton-relative slot: joint XYZ3, parent XYZ3, point-joint relative3, joint distance1, bone vector3, bone length1, point-parent relative3, parent distance1, root bit1, normalized role1, bias1.

Model: MLP `57 -> 96 -> 96 -> 1`, GELU; softmax across legal product-core controls.

Training:

- seed 1862;
- AdamW;
- LR `2e-3`;
- weight decay `1e-4`;
- 6 epochs;
- 96 deterministic-random surface points per row per epoch;
- same initialization seed per arm;
- checkpoint selected only by selection soft-label cross-entropy.

Qualification metrics: soft-label CE, weight MAE, top-1 owner agreement, GT mass in predicted top-4, and standardized deformation-probe mean/p95 errors using the frozen product hierarchy.

## Fixed oracle-count Geppetto information probe

Purpose: isolate joint-locus information content. **GT joint count is supplied as an experimental control.** This is not product Geppetto inference.

Point features: same frozen 36D point slot as above.

Model:

- point encoder `36 -> 48 -> 48`;
- 52 learned query embeddings, hidden dimension 48;
- query-to-point dot-product attention;
- query/context decoder `96 -> 48 -> 3`;
- only first `J` queries are active when GT count is `J`.

Training:

- seed 1862;
- AdamW;
- LR `1.5e-3`;
- weight decay `1e-4`;
- 6 epochs;
- 256 surface points per row;
- Hungarian matching to GT product-core joint XYZ for locus loss;
- same initialization seed per arm;
- checkpoint selected only by selection mean matched joint distance.

Qualification metrics: matched joint mean, family-p95 matched mean, joint-p95 mean, PCK@0.05 and PCK@0.08.

## Decision rule

The direct-N head is **not** promoted merely because exact N beats zero-normal B0.

Interpretation:

- If B1 closes the downstream gap to B2 to a practically negligible level on both fixed probes, normal computation stays deterministic-first in `SurfaceBuilder` and a permanent direct N head is not justified by S0.
- If B2 retains a material, reproducible advantage over B1, S0-C promotes **direct normal information** as a required IRIS output candidate, while retaining the contract-wide evidence-revision note.
- Mixed results are `INCONCLUSIVE`; do not force a binary head decision.

No final universal numerical non-inferiority margin is declared from the already-open S0-B pilot. Report paired deltas, family hard-tail and bootstrap uncertainty; classify conservatively.

## After this gate

After S0-B2, stop expanding output-head search unless this falsification exposes a major missing quantity. Record the current best-estimate RiggingSurface contract, then return research attention to the IRIS frontend/correspondence descriptor path (D1/D2 lineage, reciprocal/cycle and high-recall correspondence), consistent with the user-directed execution priority.