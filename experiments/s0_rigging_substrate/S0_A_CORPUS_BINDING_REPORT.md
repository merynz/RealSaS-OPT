# RealSaS S0-A — Corpus Binding Report

**Date:** 2026-08-22  
**Status:** `S0_A_DATA_PLANE_PASS__S0_B_PROBE_IMPLEMENTATION_NEXT`  
**Scope:** byte/provenance binding for the leading S0-B observation → skeleton/skin/deformation corpus.

## 1. Decision

The existing RealSaS corpus is sufficient to begin S0-B fixed downstream probe implementation without constructing a new corpus first.

The key result is an exact sample-ID join between the verified V19.14 observation cache and M5 V18.76 product-core influence truth:

```text
V19.14 observation rows:     1408
M5 V18.76 rows:              3456
exact intersection:          1408
observation-only rows:          0
M5-only rows:                2048
intersection split:
  train                      1024
  validation                  384
```

All 1408 joined rows have source-availability audit status:

```text
available = true
arrays_present = true
manifest_present = true
```

Therefore the earlier statement that the observation cache exact-joins to M5 is now directly re-verified from the manifests rather than inherited from a report.

## 2. Bound authorities

### Observation cache

Drive file:
`V19_14_M1_OBSERVATION_CORE_IDENTITY_CACHE_MANIFEST.json`

- file SHA-256: `dfdc7421fa7d16516f08867211f3a812e0ec125b18435d1d8641f7ddd9051f9e`
- record count: `1408`
- split: `1024 train / 384 validation`
- shards: `59`
- manifest PASS: true

This agrees with the previously recorded canonical V19.14 manifest authority.

### M5 product-core influence manifest

Drive file:
`m5_product_core_influence_row_manifest.json`

- file SHA-256: `ce4d0ca2d1397700b96f54f7685d7f23fe0fb9c5b21ba6a053a2100d692810ff`
- records: `3456`
- split: `3072 train / 384 validation`
- surface point count: `512`
- manifest PASS: true
- blockers: none
- `product_core_influence_authority = true`
- `accepted_for_compiler = false`
- `training_only = true`
- `component_contract_status = pending_surface_connectivity_owner`

The last field is important for S0: exact/source surface connectivity is **not** already a settled product authority in this corpus. Surface graph/connectivity must be tested as a derived or learned substrate component rather than assumed.

### M5 source-availability audit

Drive file:
`M5_SOURCE_AVAILABILITY_AUDIT_V18_76.json`

- requested samples: `3456`
- available samples: `3456`
- missing sample IDs: `0`
- blockers: none
- PASS: true

For all 1408 joined observation rows, both privileged teacher arrays and privileged teacher manifests are recorded present.

## 3. Joined-row census

Across the 1408 exact observation↔M5 joined rows:

- surface point count: `512 / 512` on every row;
- M5 row validation: `1408 / 1408 PASS`;
- product-core control count: min `6`, max `61`, median `21`, mean `21.7095`;
- detail control count: min `6`, max `71`, median `28`, mean `28.7315`;
- deformation probe count:
  - `16` probes on `1395` rows;
  - `14` probes on `7` rows;
  - `12` probes on `6` rows;
- maximum joined-row mass-conservation error: `3.5762786865234375e-07`;
- maximum joined-row detail-reconstruction error: `2.9802322387695312e-08`.

These are data-plane facts, not product-quality claims.

## 4. Byte-level row witness

One real M5 row was downloaded from the Drive corpus and checked against its manifest entry:

```text
row file:
ffd5399eddff8c1d0973a9c120efc9a433b347f9426dc39499e7bc05de776db4.pt

manifest row SHA-256:
0a57b61cbf2ebd89c992cd0b6e6d61cd1fe2b45efbeca278744ffe66b9a7e7b0

observed file SHA-256:
0a57b61cbf2ebd89c992cd0b6e6d61cd1fe2b45efbeca278744ffe66b9a7e7b0

SHA parity: PASS
```

The payload contains the exact object S0-B needs:

```text
surface_points                       [512,3]
surface_normals                      [512,3]
view_point_xy01                      [8,512,2]
view_point_visibility                [8,512]

product_core_node_xyz                [J,3]
product_core_parent_index            [J]
product_core_role_index              [J]
product_core_surface_skin_weights    [512,J]
product_core_support_mask            [512,J]

product_core_top_index               [512,4]
product_core_top_weight              [512,4]
product_core_top_residual            [512]

detail_node_xyz
detail_parent_index
detail_node_to_product_core_index
detail_conditional_share_within_product_core

deformation_probes
source_truth
alignment_report
```

For the byte-audited witness, `J = 18` product-core controls and `25` detail controls.

This establishes that skeleton/hierarchy truth, dense skinning truth and standardized deformation truth are co-located with the same 512-point geometric substrate format used by the M5 corpus.

## 5. Consequences for S0 design

### 5.1 We do not need a new corpus before S0-B

The leading S0-B corpus can be constructed from the existing 1408 exact join.

### 5.2 GeppettoProbe and ArachneProbe can be causally separated

The row contract directly supports:

```text
GeppettoProbe:
  substrate -> product_core_node_xyz + parent_index (+ role diagnostics)

ArachneProbe:
  substrate + GT product_core skeleton -> product_core_surface_skin_weights

JointProbe:
  predicted skeleton -> predicted weights -> frozen deformation probes
```

This isolates substrate sufficiency for skeleton and weighting before product-model architecture is chosen.

### 5.3 Surface connectivity remains scientifically open

The M5 manifest explicitly reports:

`component_contract_status = pending_surface_connectivity_owner`

Therefore exact mesh/source topology must not be smuggled into S0-B as if already canonical. The first locality representation remains a deterministic SurfaceBuilder graph constructed from admitted geometry/support. Learned connectivity is authorized only if that derived graph fails a matched ablation.

### 5.4 The current 512-point carrier format is a natural first S0-B substrate

It aligns:

- observation geometry;
- normals;
- per-view projected coordinates/support;
- product-core skeleton truth;
- dense skinning truth;
- deformation probes.

S0-B may later test denser representations, but no density increase is needed to answer the first causal field-sufficiency question.

## 6. What this report does NOT prove

It does not prove:

- that `P/N/V/U` is the final IRIS head list;
- that 512 points are sufficient for final artist-domain quality;
- that deterministic surface connectivity will be sufficient;
- that Geppetto/Arachne product models will generalize;
- that the 1408 synthetic/projected corpus closes the authored-2D domain gap;
- that M5 rows are accepted for compiler/product authority.

It proves only that the data plane required to run a controlled S0-B substrate-ablation program exists and is provenance-bound.

## 7. Next executable step

Implement **fixed-capacity S0-B probe interfaces and loaders** against this 1408-row join:

1. `S0JoinedRow` loader exposing only the selected substrate arm plus separately gated teacher outputs.
2. `GeppettoProbe` fixed skeleton/hierarchy probe.
3. `ArachneProbe` fixed skinning probe conditioned on GT product-core skeleton.
4. `JointProbe` composition into the existing M5 deformation-probe evaluator.
5. tiny open-development parity/smoke run only; no confirmatory result until split, optimizer and numerical non-inferiority margins are frozen.
