# Truth Foundry V18 — Evidence authority correction and real-truth debt

**Status:** Independent exact-cell readback PASS, *authority reclassification only*. This audit **does not** fetch any new private source, prove creator-original bytes, establish subject identity/morphology, approve legal rights, or admit a model head for training.

## Immutable inputs

- V11 inventory `SOURCE_INVENTORY_DEEP_16397_V11.csv.gz`: SHA-256 `6436a6696a55c7643818914b6ebab5a5728d4eae23409d67e6e6d210c2b17ae4`.
- V17 facet states: SHA-256 `57249761c5b93207bdbee68adc7c48c5a1360245317e13b2b1be1870433e3fbd`.
- V17 facet values: SHA-256 `4d1f0637ba6503a5e7ea0f8fc8ac0ba471fee6ee0705875e8f1df7d819d13088`.
- V17 facet delta: SHA-256 `349a8cb73cfc586d26fc5412a29fb90eddcd424f7516c65edb0769cc32850975`.

## Authority mistake discovered and repaired

V17 had assigned the state `OBSERVED_PROCESSED_PART_GEOMETRY_ONLY` to 7,028 PartVerse source records for each of three **non-geometry facets**: `attribution_obligations`, `original_asset_provider_locator`, and `original_asset_recovery_status`. The JSON values already disclaimed rights/source parity, but using the same geometry state was unsafe for downstream truth selectors.

V18 changes exactly 21,084 *states*, keeping every V17 value unchanged. It labels them, respectively, provider license class claim (attribution court pending), provider locator claim (source not verified), and historical processed part receipt (creator original not rehashed). Independent full-state/value readback verified all remaining 4,307,724 state cells and **all 4,328,808 value cells** remained identical.

## Actual coverage as of V18

| Authority category | Facet cells |
|---|---:|
| Unknown/unreviewed | 4,153,359 |
| Provider/inventory claims or historical receipts | 68,157 |
| Scoped numeric or derived historical observations (not admission) | 107,292 |
| Qualified creator-original/training admissions | 0 |
| **Total** | **4,328,808** |

The numeric/derived category is deliberately broad: *processed part meshes, historical import counts, canonical normalized derivative*, not raw creator-original parity or universal anatomical truth.

For the 16,397 source candidates, the current V11 source-gate fields have zero positive approvals in: original source-byte court, source legal training eligibility, dedup/lineage-safe split, character qualification, morphology, native pose, anatomical front, camera and 2-view pair court, attachment relation, RGBA parity, and training authorization. Provider claims are not independent rights court.

## Outputs

- `FACET_STATES_EFFECTIVE_V18_16397x264.csv.gz`: complete corrected state matrix.
- `FACET_VALUES_EFFECTIVE_V18_16397x264.csv.gz`: same cell values, byte-wise logical parity with V17.
- `AUTHORITY_CORRECTIONS_V18.csv.gz`: exactly which source/facet states changed and why.
- `SOURCE_TRUTH_GAPS_V18.csv.gz`: all 16,397 source keys, current evidence amount, and unfulfilled gates.
- `FACET_AUTHORITY_GAPS_V18.csv`: all 264 facets by authority level.
- `PROVIDER_AUTHORITY_GAPS_V18.csv`: provider-level evidence depth.
- `ADMISSION_GATE_GAPS_V18.csv`: 13 canonical source/training authority gates and observed distributions.
- `INDEPENDENT_READBACK_V18.json`: independently checked state/value parity and exact 21,084 corrections.

## Next physical truth workload

1. Preserve complete source identity / rights / lineage, quarantine explicit noncommercial and duplicate sources, and identify actual characters with pose, front orientation and 2D graphics suitability without trusting provider labels.
2. Extract typed `mesh_object`, connected component, attachment, rig/skin/weight, native action, material/texture, camera and 2-view evidence from creator-original bytes or explicitly named derivatives.
3. Accumulate independent observed source truth, unknown masks and failed gates; never convert a capability contract or representation policy into source truth.
4. Route only field-appropriate evidence to IRIS, TESSA, Semantic, Graphics, AXIS and MIRA; no cross-head leakage. Audit each staged head admissibility gate before materialization.

**Storage:** private Drive; code should be reviewed on a stacked draft PR, not merged into active training or product. No 726/16,397 completion claim.