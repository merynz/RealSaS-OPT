# Mage FIT2 CDT lineage-schema drift note — 2026-09-13

Status: **CLOSED_DIAGNOSTIC_BUG / GEOMETRY_DRIFT_NOT_ESTABLISHED**

## Trigger

The first exact 8-view CDT diagnostic replay failed on V0 with a current candidate lineage hash different from the historical sealed candidate hash, even though the renderer had already reproduced the sealed V0 source-alpha recall and face count exactly.

## Root cause

The historical CDT seal was created under the residual/report schema present at commit `f02df44b86e7ef3747b8d2cf13316ba0c532b22b` (`Contract historical CDT boundary recovery vertices`). At that point `ObservationRasterDomain.coverage()` contributed only:

- `predicted_pixel_count`
- `inside_alpha_pixel_count`
- `foreground_pixel_count`
- `precision_inside_alpha`
- `source_alpha_recall`

Later mesh-product hardening added IoU/F1 plus connected foreground/uncovered-region diagnostics to `ObservationRasterDomain.coverage()`. `MeshDiscretizationCandidateIR.candidate_lineage_hash` includes the complete `residual_report`; therefore those diagnostic-only additions changed the candidate content hash even when CDT geometry is unchanged.

The qualified mesh hash also cannot be compared directly across that boundary: the current generic supported-mesh qualifier enriched `qualification_report` and changed canonical mesh-vertex IDs from historical four-digit `MV:0000` form to the current five-digit form. These are contract/schema changes, not evidence of changed triangle geometry.

## Repair

`experiments/mage_full_subject_reclosure_v1/render_fit2_cdt_8view_diagnostic_v2.py` separates the two authorities:

1. Build current CDT candidate from exact corrected H1/GSA/camera/RGBA authority.
2. Require the current qualifier to pass current contracts.
3. Reconstruct the historical candidate hash by projecting `residual_report` to the exact seal-time field set.
4. Reconstruct the historical qualified-mesh hash with the exact seal-time identity qualifier semantics.
5. Require both reconstructed historical hashes to match the sealed `MWB2_CDT_EXACT_RECLOSURE_REPORT_20260912.json` values.
6. Require all sealed structural/coverage/kernel metrics to reproduce exactly.
7. Record current enriched candidate/mesh hashes separately.
8. Only then render the real V0..V7 CDT geometry and uncovered-alpha evidence.

Repair commit: `df3965114b04b61e5134c8149b652fa71ebd3adf`.

## Scientific interpretation

Historical sealed hashes remain historical authority under their original schema. Current enriched hashes remain current authority under the enriched schema. A direct equality test between the two is forbidden unless the payload schema is first normalized to the historical contract.

No threshold was changed. No historical result was relabeled. No geometry drift is excused by this note: V2 still fails closed if the reconstructed historical candidate hash, reconstructed historical mesh hash, any sealed structural metric, any source SHA, or the GSA lineage differs.

## Product boundary

This repair only restores exact diagnostic replay semantics. It does **not** promote the current CDT to FIT2 product mesh PASS. Product mesh qualification remains governed by the current strict observation-domain/product policy and downstream fresh Geppetto/Arachne mechanics.
