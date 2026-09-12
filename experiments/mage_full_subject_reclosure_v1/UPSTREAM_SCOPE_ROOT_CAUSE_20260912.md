# Upstream subject-scope root cause — 2026-09-12

## Historical H1 mismatch

The historical H1 scene-first Mage generator built its geometry truth / normalization contract from meshes selected by a narrow armature-modifier criterion. That geometry authority contained 3348 body vertices.

The rendered RGBA observations, however, represented a broader visual subject. Later source audits established a full normalized source of 5321 vertices / 5763 faces with additional book, weapon, hat, cape and Icosphere components. The mechanically supported subset is 5279 vertices / 5683 faces.

Thus the learned signed field was optimized under a target subject narrower than the visual observation subject.

## Why old metrics did not expose it

The historical fit metrics compared predicted geometry against the same incomplete teacher geometry and measured local first-hit / geometry distances on that authority. Therefore they could be internally good while failing full product-observation completeness.

The promotion witness did contain silhouette criteria, but the re-audit demonstrates that the artifact now used downstream does not satisfy full-subject source-alpha completeness under the exact current product observations. This contradiction reopens the authority irrespective of historical labels.

## Correct prevention

Every future geometry FIT closure must bind three mutually checked authorities:

1. observation set + exact camera set;
2. FIT/evaluation full-subject geometry/component authority;
3. predicted geometry.

The closure must report both geometry distances and product-observation silhouette/component completeness. A geometry subset may not authorize a full-subject product claim merely because predicted-to-subset metrics pass.

## Non-cause findings

The following were tested and are not the primary cause of the ~20% current product coverage:

- camera projection convention mismatch;
- raster coordinate convention mismatch;
- GSA target-node count alone;
- V5 weight simplex quality;
- MWB2 alone.

MWB2 is independently too conservative, but it acts after the larger upstream substrate loss.
