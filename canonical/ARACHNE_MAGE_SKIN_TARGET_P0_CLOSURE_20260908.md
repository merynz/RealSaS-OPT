# RealSaS — Arachne Mage Skin Target P0 Closure V2 — 2026-09-08

**Status:** `P0_PASS__EXACT_SHIPPING_SG_BINDING_REPAIRED__A0_NOT_YET_AUTHORIZED`

## Decision

The projected `950 x 22` Mage teacher skin field remains valid and unchanged. P0 V1 is superseded only at the binding-manifest layer because its recorded `surface_geometry_lineage_hash` came from a provenance-drift reconstruction rather than the exact Geppetto FIT1 shipping surface lineage.

No learned optimizer had started before this repair.

## Exact shipping binding

- surface nodes: `950`
- joint columns: `22`
- exact Geppetto FIT1 surface lineage: `67184f2cdbc3b2fca958e705d7b279d7fa5354f15d181712c2c183f8af2856eb`
- exact final QualifiedSkeletonIR.v2 lineage: `738891b236f9a261d521d17657b56d23ad47d145d9baf0f38a1bbc7d0e69c306`
- projected W content SHA-256: `c15db7b78d272ac22998071e1fb1cec4c65d7366133ef16fb72824f222c852d9`
- NPZ SHA-256: `f3db92194660f12de4425e015f85ac4ac995fcfc44a0272047b049ffb0d36d71`
- exact S/G/W binding SHA-256: `cb41eb7055b8e2646628daecdd0e31dfc079d163d5f5adaaa1a92f1ca1dfb994`

The 950 surface IDs and their row order are byte-for-byte identical between the existing P0 NPZ and the exact Geppetto FIT1 surface reconstruction. Therefore teacher projection, selected triangles, weights, confidence classes and the nine local disambiguation changes are not recomputed.

## Confidence inventory

- HIGH: `920`
- MEDIUM: `7`
- MEDIUM_LOW: `4`
- LOW: `19`

The `19` LOW rows remain explicit and are not silently promoted to teacher authority.

## Superseded V1 issue

P0 V1 recorded surface lineage `2f1e2f4988f56ea506353a4144b6e170dd730d5d883ce13229f9b0749c43f590` because the reconstruction used different provenance label/metadata. Surface IDs themselves do not depend on that provenance label, so the target row carrier remained correct; only product lineage binding was wrong.

## Next gate

`A0_MAGE_PREREG__EXACT_CONDITIONING_CACHE__CONFIDENCE_HANDLING`

A0 optimizer remains unauthorized until the exact shipping conditioning cache and LOW-row loss/evaluation policy are frozen.
