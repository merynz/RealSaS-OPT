# RealSaS Learned Models

`models/` is the semantic home of current learned mainline implementations. Models emit evidence/proposals; the Compiler owns canonical IDs and qualified product truth.

## Current learned stack

| Subsystem | Current package | Current state |
|---|---|---|
| IRIS observation/foundation | `models/iris/v2/` | promoted supporting evidence layer |
| IRIS signed geometry | `models/iris/v3/` | promoted Mage FIT1 signed-geometry witness |
| TESSA mechanical mesh | `models/tessa/v1/` | **P0 research implementation; GSA-native learned mesh proposal, not shipping authority** |
| ATLAS rig | `models/atlas/` | canonical alias namespace over current Geppetto reference-strength lineage |
| MIRA skin | `models/mira/` | canonical alias namespace over current Arachne lineage |
| Geppetto historical/current lineage | `models/geppetto/reference_strength_v1/` | FIT1-frozen/promoted skeleton proposal; migrating under ATLAS naming |
| SkinFieldCodec | `models/skin_field_codec/v1/` | retained representation/research lineage; not an Arachne V5 runtime dependency |
| Arachne historical/current lineage | `models/arachne/v5/` | FIT1-frozen/promoted K4-Z direct-simplex skin proposal; migrating under MIRA naming |

Historical packages remain source/provenance and do not become co-current merely by existing.

## Canonical learned mechanics order

`IRIS -> TESSA -> ATLAS -> MIRA`

- **IRIS**: observation-grounded surface/perception evidence.
- **TESSA**: `RiggingSurfaceIR -> learned production-mesh proposal`; Compiler support-binding/static/mechanical qualification is mandatory before downstream use.
- **ATLAS**: articulation/rig proposal on the exact sealed carrier.
- **MIRA**: skin/influence-field proposal queried on the exact sealed carrier.
- **Compiler**: sole owner of qualification, canonical identity, repair and product truth.

TESSA is intentionally upstream of final ATLAS/MIRA retraining so those models are not fitted to a carrier contract that is still moving. See `canonical/TESSA_MECHANICAL_MESH_MODEL_EXECUTION_PLAN_V1.md`.

## Arachne V5

`RiggingSurfaceIR + QualifiedSkeletonIR -> V4 backbone -> K4x512 Z -> direct row-simplex decoder -> SkinProposalIR`

The frozen Mage FIT1 checkpoint is a hash-bound composite of the exact V4 backbone and V5 decoder delta. See `models/arachne/v5/checkpoint_authority_v1.py` and `models/arachne/v5/FROZEN_MAGE_FIT1_CHECKPOINT_V1.json`.

## Import / authority firewall

- experiments may import current `models/`;
- current model packages should not depend permanently on dated experiment semantic owners;
- learned modules may create typed proposal/evidence objects but cannot claim Compiler qualification;
- source existence != scientific evidence != promotion != generalization.
