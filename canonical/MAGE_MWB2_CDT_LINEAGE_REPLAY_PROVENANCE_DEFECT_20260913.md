# RealSaS — Mage MWB2 CDT historical lineage replay provenance defect — 2026-09-13

**Status:** `OPEN_PROVENANCE_DEFECT__GEOMETRY_REPLAY_NOT_FALSIFIED`  
**Branch:** `repair/mage-full-subject-reclosure-20260912`  
**Scope:** historical `MWB2_CDT_EXACT_RECLOSURE_REPORT_20260912.json` replay only  
**Does not claim:** FIT2 product mesh PASS

## Finding

The historical exact-CDT report records per-view `candidate_lineage_hash` and `mesh_lineage_hash`, but it does **not** record the exact `camera_binding_hash` preimage used when those lineage hashes were created.

`camera_binding_hash` is part of `MeshDiscretizationCandidateIR` and therefore participates in candidate lineage. The qualified mesh also carries that camera binding and source-candidate lineage, so a missing camera-binding preimage propagates into `mesh_lineage_hash` replay.

The historical adapter identified by the seal is commit:

`f02df44b86e7ef3747b8d2cf13316ba0c532b22b`

In that adapter, `camera_binding_hash` is validated as non-empty and stored in the candidate IR, but it is not consumed to compute CDT vertices, faces, edges, alpha admission, or raster coverage. Directional geometry comes from the admitted surface raster bindings plus the exact observation-domain mask.

## Reproduction evidence

Three diagnostic attempts were intentionally fail-closed on the old lineage hashes. The strongest replay used the exact `f02df44b...` source state in an isolated process and the corrected sealed GSA8192 substrate.

For V0 it reproduced **all report-visible geometry/coverage outputs exactly**:

- visible surface nodes: `2537`
- vertices: `2525`
- edges: `7336`
- faces: `4801`
- kernel triangles: `5016`
- alpha-rejected faces: `214`
- constraint splits: `1`
- contracted boundary-recovery vertices: `1`
- post-contraction triangles: `5015`
- predicted pixels: `503782`
- foreground pixels: `544107`
- precision: `1.0`
- source-alpha recall: `0.9258877389925143`

but replayed lineage was:

- candidate: `b8f0337724ea5e1413dcaba36cc4b46aa239803ec4d415b9faf0ce53015d9ad3`
- mesh: `ed15d553c2bb186ab2f44ea08f8cad9b33fc803dd7637ae1dfb8a268cd2ea277`

while the historical seal records:

- candidate: `b063921e23a6ce6f27f15925810c62811453a4a70eef22664ba385b0d0c71b89`
- mesh: `b2e18ce5864e6e722b53bd21c66ebd7ff912e339dd69dc9fa12bd7b5fc2fad01`

Because every report-visible geometry-affecting metric reproduced exactly while the content lineage did not, the old lineage SHA cannot currently serve as a replay gate without the missing preimage.

## Decision

Do **not** weaken or overwrite the historical hashes. Preserve them as historical evidence.

For historical visual recovery only, use a separate hard geometry-replay gate:

1. exact corrected H1 artifact SHA;
2. exact V0..V7 observation/camera input SHA;
3. exact sealed GSA8192 lineage;
4. exact historical `f02df44b...` adapter and frozen historical CDT kernel;
5. exact per-view report-visible structural counts;
6. exact per-view predicted/foreground pixel counts;
7. exact recall/precision;
8. explicit proof that changing only `camera_binding_hash` leaves the geometry payload unchanged while changing candidate lineage.

If all eight views satisfy this gate, the directional CDT geometry may be rendered for diagnostic inspection. This does **not** repair the historical lineage provenance and does **not** qualify a FIT2 product mesh.

## Required forward policy

All future product/replay seals that store a lineage hash must also store every external binding preimage required to reproduce it, including the exact `camera_binding_hash` (or a typed canonical camera authority object from which it is deterministically derived).

The corrected FIT2 product mesh path remains governed by the frozen independent product qualifier; historical CDT visual recovery is diagnostic evidence only.
