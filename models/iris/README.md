# IRIS

IRIS is the RealSaS learned observation-evidence subsystem.

## Authority boundary

IRIS consumes ordered multi-view RGB observations plus admitted camera/foundation conditioning and emits learned observation evidence. Its learned geometric authority ends at forward depth/support/uncertainty. Analytic world-point construction, persistence fusion, local geometry qualification and `RiggingSurfaceIR` authority remain outside IRIS.

IRIS does **not** own canonical rig IDs, skeleton, skin, product state, proof or runtime output.

## Current source candidate

Current V2 implementation candidates are under:

`experiments/iris_reprojection_v2_20260831/`

The audited source family includes the V2 model, q-domain/evidence stack, depth/support/uncertainty heads, local refinement, foundation adapter, observation contracts/emitter, checkpoint, train and eval apparatus.

Older IRIS families under `experiments/iris_controlled_v1/`, `experiments/iris_dino_controlled_20260829/` and `experiments/g0_g1_single_pose_geometry/` are historical/candidate evidence until explicitly classified; they are not promoted by recency or name.

## Target production layout

```text
models/iris/
  README.md
  src/
    model.py
    apparatus.py
    foundation/
    evidence/
    depth/
    io/
  training/
  evaluation/
  tests/
```

Exact filenames and decomposition are audit-controlled. No executable source is copied here until its imports, truth access, camera/foundation boundary and supersession status are classified.

## External foundation distinction

A frozen external foundation backbone may be an IRIS dependency without becoming a RealSaS-owned model authority. Adapter/feature contracts belong with IRIS; third-party model weights and their provenance do not become product truth.