# IRIS

IRIS is the RealSaS learned observation-evidence subsystem.

## Current mainline

The audited current V2 source package is now promoted at:

`models/iris/v2/`

Promotion is byte-preserving for the 20 source modules listed in `restoration/MODEL_SOURCE_OWNERSHIP_AUDIT_V1_20260903.md`. The original `experiments/iris_reprojection_v2_20260831/` tree remains intact as research provenance and contains Gate-0 harnesses/results that are intentionally **not** mainline model code.

## Authority boundary

IRIS consumes ordered multi-view RGB observations plus admitted camera/foundation conditioning and emits learned observation evidence. Its learned geometric authority ends at forward depth/support/uncertainty. Analytic world-point construction, persistence fusion, local geometry qualification and `RiggingSurfaceIR` authority remain outside IRIS.

IRIS does **not** own canonical rig IDs, skeleton, skin, product state, proof or runtime output.

## Package contents

`models/iris/v2/` intentionally keeps the currently coupled V2 package flat so the promotion does not rewrite scientific behavior merely for aesthetics. It contains:

- learner/model and depth-support-uncertainty heads;
- q-domain, descriptor, view-evidence and sparse spatial evidence stack;
- exact DINOv2-S authority/adapter/apparatus code;
- observation contract and evidence emitter;
- checkpoint support;
- current training objective/regularizer;
- current scientific evaluation and DINO token-parity utilities.

Later internal sub-packaging is allowed only with byte/behavior regression coverage.

## Explicitly outside the model

`persistence_adapter_v2.py` is **not** promoted here. It consumes `ObservationEvidenceIR` and invokes Compiler surface/local-geometry authority, so it is classified `COMPILER_OWNED_PENDING_PROMOTION`.

Gate-0 corpus loaders, preflight runners, tests, preregistration and result JSON remain under `experiments/iris_reprojection_v2_20260831/` as research apparatus/evidence.

## Older IRIS lines

`experiments/iris_controlled_v1/`, `experiments/iris_dino_controlled_20260829/` and older single-pose geometry IRIS code remain historical/experimental evidence. They do not become current by name or recency.

## External foundation distinction

The frozen DINOv2 backbone is an external foundation dependency, not a RealSaS product authority. RealSaS owns the exact adapter/feature/runtime-seal contract; third-party source/weights retain their own provenance and are hash-pinned by IRIS authority code.
