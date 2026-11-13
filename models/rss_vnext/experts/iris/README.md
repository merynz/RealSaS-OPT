# IRIS

IRIS is the RealSaS learned observation-evidence subsystem.

## Current mainline — layered authority

IRIS currently has two **complementary promoted source layers**. They must not be flattened into a false V2-versus-V3 choice.

### V2 — observation/foundation evidence package

`models/iris/v2/`

V2 is the audited promoted package for the current observation/foundation evidence stack: learner/model, depth-support-uncertainty heads, q-domain/descriptors, view/sparse-spatial evidence, exact DINOv2-S authority/adapter/apparatus, observation contract, checkpoint support and current training/evaluation utilities.

Promotion is byte-preserving for the 20 source modules listed in `restoration/MODEL_SOURCE_OWNERSHIP_AUDIT_V1_20260903.md`. The original `experiments/iris_reprojection_v2_20260831/` tree remains research provenance.

### V3 — current scene-first signed-geometry head used by the Mage FIT1 witness

`models/iris/v3/`

V3 is the later promoted scene-first signed-geometry composition used to produce the current Mage FIT1 upstream witness. It consumes qualified/frozen per-patch image evidence plus exact camera context, builds one shared all-view scene memory and supports arbitrary-P signed-field queries. It deliberately forbids candidate-local image lookup and categorical absolute view identity.

Promotion/witness authority:

- source: `models/iris/v3/scene_first_signed_v3.py`;
- zero-surface decoder: `models/iris/v3/zero_surface_decoder_v3.py`;
- frozen Mage witness: `models/iris/v3/PROMOTED_MAGE_FIT_WITNESS_V1.json`;
- promotion workflow: `.github/workflows/iris_scene_first_signed_promotion_v3.yml`.

The frozen Mage witness binds checkpoint SHA-256 `766f43cefd98925ada804853bafff93bb2352e23ba4a4e77e38174ae9e6b83a2` and signed zero-surface SHA-256 `987f7d18ce202454c4ea5101225bfaed54aeb4638cba1077e70efc15f2038e9b`. Teacher mesh is not a product-inference input.

## Authority boundary

IRIS consumes ordered multi-view RGB observations plus admitted camera/foundation conditioning and emits learned observation evidence. Learned geometric authority ends at signed/depth/support/uncertainty evidence. Analytic world-point construction, persistence fusion, local geometry qualification and `RiggingSurfaceIR` authority remain outside IRIS.

IRIS does **not** own canonical rig IDs, skeleton, skin, product state, proof or runtime output.

## Compiler substrate handoff

The former experiment file `persistence_adapter_v2.py` is **not** model authority. Its audited source is promoted to:

`compiler/realsas_compiler_core/substrate/iris_v2.py`

That Compiler-owned boundary consumes `ObservationEvidenceIR`, constructs persistence groups and observed local relations, invokes current surface/local-geometry authority, and can attach qualified normals before producing `RiggingSurfaceIR` state.

The later scene-first signed substrate bridge remains Compiler/GSA-owned around the V3 learned field. The important ownership rule is stable across versions:

`learned IRIS evidence -> deterministic GSA/RiggingSurfaceIR authority`.

## Older IRIS lines

`experiments/iris_controlled_v1/`, `experiments/iris_dino_controlled_20260829/` and older single-pose geometry IRIS code remain historical/experimental evidence. They do not become current by name or recency. Historical learned results affected by privileged-input leakage remain quarantined; prospective source repair does not retroactively cleanse them.

## External foundation distinction

The frozen DINOv2 backbone is an external foundation dependency, not a RealSaS product authority. RealSaS owns the exact adapter/feature/runtime-seal contract; third-party source/weights retain their own provenance and are hash-pinned by IRIS authority code.
