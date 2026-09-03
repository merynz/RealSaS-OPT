# Model Source Ownership Audit V1 — 2026-09-03

**Status:** `IN_PROGRESS__IRIS_V2_CLOSED__GEPPETTO_CODEC_ARACHNE_PENDING`  
**Branch:** `restoration/compiler-runtime-promotion-v1-20260903`

## Purpose

Identify the current RealSaS learned implementations file-by-file before moving them out of dated research directories. Promotion changes source ownership/location, not scientific authority: models still emit evidence/proposals and the Compiler remains canonical authority.

## IRIS V2 source authority

Source tree:

`experiments/iris_reprojection_v2_20260831/`

Git tree SHA:

`597e5ccb9765e1af60c94b434a91f68de06a11a3`

### PROMOTE_MODEL_CORE

| Source file | Git blob SHA | Mainline target |
|---|---|---|
| `__init__.py` | `eabdd389a1261857b121ff4474e43e81ae3a7a63` | `models/iris/v2/__init__.py` |
| `checkpoint_v2.py` | `3d60b4d4741a090c78e89bb504c93f5a020f8d29` | same basename |
| `depth_output_head_v2.py` | `3297f479a1551d1e77cf73c244766a73bc4b46b9` | same basename |
| `dinov2_foundation_v2.py` | `d1cfed1a43905b738c2c3067941d7bc27626ce4f` | same basename |
| `evidence_field_v2.py` | `c6bf4a569f153f1ec29fc3800e2f68635dae1612` | same basename |
| `foundation_adapter_v2.py` | `90fdfa0ed918664f28879dd4146524e318abf50d` | same basename |
| `iris_apparatus_v2.py` | `de720f85b78f79f4f9b72c601b26b14f0265de53` | same basename |
| `local_refinement_v2.py` | `732c2dd2848d5f9bbfb7f4703025d6acc3965f1b` | same basename |
| `model_v2.py` | `578d866624f95081b6eb88731c11bcdcd6b3c125` | same basename |
| `observation_contract_v2.py` | `a16b8cd16cd6804976c4d3ab357a28f03e1f290f` | same basename |
| `observation_evidence_emitter_v2.py` | `5dd207427cab6717a50a568c2e0dd05393caa591` | same basename |
| `q_descriptor_sampler_v2.py` | `b66cf585984b2ba426f27ad3d52b61eacba6f1cb` | same basename |
| `q_domain_v2.py` | `09f1850c61abbae9cdf1c250317b040463eddb81` | same basename |
| `q_evidence_encoder_v2.py` | `980421f89428d4bccadbb342ee06908ac34b0357` | same basename |
| `q_spatial_graph_v2.py` | `b5c50f169358af3c03f742ccf4596fb503d5380f` | same basename |
| `ray_modes_v2.py` | `f588e0c361ff742272fe474a248939db58b51bf3` | same basename |

### PROMOTE_MODEL_TRAINING

| Source file | Git blob SHA |
|---|---|
| `train_v2.py` | `18be6d60e58ed1c062a2e306cc06a2817b7831e2` |
| `world_regularizer_v2.py` | `6d0f2b1e4e3d759e3cf39f537f22f74ae48ba53a` |

### PROMOTE_MODEL_EVALUATION

| Source file | Git blob SHA |
|---|---|
| `eval_v2.py` | `8b17657fd655fffb8270330547c9ba4caa1c8670` |
| `dino_token_parity_v2.py` | `e9365634e23237ab1e84f5a0c3fa6ccde53995cb` |

### COMPILER_OWNED_PENDING_PROMOTION

`persistence_adapter_v2.py` — blob `3a977f41182c58d33bd6b7f1d1f403115cd772d5`.

Reason: this module consumes `ObservationEvidenceIR` and invokes current Compiler `surface`, `hashing` and deterministic local-geometry authority. It is not learned-model implementation merely because it was developed beside IRIS.

### KEEP_EXPERIMENT_ONLY / EVIDENCE

- `gate0_geometry_v1.py`
- `gate0_real_corpus_v1.py`
- `gate0_real_corpus_v1_1_hashfix.py`
- `run_gate0_synthetic_preflight_v1.py`
- `test_gate0_geometry_v1.py`
- `test_gate0_real_corpus_v1.py`
- `test_gate0_real_corpus_v1_1_hashfix.py`
- `GATE0_SYNTHETIC_PREFLIGHT_RESULT_V1.json`
- `IRIS_REPROJECTION_V2_PREREG_V1.json`

These remain in the dated experiment tree. They answer experimental/gate questions rather than define the current model implementation.

## IRIS promotion invariants

1. The 20 promoted modules are linked using the **same Git blob SHA** as their source counterparts; promotion introduces no code rewrite.
2. Original experiment files remain intact as provenance.
3. `models/iris/v2/` may not import `experiments.*`.
4. Mainline `models/`, `compiler/realsas_compiler_core/`, `compiler/realsas_compiler_services/` and `runtime/reference_v4/` must converge to zero Python imports from dated experiment packages.
5. `persistence_adapter_v2.py` is not smuggled into the learned model package.
6. Frozen DINO is an external dependency pinned by authority contract, not RealSaS canonical product truth.

## Remaining learned audit

Pending file-level closure in this order:

1. Geppetto V2
2. SkinFieldCodec V1
3. Arachne V2
4. repository-wide search for any additional current RealSaS-owned learned subsystem not represented by those four semantic homes

No global learned-stack closure claim is authorized until all four steps close.
