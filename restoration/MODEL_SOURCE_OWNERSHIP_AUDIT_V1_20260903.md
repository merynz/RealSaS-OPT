# Model Source Ownership Audit V1 — 2026-09-03

**Status:** `CURRENT_V4_INFERENCE_STACK_PROMOTED__TRAINING_LANES_AND_REPO_WIDE_LEARNED_SCAN_PENDING`  
**Branch:** `restoration/compiler-runtime-promotion-v1-20260903`

## Purpose

Make the current RealSaS learned stack directly inspectable under `models/` without pretending every dated training experiment is current. Model promotion changes source ownership/location only; Compiler authority is unchanged.

## Current V4 learned stack

| Subsystem | Mainline home | Promotion state |
|---|---|---|
| IRIS V2 | `models/iris/v2/` | inference + current train/eval/checkpoint package promoted |
| Geppetto V2 | `models/geppetto/v2/` | current inference + conditioning + checkpoint promoted |
| SkinFieldCodec V1 | `models/skin_field_codec/v1/` | current codec + checkpoint promoted; config semantically split from mixed V1 stack file |
| Arachne V2 | `models/arachne/v2/` | current inference/conditioning/geometry promoted with explicit current-model dependency bridges |

All models remain evidence/proposal producers. Canonical IDs, skeleton/skin legality, product identity, qualification, proof and export remain Compiler-owned.

---

## 1. IRIS V2

Source tree: `experiments/iris_reprojection_v2_20260831/`  
Source Git tree SHA: `597e5ccb9765e1af60c94b434a91f68de06a11a3`

### Byte-preserved mainline modules

`__init__.py`, `checkpoint_v2.py`, `depth_output_head_v2.py`, `dino_token_parity_v2.py`, `dinov2_foundation_v2.py`, `eval_v2.py`, `evidence_field_v2.py`, `foundation_adapter_v2.py`, `iris_apparatus_v2.py`, `local_refinement_v2.py`, `model_v2.py`, `observation_contract_v2.py`, `observation_evidence_emitter_v2.py`, `q_descriptor_sampler_v2.py`, `q_domain_v2.py`, `q_evidence_encoder_v2.py`, `q_spatial_graph_v2.py`, `ray_modes_v2.py`, `train_v2.py`, `world_regularizer_v2.py`.

These are linked into `models/iris/v2/` with the same Git blobs as the audited source files.

### Compiler-owned

`persistence_adapter_v2.py` — blob `3a977f41182c58d33bd6b7f1d1f403115cd772d5` — `COMPILER_OWNED_PENDING_PROMOTION` because it consumes `ObservationEvidenceIR` and invokes Compiler surface/local-geometry authority.

### Experiment-only/evidence

Gate-0 geometry/corpus/preflight runners/tests plus preregistration/result JSON stay in the dated experiment tree.

---

## 2. Geppetto V2

Source tree: `experiments/geppetto_arachne_r6_20260901/`  
Source Git tree SHA: `522ec6fa2470a8236c9e78234908c87b30a1d832`

### Byte-preserved current inference

| File | Git blob SHA |
|---|---|
| `geppetto_candidate_v2.py` | `9c7572aaaa15d28f3738248ad9bbd53a9a056b0b` |
| `geppetto_conditioning_v2.py` | `9a33ab34ce050842c77dfc64407123303c4bf456` |
| `geppetto_checkpoint_v2.py` | `4991038092cad88f2b1e11cbd486af3d925e235f` |

V1 candidate/config/loss code remains research/history, not current by co-location.

### Training lane pending

`geppetto_loss_v2.py` and `geppetto_train_v2.py` still consume `training_targets_v1.py`; that shared target contract reaches the older `conditioning_v1.py` training lane. Scientific validity may survive, but ownership must be untangled before promotion.

---

## 3. SkinFieldCodec V1

### Byte-preserved implementation

| File | Git blob SHA |
|---|---|
| `skin_field_codec_v1.py` | `01d064fdbf0df1e1efb09495b788665e582e2f42` |
| `skin_field_codec_checkpoint_v1.py` | `0365d4e1434443065487c6644a68ab07a60c6cda` |

The historical `candidate_config_v1.py` mixed Geppetto V1, Codec V1 and Arachne V1 config ownership. It is **not** copied wholesale into the Codec mainline.

Mainline instead contains:

- `config_v1.py` — semantic extraction of `SkinFieldCodecConfigV1` + `SKIN_FIELD_CODEC_V1` only;
- `candidate_config_v1.py` — compatibility shim preserving the byte-identical codec module's relative import path.

Default config hash is frozen by source test as:

`24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`

### Training lane pending

Codec R6-A0 train/eval, deformation objective, tail/cooling diagnostics and shared teacher targets remain in `experiments/` until their current scientific role is classified.

---

## 4. Arachne V2

### Byte-preserved current implementation

| File | Git blob SHA |
|---|---|
| `arachne_candidate_v2.py` | `ba5bf0a2757e8d702176332fcfa91f804e0d545e` |
| `conditioning_v2.py` | `3383cb7801b22aeb634f05551a847a04e674c7c8` |
| `conditioning_v1.py` | `b3539ed6c4580ac69338bdb9bd4b93c6e78a262c` |
| `arachne_geometry_v2.py` | `0a39450b69de48da3b7748cb08972d0b69ff9700` |

Current source retains original relative dependency names. Mainline supplies two explicit bridge modules instead of duplicating learned code:

- `models/arachne/v2/skin_field_codec_v1.py` -> canonical `models.skin_field_codec.v1`;
- `models/arachne/v2/geppetto_conditioning_v2.py` -> canonical `models.geppetto.v2` conditioning.

`conditioning_v1.py` is preserved because current V2 Arachne conditioning explicitly builds on `ArachneConditioningAdapter` before adding V2 pair geometry. Its unrelated older Geppetto adapter is compatibility residue, not current Geppetto ownership.

Arachne V1 candidate, tail remediation, R6-A1 training/eval and diagnostics remain research evidence pending training-lane audit.

---

## Mainline model firewalls

1. Current model Python code must not import dated `experiments.*` packages.
2. Experiments may import current mainline; the reverse direction is migration debt/bug.
3. Byte-preserving promotions keep the original experiment source intact for provenance.
4. Compatibility shims may bridge newly separated semantic homes but may not create alternate learned implementations.
5. Model outputs remain evidence/proposals; no model promotion expands Compiler authority.
6. A current inference source promotion does not automatically promote its historical teacher/training/diagnostic apparatus.

## Remaining closure work

Before declaring **all learned source normalized**:

1. classify and promote/reject Geppetto V2 training lane;
2. classify and promote/reject SkinFieldCodec training/evaluation lane;
3. classify and promote/reject Arachne V2 training/evaluation/tail-remediation lane;
4. promote the IRIS persistence adapter into its proper Compiler semantic home;
5. repository-wide scan for any additional RealSaS-owned current `nn.Module` / learned subsystem outside IRIS, Geppetto, SkinFieldCodec and Arachne;
6. run model-mainline source gate when GitHub runner infrastructure actually launches jobs.

No global architecture refreeze or FIT authorization follows merely from source normalization.
