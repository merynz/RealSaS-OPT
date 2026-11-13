# Model Source Ownership Audit V1 — 2026-09-03

**Status:** `CURRENT_V4_MODEL_MAINLINE_PROMOTED__INFERENCE_AND_BASE_TRAINING_VISIBLE__OLD_ALTERNATIVES_REMAIN_RESEARCH`  
**Branch:** `restoration/compiler-runtime-promotion-v1-20260903`

## Purpose

Make the current RealSaS learned system directly inspectable under `models/` while preserving dated research trees as evidence/provenance. Promotion changes source ownership/location, not product authority.

## Current V4 learned inventory

| Subsystem | Mainline home | Current scope |
|---|---|---|
| IRIS V2 | `models/iris/v2/` | inference, foundation/apparatus contract, checkpoint, current train/eval |
| Geppetto V2 | `models/geppetto/v2/` | inference/conditioning/checkpoint + V2 target/loss/train/eval |
| SkinFieldCodec V1 | `models/skin_field_codec/v1/` | codec/checkpoint/config + current A0 deformation-sensitive train/eval |
| Arachne V2 | `models/arachne/v2/` | inference/conditioning/geometry + current base A1 hard-tail train/eval |

Models emit evidence/proposals only. Compiler owns canonical identity, legality, qualification, product state, proof and export.

## 1. IRIS V2

Source tree: `experiments/iris_reprojection_v2_20260831/`  
Source tree SHA: `597e5ccb9765e1af60c94b434a91f68de06a11a3`

20 current V2 modules were promoted byte-preserving into `models/iris/v2/`: model, q-domain/evidence stack, depth/support/uncertainty heads, DINO authority/adapter/apparatus, observation contract/emitter, checkpoint, current train/eval and world regularizer.

`persistence_adapter_v2.py` (blob `3a977f41182c58d33bd6b7f1d1f403115cd772d5`) is explicitly **Compiler-owned pending promotion** because it consumes observation evidence and invokes Compiler surface/local-geometry authority.

Gate-0 corpus/preflight runners/tests/prereg/results remain experiment/evidence.

## 2. Geppetto V2

Source tree: `experiments/geppetto_arachne_r6_20260901/`  
Source tree SHA: `522ec6fa2470a8236c9e78234908c87b30a1d832`

Byte-preserved current core:

- `geppetto_candidate_v2.py` — `9c7572aaaa15d28f3738248ad9bbd53a9a056b0b`
- `geppetto_conditioning_v2.py` — `9a33ab34ce050842c77dfc64407123303c4bf456`
- `geppetto_checkpoint_v2.py` — `4991038092cad88f2b1e11cbd486af3d925e235f`

Byte-preserved current V2 training/evaluation:

- `training_targets_v2.py` — `8caf6bcfc9c2cb10e2c7d0758913e550863df4f8`
- `geppetto_loss_v2.py` — `036d73690bf17348b2a299887494ebd87d230c0b`
- `geppetto_train_v2.py` — `4453810f047cc62171ab16ccab3f4c1ead0169c7`
- `geppetto_eval_v2.py` — `dfdb2f3ab754ec1530a3d489fb9c3d1207c67232`

The historical shared `training_targets_v1.py` mixed Geppetto and SkinField target types and imported V1 conditioning. Mainline uses a narrow compatibility module containing only the unchanged `GeppettoTeacherTargetV1` dataclass. This keeps current V2 loss/target source intact without promoting V1 conditioning into Geppetto V2.

Geppetto V1 candidate/loss/config, oracle-substrate harnesses, capacity/overfit/panel diagnostics remain research/history.

## 3. SkinFieldCodec V1

Byte-preserved core:

- `skin_field_codec_v1.py` — `01d064fdbf0df1e1efb09495b788665e582e2f42`
- `skin_field_codec_checkpoint_v1.py` — `0365d4e1434443065487c6644a68ab07a60c6cda`

The old `candidate_config_v1.py` mixed Geppetto V1, Codec V1 and Arachne V1 configs. Mainline semantically extracts only `SkinFieldCodecConfigV1` + `SKIN_FIELD_CODEC_V1` into `config_v1.py`; a narrow `candidate_config_v1.py` shim preserves the exact codec module's relative import. Default config hash remains:

`24c9f2580be9e80a02789e9ba35a57470145114807859057398b07bef9d58715`

Byte-preserved current A0 training/evaluation:

- `codec_deformation_loss_v1.py` — `b0462467f8346f3883058c1c47a065f4d5be5adf`
- `train_codec_r6_a0_v1.py` — `18baa39c8178b0f5f16cece0aa78aa51aac6ca00`
- `eval_codec_r6_a0_v1.py` — `c29161dca81d70209cf1dbfe1b9d99f3d5b99b3a`

Temperature/cooling/tail anatomy/pair-geometry and other diagnostic files remain experiments unless a future closure promotes them into the base codec contract.

## 4. Arachne V2

Byte-preserved current core:

- `arachne_candidate_v2.py` — `ba5bf0a2757e8d702176332fcfa91f804e0d545e`
- `conditioning_v2.py` — `3383cb7801b22aeb634f05551a847a04e674c7c8`
- `conditioning_v1.py` — `b3539ed6c4580ac69338bdb9bd4b93c6e78a262c`
- `arachne_geometry_v2.py` — `0a39450b69de48da3b7748cb08972d0b69ff9700`

Cross-model compatibility bridges point relative historical import names at the canonical current packages:

- `skin_field_codec_v1.py` -> `models.skin_field_codec.v1`
- `geppetto_conditioning_v2.py` -> `models.geppetto.v2`
- `codec_deformation_loss_v1.py` -> canonical Codec A0 deformation loss
- `train_codec_r6_a0_v1.py` -> canonical Codec A0 qualification-token contract

No duplicate learned Codec or Geppetto implementation is created inside Arachne.

Byte-preserved current base A1 training/evaluation:

- `arachne_tail_objective_v1.py` — `0abfe7c8e85135d0ce613df6463c5d4def220717`
- `train_arachne_r6_a1_v1.py` — `5db5a7b6c7e8d14e9c33bcf3127c78a445e8ca90`
- `eval_arachne_r6_a1_v1.py` — `daf233904f31161f4c504179f28d9917932c3514`

`arachne_tail_remediation_v1.py` stays in experiments: it explicitly defines a conditional post-failure remediation stage, while the current base A1 train step already carries generic family/semantic-ID-agnostic hard-tail pressure.

## Repository-wide learned scan

Code searches over `nn.Module` / `from torch import nn` and current architecture evidence found neural code in these families:

- current IRIS V2;
- older IRIS Controlled V1 and older single-pose/legacy IRIS variants;
- current Geppetto V2 plus Geppetto V1 alternative;
- current SkinFieldCodec V1;
- current Arachne V2 plus Arachne V1 alternative.

No fifth learned subsystem is authorized by current V4 architecture or current source composition. This is a **current-mainline inventory closure**, not a claim that future research cannot add another model.

## Mainline firewalls

1. `models/`, Compiler mainline and runtime reference code may not depend on dated `experiments.*` implementations.
2. Experiments may import mainline.
3. Cross-model dependencies resolve through current model homes, not the historical mixed R6 directory.
4. Teacher truth is training/evaluation-only and cannot become shipping model input or canonical identity.
5. Current source promotion does not make diagnostic/ablation/remediation experiments current automatically.
6. Models remain evidence/proposal producers; Compiler authority is unchanged.

## Remaining source-ownership work

- promote IRIS `persistence_adapter_v2.py` into the correct Compiler substrate boundary;
- run mainline source/byte/import gates when GitHub Actions runners actually launch (recent runs are infrastructure-blocked with `steps=[]`, `runner_id=0`);
- continue Compiler physical layer normalization only where dependency ownership is unambiguous;
- resume historical motion proof/playback/repair restoration after the source ownership seam is clean.

No architecture refreeze or FIT authorization follows from this source organization closure.
