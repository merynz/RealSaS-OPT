# SkinFieldCodec

SkinFieldCodec is the learned continuous per-joint influence-field representation used by the current Arachne line.

## Current mainline

`models/skin_field_codec/v1/`

Current executable source:

- `skin_field_codec_v1.py` — byte-identical promoted codec implementation;
- `skin_field_codec_checkpoint_v1.py` — byte-identical checkpoint authority;
- `config_v1.py` — clean semantic extraction of the current codec config;
- `candidate_config_v1.py` — compatibility shim required by the byte-preserved codec import path.

The compatibility shim deliberately does **not** carry the historical Geppetto V1 and Arachne V1 configs that shared the original experiment file.

## Learned vs deterministic

The codec is a learned `nn.Module`. For a fixed checkpoint, fixed inputs and deterministic backend execution its forward/decode result can be deterministic, but that does not make it a deterministic geometric solver.

It learns:

- a teacher-lane encoder from dense skin weights to per-joint latent fields;
- a shared decoder from latent fields + admitted surface/joint conditioning to dense proposal weights.

Teacher dense weights are permitted only on the training/encoding lane. Product inference must use predicted latents and admitted current conditioning.

## Current A0 training/evaluation

The current codec qualification/training lane is now visible beside the model:

- `codec_deformation_loss_v1.py` — differentiable LBS consequence loss;
- `train_codec_r6_a0_v1.py` — current A0 optimization step and qualification-token contract;
- `eval_codec_r6_a0_v1.py` — current A0 metrics and PASS/FAIL token production.

These files are byte-preserved from the audited R6 source. Broader diagnostic experiments (temperature, tail anatomy, pair-geometry ablations, cooling and similar probes) remain under `experiments/` because they investigate the codec rather than define the current base training lane.

## Authority boundary

SkinFieldCodec does not own canonical joint identity, simplex legality, sparsification, qualified skin, mesh binding, product state or proof. Dense decoded weights remain proposal/evidence consumed by Arachne/Compiler boundaries.
