# SkinFieldCodec

SkinFieldCodec is the learned continuous skin-influence representation subsystem used by the current Arachne line.

## Authority boundary

The codec learns a per-joint latent influence-field representation and a shared decoder from admitted surface + qualified-skeleton conditioning to dense proposal weights.

Teacher dense weights are permitted only on the training/encoding lane. Product inference must decode from predicted latents and admitted current conditioning. The codec never owns skin qualification, sparsification, canonical IDs or product state.

## Current source candidate

Current implementation candidate:

`experiments/geppetto_arachne_r6_20260901/skin_field_codec_v1.py`

Its current semantics explicitly separate teacher-only encoding from inference decoding and preserve a shared decoder used by Arachne. Training targets/losses/checkpoint compatibility remain audit-classified separately.

## Target production layout

```text
models/skin_field_codec/
  README.md
  src/
    codec.py
    config.py
  training/
    losses.py
    teacher_encoding.py
  evaluation/
  tests/
```

The current candidate is not copied here until the model-source ownership audit proves which adjacent config/training files are part of the current contract and which are experiment-only.