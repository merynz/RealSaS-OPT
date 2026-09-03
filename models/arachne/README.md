# Arachne

Arachne is the learned skin-proposal subsystem conditioned on admitted surface evidence and a Compiler-qualified skeleton.

## Authority boundary

Arachne predicts skin-field latents / dense influence proposals and exposes `SkinProposalIR` to the Compiler. The current line uses the shared `SkinFieldCodec` decoder.

Arachne does **not** own canonical joint identity, skeleton authority, simplex legality, sparsification, skin qualification, mesh binding, product state or proof.

## Current source candidate

Current V2 candidates are mixed into:

`experiments/geppetto_arachne_r6_20260901/`

Known current Arachne code includes the V2 candidate architecture, V2 conditioning, shared codec binding, checkpoint/training/evaluation apparatus and teacher-target adapters. V1/legacy alternatives in the same experiment tree are not assumed current and require file-level classification.

## Target production layout

```text
models/arachne/
  README.md
  src/
    model.py
    conditioning.py
    checkpoint.py
  training/
    losses.py
    train.py
  evaluation/
  tests/
```

## Dependency firewall

Arachne may depend on the production `models/skin_field_codec/` API and Compiler proposal/IR contracts. It must not depend on truth-only experiment adapters at inference. `SkinProposalIR` is evidence; `QualifiedSkinIR` is Compiler-owned.