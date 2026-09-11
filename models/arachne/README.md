# Arachne

Arachne is the learned skin-proposal subsystem conditioned on admitted `RiggingSurfaceIR` and a Compiler-qualified skeleton.

## Current mainline

`models/arachne/v5/`

Current route:

`RiggingSurfaceIR + QualifiedSkeletonIR -> A1 V4 backbone -> K4×512 Z -> V5 direct row-simplex decoder -> SkinProposalIR -> Compiler -> QualifiedSkinIR`

Mage FIT1 is closed for this route. The historical A0 continuous-field model is **not** loaded at V5 runtime.

Checkpoint authority:

- `models/arachne/v5/checkpoint_authority_v1.py`
- `models/arachne/v5/FROZEN_MAGE_FIT1_CHECKPOINT_V1.json`
- `models/arachne/v5/PROMOTED_MAGE_FIT_WITNESS_V1.json`

## Preserved historical/research lineage

`models/arachne/v2/`, V4 source, A0/SkinFieldCodec experiments and all diagnostic branches remain preserved for provenance and FIT8/LOFO failure-memory. They are not deleted, but they are superseded for current Mage FIT1 execution.

## Authority boundary

Arachne owns learned skin semantics/proposal. It does **not** own canonical skeleton identity, simplex legality, qualified skin, mesh binding, product state or proof. Those remain Compiler-owned.
