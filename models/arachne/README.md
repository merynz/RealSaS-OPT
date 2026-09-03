# Arachne

Arachne is the learned skin-proposal subsystem conditioned on admitted surface evidence and a Compiler-qualified skeleton.

## Current mainline

`models/arachne/v2/`

Byte-preserved current inference source:

- `arachne_candidate_v2.py`
- `conditioning_v2.py`
- `conditioning_v1.py` — retained compatibility base used by V2 conditioning;
- `arachne_geometry_v2.py`

Dependency bridges keep those files byte-identical while ownership remains explicit:

- `skin_field_codec_v1.py` -> canonical current SkinFieldCodec implementation/loss;
- `geppetto_conditioning_v2.py` -> canonical current Geppetto V2 conditioning contract;
- `codec_deformation_loss_v1.py` -> canonical Codec deformation-training primitive;
- `train_codec_r6_a0_v1.py` -> canonical Codec A0 qualification token contract.

No duplicate learned Codec or Geppetto implementation is created inside Arachne.

## Current A1 training/evaluation

The current base A1 lane is now visible beside Arachne:

- `arachne_tail_objective_v1.py` — generic family/semantic-ID agnostic hard-tail row objective used by current base A1 training;
- `train_arachne_r6_a1_v1.py` — byte-preserved current A1 optimization;
- `eval_arachne_r6_a1_v1.py` — byte-preserved current A1 evaluation.

`arachne_tail_remediation_v1.py` remains in `experiments/`: it explicitly describes a conditional remediation stage after a frozen A1 p95 failure, whereas the current A1 train step already carries generic hard-tail pressure in its base objective. It is preserved as research/remediation provenance, not silently promoted as a second base training authority.

## Authority boundary

Arachne predicts skin-field latents/dense influence proposals and emits `SkinProposalIR`. It does **not** own canonical joint identity, skeleton authority, simplex legality, sparsification, qualified skin, mesh binding, product state or proof.

## Compatibility residue

`conditioning_v1.py` contains both the older Geppetto and Arachne deterministic conditioning adapters because current Arachne V2 still calls the Arachne V1 base adapter before adding V2 pair geometry. The entire file is preserved for behavior stability; a later hygiene refactor may extract only the Arachne base after parity tests.
