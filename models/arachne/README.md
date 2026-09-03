# Arachne

Arachne is the learned skin-proposal subsystem conditioned on admitted surface evidence and a Compiler-qualified skeleton.

## Current mainline

`models/arachne/v2/`

Byte-preserved current source:

- `arachne_candidate_v2.py`
- `conditioning_v2.py`
- `conditioning_v1.py` — retained compatibility base used by V2 conditioning;
- `arachne_geometry_v2.py`

Two tiny dependency bridges keep the promoted V2 source byte-identical while making ownership explicit:

- `skin_field_codec_v1.py` re-exports the canonical current `models.skin_field_codec.v1` decoder;
- `geppetto_conditioning_v2.py` re-exports the canonical current `models.geppetto.v2` conditioning contract used by Arachne V2.

No duplicate learned Codec or Geppetto implementation is created inside Arachne.

## Authority boundary

Arachne predicts skin-field latents/dense influence proposals and emits `SkinProposalIR`. It does **not** own canonical joint identity, skeleton authority, simplex legality, sparsification, qualified skin, mesh binding, product state or proof.

## Compatibility residue

`conditioning_v1.py` contains both the older Geppetto and Arachne deterministic conditioning adapters because current Arachne V2 still calls the Arachne V1 base adapter before adding V2 pair geometry. The entire file is preserved for behavior stability; a later hygiene refactor may extract only the Arachne base after parity tests.

## Training status

Arachne training/evaluation/tail-remediation files remain in the R6 experiment tree while shared teacher-target, codec and diagnostic dependencies are classified. Current V2 inference ownership is now visible without pretending every research remediation script is current production code.
