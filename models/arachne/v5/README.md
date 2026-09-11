# Arachne V5 — Minimal K4 Direct Simplex

Current FIT1-frozen Arachne source home.

`RiggingSurfaceIR + QualifiedSkeletonIR -> V4 backbone -> K4×512 Z -> direct row-simplex decoder -> SkinProposalIR`

The V5 decoder imposes point-wise joint competition directly with a masked softmax. It does not load the historical A0 continuous-field model at runtime.

## Frozen Mage FIT1 witness

- architecture: `RealSaS.Arachne.A1.MinimalK4DirectSimplex.v5`
- V4 backbone checkpoint SHA-256: `95c441f97b02123de1a5bc83bdf5ad223363c4b97927e8d420a0d246efbc1763`
- V5 source SHA-256: `a66adaebe92e9181873888ef90941ad87e4b3d835d21647e65176df4441a0f9e`
- decoder delta SHA-256: `13344178bf1b3ce96c9356456db0ad2c8a3945182a5ec63617c50137b8c52137`
- GSA p95: `0.04237784981177733`
- deformation ratio: `0.019856400787830353`
- articulated deformation ratio: `0.002780771814286709`
- unseen-family claim: `false`

See `PROMOTED_MAGE_FIT_WITNESS_V1.json` and `canonical/ARACHNE_A1_V5_FIT1_EVIDENCE_MANIFEST_V1.json`.

Historical Arachne/SkinFieldCodec experiments remain evidence/provenance and are not erased by this promotion.
