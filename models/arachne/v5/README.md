# Arachne V5 — Minimal K4 Direct Simplex

**Current FIT1-frozen Arachne source home.**

`RiggingSurfaceIR + QualifiedSkeletonIR -> V4 backbone -> K4×512 Z -> direct row-simplex decoder -> SkinProposalIR -> Compiler -> QualifiedSkinIR`

The V5 decoder performs point-wise joint competition with a masked row softmax. The historical A0 continuous-field model is not a runtime dependency.

## Frozen Mage FIT1 checkpoint

The promoted checkpoint is intentionally **composite**, matching the exact passing experiment rather than repacking hundreds of MB and creating a new byte identity:

- V4 backbone checkpoint: `552,378,987` bytes, SHA-256 `95c441f97b02123de1a5bc83bdf5ad223363c4b97927e8d420a0d246efbc1763`, Drive `1HJC4GPMcWa0jEXCYgctUtW09tXn4I9ND`;
- V5 direct-simplex decoder delta: `325,313` parameters, SHA-256 `13344178bf1b3ce96c9356456db0ad2c8a3945182a5ec63617c50137b8c52137`, Drive `1_o2XWO5BOSDKGvewJDSGS16jlcMuT9Dx`;
- total model parameters: `138,378,466`;
- source SHA-256: `a66adaebe92e9181873888ef90941ad87e4b3d835d21647e65176df4441a0f9e`.

Machine authority lives in `checkpoint_authority_v1.py` and `FROZEN_MAGE_FIT1_CHECKPOINT_V1.json`. Large checkpoint bytes remain external and hash-bound, exactly like the other promoted RealSaS model evidence.

## FIT1 closure

- GSA row-L1 p95: `0.04237784981177733`;
- deformation ratio: `0.019856400787830353`;
- articulated deformation ratio: `0.002780771814286709`;
- Compiler qualification: `950/950` rows, correction L1 `2.9468642839168442e-05`;
- A0 runtime model loaded: `false`;
- unseen-family claim: `false`;
- PRODUCT_PASS claim: `false`.

See `PROMOTED_MAGE_FIT_WITNESS_V1.json`, `canonical/ARACHNE_A1_V5_FIT1_EVIDENCE_MANIFEST_V1.json`, and `canonical/ARACHNE_A1_V5_MINIMAL_K4_DIRECT_SIMPLEX_FIT1_PROMOTION_20260911.md`.

Historical A0, A1 V4 and diagnostic experiments remain preserved scientific memory for FIT8/LOFO and future debugging; promotion supersedes them for current Mage FIT1 execution but does not delete them.
