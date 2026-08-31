# SkinTokens / TokenRig — Code-Level Reference Audit V1

**Date:** 2026-08-31  
**Status:** `R0_R4_COMPLETE__SKIN_CODEC_FUNCTION_EXTRACTED__PUBLIC_TRAINING_INTEGRATION_PARTLY_INCOMPLETE__MIT`

## Scope and license

Frozen upstream: `VAST-AI-Research/SkinTokens@273b691d35989d71cd17ff2895fdc735097b92d1`.

The repository is MIT licensed. RealSaS nevertheless applies the same scientific provenance discipline used for noncommercial references: reference facts, RealSaS contract requirements and new hypotheses remain explicitly separated.

SkinTokens is audited primarily as the **Arachne representation/codec reference**, not as authority for RealSaS skeleton topology.

## R0 — frozen source inventory

Inspected public snapshot includes:

- `README.md` blob `7b9d14a47adcc29d6bc99367e6d7780980ddd055`
- `LICENSE` blob `484d9bbef8720dc2797444a18c972a4e6245a07e`
- `src/model/tokenrig.py` blob `ab35e3fb8dd850737cba7d1782f03c75846971be`
- `src/model/skin_vae_model.py` blob `a2b4e88b6997883ce5861333bb498dc9593ff80d`
- `src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py` blob `6b66d9e9486d261f0a5d67f795bbe60499fb26e3`
- `src/data/sampler.py` blob `ef7bc0d8fccd2bd4e0b2bd13887c1c96b2caa6dc`
- `src/data/transform.py` blob `3601d755766932490a584aa495848402950a0e19`
- tokenizer/rig-package/model support files in the same frozen tree
- paper: `Skin Tokens: A Learned Compact Representation for Unified Autoregressive Rigging`, arXiv:2602.04805.

The public snapshot exposes enough inference/representation code to close the functional audit, but important wrapper-level training methods remain `NotImplemented`; exact published-run integration is therefore partly `NOT_PUBLICLY_VERIFIABLE` from code alone.

## R1 — exact information contract

### Product inference geometry

The released TokenRig path conditions on sampled surface:

`P_xyz + N_xyz`.

`tokenrig.py` concatenates positions/normals for the skin-VAE geometry condition and independently passes the same P+N geometry through a learned mesh encoder for the autoregressive model.

For RealSaS the comparison object is:

`B_A = ArachneConditioningAdapter(RiggingSurfaceIR, QualifiedSkeletonIR)`.

This is stronger than comparing SkinTokens against raw IRIS output: deterministic geometry assembly, normalization, qualified local geometry and the Compiler-owned skeleton are all credited if they are functions only of admitted product information.

### Existing-skeleton / skin-only compatibility

TokenRig can start generation from an existing skeleton-token prefix. Therefore the skin problem is not intrinsically dependent on TokenRig owning skeleton generation.

This maps cleanly to the RealSaS authority split:

`Compiler-qualified skeleton -> Arachne skin proposal`.

### Uniform vs bone-specific training samples

The released sampler constructs:

1. uniform surface P/N samples and sampled skin values;
2. optional per-bone dense samples from faces with nonzero GT influence and nearby geometry.

The second path explicitly depends on ground-truth skin weights. It is therefore a **training/codec target-construction mechanism**, not product-inference information.

RealSaS may construct an independently specified equivalent active-region training sampler from authoritative dense teacher weights without claiming that those fields exist at inference.

## R2 — implementation-independent representation decomposition

### A. Per-bone skin field

The important factorization is not direct `N surface points x J joints` regression as one undifferentiated matrix. The reference represents each joint/bone's influence as a sparse spatial field conditioned on geometry.

Functional form:

`per-bone influence samples + geometry -> field encoder -> compact latent -> finite scalar quantization -> short discrete state`

and

`discrete/continuous skin state + geometry condition -> query decoder -> continuous influence at requested surface points`.

### B. Geometry-conditioned FSQ-CVAE

The released `SkinFSQCVAEModel` has separate skin/field and geometry-condition encoders, latent projections and a geometry-conditioned decoder. Its default class signature contains reference values:

- latent channels `64`;
- encoder width `512`;
- decoder width `1024`;
- encoder layers `8`;
- decoder layers `16`;
- point positional embedding from 3D coordinates;
- finite scalar quantization when an FSQ config is supplied.

These exact widths/layers are reference values, not RealSaS requirements.

### C. Unified TokenRig sequence

The released TokenRig combines learned geometry conditioning, a causal transformer, structured skeleton tokens and skin tokens. It can generate both skeleton and skin, but this unified authority topology is **not** adopted by RealSaS.

The RealSaS functional analogue is:

`QualifiedSkeletonIR deterministic serialization/conditioning + geometry -> Arachne skin-state prediction -> skin-field decode -> SkinProposalIR`.

Compiler remains skeleton and final skin authority.

## R3 — training/loss audit

### Publicly supported reference facts

The paper describes a dedicated skin-codec training stage followed by sequence-model training and later policy/reward refinement. Paper-level objectives include sparse-field-aware reconstruction terms and autoregressive token supervision; these are reference evidence, not automatically RealSaS loss requirements.

### Code-level limitation

In the frozen public source:

- `SkinVAEModel.training_step` is `NotImplemented`;
- `SkinVAEModel.get_loss_dict` is `NotImplemented` in the exposed base wrapper;
- `SkinFSQCVAEModel.forward` is not the released end-to-end training integration;
- `TokenRig.training_step` is also `NotImplemented` in the exposed base model.

Therefore exact released-run loss weighting, optimizer schedule, nested-token curriculum and RL integration must be marked `NOT_PUBLICLY_VERIFIABLE` unless independently verified from paper/checkpoint metadata. We do **not** infer exact executable training behavior from paper prose or uncalled helpers.

### Consequence for RealSaS

The audit justifies the **functional obligations**, not an exact upstream training recipe:

- sparse positive regions must receive nontrivial supervision;
- a compact skin representation must prove reconstruction capacity before predictor training;
- geometry conditioning must survive decoding back to the admitted surface;
- final evaluation must include deformation, not static scalar error alone.

Exact BCE/MSE/Dice weights, token count, FSQ levels, Qwen choice or GRPO are not frozen RealSaS mechanisms by this audit.

## R4 — learned vs deterministic/postprocess responsibility

| Function | Responsibility in released reference |
|---|---|
| compact per-bone skin representation | learned codec + FSQ discretization |
| geometry-conditioned field decode | learned |
| skin-token generation from geometry/skeleton context | learned autoregressive model |
| grammar/vocabulary switching | deterministic constrained-generation protocol |
| existing-skeleton prefix construction | deterministic tokenization/serialization |
| assembly of per-bone fields into N x J matrix | deterministic indexing/assembly around learned fields |
| sampled-surface -> original mesh transfer in VAE helper | deterministic nearest-neighbor transfer |
| optional downstream export/postprocess | deterministic and must be audited separately from neural fidelity |

## RealSaS Compiler ownership correction

Arachne is not required to produce a legally final skin matrix. The current Compiler already owns:

- exact surface-lineage binding;
- exact skeleton-lineage binding;
- legal joint/surface reference checks;
- nonnegative/finite validation;
- bounded top-k sparsification when enabled;
- bounded simplex repair/renormalization;
- fail-closed rejection when correction exceeds policy.

Therefore Arachne's job is to produce **high-quality influence evidence/fields bound to the qualified skeleton**, not to duplicate legal/simplex authority inside the model.

## Functional obligations carried into Arachne design

An independent RealSaS candidate must demonstrate:

1. geometry-conditioned per-joint influence fields;
2. explicit conditioning on the exact `QualifiedSkeletonIR` lineage;
3. a codec/representation ceiling on authoritative dense RealSaS skin truth **before** training a predictor of that code/state;
4. sufficient active-region supervision so sparse trivial solutions cannot win by aggregate error;
5. decode/evaluation on every admitted `RiggingSurfaceIR` node without hidden source-mesh completion;
6. `SkinProposalIR` output with no silent helper-weight transport or teacher identity leakage;
7. Compiler qualification after prediction;
8. deformation/motion-sensitive proof after qualification.

FSQ, exact token counts, a Qwen-style LM, unified skeleton+skin autoregression and RL refinement are only `REFERENCE_SUPPORTED` mechanisms until separately justified.

## Why SkinTokens is the primary Arachne reference

SkinTokens is materially cleaner than RigAnything's released skin path for Arachne equivalence because:

- its core representation is explicitly a geometry-conditioned skin field;
- skin-only conditioning with an existing skeleton is supported;
- GT-skin-dependent dense samples are training-only target construction rather than inference requirements;
- RealSaS already has authoritative dense weight truth for eligible corpus assets;
- RealSaS supplies a Compiler-qualified skeleton rather than asking Arachne to own skeleton truth;
- its core skin representation does not require mesh-neighbor smoothing to define the learned field itself.

The unresolved material difference is **surface coverage/accessibility**: SkinTokens operates on full 3D mesh-surface samples, while shipping RealSaS intentionally exposes partial observation-grounded geometry.

## R0-R4 verdict

`PASS_PRIMARY_ARACHNE_REFERENCE_FUNCTION_EXTRACTED`

SkinTokens provides strong external evidence that geometry-conditioned compact skin fields are a viable solution family. It does not by itself prove that the partial RealSaS substrate is input-equivalent. That question moves to the shared R5 matrix and R6 oracle-substrate ceiling.
