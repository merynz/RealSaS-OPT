# SkinTokens / TokenRig — Code-Level Reference Audit V1

**Date:** 2026-08-31  
**Status:** `R0_R4_IN_PROGRESS__REFERENCE_FACTS_ONLY__NO_ARACHNE_ARCHITECTURE_SEAL`

## Scope / license

Frozen upstream: `VAST-AI-Research/SkinTokens@273b691d35989d71cd17ff2895fdc735097b92d1`.

The public repository is MIT licensed. RealSaS still applies the same scientific clean-room/provenance discipline: reference facts are separated from RealSaS hypotheses, and no reference mechanism is promoted merely because it exists upstream.

## Sources inspected in this pass

- `README.md` blob `7b9d14a47adcc29d6bc99367e6d7780980ddd055`
- `LICENSE` blob `484d9bbef8720dc2797444a18c972a4e6245a07e`
- `src/model/tokenrig.py` blob `ab35e3fb8dd850737cba7d1782f03c75846971be`
- `src/model/spec.py` blob `fbb0e932f5dcfdb45351b63fad325018d62f84b0`
- `src/model/skin_vae_model.py` blob `a2b4e88b6997883ce5861333bb498dc9593ff80d`
- `src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py` blob `6b66d9e9486d261f0a5d67f795bbe60499fb26e3`
- `src/data/sampler.py` blob `ef7bc0d8fccd2bd4e0b2bd13887c1c96b2caa6dc`
- `src/data/transform.py` blob `3601d755766932490a584aa495848402950a0e19`
- `src/tokenizer/tokenizer_part.py` blob `5e91e3c37be61653dec2be3263bfd2b53f90e5fc`
- `demo.py` blob `a2cb3a2bc6380428ff16b74592049a2f3fb18402`
- paper: `Skin Tokens: A Learned Compact Representation for Unified Autoregressive Rigging`, arXiv:2602.04805.

## R1 — Information contract

### Product inference geometry

The released TokenRig path operates on sampled surface geometry represented as vertex/sample positions and normals. `tokenrig.py` forms geometry conditioning from `[vertices, normals]` and also passes the same P+N geometry through a learned mesh encoder.

The skin VAE's geometry conditioning uses unordered point-set geometry; the lower-level CVAE separates 3D position from accompanying point features and builds geometry latent tokens by attention over the point set.

### Existing-skeleton / skin-only mode

The released demo supports a mode that uses an existing skeleton and generates skin only. In code, TokenRig can accept pre-existing skeleton tokens; generation begins from that fixed skeleton prefix and continues into skin tokens. This is especially relevant to RealSaS because Arachne consumes `Compiler-qualified QualifiedSkeletonIR` rather than owning skeleton topology.

### Training-only skin-target geometry

The released sampler distinguishes:

1. **uniform surface samples** with positions/normals and dense skin matrix values;
2. **bone-specific dense samples** drawn from faces carrying non-zero influence for a particular bone and nearby regions.

The bone-specific sampling depends on ground-truth skin weights and therefore is a **training/codec target-construction device**, not information required from product inference. RealSaS is allowed to construct equivalent training-only target samples from authoritative skin truth without implying that such fields exist at inference.

This distinction is binding for the R5 equivalence matrix.

## R2 — Functional representation / architecture

### Stage 1: SkinTokens representation

The paper and released code implement a geometry-conditioned discrete representation of each bone's skin influence field.

Functional factorization:

```text
per-bone skin field + geometry
 -> skin-field encoder
 -> compact continuous latent tokens
 -> finite scalar quantization (FSQ)
 -> short discrete SkinToken sequence

geometry
 -> geometry/condition encoder
 -> condition tokens

SkinTokens + geometry condition
 -> decoder queried at surface points
 -> continuous per-point influence field for that bone
```

The lower-level released `SkinFSQCVAEModel` uses:

- unordered point-set attention encoders for skin/shape information;
- a geometry condition encoder;
- latent projection to a compact channel space;
- finite scalar quantization / fixed-grid discrete codes;
- a geometry-conditioned decoder queried at sampled points.

The public default class signature includes latent width `64`, encoder width `512`, decoder width `1024`, 8 encoder layers, 16 decoder layers and configurable sample-token count. These are reference implementation values, not RealSaS requirements.

### Sparse-skin objective

The paper explicitly treats dense direct regression as poorly conditioned by the extreme sparsity of the N x J matrix. SkinTokens instead compresses the **per-bone field** and trains reconstruction with a composite objective including BCE, MSE and Dice terms; Dice is used to amplify supervision on sparse positive influence regions.

The paper reports that very short token sequences can reconstruct skin fields at useful fidelity and selects a finite-scalar code configuration balancing compression and reconstruction.

This strongly supports the existing RealSaS rule:

> **Arachne must prove its skin-field representation/codec ceiling before training a predictor of that representation.**

### Stage 2: TokenRig sequence model

Released TokenRig uses:

- a learned mesh encoder producing global geometry latents;
- a causal language-model-style transformer;
- a structured skeleton-token prefix;
- a vocabulary switch after the skeleton terminator;
- a fixed number of SkinTokens per skeleton bone;
- constrained generation grammar so skeleton vocabulary and skin-token vocabulary are used in the correct sequence regions.

The skin-token block is therefore explicitly aligned to the generated/conditioned skeleton order.

For RealSaS, unifying skeleton generation and skinning into one authority is **not** adopted: the functional analogue is to condition Arachne on a deterministic serialization/embedding of `QualifiedSkeletonIR` and generate/decode one skin field per compiler-qualified joint.

### Skeleton tokenization in TokenRig

The released tokenizer quantizes skeleton coordinates and serializes branch/chain structure into discrete tokens. The code can use ordered chain/part information where available. This is a reference sequence representation, not a RealSaS canonical topology authority.

Arachne only needs a deterministic conditioning representation of the already-qualified skeleton. Geppetto remains independently audited against template-free skeleton references.

## R3 — Training/loss facts currently verifiable

### Skin codec from paper

The paper describes:

- FSQ-CVAE training for skin-field reconstruction;
- BCE + small MSE + Dice composite reconstruction objective;
- nested dropout / variable token-budget training;
- importance sampling focused on active deformation regions;
- finite scalar quantization without a learned VQ codebook;
- a long dedicated codec-training stage before TokenRig sequence learning.

### TokenRig from paper/release

The paper describes supervised next-token training for the unified sequence followed by GRPO post-training. The RL stage uses explicit rig-quality rewards, including joint coverage, bone/mesh containment, skin coverage/sparsity and deformation smoothness.

The README and paper report separate large training stages for the codec and sequence model and a short RL refinement stage.

### Public-release limitation

In the inspected repository snapshot, `TokenRig.training_step`, `SkinVAEModel.training_step`, and parts of the wrapper-level VAE encode/loss integration are intentionally/not-yet implemented in public source. Therefore exact supervised training code and every hyperparameter cannot be claimed from repository code alone.

Paper-level equations and checkpoint hyperparameters may be used as reference facts where independently verifiable; absent integration details remain `NOT_PUBLICLY_VERIFIABLE`.

## R4 — Inference and postprocessing responsibility

### Codec decode

Skin tokens are converted back to FSQ codes and decoded under geometry condition into one continuous influence value per sampled surface point and bone. Per-bone decoded fields are assembled into an N x J skin matrix.

### Surface transfer/export

The VAE prediction helper can transfer sampled skin predictions to original mesh vertices through nearest sampled-vertex lookup. The demo can additionally apply an optional voxel-based skin postprocess before export. Export also imposes a bounded group-per-vertex representation.

These deterministic transfer/postprocess steps must be separated from neural representation fidelity when comparing Arachne.

## First Arachne functional-equivalence obligations

Before treating Arachne as a solved problem class under RealSaS inputs, RealSaS must independently implement and test counterparts for the material functions below:

1. **geometry conditioning:** `ArachneConditioningAdapter(RiggingSurfaceIR, QualifiedSkeletonIR)` exposes stable P/N/local geometry and skeleton-conditioned descriptors;
2. **per-joint field factorization:** skin is representable as one sparse spatial influence field per qualified joint, without relying on authored helper identity;
3. **compact representation ceiling:** the chosen discrete/continuous codec can reconstruct authoritative RealSaS skin truth below frozen static and deformation-sensitive tolerances before any token predictor is trained;
4. **positive-region supervision:** sparse active influences receive sufficient training signal; trivial near-zero solutions cannot look good by aggregate error alone;
5. **skeleton conditioning:** skin representation/prediction is explicitly bound to the exact `QualifiedSkeletonIR` lineage and joint ordering/identity adapter;
6. **surface decoding:** compact skin state can decode back onto all admitted `RiggingSurfaceIR` nodes with no hidden source-mesh dependency;
7. **simplex/sparsity boundary:** learned outputs remain proposals; Compiler owns legal references, simplex correction, influence-limit policy and fail-closed behavior;
8. **deformation validation:** static weight error is insufficient; final qualification includes deformation/motion proof.

FSQ itself, Qwen, unified skeleton+skin autoregression, the exact token count or GRPO are **reference-supported mechanisms**, not automatically mandatory. If RealSaS uses a different mechanism, it must satisfy the same functional obligations and match or exceed the oracle-substrate/downstream ceiling.

## Important preliminary equivalence finding

SkinTokens is a particularly clean Arachne reference because its **product-inference geometry requirement is much closer to the RealSaS consumer substrate than its training pipeline first appears**:

- inference conditioning is sampled surface P+N plus skeleton/skin tokens;
- dense bone-specific samples are constructed using ground-truth influence fields for codec training, so they are not missing product-input information;
- RealSaS already owns dense authoritative skin truth in the teacher corpus and can construct training-only per-joint active-region samples;
- Arachne receives a stronger authority boundary than TokenRig's internally generated skeleton because it consumes Compiler-qualified skeleton state.

The unresolved material difference is again **coverage/topology of the surface substrate**: TokenRig/SkinTokens are trained against full 3D mesh surfaces, whereas RealSaS product geometry remains partial observation-grounded `RiggingSurfaceIR`.

## Status / next audit actions

R0-R4 are started but not closed. Next work is the exact R5 field-by-field matrix, codec-capacity design mapping onto RealSaS dense skin truth, and a clean distinction between paper-only training facts and executable released inference behavior. No Arachne architecture/loss/training seal is authorized by this draft.