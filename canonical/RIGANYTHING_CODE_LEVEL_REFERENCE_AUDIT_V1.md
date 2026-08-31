# RigAnything — Code-Level Clean-Room Reference Audit V1

**Date:** 2026-08-31  
**Status:** `R0_R4_COMPLETE__FUNCTIONAL_REFERENCE_EXTRACTED__TRAINING_INTEGRATION_PARTLY_NOT_PUBLICLY_VERIFIABLE__NO_CODE_REUSE`

## Scope and license firewall

Frozen upstream: `Isabella98Liu/RigAnything@d03cdb21dd134fa81df6b0947522469db3f78bd2`.

The upstream repository is under the Adobe Research License and is restricted to noncommercial research. This audit records independently stated behavioral facts, tensor contracts, preprocessing, functional factorization and postprocessing. No upstream implementation is admitted as RealSaS product code, no source is translated line-for-line, and no upstream package is a RealSaS product dependency.

The scientific goal is not architectural imitation. It is to identify the complete function that a RigAnything-class solution performs under its actual input substrate so RealSaS can independently satisfy the same functional obligations.

## R0 — frozen source inventory

Inspected public snapshot:

- `README.md` blob `2879b90dab26dc61c107808f113a70039c69c07e`
- `LICENSE.md` blob `ca47ac2dd8a2cac08199441e761ce07f7f9fbaa6`
- `config.yaml` blob `921d5df9d5f4005b81de004103e42e39c23612d5`
- `inference.py` blob `1ec596e0e41df019428c433091f13ced302a4d11`
- `model/ar_rig_diffusion.py` blob `862c26e6d86470e3a8074c2b2a41d44401895455`
- `model/diffloss.py` blob `a008237e04d08c79edc470ff5cdcbc3d70273bde`
- `model/utils_ar_transformer.py` blob `2e22ba4417c4f2eba0f0075501801964cb77f16f`
- repository tree at the frozen commit
- paper: `RigAnything: Template-Free Autoregressive Rigging for Diverse 3D Assets`, arXiv:2502.09615 / TOG 2025.

The frozen public repository contains inference/model components but no complete released dataset/training driver matching the published run. Exact non-paper training integration is therefore `NOT_PUBLICLY_VERIFIABLE`.

## R1 — exact product-inference information contract

### Skeleton-conditioning geometry

The released inference route:

1. loads the input mesh;
2. samples exactly `1024` points from the complete mesh surface;
3. attaches the corresponding face normal to each sample;
4. normalizes normals;
5. centers geometry by the sampled-point AABB midpoint;
6. scales by the maximum absolute centered sampled coordinate;
7. feeds `1024 x 6` `(P_xyz,N_xyz)` to the learned point tokenizer.

Thus polygon connectivity is **not a direct neural input to the skeleton generator**. Its geometric conditioning is a complete-mesh surface sample with normals.

Released config:

- point input channels: `6`;
- point count: `1024`;
- point-token hidden: `512`;
- transformer width: `1024`;
- transformer layers: `12`;
- attention head dimension config: `16`;
- released maximum joint sequence: `64`;
- diffusion sampling steps: `300`.

### Extra geometry in the skin/export route

The inference wrapper also retains full mesh vertices/normals. After skeleton generation the model can evaluate raw skin logits on full mesh points. Mesh connectivity is then used by deterministic smoothing and export transfer. These fields are **not** part of the skeleton neural input and must not be credited to the skeleton learner.

### RealSaS comparison boundary

The comparison object is not raw IRIS output. It is:

`B_G = GeppettoConditioningAdapter(RiggingSurfaceIR)`

where `RiggingSurfaceIR` is produced from the shipping-observable stack by the deterministic pre-Geppetto geometry layer (planned class name `GeometricSubstrateAssembler`). Deterministic resampling, normalization, local-normal derivation and local geometric descriptors are admissible adapter operations; source-mesh completion is not.

## R2 — implementation-independent functional decomposition

### A. Global shape conditioning

`complete-mesh surface P+N -> point tokens -> globally contextualized shape tokens`

The shape tokens self-attend and remain available to every generated skeleton step.

### B. Variable-cardinality skeleton generation

`shape context + previous generated skeleton -> next-joint context -> continuous next-joint position -> parent evidence -> update state -> endogenous stop`

The reference is template-free in the relevant sense: the number of generated joints is not supplied as a fixed semantic template at inference.

### C. Continuous joint position

The released model uses a conditional diffusion decoder for each next 3D joint location. Functionally, this supplies a continuous conditional distribution rather than a single direct mean-regression coordinate.

For RealSaS, **continuous/multimodal position evidence is a functional obligation; diffusion itself is only a reference-supported mechanism** until an independent oracle-substrate comparison shows it is necessary.

### D. Relational topology evidence

For a newly generated joint, the reference explicitly scores candidate parents from generated joint representations. The terminating event is encoded by selecting the current/new joint as its own parent; that terminating dummy joint is then removed.

RealSaS need not duplicate this authority topology. Geppetto must provide sufficiently expressive joint/root/edge evidence in `SkeletonProposalIR`; the existing Compiler global graph optimizer owns canonical root/parent selection and mints final joint identities.

### E. Skin influence head present in RigAnything

The reference pairs contextualized point tokens with generated joint tokens and predicts point/joint influence logits. This establishes another learned skinning solution family, but it is **not the preferred Arachne clean reference** because the released final skin result materially depends on mesh-topology postprocessing described below.

## R3 — training/loss audit

### Publicly supported facts

Paper-level functional training facts support:

- autoregressive skeleton serialization with explicit structural ordering;
- continuous joint-position diffusion supervision;
- explicit connectivity/parent supervision;
- skin-influence supervision;
- data drawn from rigged 3D assets with augmentation described by the paper.

The released config exposes optimizer/runtime values including LR `1e-4`, betas `.9/.95`, weight decay `.05`, warmup `500`, BF16 AMP and TF32.

### Binding public-release limitation

The frozen repository does not expose a complete authoritative training driver/data pipeline for the published model. Therefore:

- exact final loss composition/weights beyond independently verified paper statements;
- exact batching and dataset mixture;
- all augment probabilities;
- checkpoint selection semantics;
- any unpublished curriculum details

are `NOT_PUBLICLY_VERIFIABLE` and may not be reconstructed from suggestive config names.

This does **not** weaken the functional reference result because the inference contract and causal factorization are directly code-verifiable.

## R4 — neural vs deterministic responsibility

| Function | Responsibility in released reference |
|---|---|
| surface P+N contextualization | learned |
| variable joint cardinality | learned autoregressive state + explicit stop convention |
| continuous joint-position distribution | learned diffusion decoder |
| parent evidence | learned candidate scoring |
| final sequential serialization convention | deterministic protocol around learned outputs |
| raw skin influence logits | learned point/joint pair head |
| top-5 sparsification | deterministic inference wrapper |
| low-weight cutoff (`<0.068`) | deterministic inference wrapper |
| renormalization | deterministic inference wrapper |
| mesh-neighbor skin smoothing | deterministic, 10 iterations / factor .35 in released inference |
| duplicate/original GLB transfer | deterministic nearest-neighbor transfer |

The mesh-neighbor smoothing and transfer mean final RigAnything skin quality cannot be used as evidence that its raw neural skin head alone solves Arachne's problem under a topology-free partial substrate.

## Compiler ownership correction

RealSaS historically attempted model-free rig synthesis and retains substantial deterministic logic below Geppetto. The current Compiler is not a passive validator:

- Geppetto proposal IDs are non-canonical;
- proposal joints provide positions/root evidence/support;
- proposal edges provide scored relational evidence;
- Compiler builds a global graph optimization request;
- `optimize_canonical_graph_v18_98` selects the admitted root/parents;
- Compiler mints new canonical product joint IDs and rejects failed graph qualification.

Therefore the correct functional comparison is:

`RigAnything skeleton function`

versus

`independent Geppetto evidence model + existing RealSaS Compiler graph authority`.

It would be an architectural regression to duplicate the Compiler's global root/tree solver inside `GeometricSubstrateAssembler` or to require Geppetto to own the final canonical tree merely because RigAnything emits a tree sequentially.

## Functional obligations carried into the Geppetto design

A RealSaS candidate must independently satisfy all material reference functions:

1. global shape conditioning from the admitted consumer substrate;
2. variable control cardinality / explicit unsupported-overflow behavior;
3. continuous joint-position evidence that does not force unsupported multimodal means;
4. explicit root/connectivity evidence;
5. adequate evidence for the Compiler to recover a valid canonical global graph;
6. no template-specific semantic joint identity assumption;
7. no silent truncation (`160` current C0 capacity; above it abstains unless causally reopened);
8. oracle-substrate ceiling before predicted IRIS noise is introduced.

Reference-specific BFS serialization, diffusion, width `1024`, `12` layers and a `64`-joint ceiling are **not** copied requirements.

## R0-R4 verdict

`PASS_REFERENCE_FUNCTION_EXTRACTED_WITH_LICENSE_FIREWALL`

RigAnything is strong evidence that template-free skeleton generation is solvable from a complete 3D surface `P+N` substrate. The only material input-equivalence issue left for RealSaS is not tensor type but **coverage/accessibility**: RigAnything samples the complete mesh surface while shipping RealSaS geometry is deliberately partial and observation-grounded.

That coverage question moves to the shared R5 matrix and R6 oracle-substrate ceiling. Until R6, the claim `RealSaS input is equivalent to RigAnything input` remains forbidden.
