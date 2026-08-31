# RigAnything — Code-Level Clean-Room Reference Audit V1

**Date:** 2026-08-31  
**Status:** `R0_R4_IN_PROGRESS__REFERENCE_FACTS_ONLY__NO_REALSAS_ARCHITECTURE_SEAL`

## Scope / license firewall

Frozen upstream: `Isabella98Liu/RigAnything@d03cdb21dd134fa81df6b0947522469db3f78bd2`.

The upstream repository is Adobe Research License / noncommercial research only. This artifact records independently stated behavioral facts, equations, tensor contracts, preprocessing and postprocessing. It is not a source transplant. No upstream product code is admitted into RealSaS product implementation.

## Sources inspected in this pass

- `README.md` blob `2879b90dab26dc61c107808f113a70039c69c07e`
- `LICENSE.md`
- `config.yaml` blob `921d5df9d5f4005b81de004103e42e39c23612d5`
- `scripts/inference.sh` blob `0bb52089be51e854649041d2673a2d0fdf348783`
- `inference.py` blob `1ec596e0e41df019428c433091f13ced302a4d11`
- `model/ar_rig_diffusion.py` blob `862c26e6d86470e3a8074c2b2a41d44401895455`
- `model/utils_ar_transformer.py` blob `2e22ba4417c4f2eba0f0075501801964cb77f16f`
- `model/diffloss.py` blob `a008237e04d08c79edc470ff5cdcbc3d70273bde`
- paper: `RigAnything: Template-Free Autoregressive Rigging for Diverse 3D Assets`, arXiv:2502.09615 / TOG 2025.

## R1 — Product-inference information contract

### External input

Public inference accepts `.glb` / `.obj`; the wrapper can optionally simplify the mesh before model inference.

### Geometry actually consumed by the neural path

The inference implementation constructs:

1. a **1024-sample surface point set** sampled from the mesh surface;
2. one corresponding surface normal per sampled point;
3. a normalized object frame using the sampled-point bounding-box midpoint as center and maximum absolute coordinate as scale;
4. optionally/full-resolution mesh vertex positions + vertex normals for dense skinning evaluation after skeleton generation.

The skeleton-generation neural input is therefore functionally a set/sequence of `1024 x (P_xyz, N_xyz)` rather than raw polygon connectivity.

Mesh faces/topology remain available outside the skeleton network and become relevant in downstream skinning transfer/smoothing/postprocessing.

### Current preliminary reference-to-RealSaS mapping target

The correct RealSaS comparison is **not raw IRIS depth**. Per the consumer-substrate boundary amendment, RigAnything's `(surface point, normal)` condition must be compared against a declared deterministic `GeppettoConditioningAdapter(RiggingSurfaceIR)`.

Current RealSaS `RiggingSurfaceIR` already exposes analytic `P`, support/provenance/raster bindings, optional qualified derived normal and local relations. Whether its **partial observation-supported coverage** is sufficient relative to RigAnything's mesh-wide surface sample remains OPEN and must be measured.

## R2 — Functional architecture decomposition

### Shape encoding

Reference structure:

```text
1024 x 6D surface P+N
 -> point MLP
 -> width-d shape tokens
 -> global self-attention among shape tokens
```

Public config:

- input point channels `6`;
- point-tokenizer hidden `512`;
- transformer width `1024`;
- transformer layers `12`;
- attention head dimension `16`;
- maximum joint sequence `64` in the released checkpoint/config.

### Skeleton generation

The skeleton is represented as a variable-length autoregressive sequence of joint positions plus parents. The paper serializes ground-truth trees in BFS order and randomizes sibling order during training to avoid treating arbitrary sibling ordering as unique truth.

Functional chain:

```text
shape tokens + previous skeleton state
 -> hybrid transformer
 -> next-joint context
 -> continuous probabilistic next-joint position
 -> explicit parent-candidate scores
 -> update skeleton state
 -> endogenous stop/count
```

The public implementation uses:

- shape tokens that self-attend globally;
- skeleton tokens that attend all shape tokens and causally attend prior skeleton tokens;
- a diffusion-conditioned continuous 3D position decoder;
- explicit pairwise parent scoring over generated joint representations;
- a self-parent event for the newly generated joint as the sequence termination signal; the terminating dummy joint is removed.

### Continuous joint distribution

The reference uses conditional diffusion rather than direct coordinate regression. Public `DiffLoss` implements Gaussian diffusion training loss machinery and a reverse sampler; the paper specifies noise-prediction MSE for ground-truth joint coordinates.

This is `REFERENCE_SUPPORTED`; it is **not yet** classified as a mandatory RealSaS mechanism. The functional requirement is to represent genuine joint-position ambiguity without forcing mean-seeking collapse. Whether diffusion itself is necessary must be established by oracle-substrate comparison/ablation before the Geppetto seal.

### Parent / topology representation

The reference does not infer topology with a post-hoc MST. Each newly generated joint receives parent probabilities against prior generated joints. The paper supervises connectivity explicitly; inference selects a parent from the predicted distribution.

For RealSaS, the functionally corresponding output is not a reference-style final tree authority. Geppetto must emit sufficiently expressive parent/root/edge evidence into `SkeletonProposalIR`; the Compiler remains the sole topology authority.

### Skinning path present in RigAnything

After skeleton generation, the reference pairs each surface/point token with each generated joint token and maps each pair to an influence logit. Softmax across joints yields a per-surface influence distribution in the paper.

The released inference wrapper additionally applies deterministic operations before final GLB export:

- retain top-5 joint logits per point;
- softmax;
- zero very small weights using the released threshold;
- renormalize;
- mesh-neighbor smoothing for multiple iterations;
- nearest-neighbor transfer to duplicate/original GLB vertices;
- final renormalization.

These operations must not be credited to the neural skinning head when comparing model capability.

## R3 — Training/loss facts currently verifiable

### From paper

Verified functional objectives:

- joint position: diffusion noise-prediction MSE;
- connectivity: explicit supervised connectivity classification loss;
- skinning: weighted cross-entropy using ground-truth influence weights as target mass;
- skeleton serialization: BFS; sibling order randomized during training;
- training data: RigNet plus curated rigged Objaverse; random pose augmentation is reported.

### From public config

The released config records Adam-like optimizer hyperparameters (`beta1=.9`, `beta2=.95`), LR `1e-4`, weight decay `.05`, warmup `500`, BF16 AMP/TF32 and the architecture values above.

### Public-release limitation

The current public repository presents inference code and model components but does **not** expose a complete training forward/integration path for the released model. An open upstream issue also asks for training code release. Therefore exact loss weighting, batching/data implementation and any non-paper training details are `NOT_PUBLICLY_VERIFIABLE` unless recovered from an authoritative paper/source/checkpoint artifact.

Config keys alone must not be treated as proof that a particular loss was active in the released training run.

## R4 — Learned vs deterministic/postprocess responsibility

| Capability | Reference source |
|---|---|
| global surface context | learned transformer |
| variable joint count | autoregressive learned generation + explicit stop convention |
| joint-position multimodality | learned diffusion distribution |
| parent evidence | learned explicit candidate scoring |
| final skeleton serialization | deterministic sequence convention |
| raw skin influence logits | learned point/joint pair head |
| top-k sparsification | deterministic inference postprocess |
| small-weight cutoff | deterministic inference postprocess |
| mesh-neighbor smoothing | deterministic mesh-topology postprocess |
| transfer to original GLB vertices | deterministic nearest-neighbor postprocess |

## First Geppetto functional-equivalence obligations

Before calling the Geppetto problem externally supported under RealSaS inputs, RealSaS must demonstrate a counterpart for every material function below:

1. **shape conditioning:** a deterministic adapter can expose an adequate global P+N surface sample/descriptor from `RiggingSurfaceIR`;
2. **coverage:** partial observed surface retains enough global structural evidence for the reference-class task;
3. **variable cardinality:** no fixed-template joint count assumption;
4. **structural ambiguity:** sibling/topology ambiguity is not collapsed into a unique arbitrary teacher serialization;
5. **continuous position distribution:** genuinely multimodal joint positions can remain multimodal/uncertain;
6. **relational topology evidence:** each proposed control has explicit relational parent/root evidence sufficient for Compiler qualification;
7. **termination/count:** the model can express when no further supported controls should be proposed;
8. **oracle ceiling:** the independently implemented candidate can fit/solve exact RealSaS consumer substrate before predicted-IRIS error is introduced.

Exact BFS serialization, diffusion, transformer width 1024 or a 64-joint cap are **reference mechanisms**, not automatically RealSaS requirements. A different independent mechanism is admissible only if it fulfills the same function and passes the same oracle/downstream tests.

## Key open equivalence question

The current central uncertainty is no longer field type (`P+N` is plausibly derivable) but **surface coverage/accessibility**:

> Can observation-grounded `RiggingSurfaceIR`, after deterministic SurfaceBuilder and conditioning adapter, expose enough of the object-wide surface organization that a RigAnything-class template-free skeleton generator remains solvable without hidden completed mesh geometry?

This must be answered empirically; it may not be inferred from semantic similarity.

## Status / next audit actions

R0-R4 are started but not closed. Remaining work includes paper equation/ablation cross-check, checkpoint/config provenance where accessible, exact preprocessing edge cases, and the R5 field-by-field RealSaS equivalence matrix. No Geppetto architecture/loss/training seal is authorized by this draft.