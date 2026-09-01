# RealSaS — Full-Product Model Implementation Plan V1

**Binding:** `FULL_PRODUCT_SINGLE_SPECIMEN_FIT_CONTRACT_V1.md`  
**Topology authority:** current architecture V3  
**Specimen state:** `NOT_SELECTED`  

## Goal

Implement the actual current RealSaS product-intent learned stack before binding a specimen. The final investor demo differs from the generic program only in optimizer data distribution: one full-truth specimen.

## Source namespace

Final source lives under:

`demo/investor_e2e/product_fit_v3/`

The earlier `realsas_demo_e2e/*DemoV1` code is quarantined smoke history and cannot be imported by final fit/inference entrypoints.

---

## 1. IRIS Reprojection-Centered V2-A

### 1.1 Observation / camera layer

Files:

- `iris_v2a/observation.py`
- `iris_v2a/camera.py`

Responsibilities:

- ObservationContract V1 Mode-G validation;
- exactly 8 ordered views;
- native 1024 RGBA authority;
- exact orthographic camera basis;
- native alpha typed support;
- deterministic 518 foundation resize with whole-canvas framing;
- no adaptive crop or per-view zoom.

### 1.2 Frozen foundation path

File:

- `iris_v2a/foundation_dinov2.py`

Responsibilities:

- DINOv2-S/14 frozen backbone adapter;
- 518x518 / 37x37 patch grid;
- multi-level features exposed through a stable tensor contract;
- frozen feature cache support with model/checkpoint/hash provenance;
- zero gradient into foundation weights for the initial fitted architecture.

A synthetic test provider may implement the same interface without DINO weights. It is test apparatus only and cannot be used in the final run.

### 1.3 Native high-resolution path

File:

- `iris_v2a/native_pyramid.py`

Responsibilities:

- 1024 RGBA shared per-view pyramid;
- preserve sub-patch-thin detail;
- multi-scale native features;
- learned, trainable, same weights across all 8 views.

### 1.4 Visual-hull / q-domain geometry

Files:

- `iris_v2a/hull.py`
- `iris_v2a/q_lattice.py`
- `iris_v2a/reprojection.py`

Responsibilities:

- conservative alpha-based visual hull candidate domain;
- canonical world q lattice;
- exact q -> every-view reprojection;
- foreground/validity gating;
- deterministic geometry only;
- no learned camera or hidden completion.

### 1.5 Reprojection evidence learner

Files:

- `iris_v2a/evidence_field.py`
- `iris_v2a/model.py`

For each admissible q:

- sample frozen foundation descriptors in every view;
- sample learned native multi-scale features in every view;
- preserve per-view tokens and validity until q-state exists;
- robust attention/aggregation across valid views;
- construct trainable compact q evidence;
- scatter to regular 3D field;
- world-space approximately isotropic 3D regularization;
- extract supported/ambiguous depth modes along exact camera rays;
- local continuous depth refinement conditioned on high-resolution native features;
- emit forward depth, support/validity, uncertainty/risk evidence.

External learned geometry authority remains depth d.

### 1.6 IRIS losses

File:

- `iris_v2a/losses.py`

Training-only truth may supervise:

- supported forward depth robust NLL;
- alpha/support consistency;
- q-field support occupancy only where observation-equivalent truth is valid;
- exact multi-view reprojection consistency;
- continuous refinement residual;
- calibrated geometric uncertainty.

Forbidden loss inputs:

- joint/skin/part/product-importance labels;
- compiler failure labels;
- source rig identity.

---

## 2. Geppetto R6

### 2.1 Conditioning adapter

File:

- `geppetto_r6/conditioning.py`

Input: `RiggingSurfaceIR S` only.

Deterministically derive:

- normalized P;
- frozen operational local normal/differential descriptors;
- local kNN graph descriptors;
- support-view statistics;
- raster/provenance summaries;
- masks and resampling/packaging.

No source mesh, source bone ID/tail, teacher identity, hidden completion, or fixed product K.

### 2.2 Surface encoder

File:

- `geppetto_r6/surface_encoder.py`

Architecture:

- local point attention over geometric neighborhoods;
- global set/transformer context;
- permutation-safe surface representation;
- dynamic surface count with deterministic compute-budget resampling only when needed.

### 2.3 Variable-cardinality mechanical referent generator

File:

- `geppetto_r6/model.py`

Temporary autoregressive/contextual state includes:

- generated joint location state;
- uncertainty;
- selected/teacher-forced parent relation state;
- root evidence;
- structural history;
- global surface context.

At each step predict:

- continuous joint/control location distribution;
- spatial uncertainty;
- root score;
- parent score over already generated temporary referents;
- stop / unsupported probability.

After the temporary set is generated, a pairwise relation refiner emits full directed parent/edge evidence among all generated proposals.

No fixed output cardinality. Inference terminates by learned stop; only a general resource safety bound derived from current input size prevents infinite loops.

Output: anonymous `SkeletonProposalIR` only.

### 2.4 Geppetto losses / evaluator

File:

- `geppetto_r6/losses.py`

Training-only anonymous R6 teacher projection:

- deterministic BFS serialization plus sibling randomization;
- location heteroscedastic NLL / robust residual;
- parent cross entropy;
- root evidence loss;
- learned stop loss;
- full directed relation loss;
- permutation/serialization non-regression tests.

Final fit PASS requires Compiler-qualified structure, not teacher-forced loss alone.

---

## 3. SkinFieldCodec

Files:

- `arachne_r6/codec.py`
- `arachne_r6/codec_losses.py`

Teacher/training encoder:

`S context + G context + dense teacher influence field for one control -> compact latent`.

Codec:

- per-control field latent;
- factorized scalar quantization / FSQ-style straight-through quantizer;
- dynamic control count;
- no source control ID in latent semantics.

Decoder:

`S context + G context + per-control latent -> dense nonnegative field evidence`.

Decoded fields are normalized only in the differentiable training/evaluation path; Compiler remains final product simplex/reference authority.

Codec gates:

- synthetic sparse-field reconstruction;
- heterogeneous synthetic structure reconstruction;
- deformation-sensitive ceiling;
- later selected-specimen reconstruction ceiling before Arachne predictor fit.

---

## 4. Arachne R6

### 4.1 Conditioning

Files:

- `arachne_r6/conditioning.py`
- `arachne_r6/tree_encoder.py`

Input: `RiggingSurfaceIR S + QualifiedSkeletonIR G` only.

Features:

- surface contextual tokens;
- tree-aware joint/control tokens;
- current parent relations;
- point-to-joint relative geometry;
- point-to-parent-child/bone-segment distance;
- local surface differential information;
- no source bone tails/IDs/teacher columns.

### 4.2 Predictor

File:

- `arachne_r6/model.py`

Architecture:

- bidirectional surface/control cross-attention;
- tree-context message passing;
- per-control latent prediction in the same SkinFieldCodec latent space;
- predicted latent uncertainty;
- frozen/current codec decoder -> dense influence evidence;
- dynamic N and J.

Output: `SkinProposalIR` only.

### 4.3 Functional deformation training

Files:

- `arachne_r6/deformation_surrogate.py`
- `arachne_r6/losses.py`

Training/evaluator-only differentiable generic probe bank:

- bounded rotations about current qualified controls;
- descendant transform propagation;
- multiple world axes;
- both signs;
- predicted-vs-teacher dense deformation residual;
- view-projected residual over all 8 exact cameras when available;
- weight-field reconstruction and uncertainty objectives.

This surrogate shapes learned weights but never becomes canonical deformation authority. Product proof still runs the exact Compiler/historical downstream route.

---

## 5. Product / runtime integration

Files:

- `pipeline/compiler_bridge.py`
- `pipeline/historical_downstream_bridge.py`
- `pipeline/image_only_inference.py`

Requirements:

- exact current typed S/G/W/M/B lineage;
- Compiler qualification after every proposal stage;
- CanonicalPuppetGraph.v2 assembly;
- exact product-state proof binding;
- SHA-verified historical downstream authority for deformation/probe/runtime;
- no branch-local replacement auto-rigger.

---

## 6. Fitting

Files:

- `fit/data_contract.py`
- `fit/fit_iris.py`
- `fit/fit_geppetto.py`
- `fit/fit_skin_codec.py`
- `fit/fit_arachne.py`
- `fit/fit_all.py`

The same model source is usable for future generic training. The demo runner simply supplies one specimen repeatedly.

No fit stage may use a model architecture that contains the selected specimen's count, identity, topology, coordinates, weights, thresholds or output arrays.

---

## 7. Synthetic pre-specimen gates

Before specimen selection:

1. IRIS: coherent analytic multi-view synthetic object -> exact depth fit under q/reprojection architecture.
2. Geppetto: several synthetic trees with different counts -> one architecture demonstrates variable-count/stop/root/parent/loci capacity.
3. Codec: several generated sparse fields with different N/J -> reconstruction + deformation ceiling.
4. Arachne: several synthetic geometry/tree/field systems -> predicted codec-latent skin and deformation ceiling.
5. Full typed closure: synthetic image-only fixture -> S -> G -> W -> M/B -> canonical product -> exact-state proof/runtime.

These fixtures exist to test architecture capacity and shape contracts. They are not the later investor specimen.

---

## 8. Forbidden shortcuts

- fixed output joint/control count;
- oracle K at final inference;
- source skeleton/weight/mesh truth crossing final inference;
- specimen-conditioned code branches;
- direct copied output tables;
- simple demo-specific U-Net substituted for IRIS V2-A;
- direct pairwise MLP substituted for full Arachne+codec path;
- toy LBS/animation substituted for recovered exact downstream authority;
- changing source after specimen selection without invalidating READY.