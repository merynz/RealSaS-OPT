# IRIS → TESSA V2 Structuralization Plan — 2026-10-19

Status: **ACTIVE DESIGN / SOURCE IMPLEMENTATION BRANCH / NO TRAINING AUTHORIZATION**

Branch: `research/iris-vnext-hybrid-tessa-v2-20261019`

Base source: `f78b6fb647bd6e514b7fbe5e8b5026e06c9d5c5b`

## 1. Decision

TESSA V2 is not a second surface-reconstruction model and it is not merely a small IRIS head.

The preferred split is:

```text
8-view source observations
        ↓
IRIS shared continuous scene carrier H / triplanes
        ↓
high-fidelity continuous field + sparse/high-resolution surface witness
        ↓
TESSA V2 structuralization / learned retopology layer
        ↓
learned mesh candidate set
        ↓
Compiler qualification / support binding / canonical M
```

IRIS owns learned continuous scene evidence. TESSA owns learned explicit structural mesh proposals. Compiler remains the only product authority that may mint canonical M.

This is intentionally the intersection of two different research lessons rather than a forced concatenation of two papers:

- StdGEN++ / InstantMesh lineage: preserve geometric/raster supervision through representation transitions; use high-resolution active/sparse surface computation; treat thin-structure continuity as an explicit optimization problem.
- MeshAnything V2 lineage: dense/extracted surfaces are not automatically good artist/production meshes; autoregressive shape-conditioned mesh generation can learn explicit connectivity and much lower-face structural meshes; AMT improves sequence efficiency by exploiting adjacency.
- MeshRipple / 2026 mesh-generation evidence: truncated/local autoregression can break long-range topology; topological traversal order and global structural memory matter for large meshes.
- FACE / 2026 evidence: raw coordinate-token streams are not the only viable granularity; sequence compression must be treated as a first-class scalability problem.

The branch does **not** claim any of those external systems as RealSaS authority. They are architecture evidence only.

## 2. What stays from TESSA V1

Keep unless a court falsifies it:

- 1024 coordinate bins in canonical `[-0.5, 0.5]` space;
- explicit COMPONENT boundaries;
- adjacency-aware mesh serialization rather than independent-face soup;
- proposal-only authority;
- Compiler support binding and qualification after decoding;
- max policy of 32768 vertices / 32768 faces as an admission ceiling, not a demonstrated capability claim;
- fail-closed quantized-identity collision policy;
- no teacher topology, rig, skin or privileged product state in inference input.

## 3. What changes in TESSA V2

### 3.1 Role

TESSA moves from an IRIS `structure head` abstraction to a separately typed **structuralization layer**.

It consumes only model-produced / admissible IRIS evidence:

- shared scene planes;
- high-resolution or best-available SurfaceWitness XYZ;
- witness normals;
- raw local H features;
- field uncertainty;
- mechanical evidence;
- later, only explicitly admitted structural-continuity evidence.

It must not consume source/teacher topology as inference conditioning.

### 3.2 Capacity

Do not start from a reduced diagnostic model.

Current integrated IRIS topology path is only:

- width 512;
- 8 heads;
- 8 decoder layers;
- 256 memory tokens.

Standalone historical TESSA is larger at 768 / 12 heads / 12 layers / 384 memory tokens. MeshAnything V2 uses an OPT-350M-class decoder. Public MeshRipple 20k inference uses feature width 1024 and 16 heads.

TESSA V2 target class therefore starts at:

- model width: **1024**;
- attention heads: **16**;
- decoder depth: **24**;
- feed-forward width: **4096**;
- coordinate bins: **1024**;
- typed scene-memory tokens: **512**;
- typed surface/mechanics-memory tokens: **512**;
- total conditioning memory: **1024 tokens**;
- local causal context: at least **4096 tokens** plus a separate global structural-history mechanism before any >local-window scale claim;
- BF16 + activation checkpointing / memory-efficient attention expected for training.

These are research defaults, not a mobile runtime promise.

### 3.3 Remove the replacement bottleneck

The current V8 structure path concatenates scene and witness source tokens and resamples them into only 256 learned topology-memory tokens. V2 keeps scene and surface/mechanics as separately typed streams before fusion.

Required invariant:

```text
rich IRIS scene evidence ──→ scene memory ─────┐
                                               ├─→ TESSA decoder
surface + H + mechanics ───→ surface memory ──┘
```

Ablations may later reduce memory, but the first serious model must not prove capability through an arbitrarily narrow bridge.

### 3.4 Large-mesh topology context

The V1 trailing-window self-attention is not sufficient evidence for 32768-face capability.

TESSA V2 must add at least one of:

1. frontier-aware topology serialization that keeps the active neighborhood near the sequence tail;
2. persistent global structural-history memory over already generated topology;
3. a face-level/hierarchical token stream that reduces coordinate-sequence length.

The first implementation should preserve V1 AMT compatibility while adding explicit global-history support; tokenizer replacement is allowed only behind a versioned V2 grammar and A/B court.

## 4. Loss redesign

### 4.1 Topology

Token cross entropy remains necessary but is not sufficient.

The first V2 objective is:

```text
L_tessa =
    λ_token      * CE(next token)
  + λ_coord      * ordinal/metric coordinate loss
  + λ_special    * structural-token loss
```

Coordinate tokens are ordered geometry. Plain CE treats a one-bin miss and a five-hundred-bin miss as equally wrong. V2 therefore adds a metric loss on the expected coordinate value for coordinate-token targets.

Free-running decoded meshes are evaluated outside the gradient path by an independent court:

- grammar validity;
- EOS/completion validity;
- valid and degenerate faces;
- duplicate faces;
- component-count error;
- boundary loops / holes;
- manifold diagnostics where applicable;
- canonical-surface distance / Hausdorff-like worst case;
- normal consistency;
- support-binding success;
- authored-motion mechanical consequence;
- Compiler qualification rate;
- cohort worst subject.

A falling teacher-forced CE is never a topology PASS.

### 4.2 Geometry / extraction

The 24^3 differentiable V8 training court must not be confused with final spatial capability. The repository already contains an older sparse T512→T1024 zero-surface decoder. The post-V8 redesign should recover a sparse high-resolution training/evaluation path rather than raising one dense cube blindly.

Transferable StdGEN++ supervision:

- silhouette/mask;
- depth;
- normal;
- surface distance;
- FlexiCubes deviation;
- continuity/hole regularization only under source-defensible selectors.

### 4.3 Relation/order

The relation redesign follows a strict rule:

> If geometry is deterministically available, do not force a latent head to rediscover it.

Current camera features already contain normalized camera origin, right, screen-up, forward and normalized half extent. V3 relation prediction therefore receives, in addition to H(A)/H(B):

- XYZ_A / XYZ_B;
- ΔXYZ and distance;
- camera-space depth_A / depth_B / Δdepth;
- projected UV_A / UV_B / ΔUV;
- same-ray residual;
- camera embedding.

Front/order remains pairwise. Visibility is removed from the pairwise abstraction and belongs to a ray-stack/view head or deterministic raster authority, because global visibility depends on all hits on the ray.

### 4.4 No orphan authority outputs

Every inference output must satisfy one of:

- directly supervised;
- calibrated against an explicit target;
- deterministic / derived;
- typed diagnostic-only and forbidden from product authority.

Immediate consequences:

- unsupervised relation `confidence` must not gate Compiler authority;
- appearance confidence must be calibrated or removed;
- appearance descriptor must receive a real objective or become diagnostic-only;
- mechanical uncertainty must receive a heteroscedastic/calibration objective or be removed;
- authored `contact_persistence` must be retained as an eligible mechanical/support target instead of being silently dropped.

## 5. StdGEN++ vs mesh autoregression: why both can be useful

StdGEN++ demonstrates high-fidelity explicit surfaces extracted from learned fields, including high-resolution thin/hollow structures and decomposition-aware mesh training. It does not solve the same problem as artist-retopology models: its FlexiCubes output remains an isosurface extraction product whose face layout is driven by the extraction lattice.

MeshAnything V2 / EdgeRunner / MeshRipple-class models add value only if RealSaS needs a different structural mesh from that dense witness: lower face count, explicit learned connectivity, more useful topology and controlled production structure.

Therefore TESSA V2 is justified only by an A/B court:

```text
A: high-resolution IRIS/FlexiCubes witness → Compiler directly
B: same witness/evidence → TESSA V2 → Compiler
```

TESSA survives only if B improves the properties RealSaS actually needs without losing geometric/mechanical fidelity.

## 6. Scientific admission

No optimizer step is authorized by this document.

Before a TESSA V2 fit:

1. source implementation + tests green;
2. exact teacher topology corpus and free-running court materialized;
3. V8/old-TESSA baseline reproducible;
4. capacity/resource preflight on the intended accelerator;
5. thresholds preregistered;
6. decoded-mesh evaluator independent from the training loss;
7. Compiler qualification path wired but not allowed to fabricate missing truth.

## 7. Immediate implementation order

1. add TESSA V2 capacity contract and objective;
2. add explicit relation geometry conditioning V3 and remove neural confidence from the new authority API;
3. add typed dual-stream TESSA scene/surface memory;
4. add grammar-aware free-running decode and topology court;
5. restore/adapt sparse high-resolution surface witness path;
6. materialize source-defensible losses and contact-persistence truth;
7. run A/B against direct high-resolution FlexiCubes→Compiler before promoting TESSA as mandatory.
